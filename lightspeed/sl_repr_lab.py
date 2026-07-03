"""Map-representation SL lab: which obs/choice encodings make map reasoning easy?

Trains fresh probe nets (RL-sized transformer) on real dumped path-decision states under a
matrix of (representation variant) x (objective). Objectives cover both policy imitation
(synthetic "take the elite/rest" rules) and map-understanding regression queries computed
by exact DAG dynamic programming over the observed map (min/max elites still reachable,
steps to the closest rest site).

Representation variants are built ON TOP of the production collate_fn output, so the
baseline arm is exactly the RL encoding and each variant is an additive, individually
attributable change:
  R0 base      : production encoding (path option = bare x; map node = room/is_current/pos/path_xs)
  R1 dest      : path option token gains the DESTINATION room type (collapses the
                 option->map lookup the net otherwise has to learn)
  R2 nextrow   : map nodes gain an is-next-row flag (removes the y+1 hop)
  R3 ego       : map nodes gain ego-relative (dx, dy) coords + reachable-from-here flag
  R4 dest_ego  : R1 + R3
  R5 oracle    : map nodes gain precomputed (min_elites, max_elites, dist_rest) aggregates
                 (upper bound / feature-engineering arm)

Results append to sl_repr_results.csv; one fresh net per (repr, task) cell.
"""
import argparse
import csv
import glob
import os
import time
from enum import IntEnum

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # before torch (teardown race)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import slaythespire as sts
from lightspeed import network as nw
from lightspeed.network import (
    collate_fn, obs_space, choice_space, TransformerBlock, RMSNorm, ModelHP,
    MAX_DECK_SIZE, MAX_FIXED_ACTIONS, MAX_MAP_NODES,
)
from lightspeed.inputs import (
    DictSpace, DictAddSpace, SequenceSpace, FixedVecSpace, EnumSpace, EmbedCache, ScalarSpace,
)

PATHS_OFFSET = MAX_DECK_SIZE + 3 + sts.MAX_POTION_CAPACITY + MAX_FIXED_ACTIONS
INF_DIST = 99
DIST_CAP = 15


class Flag2(IntEnum):
    NO = 0
    YES = 1


class LearnedVecEmbedding(nn.Module):
    """Sum of per-component learned lookup tables for small-integer feature vectors --
    the direct alternative to sinusoid+projection for low-cardinality coordinates."""

    def __init__(self, dim, specs):
        super().__init__()
        self.specs = specs  # list of (offset, cardinality); index = clamp(value + offset)
        self.tables = nn.ModuleList([nn.Embedding(c, dim) for _, c in specs])
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)

    def forward(self, xs):
        out = None
        for i, (off, c) in enumerate(self.specs):
            idx = (xs[..., i].long() + off).clamp(0, c - 1)
            e = self.tables[i](idx)
            out = e if out is None else out + e
        return out


class LearnedVecSpace(ScalarSpace):
    def __init__(self, specs):
        self.specs = specs
        super().__init__()

    def build_embed(self, dim, cache):
        return cache.build(self, dim, lambda: LearnedVecEmbedding(dim, self.specs))

    def sample(self, rng):
        return np.array([int(rng.integers(0, c)) - off for off, c in self.specs])


# ---- shared learned column embedding (the "sum the x embeddings" arm) -----------
# One learned table over the 7 map columns is shared by BOTH the path-choice token's x
# (single lookup) and a node's reach_via (sum of the columns that can reach it). Sharing the
# table means a node's reach_via lives in the same space as the option's own x embedding, so
# attention binds them by identity rather than having to learn a coordinate match.
NUM_COLS = 7


class _ColTable:
    """Lazy holder so the lookup and bag spaces below resolve to ONE nn.Embedding instance."""
    def __init__(self):
        self.table = None

    def get(self, dim):
        if self.table is None:
            self.table = nn.Embedding(NUM_COLS, dim)
            nn.init.normal_(self.table.weight, std=0.02)
        return self.table


class _ColLookupEmbed(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = table

    def forward(self, x):  # x: [..., 1] int column (-1 padding, masked downstream)
        return self.table(x[..., 0].long().clamp(min=0))


class _ColBagEmbed(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = table

    def forward(self, x):  # x: [..., NUM_COLS] multi-hot -> sum of the set columns' embeddings
        return x.float() @ self.table.weight


class ColLookupSpace(ScalarSpace):
    def __init__(self, holder):
        self.holder = holder
        super().__init__()

    def build_embed(self, dim, cache):
        return _ColLookupEmbed(self.holder.get(dim))

    def sample(self, rng):
        return np.array([int(rng.integers(0, NUM_COLS))])


class ColBagSpace(ScalarSpace):
    def __init__(self, holder):
        self.holder = holder
        super().__init__()

    def build_embed(self, dim, cache):
        return _ColBagEmbed(self.holder.get(dim))

    def sample(self, rng):
        return np.array([int(rng.integers(0, 2)) for _ in range(NUM_COLS)])


# ---------------------------------------------------------------- map ground truth

def map_info(r):
    """Exact DAG info for one row: per-node aggregates + per-offered-option destinations.

    Returns None if any offered option's destination is unresolvable (boss transitions).
    """
    xs = [int(v) for v in r['obs.map.xs']]
    ys = [int(v) for v in r['obs.map.ys']]
    rts = [int(v) for v in r['obs.map.roomTypes']]
    pxs = r['obs.map.pathXs']
    idx = {(x, y): i for i, (x, y) in enumerate(zip(xs, ys))}
    n = len(xs)
    succ = [[] for _ in range(n)]
    for i in range(n):
        for e in pxs[i]:
            e = int(e)
            j = idx.get((e, ys[i] + 1))
            if e >= 0 and j is not None:
                succ[i].append(j)
    ELITE, REST = int(sts.Room.ELITE), int(sts.Room.REST)
    minE, maxE, dR = [0] * n, [0] * n, [INF_DIST] * n
    for i in sorted(range(n), key=lambda i: -ys[i]):
        e = 1 if rts[i] == ELITE else 0
        minE[i] = e + (min(minE[j] for j in succ[i]) if succ[i] else 0)
        maxE[i] = e + (max(maxE[j] for j in succ[i]) if succ[i] else 0)
        if rts[i] == REST:
            dR[i] = 0
        elif succ[i]:
            dR[i] = min(INF_DIST, 1 + min(dR[j] for j in succ[i]))

    ydest = int(r['obs.mapY']) + 1
    opts = [idx.get((int(x), ydest)) for x in r['paths_offered']]
    if None in opts or len(opts) < 2:
        return None
    # forward reachability from the offered frontier (binary) AND per-column: reach_col[j] is a
    # bitmask over the frontier COLUMN (map x, 0..6) of each option that can forward-reach node j.
    # Frontier options occupy distinct columns, so column indexing is unambiguous and aligns with
    # the path-choice token's own x coordinate.
    reach = [False] * n
    reach_col = [0] * n
    for o in opts:
        c = xs[o]
        seen, stack = set(), [o]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(succ[i])
        for i in seen:
            reach[i] = True
            reach_col[i] |= (1 << c)

    # Burning-elite node (emerald key), if its position was recorded in this obs (older dumps
    # predate the field -> None). reaches_burn[o] tells whether option o's cone contains it.
    bx, by = int(r.get('obs.map.burningEliteX', -1)), int(r.get('obs.map.burningEliteY', -1))
    burn = idx.get((bx, by)) if bx >= 0 else None
    return dict(
        xs=xs, ys=ys, rts=rts, n=n, opts=opts, ydest=ydest, succ=succ,
        minE=minE, maxE=maxE, dR=dR, reach=reach, reach_col=reach_col, burn=burn,
        cur=(int(r['obs.mapX']), int(r['obs.mapY'])),
    )


# ---------------------------------------------------------------- representations

def _map_element_space(extra: dict, drop=()):
    base = {
        'room': EnumSpace(sts.Room),
        'is_current': EnumSpace(nw.IsCurrentNode),
        'pos': FixedVecSpace([7, 16]),
        'path_xs': FixedVecSpace([7, 7, 7]),
    }
    for k in drop:
        del base[k]
    base.update(extra)
    return SequenceSpace(DictAddSpace(base))


def _obs_space_with_map(extra: dict, drop=()):
    return DictSpace({**obs_space.spaces, 'map_nodes': _map_element_space(extra, drop)})


def _choice_space_with_paths(paths_elem):
    return DictSpace({**choice_space.spaces, 'paths': SequenceSpace(paths_elem)})


_PATHS_DEST = DictAddSpace({'x': FixedVecSpace([7]), 'room': EnumSpace(sts.Room)})


# Composable augmentation pieces for the reach_via / per-choice arms (each mutates the collated
# batch in place; order matters only in that _aug_path_cone needs _aug_paths_room run first to
# turn the bare path-x tensor into the {'x', 'room', ...} dict it extends).
def _aug_paths_room(bt, infos):
    x = bt['choices']['paths']['value']
    if isinstance(x, dict):
        x = x['x']
    room = torch.zeros(x.shape[0], x.shape[1], dtype=torch.int32)
    for i, mi in enumerate(infos):
        for k, o in enumerate(mi['opts'][:x.shape[1]]):
            room[i, k] = mi['rts'][o]
    bt['choices']['paths']['value'] = {'x': x, 'room': room}


def _aug_node_rel(bt, infos):
    rel = torch.zeros(len(infos), MAX_MAP_NODES, 2, dtype=torch.int32)
    for i, mi in enumerate(infos):
        cx, cy = mi['cur']
        for j in range(mi['n']):
            rel[i, j, 0] = mi['xs'][j] - cx
            rel[i, j, 1] = mi['ys'][j] - cy
    bt['map_nodes']['value']['rel'] = rel


def _aug_node_reachable(bt, infos):
    reach = torch.zeros(len(infos), MAX_MAP_NODES, dtype=torch.int32)
    for i, mi in enumerate(infos):
        for j in range(mi['n']):
            reach[i, j] = 1 if mi['reach'][j] else 0
    bt['map_nodes']['value']['reachable'] = reach


def _aug_node_reach_via(bt, infos):
    rv = torch.zeros(len(infos), MAX_MAP_NODES, NUM_COLS, dtype=torch.float32)
    for i, mi in enumerate(infos):
        for j in range(mi['n']):
            m = mi['reach_col'][j]
            for c in range(NUM_COLS):
                if (m >> c) & 1:
                    rv[i, j, c] = 1.0
    bt['map_nodes']['value']['reach_via'] = rv


def _aug_node_agg(bt, infos):
    agg = torch.zeros(len(infos), MAX_MAP_NODES, 3, dtype=torch.float32)
    for i, mi in enumerate(infos):
        for j in range(mi['n']):
            agg[i, j, 0] = mi['minE'][j] / DIST_CAP
            agg[i, j, 1] = mi['maxE'][j] / DIST_CAP
            agg[i, j, 2] = min(mi['dR'][j], DIST_CAP) / DIST_CAP
    bt['map_nodes']['value']['agg'] = agg


def _aug_path_cone(bt, infos):
    """Forward-cone summary on each path-choice token: the frontier node's (minE, maxE,
    dist_rest) aggregates (scaled) plus a 'reaches_burn' bit (does this option's cone contain
    the burning elite). Precomputes the lookahead the routing otherwise has to attend out."""
    pv = bt['choices']['paths']['value']
    W = pv['x'].shape[1]
    cone = torch.zeros(len(infos), W, 3, dtype=torch.float32)
    rb = torch.zeros(len(infos), W, dtype=torch.int32)
    for i, mi in enumerate(infos):
        for k, o in enumerate(mi['opts'][:W]):
            cone[i, k, 0] = mi['minE'][o] / DIST_CAP
            cone[i, k, 1] = mi['maxE'][o] / DIST_CAP
            cone[i, k, 2] = min(mi['dR'][o], DIST_CAP) / DIST_CAP
            if mi['burn'] is not None and (mi['reach_col'][mi['burn']] >> mi['xs'][o]) & 1:
                rb[i, k] = 1
    pv['cone'] = cone
    pv['reaches_burn'] = rb


_PATHS_CONE = DictAddSpace({'x': FixedVecSpace([7]), 'room': EnumSpace(sts.Room),
                            'cone': FixedVecSpace([2, 2, 2]), 'reaches_burn': EnumSpace(Flag2)})


def build_repr(name):
    """-> (obs_space_v, choice_space_v, augment(bt, infos) -> None)."""
    if name == 'R0':
        return obs_space, choice_space, lambda bt, infos: None

    if name == 'R1':
        def aug(bt, infos):
            x = bt['choices']['paths']['value']  # [B, 7, 1]
            room = torch.zeros(x.shape[0], x.shape[1], dtype=torch.int32)
            for i, mi in enumerate(infos):
                for k, o in enumerate(mi['opts']):
                    room[i, k] = mi['rts'][o]
            bt['choices']['paths']['value'] = {'x': x, 'room': room}
        return obs_space, _choice_space_with_paths(_PATHS_DEST), aug

    if name == 'R2':
        def aug(bt, infos):
            v = torch.zeros(len(infos), MAX_MAP_NODES, dtype=torch.int32)
            for i, mi in enumerate(infos):
                for j in range(mi['n']):
                    v[i, j] = 1 if mi['ys'][j] == mi['ydest'] else 0
            bt['map_nodes']['value']['is_next'] = v
        return _obs_space_with_map({'is_next': EnumSpace(Flag2)}), choice_space, aug

    if name == 'R3':
        def aug(bt, infos):
            rel = torch.zeros(len(infos), MAX_MAP_NODES, 2, dtype=torch.int32)
            reach = torch.zeros(len(infos), MAX_MAP_NODES, dtype=torch.int32)
            for i, mi in enumerate(infos):
                cx, cy = mi['cur']
                for j in range(mi['n']):
                    rel[i, j, 0] = mi['xs'][j] - cx
                    rel[i, j, 1] = mi['ys'][j] - cy
                    reach[i, j] = 1 if mi['reach'][j] else 0
            bt['map_nodes']['value']['rel'] = rel
            bt['map_nodes']['value']['reachable'] = reach
        return (_obs_space_with_map({'rel': FixedVecSpace([15, 31]), 'reachable': EnumSpace(Flag2)}),
                choice_space, aug)

    if name == 'R4':  # R1 + R3
        o_sp, _, aug3 = build_repr('R3')
        _, c_sp, aug1 = build_repr('R1')
        def aug(bt, infos):
            aug1(bt, infos); aug3(bt, infos)
        return o_sp, c_sp, aug

    if name == 'R5':
        def aug(bt, infos):
            agg = torch.zeros(len(infos), MAX_MAP_NODES, 3, dtype=torch.int32)
            for i, mi in enumerate(infos):
                for j in range(mi['n']):
                    agg[i, j, 0] = mi['minE'][j]
                    agg[i, j, 1] = mi['maxE'][j]
                    agg[i, j, 2] = min(mi['dR'][j], DIST_CAP)
            bt['map_nodes']['value']['agg'] = agg
        return _obs_space_with_map({'agg': FixedVecSpace([8, 8, 16])}), choice_space, aug

    if name == 'R5b':  # R4 + aggregates scaled to [0,1] (avoid drowning the embedding sum)
        o_sp4, c_sp4, aug4 = build_repr('R4')
        o_sp = DictSpace({**o_sp4.spaces, 'map_nodes': _map_element_space({
            'rel': FixedVecSpace([15, 31]), 'reachable': EnumSpace(Flag2),
            'agg': FixedVecSpace([2, 2, 2])})})
        def aug(bt, infos):
            aug4(bt, infos)
            agg = torch.zeros(len(infos), MAX_MAP_NODES, 3, dtype=torch.float32)
            for i, mi in enumerate(infos):
                for j in range(mi['n']):
                    agg[i, j, 0] = mi['minE'][j] / DIST_CAP
                    agg[i, j, 1] = mi['maxE'][j] / DIST_CAP
                    agg[i, j, 2] = min(mi['dR'][j], DIST_CAP) / DIST_CAP
            bt['map_nodes']['value']['agg'] = agg
        return o_sp, c_sp4, aug

    if name == 'R6':  # R4 without path_xs (is the raw edge encoding still needed?)
        _, c_sp4, aug4 = build_repr('R4')
        o_sp = _obs_space_with_map({'rel': FixedVecSpace([15, 31]), 'reachable': EnumSpace(Flag2)},
                                   drop=('path_xs',))
        def aug(bt, infos):
            aug4(bt, infos)
            del bt['map_nodes']['value']['path_xs']
        return o_sp, c_sp4, aug

    # ---- choice-dependent reachability arms (all R5b + a routing-signal change) ----
    # Baseline for these is R5b (the production encoding). reach_via replaces R5b's binary
    # `reachable` with a per-frontier-column multi-hot; PC adds a forward-cone summary to the
    # path-choice token. RV = multi-hot (sinusoidal FixedVec); RVe = the same set as a SUM of a
    # learned per-column embedding table SHARED with the path token's x (identity binding).
    _RV_MAP = {'rel': FixedVecSpace([15, 31]), 'reach_via': FixedVecSpace([2] * NUM_COLS),
               'agg': FixedVecSpace([2, 2, 2])}
    _BASE_MAP = {'rel': FixedVecSpace([15, 31]), 'reachable': EnumSpace(Flag2),
                 'agg': FixedVecSpace([2, 2, 2])}

    if name == 'Rbase':  # production R5b features, rebuilt via the robust aug helpers so the
        # only delta vs RV is reachable->reach_via and vs PC is the absent cone. The apples-to-
        # apples baseline (the lab's stale R1-R6 augs assume a pre-R5b bare-tensor paths value).
        def aug(bt, infos):
            _aug_paths_room(bt, infos); _aug_node_rel(bt, infos)
            _aug_node_reachable(bt, infos); _aug_node_agg(bt, infos)
        return _obs_space_with_map(_BASE_MAP), _choice_space_with_paths(_PATHS_DEST), aug

    if name == 'RV':
        def aug(bt, infos):
            _aug_paths_room(bt, infos); _aug_node_rel(bt, infos)
            _aug_node_reach_via(bt, infos); _aug_node_agg(bt, infos)
        return _obs_space_with_map(_RV_MAP), _choice_space_with_paths(_PATHS_DEST), aug

    if name == 'RVe':
        holder = _ColTable()
        def aug(bt, infos):
            _aug_paths_room(bt, infos); _aug_node_rel(bt, infos)
            _aug_node_reach_via(bt, infos); _aug_node_agg(bt, infos)
        o_sp = _obs_space_with_map({'rel': FixedVecSpace([15, 31]), 'reach_via': ColBagSpace(holder),
                                    'agg': FixedVecSpace([2, 2, 2])})
        c_sp = _choice_space_with_paths(DictAddSpace({'x': ColLookupSpace(holder),
                                                      'room': EnumSpace(sts.Room)}))
        return o_sp, c_sp, aug

    if name == 'PC':
        def aug(bt, infos):
            _aug_paths_room(bt, infos); _aug_node_rel(bt, infos)
            _aug_node_reachable(bt, infos); _aug_node_agg(bt, infos); _aug_path_cone(bt, infos)
        o_sp = _obs_space_with_map({'rel': FixedVecSpace([15, 31]), 'reachable': EnumSpace(Flag2),
                                    'agg': FixedVecSpace([2, 2, 2])})
        return o_sp, _choice_space_with_paths(_PATHS_CONE), aug

    if name == 'RV+PC':
        def aug(bt, infos):
            _aug_paths_room(bt, infos); _aug_node_rel(bt, infos)
            _aug_node_reach_via(bt, infos); _aug_node_agg(bt, infos); _aug_path_cone(bt, infos)
        return _obs_space_with_map(_RV_MAP), _choice_space_with_paths(_PATHS_CONE), aug

    if name == 'RVe+PC':
        holder = _ColTable()
        def aug(bt, infos):
            _aug_paths_room(bt, infos); _aug_node_rel(bt, infos)
            _aug_node_reach_via(bt, infos); _aug_node_agg(bt, infos); _aug_path_cone(bt, infos)
        o_sp = _obs_space_with_map({'rel': FixedVecSpace([15, 31]), 'reach_via': ColBagSpace(holder),
                                    'agg': FixedVecSpace([2, 2, 2])})
        c_sp = _choice_space_with_paths(DictAddSpace({
            'x': ColLookupSpace(holder), 'room': EnumSpace(sts.Room),
            'cone': FixedVecSpace([2, 2, 2]), 'reaches_burn': EnumSpace(Flag2)}))
        return o_sp, c_sp, aug

    # Learned-table arms: same tensors/structure, lookup embeddings instead of sinusoids
    # for the small-integer coordinate features.
    _POS_L = LearnedVecSpace([(0, 7), (0, 16)])
    _PXS_L = LearnedVecSpace([(1, 8), (1, 8), (1, 8)])      # path_xs in -1..6
    _REL_L = LearnedVecSpace([(6, 13), (15, 31)])           # dx in -6..6, dy in -15..15
    _X_L = LearnedVecSpace([(0, 7)])

    if name == 'R7':  # baseline structure, learned embeddings only
        o_sp = _obs_space_with_map({'pos': _POS_L, 'path_xs': _PXS_L}, drop=('pos', 'path_xs'))
        c_sp = _choice_space_with_paths(_X_L)
        return o_sp, c_sp, lambda bt, infos: None

    if name == 'R4L':  # R4 features, learned embeddings
        _, _, aug4 = build_repr('R4')
        o_sp = _obs_space_with_map({'pos': _POS_L, 'path_xs': _PXS_L, 'rel': _REL_L,
                                    'reachable': EnumSpace(Flag2)}, drop=('pos', 'path_xs'))
        c_sp = _choice_space_with_paths(
            DictAddSpace({'x': _X_L, 'room': EnumSpace(sts.Room)}))
        return o_sp, c_sp, aug4

    if name == 'R5L':  # R4L + aggregates as learned tables (vs R5 sinusoid / R5b scaled)
        o_sp4l, c_sp4l, aug4 = build_repr('R4L')
        o_sp = _obs_space_with_map({'pos': _POS_L, 'path_xs': _PXS_L, 'rel': _REL_L,
                                    'reachable': EnumSpace(Flag2),
                                    'agg': LearnedVecSpace([(0, 8), (0, 8), (0, 16)])},
                                   drop=('pos', 'path_xs'))
        def aug(bt, infos):
            aug4(bt, infos)
            agg = torch.zeros(len(infos), MAX_MAP_NODES, 3, dtype=torch.int32)
            for i, mi in enumerate(infos):
                for j in range(mi['n']):
                    agg[i, j, 0] = mi['minE'][j]
                    agg[i, j, 1] = mi['maxE'][j]
                    agg[i, j, 2] = min(mi['dR'][j], DIST_CAP)
            bt['map_nodes']['value']['agg'] = agg
        return o_sp, c_sp4l, aug

    raise ValueError(name)


# ---------------------------------------------------------------- tasks

def task_spec(name):
    """-> (kind, fn(row, mi) -> label or None). CE labels are option indices k;
    REG labels are floats."""
    ELITE, REST, MONSTER = int(sts.Room.ELITE), int(sts.Room.REST), int(sts.Room.MONSTER)

    def take(target):
        def f(r, mi):
            tn = [mi['rts'][o] for o in mi['opts']]
            cand = [k for k, t in enumerate(tn) if t == target]
            if not cand or len(cand) == len(tn):
                return None
            xs = [int(x) for x in r['paths_offered']]
            return min(cand, key=lambda k: xs[k])
        return f

    if name == 'elite':
        return 'ce', take(ELITE)
    if name == 'rest':
        return 'ce', take(REST)
    if name == 'monster':
        return 'ce', take(MONSTER)
    if name == 'min_elites':
        return 'reg', lambda r, mi: float(min(mi['minE'][o] for o in mi['opts']))
    if name == 'max_elites':
        return 'reg', lambda r, mi: float(max(mi['maxE'][o] for o in mi['opts']))
    if name == 'dist_rest':
        return 'reg', lambda r, mi: float(min(DIST_CAP, 1 + min(mi['dR'][o] for o in mi['opts'])))

    def route(score_fn):
        """CE: pick the option with the lowest score (lowest x breaks ties); all-tie rows excluded."""
        def f(r, mi):
            scores = [score_fn(mi, o) for o in mi['opts']]
            if len(set(scores)) < 2:
                return None
            best = min(scores)
            cand = [k for k, s in enumerate(scores) if s == best]
            xs = [int(x) for x in r['paths_offered']]
            return min(cand, key=lambda k: xs[k])
        return f

    if name == 'route_rest':
        return 'ce', route(lambda mi, o: min(DIST_CAP, 1 + mi['dR'][o]))
    if name == 'route_avoid_elite':
        return 'ce', route(lambda mi, o: mi['minE'][o])
    if name == 'elites_within_3':
        ELITE3 = int(sts.Room.ELITE)
        def f(r, mi):
            seen, frontier = set(mi['opts']), set(mi['opts'])
            for _ in range(2):  # opts are step 1; expand to step 3
                frontier = {j for i in frontier for j in mi['succ'][i]} - seen
                seen |= frontier
            return float(sum(1 for i in seen if mi['rts'][i] == ELITE3))
        return 'reg', f

    # Multi-hop routing: the correct option is distinguished by what's reachable BEYOND its
    # immediate destination -- exactly the lookahead one-hop 'elite'/'rest' tasks don't exercise
    # and the binary `reachable` bit collapses. These are the burning-elite-routing analogues.
    if name == 'route_deep_elite':
        def f(r, mi):
            # deep[o] = is an elite reachable from option o at depth >= 2 (not o itself)?
            deep = [1 if max((mi['maxE'][j] for j in mi['succ'][o]), default=0) >= 1 else 0
                    for o in mi['opts']]
            if sum(deep) in (0, len(deep)):
                return None  # need it present AND distinguishable
            xs = [int(x) for x in r['paths_offered']]
            return min((k for k, d in enumerate(deep) if d), key=lambda k: xs[k])
        return 'ce', f
    if name == 'route_burning':
        def f(r, mi):
            if mi['burn'] is None:
                return None
            rc = mi['reach_col'][mi['burn']]
            reaches = [1 if (rc >> mi['xs'][o]) & 1 else 0 for o in mi['opts']]
            if sum(reaches) in (0, len(reaches)):
                return None
            xs = [int(x) for x in r['paths_offered']]
            return min((k for k, b in enumerate(reaches) if b), key=lambda k: xs[k])
        return 'ce', f
    raise ValueError(name)


# ---------------------------------------------------------------- probe net

class ProbeNN(nn.Module):
    """NN-architecture probe parameterized by spaces; CE head over choice tokens and a
    masked-mean-pooled regression head."""

    def __init__(self, o_sp, c_sp, dim, n_layers):
        super().__init__()
        H = ModelHP(use_value_head=False, dim=dim, n_layers=n_layers)
        cache = EmbedCache()
        self.obs_embed = o_sp.build_embed(H.dim, cache)
        self.choice_embed = c_sp.build_embed(H.dim, cache)
        self.layers = nn.ModuleList([TransformerBlock(H=H) for _ in range(H.n_layers)])
        self.norm = RMSNorm(H.dim, H.norm_eps)
        self.choice_logits = nn.Linear(H.dim, 1, bias=True)
        nn.init.uniform_(self.choice_logits.weight, -0.01, 0.01)
        nn.init.zeros_(self.choice_logits.bias)
        self.reg_head = nn.Linear(H.dim, 1, bias=True)
        # Auxiliary heads: per-path-option destination room classifier (grounding circuit)
        # and pooled (dist_rest, max_elites) regressors (aggregation circuit).
        self.aux_room = nn.Linear(H.dim, len(sts.Room), bias=True)
        self.aux_reg = nn.Linear(H.dim, 2, bias=True)

    def forward(self, batch):
        choices = batch['choices']
        obs_e, obs_m = self.obs_embed(batch)
        ch_e, ch_m = self.choice_embed(choices)
        x = torch.cat([obs_e, ch_e], dim=1)
        m = torch.cat([obs_m, ch_m], dim=1)
        for l in self.layers:
            x = l(x, m)
        xn = self.norm(x)
        action_xs = xn[:, obs_m.size(1):, :]
        logits = self.choice_logits(action_xs).squeeze(-1).float()
        logits = logits.masked_fill(ch_m, float('-inf'))
        valid = (~m).unsqueeze(-1).float()
        pooled = (xn * valid).sum(1) / valid.sum(1).clamp(min=1)
        reg = self.reg_head(pooled).squeeze(-1).float()
        path_tokens = action_xs[:, PATHS_OFFSET:PATHS_OFFSET + 7, :]
        aux_room = self.aux_room(path_tokens).float()       # [B, 7, n_rooms]
        aux_reg = self.aux_reg(pooled).float()              # [B, 2]
        return logits, reg, aux_room, aux_reg


# ---------------------------------------------------------------- protocol

def index_batch(bt, idx):
    if isinstance(bt, dict):
        return {k: index_batch(v, idx) for k, v in bt.items()}
    return bt[idx]


def to_device(bt, device):
    if isinstance(bt, dict):
        return {k: to_device(v, device) for k, v in bt.items()}
    return bt.to(device)


def is_valid_seed(seed, frac=0.15):
    return (((int(seed) * 1327217885) & 0xFFFFFFFF) / 0xFFFFFFFF) < frac


def aux_targets(rows, infos, device):
    """Self-supervised aux labels: per-option destination room (-100 = unoffered slot) and
    state-level (dist_rest, max_elites). All computable from the obs alone (free in RL)."""
    n = len(rows)
    room = torch.full((n, 7), -100, dtype=torch.long)
    reg = torch.zeros(n, 2)
    for i, mi in enumerate(infos):
        for k, o in enumerate(mi['opts'][:7]):
            room[i, k] = mi['rts'][o]
        reg[i, 0] = min(DIST_CAP, 1 + min(mi['dR'][o] for o in mi['opts']))
        reg[i, 1] = max(mi['maxE'][o] for o in mi['opts'])
    return room, reg


def run_cell(repr_name, task_name, rows, infos, bt, args, device, writer, fout):
    kind, fn = task_spec(task_name)
    aux = [a for a in args.aux.split(',') if a]
    aux_room_y, aux_reg_y = (aux_targets(rows, infos, device) if aux else (None, None))
    labels, keep = [], []
    for i, (r, mi) in enumerate(zip(rows, infos)):
        lab = fn(r, mi)
        if lab is None:
            continue
        keep.append(i)
        labels.append(PATHS_OFFSET + lab if kind == 'ce' else lab)
    keep = np.array(keep)
    y = torch.tensor(labels, dtype=torch.long if kind == 'ce' else torch.float32)
    val = np.array([is_valid_seed(rows[i]['seed']) for i in keep])
    tr = torch.tensor(keep[~val]); va = torch.tensor(keep[val])
    if args.data_frac < 1.0:  # low-data regime: subsample TRAIN only (valid stays full)
        g0 = torch.Generator().manual_seed(12345)
        tr = tr[torch.randperm(len(tr), generator=g0)[:max(1, int(len(tr) * args.data_frac))]]
    y_by_row = torch.zeros(len(rows), dtype=y.dtype)
    y_by_row[torch.tensor(keep)] = y

    if kind == 'ce':
        nopt = np.array([len(rows[i]['paths_offered']) for i in keep[val]])
        baseline = float(np.mean(1.0 / nopt))  # chance
        epochs = args.epochs_ce
    else:
        mu = y_by_row[tr].mean().item()
        baseline = (y_by_row[va] - mu).abs().mean().item()  # predict-train-mean MAE
        epochs = args.epochs_reg

    torch.manual_seed(args.torch_seed)
    o_sp, c_sp, _ = build_repr(repr_name)
    net = ProbeNN(o_sp, c_sp, args.dim, args.n_layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, (len(tr) + args.batch_size - 1) // args.batch_size) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                                                pct_start=0.1)

    def evaluate():
        hits = se = ae = tot = 0
        with torch.no_grad():
            for i in range(0, len(va), args.batch_size):
                idx = va[i:i + args.batch_size]
                logits, reg, _, _ = net(to_device(index_batch(bt, idx), device))
                yy = y_by_row[idx].to(device)
                if kind == 'ce':
                    hits += (logits.argmax(-1) == yy).sum().item()
                else:
                    ae += (reg - yy).abs().sum().item()
                    se += ((reg - yy) ** 2).sum().item()
                tot += len(idx)
        return (hits / tot, None) if kind == 'ce' else (ae / tot, se / tot)

    g = torch.Generator().manual_seed(args.torch_seed)
    print(f"\n=== {repr_name} x {task_name} [{kind}]: {len(tr)} train / {len(va)} valid, "
          f"baseline {baseline:.4f} ===", flush=True)
    t0 = time.time()
    best = None
    for ep in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits, reg, a_room, a_reg = net(to_device(index_batch(bt, idx), device))
            yy = y_by_row[idx].to(device)
            loss = F.cross_entropy(logits, yy) if kind == 'ce' else F.mse_loss(reg, yy)
            if 'dest_room' in aux:
                ar = aux_room_y[idx].to(device)
                loss = loss + args.aux_weight * F.cross_entropy(
                    a_room.reshape(-1, a_room.shape[-1]), ar.reshape(-1), ignore_index=-100)
            if 'queries' in aux:
                loss = loss + args.aux_weight * F.mse_loss(a_reg, aux_reg_y[idx].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
        net.eval()
        m1, m2 = evaluate()
        best = m1 if best is None else (max(best, m1) if kind == 'ce' else min(best, m1))
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1:
            tag = f"acc {m1:.4f}" if kind == 'ce' else f"MAE {m1:.4f} MSE {m2:.4f}"
            print(f"[{repr_name}/{task_name}] epoch {ep + 1}/{epochs}: {tag}", flush=True)
    wall = time.time() - t0
    print(f"[{repr_name}/{task_name}] FINAL {m1:.4f} (best {best:.4f}, baseline {baseline:.4f}, "
          f"{wall:.0f}s)", flush=True)
    writer.writerow([repr_name, task_name, kind, len(tr), len(va), epochs,
                     round(baseline, 4), round(m1, 4), round(best, 4), round(wall),
                     args.torch_seed, args.data_frac, args.aux])
    fout.flush()
    del net
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes-glob', default='runs/ppo_hient.pt.episodes/*.parquet')
    ap.add_argument('--max-files', type=int, default=12)
    ap.add_argument('--reprs', default='R0,R1,R2,R3')
    ap.add_argument('--tasks', default='elite,max_elites,dist_rest,min_elites')
    ap.add_argument('--epochs-ce', type=int, default=60)
    ap.add_argument('--epochs-reg', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--out-csv', default='sl_repr_results.csv')
    ap.add_argument('--torch-seed', type=int, default=0)
    ap.add_argument('--data-frac', type=float, default=1.0)
    ap.add_argument('--aux', default='', help="comma list of {dest_room, queries}")
    ap.add_argument('--aux-weight', type=float, default=0.3)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    files = sorted(glob.glob(args.episodes_glob))[:args.max_files]
    print(f"{len(files)} episode files on {device}", flush=True)

    rows, infos = [], []
    for f in files:
        df = pd.read_parquet(f)
        for _, r in df[df.choice_type == 3].iterrows():
            mi = map_info(r)
            if mi is not None:
                rows.append(r.to_dict()); infos.append(mi)
    print(f"{len(rows)} usable path decisions", flush=True)

    exists = os.path.exists(args.out_csv)
    fout = open(args.out_csv, 'a', newline='')
    writer = csv.writer(fout)
    if not exists:
        writer.writerow(['repr', 'task', 'kind', 'n_train', 'n_valid', 'epochs',
                         'baseline', 'final', 'best', 'wall_s', 'torch_seed', 'data_frac', 'aux'])

    import copy
    bt_base = collate_fn(rows)  # collate once; each repr augments a deep copy
    for repr_name in args.reprs.split(','):
        _, _, aug = build_repr(repr_name)
        bt = copy.deepcopy(bt_base)
        aug(bt, infos)
        for task_name in args.tasks.split(','):
            run_cell(repr_name, task_name, rows, infos, bt, args, device, writer, fout)
        del bt

    fout.close()


if __name__ == '__main__':
    main()
