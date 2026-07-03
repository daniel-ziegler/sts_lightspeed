# %%
# Generate battle-outcome training data: (pre-battle state obs, encounter) -> ΔHP.
#
# Plays full games (NN policy out-of-combat, MCTS agent in battles) like
# collect_states.py. At each battle entry the GameContext is snapshotted and a set
# of variants is simulated to completion, each with freshly rerolled RNG (all battle
# randomness, env + search, derives from gc.seed + floorNum):
#   - the real encounter on the unmutated state  (--real-sims rerolls)
#   - mutated decks (remove / remove-all-copies / add / duplicate cards)
#   - alternative encounters drawn from the empirical (act, kind) pool seen so far
# Each simulated battle is one row: episode-style obs.* columns (collate_fn-
# compatible) + encounter + ΔHP labels (see battle_buckets). The canonical game then
# continues with an un-overridden playout so later floors are reached.
#
# Validation sets: --real-sims 32 --mutations 0 --alt-encounters 2 --alt-sims 32
# gives the empirical outcome distribution per (state, encounter) for
# distribution-level eval (rows group by state_id).
from __future__ import annotations

import argparse
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
from tqdm.auto import tqdm

import slaythespire as sts
from lightspeed.battle_buckets import to_bucket
from lightspeed.collect_states import load_policy_net
from lightspeed.playouts import NNService, construct_choice, pick_card_with_net

_E = sts.MonsterEncounter
ELITES = {_E.GREMLIN_NOB, _E.LAGAVULIN, _E.THREE_SENTRIES, _E.GREMLIN_LEADER, _E.SLAVERS,
          _E.BOOK_OF_STABBING, _E.GIANT_HEAD, _E.NEMESIS, _E.REPTOMANCER, _E.SHIELD_AND_SPEAR}
BOSSES = {_E.SLIME_BOSS, _E.THE_GUARDIAN, _E.HEXAGHOST, _E.AUTOMATON, _E.COLLECTOR, _E.CHAMP,
          _E.AWAKENED_ONE, _E.TIME_EATER, _E.DONU_AND_DECA, _E.THE_HEART}
EVENTS = {_E.LAGAVULIN_EVENT, _E.COLOSSEUM_EVENT_SLAVERS, _E.COLOSSEUM_EVENT_NOBS,
          _E.MASKED_BANDITS_EVENT, _E.MUSHROOMS_EVENT, _E.MYSTERIOUS_SPHERE_EVENT}

# Cards eligible for the add mutation: the engine's own obtainable pools (what card
# rewards draw from), so every entry is class-correct and battle-implemented.
ADD_POOL = [c
            for t in (sts.CardType.ATTACK, sts.CardType.SKILL, sts.CardType.POWER)
            for r in (sts.CardRarity.COMMON, sts.CardRarity.UNCOMMON, sts.CardRarity.RARE)
            for c in sts.get_card_pool(sts.CharacterClass.IRONCLAD, t, r)] \
           + list(sts.get_colorless_card_pool())

MIN_DECK = 5  # never mutate a deck below this size


def encounter_kind(enc) -> str:
    if enc in BOSSES:
        return 'boss'
    if enc in ELITES:
        return 'elite'
    if enc in EVENTS:
        return 'event'
    return 'normal'


def flatten_obs(gc) -> dict:
    """Episode-parquet-style obs.* columns from a GameContext. 2-D arrays
    (obs.map.pathXs) become nested lists: pyarrow rejects 2-D ndarray cells."""
    flat = {}
    for k, v in sts.getNNRepresentation(gc).as_dict().items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f'obs.{k}.{sk}'] = sv
        else:
            flat[f'obs.{k}'] = v
    return {k: v.tolist() if hasattr(v, 'ndim') and v.ndim >= 2 else v
            for k, v in flat.items()}


# ---- deck mutations (applied to a fresh copy; return a tag or None if not applicable) ----

def mut_remove_random(g, rng):
    n = rng.randint(1, 3)
    if len(g.deck) - n < MIN_DECK:
        return None
    for _ in range(n):
        g.remove_card(rng.randrange(len(g.deck)))
    return f'remove{n}'


def mut_remove_copies(g, rng):
    ids = sorted({int(c.id) for c in g.deck})
    rng.shuffle(ids)
    n_kinds = rng.randint(1, 2)
    removed = 0
    for cid in ids[:n_kinds]:
        idxs = [i for i, c in enumerate(g.deck) if int(c.id) == cid]
        if len(g.deck) - removed - len(idxs) < MIN_DECK:
            continue
        for i in reversed(idxs):
            g.remove_card(i)
        removed += len(idxs)
    return f'remove_copies{removed}' if removed else None


def mut_add_random(g, rng):
    n = rng.randint(1, 2)
    for _ in range(n):
        up = 1 if rng.random() < 0.25 else 0
        g.obtain_card(sts.Card(rng.choice(ADD_POOL), up))
    return f'add{n}'


def mut_duplicate(g, rng):
    n = rng.randint(1, 2)
    deck = g.deck
    for _ in range(n):
        c = deck[rng.randrange(len(deck))]
        g.obtain_card(sts.Card(c.id, c.upgrade_count))
    return f'dup{n}'


MUTATIONS = [mut_remove_random, mut_remove_copies, mut_add_random, mut_duplicate]


def make_agent(sim_count: int):
    ag = sts.Agent()
    ag.simulation_count_base = sim_count
    ag.verbosity_level = 0
    return ag


def simulate(base_gc, hp0: int, encounter, sim_count: int, sim_seed: int):
    """One battle sim off a snapshot copy with rerolled RNG. Returns (ΔHP, died)."""
    g = base_gc.copy()
    g.seed = sim_seed
    make_agent(sim_count).playout_battle(g, encounter)
    died = g.outcome == sts.GameOutcome.PLAYER_LOSS
    return g.cur_hp - hp0, died


class EncounterPools:
    """Empirical alt-encounter pools keyed by (act, kind), filled as games run."""

    def __init__(self):
        self._pools: dict[tuple[int, str], set] = {}
        self._lock = threading.Lock()

    def add(self, act: int, enc):
        kind = encounter_kind(enc)
        if kind == 'event':
            return
        with self._lock:
            self._pools.setdefault((act, kind), set()).add(enc)

    def sample_alts(self, act: int, real_enc, n: int, rng) -> list:
        kind = encounter_kind(real_enc)
        if kind == 'event':
            return []
        with self._lock:
            cands = sorted(self._pools.get((act, kind), set()) - {real_enc}, key=int)
        if len(cands) < 2:   # thin pool early on; skip rather than oversample
            return []
        return rng.sample(cands, min(n, len(cands)))


def record_battle(gc, battle_idx: int, rng, args, pools: EncounterPools) -> list[dict]:
    snap = gc.copy()
    hp0, max_hp = snap.cur_hp, snap.max_hp
    floor, act, asc = snap.floor_num, snap.act, snap.ascension
    real_enc = snap.encounter
    pools.add(act, real_enc)

    state_id = f'{snap.seed}_{battle_idx}'

    # (variant gc, encounter, mutation tag, obs row) — obs re-derived for mutated decks
    base_obs = flatten_obs(snap)
    variants = [(snap, real_enc, 'real', base_obs)] * args.real_sims

    if args.mutations > 0:
        muts = [MUTATIONS[i % len(MUTATIONS)] for i in range(args.mutations)]
        rng.shuffle(muts)
        for mut in muts:
            g = snap.copy()
            try:
                tag = mut(g, rng)
            except Exception:
                continue
            if tag is not None:
                variants.append((g, real_enc, tag, flatten_obs(g)))

    for alt in pools.sample_alts(act, real_enc, args.alt_encounters, rng):
        for _ in range(args.alt_sims):
            variants.append((snap, alt, 'alt_encounter', base_obs))

    rows = []
    for g, enc, tag, obs in variants:
        sim_seed = rng.getrandbits(48)
        try:
            delta, died = simulate(g, hp0, enc, args.mcts_simulations, sim_seed)
        except Exception:
            continue
        rows.append({
            **obs,
            'screen_state': int(sts.ScreenState.BATTLE),
            'select_screen_type': 0,
            'state_id': state_id,
            'encounter': int(enc),
            'real_encounter': int(real_enc),
            'encounter_kind': encounter_kind(enc),
            'mutation': tag,
            'hp_before': hp0,
            'max_hp': max_hp,
            'hp_delta': delta,
            'hp_frac_delta': delta / max_hp,
            'died': died,
            'bucket': to_bucket(delta, max_hp, died),
            'floor': floor,
            'act': act,
            'ascension': asc,
            'game_seed': snap.seed,
            'sim_seed': sim_seed,
        })
    return rows


def run_game(game_seed: int, service, args, pools: EncounterPools) -> list[dict]:
    rng = random.Random(game_seed ^ 0x9E3779B97F4A7C15)
    asc = rng.randint(0, args.max_ascension)
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, game_seed, asc)
    agent = make_agent(args.mcts_simulations)

    rows: list[dict] = []
    battle_idx = 0
    while gc.outcome == sts.GameOutcome.UNDECIDED:
        if gc.screen_state == sts.ScreenState.BATTLE:
            rows.extend(record_battle(gc, battle_idx, rng, args, pools))
            battle_idx += 1
            agent.playout_battle(gc)   # canonical continuation
        else:
            obs = sts.getNNRepresentation(gc)
            actions = sts.GameAction.getAllActionsInState(gc)
            choice = construct_choice(gc, obs, actions)
            if service is not None and choice is not None and len(actions) > 1:
                action, _ = pick_card_with_net(service, choice, actions,
                                               temperature=args.temperature, rng=rng)
            else:
                action = agent.pick_gameaction(gc)
            assert action.isValidAction(gc), f'invalid action (seed {game_seed})'
            action.execute(gc)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, help='policy net for out-of-combat decisions')
    ap.add_argument('--out-dir', required=True, help='directory for parquet shards')
    ap.add_argument('--num-games', type=int, default=1000)
    ap.add_argument('--start-seed', type=int, default=0)
    ap.add_argument('--max-ascension', type=int, default=20)
    ap.add_argument('--mcts-simulations', type=int, default=1000)
    ap.add_argument('--real-sims', type=int, default=2, help='rerolled sims of the real encounter')
    ap.add_argument('--mutations', type=int, default=6, help='mutated-deck variants per battle')
    ap.add_argument('--alt-encounters', type=int, default=2, help='alternative encounters per battle')
    ap.add_argument('--alt-sims', type=int, default=1, help='rerolled sims per alt encounter')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--num-threads', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--shard-rows', type=int, default=20000)
    ap.add_argument('--torch-compile', default='default')
    args = ap.parse_args()

    torch.set_float32_matmul_precision('high')
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_policy_net(args.checkpoint, torch_compile_mode=args.torch_compile)
    service = NNService(model, batch_size=args.batch_size,
                        batch_size_factor=min(min(8, args.batch_size),
                                              max(1, (args.num_threads + 1) // 2)),
                        torch_compile_mode=args.torch_compile)

    pools = EncounterPools()
    pending: list[dict] = []
    shard_idx = 0
    total_rows = 0

    def flush(force=False):
        nonlocal pending, shard_idx
        while len(pending) >= args.shard_rows or (force and pending):
            chunk, pending = pending[:args.shard_rows], pending[args.shard_rows:]
            path = os.path.join(args.out_dir, f'shard_{shard_idx:05d}.parquet')
            pd.DataFrame(chunk).to_parquet(path, engine='pyarrow')
            print(f'wrote {path} ({len(chunk)} rows)')
            shard_idx += 1

    seeds = range(args.start_seed, args.start_seed + args.num_games)
    pbar = tqdm(total=args.num_games, desc='games')
    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        futures = {ex.submit(run_game, s, service, args, pools): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f'seed {seed} failed: {type(e).__name__}: {e}')
                rows = []
            pending.extend(rows)
            total_rows += len(rows)
            pbar.update(1)
            pbar.set_postfix(rows=total_rows)
            flush()
    pbar.close()
    flush(force=True)
    service.stop()
    print(f'done: {total_rows} rows across {shard_idx} shards in {args.out_dir}')


if __name__ == '__main__':
    main()
