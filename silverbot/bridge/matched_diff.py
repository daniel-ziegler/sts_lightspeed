"""Matched-seed live-vs-offline trajectory diff (phase 3 of the gap-diagnosis harness).

Aligns a live comm-capture game with the offline eval_hero --trace game played on the SAME seed
(overworld RNG is seed-faithful, so map/rewards/events must match until play diverges) and prints
a per-floor table plus the FIRST divergence with a class guess:

  ROUTE    -- different room type at the same floor (an upstream path pick differed)
  CONTENT  -- same room, different offered items (engine content-RNG infidelity; serious)
  PICK     -- same offer, hp equal, but the trajectories separate right after (policy diff)
  COMBAT   -- offers match but hp differs at the floor entry (the preceding fight went differently)

At temperature 0 everything downstream of the first true divergence cascades, so only the first
mismatch per seed carries signal; the rest of the table is context.

Usage:
  python -m silverbot.bridge.matched_diff --capture runs/comm_capture_a20h10k.jsonl \
      --trace runs/eval_a20_matched_trace.jsonl.trace.jsonl [--seed 2JI0C18TU6KA8]
"""
import argparse
import json
import re
from collections import defaultdict

from silverbot.bridge.seeds import seed_long_to_string


def norm_token(s: str) -> str:
    """Normalize a card/relic/potion/path name from either side to a comparable token:
    lowercase, spaces for underscores, upgrade counts collapsed to '+', 'x=3' -> 'x3'."""
    s = s.strip().lower().replace('_', ' ')
    s = re.sub(r'\+\d*$', '+', s)
    s = re.sub(r'^x=', 'x', s)
    return s


def live_games(capture_path):
    """Split a comm capture into per-seed decision lists (out-of-combat records only -- the
    capture holds one record per state comm.py acted on)."""
    games = defaultdict(list)
    with open(capture_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            gs = d['raw']['game_state']
            games[gs['seed']].append({
                'floor': gs['floor'],
                'screen': d['screen_type'].split('.')[-1],
                'room': gs.get('room_type', ''),
                'choices': [norm_token(c) for c in gs.get('choice_list', [])],
                'hp': gs['current_hp'],
                'max_hp': gs['max_hp'],
                'gold': gs['gold'],
                'deck': len(gs.get('deck', [])),
            })
    return games


def offline_decisions(trace_rec):
    """Flatten one eval_hero trace game into per-decision dicts shaped like the live side."""
    out = []
    for dec in trace_rec['decisions']:
        off = dec['offered']
        toks = ([norm_token(c) for c in off['cards']]
                + [norm_token(r) for r in off['relics']]
                + [norm_token(p) for p in off['potions']]
                + [f'x{x}' for x in off['paths']]
                + [norm_token(a) for a in off['fixed']])
        out.append({
            'floor': dec['floor'],
            'choices': toks,
            'picked': norm_token(dec['picked'].strip('<>')),
            'hp': dec['hp'],
            'max_hp': dec['max_hp'],
        })
    return out


def floor_hp(decs):
    """hp at each floor's first decision (floor -> hp)."""
    out = {}
    for d in decs:
        out.setdefault(d['floor'], d['hp'])
    return out


def multi(decs):
    """Only multi-choice decisions (the live capture includes single-choice screens like
    'talk'/'open'/proceed that the offline trace never records)."""
    return [d for d in decs if len(d['choices']) > 1]


def diff_seed(seed_str, live, off_rec, live_kind=None):
    off = offline_decisions(off_rec)
    l_hp, o_hp = floor_hp(live), floor_hp(off)
    l_multi = multi(live)
    o_multi = off  # trace already holds only multi-choice decisions

    print(f"\n=== seed {seed_str} ===")
    print(f"  live: {'?' if live_kind is None else live_kind} (max floor {max(l_hp) if l_hp else 0})"
          f"  |  offline: {'WIN' if off_rec['won'] == 1 else 'loss'} floor {off_rec['floor']}")

    first = None  # (floor, class, detail)
    # HP at floor entry: the earliest floor where they differ marks the preceding fight/event.
    for f in sorted(set(l_hp) & set(o_hp)):
        if first is None and l_hp[f] != o_hp[f]:
            first = (f, 'COMBAT', f"hp at floor {f}: live {l_hp[f]} vs offline {o_hp[f]}")
            break

    # Offered content: zip multi-choice decisions per floor in order.
    l_by_floor, o_by_floor = defaultdict(list), defaultdict(list)
    for d in l_multi:
        l_by_floor[d['floor']].append(d)
    for d in o_multi:
        o_by_floor[d['floor']].append(d)
    for f in sorted(set(l_by_floor) & set(o_by_floor)):
        if first is not None and f >= first[0]:
            break
        for ld, od in zip(l_by_floor[f], o_by_floor[f]):
            if sorted(ld['choices']) != sorted(od['choices']):
                cls = 'ROUTE' if set(ld['choices']) & set(od['choices']) == set() else 'CONTENT'
                first = (f, cls, f"offered at floor {f}: live {ld['choices']} vs offline {od['choices']}")
                break
        if first is not None and first[0] == f:
            break

    # Per-floor table around the divergence (or the whole game if none found).
    lo = 0 if first is None else max(0, first[0] - 3)
    hi = 10**9 if first is None else first[0] + 3
    print(f"  {'floor':>5}  {'live hp':>8}  {'off hp':>7}  live offered / offline offered+pick")
    for f in sorted(set(l_hp) | set(o_hp)):
        if not (lo <= f <= hi):
            continue
        lh = l_hp.get(f, '-')
        oh = o_hp.get(f, '-')
        lo_choices = [d['choices'] for d in l_by_floor.get(f, [])]
        of_choices = [(d['choices'], d['picked']) for d in o_by_floor.get(f, [])]
        mark = ' <== FIRST DIVERGENCE' if first is not None and f == first[0] else ''
        print(f"  {f:>5}  {lh!s:>8}  {oh!s:>7}  {lo_choices} / {of_choices}{mark}")

    if first is None:
        print("  no divergence found on shared floors "
              f"(shared up to floor {max(set(l_hp) & set(o_hp), default=0)})")
    else:
        print(f"  FIRST DIVERGENCE: floor {first[0]} [{first[1]}] {first[2]}")
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True, help='live comm capture jsonl (multi-game)')
    ap.add_argument('--trace', required=True, help='offline eval_hero .trace.jsonl')
    ap.add_argument('--results', default=None,
                    help='live grind results txt (adds live outcome kind per seed)')
    ap.add_argument('--seed', default=None, help='only this base-35 seed')
    args = ap.parse_args()

    live = live_games(args.capture)  # numeric seed -> records
    live_by_str = {seed_long_to_string(s): recs for s, recs in live.items()}

    kinds = {}
    if args.results:
        for line in open(args.results):
            m = re.search(r'seed=([0-9A-Z]+) .*kind=([A-Za-z0-9:]+)', line)
            if m:
                kinds[m.group(1)] = m.group(2)

    summary = []
    with open(args.trace) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            s_str = seed_long_to_string(rec['seed'])
            if args.seed and s_str != args.seed:
                continue
            if s_str not in live_by_str:
                print(f"seed {s_str}: in trace but not in capture, skipping")
                continue
            first = diff_seed(s_str, live_by_str[s_str], rec, kinds.get(s_str))
            summary.append((s_str, first))

    print("\n=== SUMMARY ===")
    for s_str, first in summary:
        if first is None:
            print(f"  {s_str}: no divergence detected")
        else:
            print(f"  {s_str}: floor {first[0]} [{first[1]}] {first[2]}")


if __name__ == '__main__':
    main()
