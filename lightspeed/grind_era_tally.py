"""Behavior-era-corrected tally for the a20h10k live grind.

Bridge fixes shipped MID-GRIND change agent behavior, so a raw winrate over all games mixes
different agents. Protocol (per game, on its realized trajectory):

  KEEP     the game's era already has current behavior, OR the trajectory never entered a state
           where its era's code and current code differ -- then the outcome is identically
           distributed under current code (the code computes the same search distribution at
           every state actually visited).
  DISCARD  the trajectory entered a fix-affected state (outcome came from a different agent).
  REDO     crashed (no outcome); replay the pre-committed seed under current code. Crashes are
           NOT ignorable: they correlate with depth/state, so dropping them would bias the rate.

Era boundaries (a game's era = bridge code at its comm.py launch; launch time = the previous
game's errlog-archive timestamp):

  E0  g1-g33   pre-fix
  E1  g34      5831f2f (Lagavulin wake park) only
  F1  g35-g38  + 0200440 (Surrounded facing); ASLEEP seeding still Metallicize-keyed
  F2  g39+     current (a510d70 byte-keyed seeding; c740d3e forensics is logging-only)

Fix-affected states, matched against the per-decision battle captures:
  LAGA46      a decision saw Lagavulin with live move byte 4 (STUNNED) or 6 (OPEN_NATURAL)
              [taints E0 -- the old park made the search brace for a phantom attack]
  LAGA46_MET  same, with Metallicize present live (burning elite)
              [additionally taints F1 -- the Metallicize heuristic mis-seeded ASLEEP]
  SPIRE       any Spire Shield / Spire Spear decision
              [taints E0+E1 -- facing was never restored, mis-siding the +50% back-attack]

Usage: python -m lightspeed.grind_era_tally [results_txt] [battle_capture_jsonl]
"""
import json
import re
import sys

from lightspeed.bridge.seeds import seed_string_to_long


def era_of(n: int) -> str:
    if n <= 33:
        return 'E0'
    if n == 34:
        return 'E1'
    if n <= 38:
        return 'F1'
    return 'F2'


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else 'runs/grind_a20h10k_results.txt'
    capture = sys.argv[2] if len(sys.argv) > 2 else 'runs/comm_capture_a20h10k.jsonl.battle.jsonl'

    games = []
    for line in open(results):
        m = re.match(r'g(\d+) seed=([0-9A-Z]+) asc=\d+ kind=([A-Za-z0-9:]*)', line)
        if m:
            games.append((int(m.group(1)), m.group(2), m.group(3)))
    seed_nums = {seed_string_to_long(s) for _, s, _ in games}

    marks = {}
    for line in open(capture):
        d = json.loads(line)
        gs = d.get('raw', d).get('game_state', {})
        sd = gs.get('seed')
        if sd not in seed_nums:
            continue
        for mn in gs.get('combat_state', {}).get('monsters', []):
            nm = mn['name'].replace(' ', '')
            if nm == 'Lagavulin' and mn.get('move_id') in (4, 6):
                met = any(p['id'] == 'Metallicize' for p in mn.get('powers', []))
                marks.setdefault(sd, set()).add('LAGA46_MET' if met else 'LAGA46')
            if nm in ('SpireShield', 'SpireSpear'):
                marks.setdefault(sd, set()).add('SPIRE')

    clean, redo, discard = [], [], []
    print(f"{'g':>4} {'seed':<14} {'kind':<10} era  marks -> verdict")
    for n, s, k in games:
        sd = seed_string_to_long(s)
        era = era_of(n)
        mk = marks.get(sd, set())
        if k.startswith(('CRASH', 'HANG', 'TIMEOUT')):
            v = 'REDO'
            redo.append(s)
        else:
            taint = set()
            if era == 'E0':
                taint = mk & {'LAGA46', 'LAGA46_MET', 'SPIRE'}
            elif era == 'E1':
                taint = mk & {'SPIRE'}
            elif era == 'F1':
                taint = mk & {'LAGA46_MET'}
            if taint:
                v = f"DISCARD ({'+'.join(sorted(taint))})"
                discard.append(s)
            else:
                v = 'KEEP'
                clean.append((n, s, k))
        print(f"{n:>4} {s:<14} {k:<10} {era}  {','.join(sorted(mk)) or '-':<18} {v}")

    wins = [(n, s, k) for n, s, k in clean if k.startswith(('heart', 'act3'))]
    hearts = [x for x in wins if x[2].startswith('heart')]
    print(f"\nclean: {len(clean)} | hearts {len(hearts)} | act3 {len(wins) - len(hearts)} "
          f"| losses {len(clean) - len(wins)}")
    print(f"redo ({len(redo)}): {' '.join(redo)}")
    print(f"discarded ({len(discard)}): {' '.join(discard)}")


if __name__ == '__main__':
    main()
