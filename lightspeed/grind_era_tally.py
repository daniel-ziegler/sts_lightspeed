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

SELECTION WARNING: pre-fix KEEPs are samples of the distribution CONDITIONAL on never entering
an affected state -- an event anti-correlated with winning (heart kills must fight the Spire
elites, so deep pre-fix runs are tainted by construction). Do NOT pool them with current-era
games or compare their win rate to an unconditional benchmark. "Keep-if-unaffected else redo"
is also biased (the redo draw is unconditional; the retention event is not). The unconditional
estimators are: current-era games only, or redo ALL pre-fix seeds ignoring their old outcomes.
The tally below therefore reports current-era games as the headline and pre-fix keeps as a
separate conditional stratum.

Era boundaries (a game's era = bridge code at its comm.py launch; launch time = the previous
game's errlog-archive timestamp):

  E0  g1-g33   pre-fix
  E1  g34      5831f2f (Lagavulin wake park) only
  F1  g35-g38  + 0200440 (Surrounded facing); ASLEEP seeding still Metallicize-keyed
  F2  g39-g43  + a510d70 (byte-keyed ASLEEP seeding; c740d3e forensics is logging-only)
  F3  g44-g100 effectiveGold Ectoplasm fix -- thief-held stolen gold is unrecoverable
               under Ectoplasm, so the search no longer pays a phantom bonus to kill the thief;
               the original g44 was killed mid-game for this fix and replays under F3
  F4  (redo g5+) e9f82d8 reconcile keeps live-observed uniquePower0/1 (Time Warp counter et al);
               transplant only Hexaghost's hidden sequence counter. No main-grind games ran F4;
               redo-run eras: g1-g4 F3-code (kept iff no TE fight), g5+ F4.
  F5  (redo g23+) fec405c Perfected Strike duplicate plays keep the full strike count -- the
               autoplay -1 (Havoc/Mayhem-only in live) was also applied to Necronomicon/Double
               Tap purge-duplicates, under-dealing the dup hit by one strike bonus. No main-grind
               or redo g1-g22 games ran F5 (g22 launched 08:43, .so landed 08:55).
  F6  (redo g25+) dd0bc8e Runic Dome fights fixed -- conversion parked no current move under a
               hidden intent, so firstTurn() made every deferred roll re-issue the fight OPENER
               in every search sim (and the pbc advance re-executed it: the redo-g23 phantom
               PLAYER_LOSS crash); pbc END_TURN now defers and replays observed moves. No
               main-grind or redo g1-g24 games ran F6 (g24 launched 09:27, fix landed ~10:05).

Fix-affected states, matched against the per-decision battle captures:
  LAGA46      a decision saw Lagavulin with live move byte 4 (STUNNED) or 6 (OPEN_NATURAL)
              [taints E0 -- the old park made the search brace for a phantom attack]
  LAGA46_MET  same, with Metallicize present live (burning elite)
              [additionally taints F1 -- the Metallicize heuristic mis-seeded ASLEEP]
  SPIRE       any Spire Shield / Spire Spear decision
              [taints E0+E1 -- facing was never restored, mis-siding the +50% back-attack]
  ECTO_THIEF  a decision saw a Looter/Mugger while the player held Ectoplasm
              [taints E0-F2 -- effectiveGold counted thief-held gold as recoverable-by-kill]
  TE_FIGHT    any Time Eater combat decision
              [taints E0-F3 -- the reconcile transplanted a drifting engine Time Warp counter
              over the live-observed one, mis-planning around the forced end-of-turn; fatal in
              the g38/redo-g3 phantom, distorting in any driven TE fight]
  PS_DUP      a combat decision with Perfected Strike in the deck alongside a duplication
              source (Necronomicon relic, Double Tap / Echo Form in deck, Duplication potion)
              [taints E0-F4 -- the search under-valued the duplicated PS hit by one strike
              bonus in every sim, whether or not live ever played the line]
  DOME        a combat decision with Runic Dome held
              [taints E0-F5 -- with intents hidden the conversion parked no current move, so
              the search modeled every monster as re-issuing its fight OPENER every turn]

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
    if n <= 43:
        return 'F2'
    return 'F3'


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
        ecto = any(r.get('id') == 'Ectoplasm' for r in gs.get('relics', []))
        deck_ids = {c.get('id') for c in gs.get('deck', [])}
        if gs.get('combat_state') and any(r.get('id') == 'Runic Dome' for r in gs.get('relics', [])):
            marks.setdefault(sd, set()).add('DOME')
        if 'Perfected Strike' in deck_ids and gs.get('combat_state') and (
                any(r.get('id') == 'Necronomicon' for r in gs.get('relics', []))
                or deck_ids & {'Double Tap', 'Echo Form'}
                or any(p.get('id') == 'DuplicationPotion' for p in gs.get('potions', []))):
            marks.setdefault(sd, set()).add('PS_DUP')
        for mn in gs.get('combat_state', {}).get('monsters', []):
            nm = mn['name'].replace(' ', '')
            if nm == 'Lagavulin' and mn.get('move_id') in (4, 6):
                met = any(p['id'] == 'Metallicize' for p in mn.get('powers', []))
                marks.setdefault(sd, set()).add('LAGA46_MET' if met else 'LAGA46')
            if nm in ('SpireShield', 'SpireSpear'):
                marks.setdefault(sd, set()).add('SPIRE')
            if ecto and nm in ('Looter', 'Mugger'):
                marks.setdefault(sd, set()).add('ECTO_THIEF')
            if nm == 'TimeEater':
                marks.setdefault(sd, set()).add('TE_FIGHT')

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
                taint = mk & {'LAGA46', 'LAGA46_MET', 'SPIRE', 'ECTO_THIEF', 'TE_FIGHT', 'PS_DUP', 'DOME'}
            elif era == 'E1':
                taint = mk & {'SPIRE', 'ECTO_THIEF', 'TE_FIGHT', 'PS_DUP', 'DOME'}
            elif era == 'F1':
                taint = mk & {'LAGA46_MET', 'ECTO_THIEF', 'TE_FIGHT', 'PS_DUP', 'DOME'}
            elif era == 'F2':
                taint = mk & {'ECTO_THIEF', 'TE_FIGHT', 'PS_DUP', 'DOME'}
            elif era == 'F3':
                taint = mk & {'TE_FIGHT', 'PS_DUP', 'DOME'}
            if taint:
                v = f"DISCARD ({'+'.join(sorted(taint))})"
                discard.append(s)
            else:
                v = 'KEEP'
                clean.append((n, s, k))
        print(f"{n:>4} {s:<14} {k:<10} {era}  {','.join(sorted(mk)) or '-':<18} {v}")

    def tally(rows, label):
        wins = [(n, s, k) for n, s, k in rows if k.startswith(('heart', 'act3'))]
        hearts = [x for x in wins if x[2].startswith('heart')]
        print(f"{label}: {len(rows)} | hearts {len(hearts)} | act3 {len(wins) - len(hearts)} "
              f"| losses {len(rows) - len(wins)}")

    current = [(n, s, k) for n, s, k in clean if era_of(n) == 'F3']   # F3 keeps = no-TE trajectories,
    stratum = [(n, s, k) for n, s, k in clean if era_of(n) != 'F3']   # identically distributed under F4
    print()
    tally(current, "HEADLINE current-era (unconditional)")
    tally(stratum, "pre-fix keeps (CONDITIONAL stratum -- not comparable to a benchmark)")
    print(f"redo ({len(redo)}): {' '.join(redo)}")
    print(f"discarded ({len(discard)}): {' '.join(discard)}")


if __name__ == '__main__':
    main()
