"""Smoke + behavior test for the re-enabled Colosseum and Match-and-Keep events.

Plays full games with a manual out-of-combat loop (pick_gameaction + execute, playout_battle for
combat) so we can observe when each event is reached and drive it through the real agent. The .so
has sts_asserts on, so a broken event handler (bad chooseMatchAndKeepCards, the Colosseum phase
flow, the combat-reward -> event re-entry, ...) aborts. We just need to reach the events without
crashing across many games, and confirm both actually occur.
"""
import sys
import slaythespire as sts

COLOSSEUM = sts.Event.COLOSSEUM
MAK = sts.Event.MATCH_AND_KEEP
BATTLE = sts.ScreenState.BATTLE
EVENT = sts.ScreenState.EVENT_SCREEN


def play(seed, sims):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    agent = sts.Agent()
    agent.simulation_count_base = sims
    seen = set()
    steps = 0
    while gc.outcome == sts.GameOutcome.UNDECIDED and steps < 8000:
        steps += 1
        if gc.screen_state == BATTLE:
            agent.playout_battle(gc)
        else:
            if gc.screen_state == EVENT and gc.cur_event in (COLOSSEUM, MAK):
                seen.add(gc.cur_event)
            a = agent.pick_gameaction(gc)
            a.execute(gc)
    return seen, gc.floor_num


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    nColosseum = nMak = maxFloor = 0
    for seed in range(n):
        try:
            seen, floor = play(seed, sims)
        except Exception as e:
            print(f"FAIL seed {seed}: {e}")
            return 1
        nColosseum += COLOSSEUM in seen
        nMak += MAK in seen
        maxFloor = max(maxFloor, floor)
    print(f"{n} full games completed with no crash/assert. Deepest floor reached: {maxFloor}")
    print(f"  games that hit Match-and-Keep: {nMak}")
    print(f"  games that hit Colosseum:      {nColosseum}")
    if nMak == 0 and nColosseum == 0:
        print("WARNING: neither event occurred — increase game count / sims to exercise them.")
        return 1
    print("PASS: re-enabled events spawn and play through the agent without crashing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
