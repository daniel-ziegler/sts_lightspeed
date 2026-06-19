"""Functional test for sequential in-combat multi-select (GAMBLE / EXHAUST_MANY).

Gambling Chip forces a GAMBLE card-select at the start of every combat, so playing battles with
it held (and the deck clogged with Clumsy curses worth gambling away) drives the new
one-card-at-a-time selection through the real searcher. The dbg build has sts_asserts + plain
asserts on, so an invalid action or an infinite re-open loop would abort.

The searcher prints each chosen action; we capture stdout (the C++ std::cout, via an fd redirect)
and confirm the sequential path is exercised:
  - "{ GAMBLE (i)  }"                       -> a SINGLE_CARD_SELECT pick (one card added)
  - "{ GAMBLE (..) Card, Card, ... }"       -> a MULTI_CARD_SELECT confirm applying the whole set
"""
import os
import re
import sys
import tempfile
import slaythespire as sts


def run_all(n, sims):
    """Run n Gambling-Chip battles, returning the captured searcher stdout."""
    cap = tempfile.TemporaryFile()
    saved = os.dup(1)
    os.dup2(cap.fileno(), 1)
    try:
        for seed in range(n):
            gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
            gc.obtain_relic(sts.RelicId.GAMBLING_CHIP)
            for _ in range(6):
                gc.obtain_card(sts.Card(sts.CardId.CLUMSY, 0))  # dead cards worth gambling away
            agent = sts.Agent()
            agent.simulation_count_base = sims
            # synthetic gc has no regainControlAction -> exitBattle() throws after the battle
            # itself completes; any invalid in-battle action would have aborted earlier (asserts on).
            try:
                agent.playout_battle(gc, sts.MonsterEncounter.JAW_WORM)
            except Exception:
                pass
    finally:
        os.dup2(saved, 1)
        os.close(saved)
    cap.seek(0)
    return cap.read().decode("utf-8", "replace")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    log = run_all(n, sims)

    # A pick prints the task + a single index and no card name; a confirm prints the selected cards.
    picks = len(re.findall(r"\{ GAMBLE \(\d+\)\s*\}", log))
    confirms_with_cards = len(re.findall(r"\{ GAMBLE \(\d+\)\s+[A-Za-z]", log))
    confirm_none = len(re.findall(r"\{ GAMBLE none", log))

    print(f"{n} Gambling Chip battles completed with no crash/assert/hang.")
    print(f"  sequential SINGLE_CARD_SELECT picks:        {picks}")
    print(f"  MULTI_CARD_SELECT confirms (>=1 card):      {confirms_with_cards}")
    print(f"  confirms with nothing selected:             {confirm_none}")

    if picks == 0 or confirms_with_cards == 0:
        print("FAIL: the sequential multi-pick path was not exercised.")
        return 1
    print("PASS: searcher picks cards one at a time and applies the multi-card gamble.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
