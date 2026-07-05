#!/usr/bin/env python3
"""Paired boss-fight comparison of the post-heal objective fix (before vs after arms).

Both arms play identically until the act-1 boss (the fix only changes boss-encounter search),
so act-1 boss fights are perfectly paired by seed. Act-2/3 comparisons are distributional
(games diverge after the first boss).
"""
import csv
import sys
from math import comb

BOSS_ENC = {18, 19, 20, 37, 38, 39, 52, 53, 54, 56}
ENC_NAMES = {18: "SLIME_BOSS", 19: "THE_GUARDIAN", 20: "HEXAGHOST",
             37: "AUTOMATON", 38: "COLLECTOR", 39: "CHAMP",
             52: "AWAKENED_ONE", 53: "TIME_EATER", 54: "DONU_AND_DECA", 56: "THE_HEART"}


def load(fn):
    by_seed = {}
    for r in csv.DictReader(open(fn)):
        by_seed.setdefault(int(r["seed"]), []).append(r)
    return by_seed


def boss_fights(battles, act):
    return [b for b in battles if int(b["encounter"]) in BOSS_ENC and b["act"] == str(act)]


def mcnemar(b, c):
    m = b + c
    if m == 0:
        return 1.0
    p = sum(comb(m, k) for k in range(min(b, c) + 1)) * 2 / 2 ** m
    return min(p, 1.0)


def main(before_fn, after_fn):
    before, after = load(before_fn), load(after_fn)
    seeds = sorted(set(before) & set(after))
    print(f"paired games: {len(seeds)}")

    # --- Act-1 boss: perfectly paired ---
    print("\n=== ACT-1 BOSS (perfectly paired: identical entering state) ===")
    paired = []
    for s in seeds:
        b1 = boss_fights(before[s], 1)
        a1 = boss_fights(after[s], 1)
        if b1 and a1:
            paired.append((s, b1[0], a1[0]))
    n = len(paired)
    print(f"games reaching act-1 boss in both arms: {n}")
    # consistency check: same encounter both arms (entering state identical)
    enc_mismatch = sum(1 for _, b, a in paired if b["encounter"] != a["encounter"])
    print(f"encounter mismatches (should be 0): {enc_mismatch}")

    surv_b = sum(1 for _, b, _ in paired if int(b["curHp"]) > 0)
    surv_a = sum(1 for _, _, a in paired if int(a["curHp"]) > 0)
    flips_to_surv = [(s) for s, b, a in paired if int(b["curHp"]) <= 0 < int(a["curHp"])]
    flips_to_death = [(s) for s, b, a in paired if int(a["curHp"]) <= 0 < int(b["curHp"])]
    print(f"survival before: {surv_b}/{n} = {surv_b/n:.1%}")
    print(f"survival after:  {surv_a}/{n} = {surv_a/n:.1%}")
    print(f"flips death->survive: {len(flips_to_surv)}")
    print(f"flips survive->death: {len(flips_to_death)}")
    print(f"McNemar exact p = {mcnemar(len(flips_to_surv), len(flips_to_death)):.4f}")

    # among fights where both survived: HP and potions carried out
    both_surv = [(b, a) for _, b, a in paired if int(b["curHp"]) > 0 and int(a["curHp"]) > 0]
    if both_surv:
        hp_b = sum(int(b["curHp"]) for b, _ in both_surv) / len(both_surv)
        hp_a = sum(int(a["curHp"]) for _, a in both_surv) / len(both_surv)
        po_b = sum(int(b["potions"]) for b, _ in both_surv) / len(both_surv)
        po_a = sum(int(a["potions"]) for _, a in both_surv) / len(both_surv)
        print(f"both-survived ({len(both_surv)}): post-fight HP {hp_b:.1f} -> {hp_a:.1f} | potions {po_b:.2f} -> {po_a:.2f}")

    # --- Act 2/3 bosses: distributional (games diverged) ---
    for act in (2, 3):
        bf = [b for s in seeds for b in boss_fights(before[s], act)]
        af = [a for s in seeds for a in boss_fights(after[s], act)]
        if not bf or not af:
            continue
        sb = sum(1 for b in bf if int(b["curHp"]) > 0)
        sa = sum(1 for a in af if int(a["curHp"]) > 0)
        print(f"\n=== ACT-{act} BOSS (distributional) ===")
        print(f"before: {sb}/{len(bf)} = {sb/len(bf):.1%} survive | after: {sa}/{len(af)} = {sa/len(af):.1%} survive")

    # game-level recap
    won_b = sum(1 for s in seeds if before[s][-1]["game_won"] == "1")
    won_a = sum(1 for s in seeds if after[s][-1]["game_won"] == "1")
    print(f"\ngame wins: before {won_b}/{len(seeds)}  after {won_a}/{len(seeds)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "states_h2/bo_before.csv",
         sys.argv[2] if len(sys.argv) > 2 else "states_h2/bo_after.csv")
