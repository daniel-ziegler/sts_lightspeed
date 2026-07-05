#!/usr/bin/env python3
"""Full-game win-rate sweep over tuned candidates, to check how well the per-battle
tuning objective predicts full-game performance.

Runs `./test winrate_mt` for each candidate (top + spread from the Optuna study, plus
default and 3-knob anchors), parsing win-rate and avg floor reached. Reports them next to
each candidate's per-battle subset score.
"""
import argparse, csv, re, subprocess, sys

PARAMS = ["exploration", "wideningC", "wideningAlpha", "winBonus", "potionWeight",
          "monsterDamage", "aliveWeight", "energyWaste", "turnSurvival"]

ANCHORS = [
    ("default", "", {}),  # all defaults
    ("3knob",   "", {"exploration": 6.7176, "wideningC": 1.9752, "wideningAlpha": 0.9971}),
]


def run(test_bin, threads, ngames, budget, params):
    cmd = [test_bin, "winrate_mt", str(threads), "1", str(ngames), str(budget), "0"]
    cmd += [f"{k}={v}" for k, v in params.items()]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    win = re.search(r"percentWin:\s*([\d.]+)%", out)
    flr = re.search(r"avgFloorReached:\s*([\d.]+)", out)
    if not win or not flr:
        sys.stderr.write(out + "\n")
        raise RuntimeError("could not parse winrate_mt output")
    return float(win.group(1)), float(flr.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-bin", default="./test")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--ngames", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--candidates", default="winrate_candidates.csv")
    ap.add_argument("--out", default="winrate_results.csv")
    args = ap.parse_args()

    cands = []
    with open(args.candidates) as f:
        for d in csv.DictReader(f):
            p = {k: float(d[k]) for k in PARAMS}
            cands.append((d["label"], float(d["subset_mean"]), float(d["subset_win"]), p))

    w = csv.writer(open(args.out, "w"))
    w.writerow(["label", "subset_mean", "subset_battlewin", "fullgame_winpct", "avg_floor"])
    print(f"{'label':12s} {'subsetScore':>11s} {'gameWin%':>9s} {'avgFloor':>9s}", flush=True)

    for label, sub_mean, sub_win, p in cands:
        wp, fl = run(args.test_bin, args.threads, args.ngames, args.budget, p)
        w.writerow([label, sub_mean, sub_win, wp, fl]);
        print(f"{label:12s} {sub_mean:11.2f} {wp:9.2f} {fl:9.2f}", flush=True)

    for label, _, p in ANCHORS:
        wp, fl = run(args.test_bin, args.threads, args.ngames, args.budget, p)
        w.writerow([label, "", "", wp, fl])
        print(f"{label:12s} {'-':>11s} {wp:9.2f} {fl:9.2f}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
