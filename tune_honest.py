#!/usr/bin/env python3
"""Coarse search-knob tuning for the honest (CardPile) engine on fresh games.

Objective: winrate_mt over N seeds at fixed sims -> avgFloor + floorWeightedWin.
This runs at SimpleAgent-out-of-combat strength, i.e. a WEAKER distribution than deployment
(the boss-tuning round-2 trap) -- treat results as landscape exploration; the top candidates
must be confirmed with eval_hero deployment arms before any default changes.

Resumable: sqlite storage + COMPLETE-trial counting; rerun with the same args to continue.
"""
import argparse
import csv
import os
import re
import subprocess

import optuna

WARM_STARTS = [
    {'exploration': 9.9, 'wideningC': 4.6, 'wideningAlpha': 0.37},   # cheat-mode-tuned defaults
    {'exploration': 9.9, 'wideningC': 2.0, 'wideningAlpha': 0.37},   # narrow-widening hint
    {'exploration': 4.24, 'wideningC': 1.0, 'wideningAlpha': 0.5},   # legacy-era knobs
]


def run_winrate(test_bin, threads, seed_start, n_games, sims, params):
    cmd = [test_bin, 'winrate_mt', str(threads), str(seed_start), str(n_games), str(sims), '0',
           f"exploration={params['exploration']}",
           f"wideningC={params['wideningC']}",
           f"wideningAlpha={params['wideningAlpha']}"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600).stdout
    m = re.search(r'w/l: \((\d+), (\d+)\).*?avgFloorReached: ([\d.]+)', out, re.S)
    if not m:
        raise RuntimeError(f"unparseable winrate_mt output: {out[-500:]}")
    wins, losses, floor = int(m.group(1)), int(m.group(2)), float(m.group(3))
    win_rate = wins / (wins + losses)
    return floor, win_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-bin', required=True)
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--seed-start', type=int, default=9500000)
    ap.add_argument('--n-games', type=int, default=400)
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--n-trials', type=int, default=50)
    ap.add_argument('--storage', default='sqlite:///tune_honest.db')
    ap.add_argument('--study-name', default='honest_knobs_v1')
    ap.add_argument('--log', default='tune_honest_evals.csv')
    args = ap.parse_args()

    study = optuna.create_study(direction='maximize', storage=args.storage,
                                study_name=args.study_name, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(multivariate=True, seed=1234))

    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    enqueued = {tuple(sorted(t.params.items())) for t in study.trials}
    for ws in WARM_STARTS:
        if tuple(sorted(ws.items())) not in enqueued:
            study.enqueue_trial(ws)

    log_exists = os.path.exists(args.log)
    logf = open(args.log, 'a')
    logw = csv.writer(logf)
    if not log_exists:
        logw.writerow(['trial', 'exploration', 'wideningC', 'wideningAlpha',
                       'floor', 'win', 'objective'])

    def objective(trial):
        params = {
            'exploration': trial.suggest_float('exploration', 2.0, 20.0, log=True),
            'wideningC': trial.suggest_float('wideningC', 0.5, 10.0, log=True),
            'wideningAlpha': trial.suggest_float('wideningAlpha', 0.05, 1.0),
        }
        floor, win = run_winrate(args.test_bin, args.threads, args.seed_start,
                                 args.n_games, args.sims, params)
        # floor carries most of the signal at this strength; wins are rare but each one
        # implies ~22 extra floors of play, weight them accordingly
        obj = floor + 25.0 * win
        logw.writerow([trial.number, params['exploration'], params['wideningC'],
                       params['wideningAlpha'], floor, win, obj])
        logf.flush()
        print(f"trial {trial.number}: floor={floor:.2f} win={win:.3f} obj={obj:.2f} "
              f"| {params}", flush=True)
        return obj

    remaining = max(0, args.n_trials - done)
    if remaining:
        study.optimize(objective, n_trials=remaining)

    print("\n=== top 5 by objective ===")
    top = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                 key=lambda t: t.value or 0.0, reverse=True)[:5]
    for t in top:
        print(f"  obj={t.value:.2f}  {t.params}")


if __name__ == '__main__':
    main()
