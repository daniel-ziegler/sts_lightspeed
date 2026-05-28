#!/usr/bin/env python3
"""Optuna tuner for the battle MCTS: 3 search knobs + 6 evaluateEndState weights.

Objective = mean battle outcome from `./test eval_states` (curHp + 10*potions, -200 on death).
Searches at reduced fidelity (sim budget + state subset); the best trials are then
re-validated at full fidelity on every state. Warm-started around the current best config.
Uses a persistent sqlite study so it resumes after interruption (spot/restart).
"""

import argparse
import csv
import os
import subprocess
import sys
import threading

import optuna

# name -> (low, high, log)
SPACE = {
    "exploration":   (0.5, 12.0, True),
    "wideningC":     (0.3, 8.0, True),
    "wideningAlpha": (0.1, 1.0, False),
    "winBonus":      (20.0, 300.0, True),
    "potionWeight":  (0.0, 30.0, False),
    "monsterDamage": (0.0, 40.0, False),
    "aliveWeight":   (0.0, 10.0, False),
    "energyWaste":   (0.0, 2.0, False),
    "turnSurvival":  (0.0, 2.0, False),
}

# current best 3-knob config (from the CMA-ES run) + eval defaults
WARM_BEST = {
    "exploration": 6.57, "wideningC": 3.14, "wideningAlpha": 0.97,
    "winBonus": 100.0, "potionWeight": 10.0, "monsterDamage": 10.0,
    "aliveWeight": 1.0, "energyWaste": 0.2, "turnSurvival": 0.2,
}
# stock defaults
WARM_DEFAULT = {
    "exploration": 4.2426, "wideningC": 1.0, "wideningAlpha": 0.5,
    "winBonus": 100.0, "potionWeight": 10.0, "monsterDamage": 10.0,
    "aliveWeight": 1.0, "energyWaste": 0.2, "turnSurvival": 0.2,
}


def run_eval(test_bin, state_file, threads, params, budget, limit, logf=None):
    cmd = [test_bin, "eval_states", str(threads), state_file, str(budget), str(limit)]
    cmd += [f"{k}={v:.6f}" for k, v in params.items()]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("SCORE ")), None)
    if line is None:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"no SCORE from: {' '.join(cmd)}")
    _, mean, winrate, avghp, n = line.split()
    if logf:
        with _loglock:
            logf.writerow([budget, limit] + [f"{params[k]:.6f}" for k in SPACE] + [mean, winrate, avghp, n])
            if _logfile:
                _logfile.flush()
    return float(mean), float(winrate), float(avghp), int(n)


_logfile = None
_loglock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-file", default="states2000.txt")
    ap.add_argument("--test-bin", default="./test")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--search-budget", type=int, default=5000)
    ap.add_argument("--search-limit", type=int, default=600)
    ap.add_argument("--valid-budget", type=int, default=5000)
    ap.add_argument("--valid-limit", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=1, help="concurrent trials (each uses --threads)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--storage", default="sqlite:///tune_optuna.db")
    ap.add_argument("--study-name", default="mcts")
    ap.add_argument("--log", default="tune_optuna_evals.csv")
    args = ap.parse_args()

    global _logfile
    _logfile = open(args.log, "a", newline="")
    logf = csv.writer(_logfile)
    if os.stat(args.log).st_size == 0:
        logf.writerow(["budget", "limit"] + list(SPACE) + ["mean", "winrate", "avghp", "n"])
        _logfile.flush()

    def objective(trial):
        params = {}
        for name, (lo, hi, log) in SPACE.items():
            params[name] = trial.suggest_float(name, lo, hi, log=log)
        mean, winrate, avghp, n = run_eval(
            args.test_bin, args.state_file, args.threads, params,
            args.search_budget, args.search_limit, logf)
        trial.set_user_attr("winrate", winrate)
        return mean

    study = optuna.create_study(
        direction="maximize", study_name=args.study_name,
        storage=args.storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=1, multivariate=True, n_startup_trials=20),
    )
    # warm-start (only if this is a fresh study)
    if len(study.trials) == 0:
        study.enqueue_trial(WARM_BEST)
        study.enqueue_trial(WARM_DEFAULT)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = len(study.trials)

    def cb(st, tr):
        b = st.best_trial
        print(f"trial {tr.number}: {tr.value:.3f}  | best {b.value:.3f} "
              f"win {b.user_attrs.get('winrate', 0):.3f} @ "
              + " ".join(f"{k}={b.params[k]:.3f}" for k in SPACE), flush=True)

    study.optimize(objective, n_trials=max(0, args.n_trials - done), n_jobs=args.n_jobs, callbacks=[cb])

    print("\n=== search done; validating top candidates at full fidelity ===", flush=True)
    seen = sorted(study.trials, key=lambda t: -(t.value if t.value is not None else -1e9))
    cands, keys = [], set()
    for t in seen:
        key = tuple(round(t.params[k], 4) for k in SPACE)
        if key in keys:
            continue
        keys.add(key)
        cands.append(t.params)
        if len(cands) >= args.top_k:
            break
    cands.append(WARM_BEST)
    cands.append(WARM_DEFAULT)

    results = []
    for p in cands:
        mean, winrate, avghp, n = run_eval(
            args.test_bin, args.state_file, args.threads, p,
            args.valid_budget, args.valid_limit, logf)
        tag = " (3-knob best)" if p is WARM_BEST else " (default)" if p is WARM_DEFAULT else ""
        results.append((mean, winrate, p, tag))
        print(f"  mean={mean:8.3f} win={winrate:.3f} n={n}  "
              + " ".join(f"{k}={p[k]:.3f}" for k in SPACE) + tag, flush=True)

    results.sort(key=lambda r: -r[0])
    best = results[0]
    print("\n=== best full-fidelity config ===", flush=True)
    print(" ".join(f"{k}={best[2][k]:.4f}" for k in SPACE) + best[3], flush=True)
    print(f"mean={best[0]:.3f} win={best[1]:.3f}", flush=True)
    _logfile.close()


if __name__ == "__main__":
    main()
