#!/usr/bin/env python3
"""Optuna tuner that optimizes FULL-GAME performance directly (avg floor reached),
rather than the per-battle objective. Drives `./test winrate_mt` over full games.

Goal: find the eval-weight / search-knob heuristic that best serves whole-run play, and
see how far it diverges from the per-battle-optimal config. avg floor is the objective
because asc-0 full-run win-rate is too sparse to optimize with feasible game counts; the
win-rate is recorded alongside.
"""
import argparse, csv, os, re, subprocess, sys, threading
import optuna

SPACE = {
    "exploration":   (0.5, 14.0, True),
    "wideningC":     (0.3, 8.0, True),
    "wideningAlpha": (0.1, 1.0, False),
    "winBonus":      (20.0, 400.0, True),
    "potionWeight":  (0.0, 30.0, False),
    "monsterDamage": (0.0, 60.0, False),   # widened: per-battle search pinned this near its cap
    "aliveWeight":   (0.0, 12.0, False),
    "energyWaste":   (0.0, 3.0, False),
    "turnSurvival":  (0.0, 3.0, False),
}

# per-battle Optuna winner (does the per-battle-optimal config also play full games well?)
WARM_PERBATTLE = {
    "exploration": 9.93, "wideningC": 4.61, "wideningAlpha": 0.372, "winBonus": 53.43,
    "potionWeight": 10.97, "monsterDamage": 37.12, "aliveWeight": 3.37,
    "energyWaste": 1.74, "turnSurvival": 1.51,
}
WARM_3KNOB = {
    "exploration": 6.7176, "wideningC": 1.9752, "wideningAlpha": 0.9971, "winBonus": 100.0,
    "potionWeight": 10.0, "monsterDamage": 10.0, "aliveWeight": 1.0,
    "energyWaste": 0.2, "turnSurvival": 0.2,
}
WARM_DEFAULT = {
    "exploration": 4.2426, "wideningC": 1.0, "wideningAlpha": 0.5, "winBonus": 100.0,
    "potionWeight": 10.0, "monsterDamage": 10.0, "aliveWeight": 1.0,
    "energyWaste": 0.2, "turnSurvival": 0.2,
}

_logfile = None
_loglock = threading.Lock()


def run_eval(test_bin, threads, ngames, budget, params, logf):
    cmd = [test_bin, "winrate_mt", str(threads), "1", str(ngames), str(budget), "0"]
    cmd += [f"{k}={v:.6f}" for k, v in params.items()]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    win = re.search(r"percentWin:\s*([\d.]+)%", out)
    flr = re.search(r"avgFloorReached:\s*([\d.]+)", out)
    if not (win and flr):
        sys.stderr.write(out + "\n")
        raise RuntimeError(f"could not parse: {' '.join(cmd)}")
    winpct, floor = float(win.group(1)), float(flr.group(1))
    if logf:
        with _loglock:
            logf.writerow([ngames] + [f"{params[k]:.6f}" for k in SPACE] + [floor, winpct])
            if _logfile is not None:
                _logfile.flush()
    return floor, winpct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-bin", default="./test")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--ngames", type=int, default=400)
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--n-trials", type=int, default=120)
    ap.add_argument("--valid-ngames", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--storage", default="sqlite:///tune_fullgame.db")
    ap.add_argument("--study-name", default="fullgame")
    ap.add_argument("--log", default="tune_fullgame_evals.csv")
    args = ap.parse_args()

    global _logfile
    _logfile = open(args.log, "a", newline="")
    logf = csv.writer(_logfile)
    if os.stat(args.log).st_size == 0:
        logf.writerow(["ngames"] + list(SPACE) + ["floor", "winpct"])
        _logfile.flush()

    def objective(trial):
        params = {n: trial.suggest_float(n, lo, hi, log=log) for n, (lo, hi, log) in SPACE.items()}
        floor, winpct = run_eval(args.test_bin, args.threads, args.ngames, args.budget, params, logf)
        trial.set_user_attr("winpct", winpct)
        return floor

    study = optuna.create_study(
        direction="maximize", study_name=args.study_name, storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=1, multivariate=True, n_startup_trials=20))
    if len(study.trials) == 0:
        for w in (WARM_PERBATTLE, WARM_3KNOB, WARM_DEFAULT):
            study.enqueue_trial(w)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = len(study.trials)

    def cb(st, tr):
        b = st.best_trial
        print(f"trial {tr.number}: floor {tr.value:.2f} win {tr.user_attrs.get('winpct',0):.2f}% "
              f"| best floor {b.value:.2f} win {b.user_attrs.get('winpct',0):.2f}% @ "
              + " ".join(f"{k}={b.params[k]:.2f}" for k in SPACE), flush=True)

    study.optimize(objective, n_trials=max(0, args.n_trials - done), n_jobs=args.n_jobs, callbacks=[cb])

    print("\n=== validating top candidates at full game count ===", flush=True)
    ts = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)
    cands, keys = [], set()
    for t in ts:
        k = tuple(round(t.params[x], 3) for x in SPACE)
        if k in keys:
            continue
        keys.add(k); cands.append(t.params)
        if len(cands) >= args.top_k:
            break
    cands += [WARM_PERBATTLE, WARM_3KNOB, WARM_DEFAULT]
    results = []
    for p in cands:
        floor, winpct = run_eval(args.test_bin, 64, args.valid_ngames, args.budget, p, logf)
        tag = (" (per-battle)" if p is WARM_PERBATTLE else " (3knob)" if p is WARM_3KNOB
               else " (default)" if p is WARM_DEFAULT else "")
        results.append((floor, winpct, p, tag))
        print(f"  floor={floor:6.2f} win={winpct:5.2f}%  "
              + " ".join(f"{k}={p[k]:.2f}" for k in SPACE) + tag, flush=True)
    results.sort(key=lambda r: -r[0])
    best = results[0]
    print("\n=== best full-game config ===", flush=True)
    print(" ".join(f"{k}={best[2][k]:.4f}" for k in SPACE) + best[3], flush=True)
    print(f"floor={best[0]:.2f} win={best[1]:.2f}%", flush=True)
    _logfile.close()


if __name__ == "__main__":
    main()
