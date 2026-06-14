"""Optuna tuning of the rollout/eval gradation knobs against a mixed (boss A>=10 + random A>=10)
battle-state set. Tunes three default-off knobs introduced for the "blind search on sharp
positions" failure (state 98 / Slime Boss):

  * lossDamageWeight  -- eval_states float param: loss-branch reward on cumulative monster damage.
  * STS_ROLLOUT_CARD_EPSILON   -- env int (per-mille): random card-order variety in rollouts.
  * STS_ROLLOUT_POTION_EPSILON -- env int (per-mille): drink a potion a small fraction of rollouts.

The PRODUCTION search config is pinned (FIXED below) because eval_states' own defaults are stale
(exploration 9.9 / widening 4.6,0.37) vs the deployed SearchAgent (25 / 3.7028,0.52389) -- tuning
the rollout knobs against the wrong search would not transfer. Trial 0 is the all-off baseline
(== current production), so any reported gain is over the live engine. Top configs are validated
on a disjoint holdout (dev gains have historically overfit the state mix). Persistent sqlite study.
"""
import argparse
import csv
import os
import subprocess
import sys
import threading

import optuna

# Float eval_states params to tune.
SPACE = {
    "lossDamageWeight": (0.0, 0.2, False),
}
# Env knobs to tune (int, per-mille). name -> (env var, lo, hi).
ENV_SPACE = {
    "cardEps":   ("STS_ROLLOUT_CARD_EPSILON",   0, 400),
    "potionEps": ("STS_ROLLOUT_POTION_EPSILON", 0, 400),
}
# Production search config -- eval_states defaults are stale, so pin the deployed values.
FIXED = {
    "exploration": 25.0, "explorationChance": 25.0,
    "wideningC": 3.7028, "wideningAlpha": 0.52389,
    "endTurnWideningC": 3.7028, "endTurnWideningAlpha": 0.52389,
}
# Baseline (all knobs off == current production); enqueued first so trial 0 is the value to beat.
WARM_BASELINE = {"lossDamageWeight": 0.0, "cardEps": 0, "potionEps": 0}

_logfile = None
_loglock = threading.Lock()


def run_eval(test_bin, state_file, threads, params, env_params, budget, limit, logf=None):
    cmd = [test_bin, "eval_states", str(threads), state_file, str(budget), str(limit)]
    cmd += [f"{k}={v:.6f}" for k, v in {**FIXED, **params}.items()]
    env = dict(os.environ)
    for name, (var, _lo, _hi) in ENV_SPACE.items():
        env[var] = str(env_params[name])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("SCORE ")), None)
    if line is None:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"no SCORE from: {' '.join(cmd)}")
    _, mean, winrate, avghp, n = line.split()
    if logf:
        with _loglock:
            row = ([budget, limit]
                   + [f"{params[k]:.6f}" for k in SPACE]
                   + [env_params[k] for k in ENV_SPACE]
                   + [mean, winrate, avghp, n])
            logf.writerow(row)
            if _logfile:
                _logfile.flush()
    return float(mean), float(winrate), float(avghp), int(n)


def split_params(allp):
    fp = {k: allp[k] for k in SPACE}
    ep = {k: allp[k] for k in ENV_SPACE}
    return fp, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-file", default="set_dev.txt")
    ap.add_argument("--holdout-file", default="set_mix.txt")
    ap.add_argument("--test-bin", default="../build_native/test")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--search-budget", type=int, default=1000)
    ap.add_argument("--search-limit", type=int, default=1000)
    ap.add_argument("--holdout-limit", type=int, default=1000)
    ap.add_argument("--n-trials", type=int, default=80)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--storage", default="sqlite:///tune_rollout.db")
    ap.add_argument("--study-name", default="rollout_v1")
    ap.add_argument("--log", default="tune_rollout_evals.csv")
    args = ap.parse_args()

    global _logfile
    _logfile = open(args.log, "a", newline="")
    logf = csv.writer(_logfile)
    if os.stat(args.log).st_size == 0:
        logf.writerow(["budget", "limit"] + list(SPACE) + list(ENV_SPACE)
                      + ["mean", "winrate", "avghp", "n"])
        _logfile.flush()

    def objective(trial):
        params = {name: trial.suggest_float(name, lo, hi, log=log)
                  for name, (lo, hi, log) in SPACE.items()}
        env_params = {name: trial.suggest_int(name, lo, hi)
                      for name, (_var, lo, hi) in ENV_SPACE.items()}
        mean, winrate, avghp, n = run_eval(
            args.test_bin, args.state_file, args.threads, params, env_params,
            args.search_budget, args.search_limit, logf)
        trial.set_user_attr("winrate", winrate)
        return mean

    study = optuna.create_study(
        direction="maximize", study_name=args.study_name,
        storage=args.storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=1, multivariate=True, n_startup_trials=20),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(dict(WARM_BASELINE))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    def cb(st, tr):
        b = st.best_trial
        cfg = " ".join(f"{k}={b.params[k]:.3f}" for k in SPACE)
        cfg += " " + " ".join(f"{k}={b.params[k]}" for k in ENV_SPACE)
        print(f"trial {tr.number}: {tr.value:.3f} win {tr.user_attrs.get('winrate',0):.3f} "
              f"| best {b.value:.3f} win {b.user_attrs.get('winrate', 0):.3f} @ {cfg}", flush=True)

    study.optimize(objective, n_trials=max(0, args.n_trials - done),
                   n_jobs=args.n_jobs, callbacks=[cb])

    print("\n=== search done; validating top candidates on holdout ===", flush=True)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    seen = sorted(completed, key=lambda t: -t.value)[:8]
    results = []
    for t in seen:
        fp, ep = split_params(t.params)
        ho = None
        if args.holdout_file and os.path.exists(args.holdout_file):
            ho = run_eval(args.test_bin, args.holdout_file, args.threads, fp, ep,
                          args.search_budget, args.holdout_limit)
        cfg = " ".join(f"{k}={t.params[k]:.3f}" for k in SPACE) + " " \
              + " ".join(f"{k}={t.params[k]}" for k in ENV_SPACE)
        ho_str = f" | holdout mean={ho[0]:.3f} win={ho[1]:.3f}" if ho else ""
        print(f"  dev mean={t.value:8.3f} win={t.user_attrs.get('winrate',0):.3f}{ho_str}  @ {cfg}",
              flush=True)
        results.append((t, ho))

    # Baseline on holdout for reference.
    if args.holdout_file and os.path.exists(args.holdout_file):
        b = run_eval(args.test_bin, args.holdout_file, args.threads,
                     {"lossDamageWeight": 0.0}, {"cardEps": 0, "potionEps": 0},
                     args.search_budget, args.holdout_limit)
        print(f"  BASELINE holdout mean={b[0]:.3f} win={b[1]:.3f}", flush=True)

    best = max(results, key=lambda r: (r[1][0] if r[1] else r[0].value))
    fp, ep = split_params(best[0].params)
    print(f"\nBEST (by holdout): {fp} {ep}", flush=True)


if __name__ == "__main__":
    main()
