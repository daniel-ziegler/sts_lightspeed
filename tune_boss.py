#!/usr/bin/env python3
"""Optuna tuner for boss battles: search knobs + evaluateEndState weights + rollout potion mode.

Same shape as tune_optuna.py but tuned against a boss-only state set (every record is a
pre-boss-fight GameContext) and with two boss-specific additions:
  * STS_ROLLOUT_POTION_MODE swept as a categorical env knob (0=auto, 1=never, 2=always-dump).
  * the objective is the boss eval_states SCORE, which scores victories on post-act-transition
    heal HP (BattleContext::postBattleHealedHp), so the tuner doesn't chase HP the game restores.

Searches at reduced fidelity (sim budget + state subset); the best trials are re-validated at
full fidelity on every state. Held-out validation against a separate boss set is done by re-running
the reported config on it (--holdout-file), since dev gains have historically overfit state mix.
Persistent sqlite study so it resumes after interruption.
"""

import argparse
import csv
import os
import subprocess
import sys
import threading

import optuna

# Round 3 (full-strength states): only the boss-gated widening pair is searched; exploration and
# eval weights stay pinned at the jointly-tuned defaults (FIXED below), and rollouts never drink
# potions (settled in round 2). On a boss-only state set, eval_states' wideningC/Alpha args ARE
# the boss widening.
SPACE = {
    "wideningC":     (0.5, 8.0, True),
    "wideningAlpha": (0.05, 1.0, False),
}
FIXED = {
    "exploration": 9.9, "winBonus": 53.0, "potionWeight": 11.0, "monsterDamage": 37.0,
    "aliveWeight": 3.4, "energyWaste": 1.75, "turnSurvival": 1.5,
}
ENV_SPACE = {}

# Warm starts: the general widening (current default = no boss specialization) and the round-1/2
# optimum that regressed at deployment strength.
WARM_GENERAL = {"wideningC": 4.6, "wideningAlpha": 0.37}
WARM_OLD_OPT = {"wideningC": 6.46, "wideningAlpha": 0.8495}

# Baseline row for the final validation table (== WARM_GENERAL; the value to beat on held-out).
ORIG_DEFAULTS = dict(WARM_GENERAL)

_logfile = None
_loglock = threading.Lock()


def run_eval(test_bin, state_file, threads, params, env_params, budget, limit, logf=None):
    cmd = [test_bin, "eval_states", str(threads), state_file, str(budget), str(limit)]
    cmd += [f"{k}={v:.6f}" for k, v in {**FIXED, **params}.items()]
    env = dict(os.environ)
    for name, (var, _choices) in ENV_SPACE.items():
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
    """Split a combined param dict into (float eval params, categorical env params)."""
    fp = {k: allp[k] for k in SPACE}
    ep = {k: allp[k] for k in ENV_SPACE}
    return fp, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-file", default="states_h2_boss_dev.txt")
    ap.add_argument("--holdout-file", default="states_h2_boss_holdout.txt")
    ap.add_argument("--test-bin", default="./test")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--search-budget", type=int, default=2000)
    ap.add_argument("--search-limit", type=int, default=0)
    ap.add_argument("--valid-budget", type=int, default=5000)
    ap.add_argument("--valid-limit", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=1, help="concurrent trials (each uses --threads)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--storage", default="sqlite:///tune_boss.db")
    ap.add_argument("--study-name", default="boss_v2")
    ap.add_argument("--log", default="tune_boss_evals.csv")
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
        env_params = {name: trial.suggest_categorical(name, choices)
                      for name, (_var, choices) in ENV_SPACE.items()}
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
        study.enqueue_trial(dict(WARM_GENERAL))
        study.enqueue_trial(dict(WARM_OLD_OPT))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Only completed trials count toward the budget; enqueued-but-waiting ones still need to run.
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    def cb(st, tr):
        b = st.best_trial
        cfg = " ".join(f"{k}={b.params[k]:.3f}" for k in SPACE)
        cfg += " " + " ".join(f"{k}={b.params[k]}" for k in ENV_SPACE)
        print(f"trial {tr.number}: {tr.value:.3f}  | best {b.value:.3f} "
              f"win {b.user_attrs.get('winrate', 0):.3f} @ {cfg}", flush=True)

    study.optimize(objective, n_trials=max(0, args.n_trials - done),
                   n_jobs=args.n_jobs, callbacks=[cb])

    print("\n=== search done; validating top candidates at full fidelity ===", flush=True)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    seen = sorted(completed, key=lambda t: -t.value)
    cands, keys = [], set()
    for t in seen:
        key = tuple(round(t.params[k], 4) for k in SPACE) + tuple(t.params[k] for k in ENV_SPACE)
        if key in keys:
            continue
        keys.add(key)
        cands.append(dict(t.params))
        if len(cands) >= args.top_k:
            break
    cands.append(dict(ORIG_DEFAULTS))

    results = []
    for allp in cands:
        fp, ep = split_params(allp)
        mean, winrate, avghp, n = run_eval(
            args.test_bin, args.state_file, args.threads, fp, ep,
            args.valid_budget, args.valid_limit, logf)
        # held-out re-eval of the same config
        ho_mean = ho_win = None
        if args.holdout_file and os.path.exists(args.holdout_file):
            ho_mean, ho_win, _, _ = run_eval(
                args.test_bin, args.holdout_file, args.threads, fp, ep,
                args.valid_budget, args.valid_limit, logf)
        tag = " (orig defaults)" if allp == ORIG_DEFAULTS else ""
        results.append((mean, winrate, ho_mean, ho_win, allp, tag))
        ho_str = f" | holdout mean={ho_mean:.3f} win={ho_win:.3f}" if ho_mean is not None else ""
        print(f"  dev mean={mean:8.3f} win={winrate:.3f} n={n}{ho_str}  "
              + " ".join(f"{k}={allp[k]:.3f}" for k in SPACE)
              + " " + " ".join(f"{k}={allp[k]}" for k in ENV_SPACE) + tag, flush=True)

    # rank by held-out when available, else dev
    results.sort(key=lambda r: -(r[2] if r[2] is not None else r[0]))
    best = results[0]
    print("\n=== best config (ranked by held-out) ===", flush=True)
    print(" ".join(f"{k}={best[4][k]:.4f}" for k in SPACE)
          + " " + " ".join(f"{k}={best[4][k]}" for k in ENV_SPACE) + best[5], flush=True)
    print(f"dev mean={best[0]:.3f} win={best[1]:.3f}"
          + (f" | holdout mean={best[2]:.3f} win={best[3]:.3f}" if best[2] is not None else ""),
          flush=True)
    _logfile.close()


if __name__ == "__main__":
    main()
