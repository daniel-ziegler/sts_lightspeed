#!/usr/bin/env python3
"""CMA-ES multi-fidelity tuner for the battle MCTS hyperparameters.

Optimizes (explorationParameter, chanceWideningC, chanceWideningAlpha) against a
fixed set of pre-battle states produced by `./test gen_states`. The objective is
the mean battle outcome reported by `./test eval_states`:
    score = -200 on death, else curHp + 10 * remaining potions.

Search runs at reduced fidelity (small sim budget + state subset); the best
candidates are then re-validated at full fidelity on every state.
"""

import argparse
import csv
import os
import subprocess
import sys

import cma

# (name, low, high, default)
PARAMS = [
    ("exploration", 0.1, 12.0, 3.0 * (2.0 ** 0.5)),
    ("wideningC", 0.1, 5.0, 1.0),
    ("wideningAlpha", 0.05, 1.0, 0.5),
]


def to_real(x_norm):
    """Map a normalized [0,1]^3 vector to real parameter values (clipped)."""
    out = []
    for xi, (_, lo, hi, _) in zip(x_norm, PARAMS):
        xi = min(1.0, max(0.0, xi))
        out.append(lo + xi * (hi - lo))
    return out


def to_norm(real):
    return [(v - lo) / (hi - lo) for v, (_, lo, hi, _) in zip(real, PARAMS)]


class Evaluator:
    def __init__(self, test_bin, state_file, threads, logf):
        self.test_bin = test_bin
        self.state_file = state_file
        self.threads = threads
        self.cache = {}
        self.logf = logf
        self.n_evals = 0

    def eval_real(self, real, budget, limit):
        key = (round(real[0], 4), round(real[1], 4), round(real[2], 4), budget, limit)
        if key in self.cache:
            return self.cache[key]

        cmd = [
            self.test_bin, "eval_states", str(self.threads), self.state_file,
            f"{real[0]:.6f}", f"{real[1]:.6f}", f"{real[2]:.6f}",
            str(budget), str(limit),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = None
        for ln in proc.stdout.splitlines():
            if ln.startswith("SCORE "):
                line = ln
                break
        if line is None:
            sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
            raise RuntimeError(f"eval_states produced no SCORE line for {cmd}")

        _, mean, winrate, avghp, n = line.split()
        result = (float(mean), float(winrate), float(avghp), int(n))
        self.cache[key] = result

        self.n_evals += 1
        self.logf.writerow([self.n_evals, budget, limit,
                            f"{real[0]:.6f}", f"{real[1]:.6f}", f"{real[2]:.6f}",
                            mean, winrate, avghp, n])
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-file", default="states.txt")
    ap.add_argument("--test-bin", default="./test")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--search-budget", type=int, default=2000)
    ap.add_argument("--search-limit", type=int, default=500)
    ap.add_argument("--valid-budget", type=int, default=5000)
    ap.add_argument("--valid-limit", type=int, default=0)
    ap.add_argument("--max-fevals", type=int, default=120)
    ap.add_argument("--sigma0", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--log", default="tune_log.csv")
    args = ap.parse_args()

    logfile = open(args.log, "w", newline="")
    logf = csv.writer(logfile)
    logf.writerow(["eval", "budget", "limit", "exploration", "wideningC",
                   "wideningAlpha", "mean", "winrate", "avghp", "n"])

    ev = Evaluator(args.test_bin, args.state_file, args.threads, logf)

    defaults = [p[3] for p in PARAMS]
    print(f"defaults {defaults}")
    base = ev.eval_real(defaults, args.search_budget, args.search_limit)
    print(f"  search-fidelity default score: mean={base[0]:.3f} "
          f"winrate={base[1]:.3f} avghp={base[2]:.2f} n={base[3]}")
    logfile.flush()

    x0 = to_norm(defaults)
    es = cma.CMAEvolutionStrategy(
        x0, args.sigma0,
        {"bounds": [0.0, 1.0], "maxfevals": args.max_fevals, "seed": args.seed,
         "verb_disp": 1},
    )

    seen = {}  # rounded-real-tuple -> (score, real)
    while not es.stop():
        solutions = es.ask()
        costs = []
        for x in solutions:
            real = to_real(x)
            score = ev.eval_real(real, args.search_budget, args.search_limit)[0]
            costs.append(-score)  # CMA minimizes
            seen[tuple(round(v, 4) for v in real)] = (score, real)
        es.tell(solutions, costs)
        es.disp()
        logfile.flush()

    print("\n=== search complete; validating top candidates at full fidelity ===")
    # Best distinct candidates from the search phase, plus the defaults.
    ranked = sorted(seen.values(), key=lambda sr: -sr[0])
    candidates = [sr[1] for sr in ranked[:args.top_k]]
    candidates.append(defaults)

    results = []
    for real in candidates:
        mean, winrate, avghp, n = ev.eval_real(real, args.valid_budget, args.valid_limit)
        is_default = all(abs(a - b) < 1e-9 for a, b in zip(real, defaults))
        tag = " (default)" if is_default else ""
        results.append((mean, winrate, avghp, n, real, tag))
        print(f"  mean={mean:8.3f} winrate={winrate:.3f} avghp={avghp:6.2f} n={n}  "
              f"explore={real[0]:.4f} C={real[1]:.4f} alpha={real[2]:.4f}{tag}")
        logfile.flush()

    results.sort(key=lambda r: -r[0])
    best = results[0]
    print("\n=== best full-fidelity config ===")
    print(f"  exploration={best[4][0]:.4f}  wideningC={best[4][1]:.4f}  "
          f"wideningAlpha={best[4][2]:.4f}{best[5]}")
    print(f"  mean={best[0]:.3f} winrate={best[1]:.3f} avghp={best[2]:.2f} n={best[3]}")

    logfile.close()


if __name__ == "__main__":
    main()
