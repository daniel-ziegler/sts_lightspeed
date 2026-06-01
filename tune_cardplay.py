"""Greedy random-search tuner for cardPlayMap.

Reads dumped default priorities + CardId enum names, then perturbs a few cards per iteration,
evaluates 4-way parallel via STS_CARDPLAY_OVERRIDE, keeps the global best. Final winner is
saved to /tmp/best_override.txt for held-out validation.
"""
import argparse
import os
import random
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--bin",       default="build_tune/test")
p.add_argument("--states",    default="states2000.txt")
p.add_argument("--n-states",  type=int, default=500)
p.add_argument("--sims",      type=int, default=5000)
p.add_argument("--iters",     type=int, default=60, help="batches of 4 candidates")
p.add_argument("--perturb-k", type=int, default=4, help="cards perturbed per candidate")
p.add_argument("--cores",     type=int, default=4)
p.add_argument("--seed",      type=int, default=42)
p.add_argument("--restart-every", type=int, default=15, help="random restart period")
p.add_argument("--default",   default="/tmp/cardplay_default.txt")
p.add_argument("--id-map",    default="/tmp/cardid_map.txt")
p.add_argument("--out",       default="/tmp/best_override.txt")
args = p.parse_args()

default_prios = {}
with open(args.default) as f:
    for line in f:
        cid, pr = line.split()
        default_prios[int(cid)] = int(pr)
tunable = sorted([cid for cid, pr in default_prios.items() if pr > 0])
id2name = {}
with open(args.id_map) as f:
    for line in f:
        cid, name = line.split()
        id2name[int(cid)] = name
print(f"tunable cards: {len(tunable)} (priority > 0)")
print(f"workload: {args.states} {args.n_states} states @ {args.sims} sims, {args.iters} batches x {args.cores} parallel")

def write_override(path, overrides):
    with open(path, "w") as f:
        for cid, prio in overrides.items():
            f.write(f"{cid} {prio}\n")

def eval_parallel(configs):
    procs = []
    for i, cfg in enumerate(configs):
        path = f"/tmp/override_{i}.txt"
        write_override(path, cfg)
        env = {**os.environ, "STS_CARDPLAY_OVERRIDE": path}
        cmd = ["taskset", "-c", str(i % args.cores), args.bin,
               "eval_states", "1", args.states, str(args.sims), str(args.n_states)]
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
    scores = []
    for p_ in procs:
        out, _ = p_.communicate()
        sc = float("-inf")
        for line in out.splitlines():
            if line.startswith("SCORE "):
                sc = float(line.split()[1])
                break
        scores.append(sc)
    return scores

rng = random.Random(args.seed)

# Baseline (no overrides)
print("evaluating baseline...")
baseline = eval_parallel([{}])[0]
print(f"baseline SCORE: {baseline:.3f}")
best = {}
best_score = baseline
since_improve = 0
t0 = time.time()

for batch in range(args.iters):
    # Restart every K batches without improvement to avoid getting stuck
    base_cfg = best if (since_improve < args.restart_every) else {}
    if since_improve >= args.restart_every:
        print(f"  [batch {batch}] random restart (no improve in {args.restart_every} batches)")
        since_improve = 0
    cands = []
    for _ in range(args.cores):
        cand = dict(base_cfg)
        for _ in range(args.perturb_k):
            cid = rng.choice(tunable)
            cand[cid] = rng.randint(1, 200)
        cands.append(cand)
    scores = eval_parallel(cands)
    improved = False
    for cand, sc in zip(cands, scores):
        if sc > best_score:
            best, best_score = cand, sc
            improved = True
    elapsed = time.time() - t0
    s_str = " ".join(f"{s:.2f}" for s in scores)
    print(f"batch {batch:3d} t={elapsed:6.1f}s scores=[{s_str}] best={best_score:.3f} ({len(best)} overrides) {'*NEW*' if improved else ''}", flush=True)
    since_improve = 0 if improved else since_improve + 1
    # Checkpoint best every batch (cheap; survives spot reclaim).
    write_override(args.out, best)
    with open(args.out + ".meta", "w") as f:
        f.write(f"batch={batch} best_score={best_score:.4f} baseline={baseline:.4f} elapsed={elapsed:.1f}s\n")

print(f"\n=== FINAL: baseline={baseline:.3f}, best={best_score:.3f}, gain={best_score-baseline:+.3f} ===")
print(f"Best overrides ({len(best)} cards):")
for cid in sorted(best):
    print(f"  {id2name.get(cid, str(cid))} ({cid}): default={default_prios[cid]} -> {best[cid]}")
write_override(args.out, best)
print(f"\nSaved best override to {args.out}")
