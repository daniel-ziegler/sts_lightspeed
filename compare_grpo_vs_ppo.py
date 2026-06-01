#!/usr/bin/env python3
"""Compare from-scratch GRPO runs (grpo_a/grpo_b) against the from-scratch PPO baseline
(hero v1, iters 1-99) at matched cumulative wall time.

Wall time = sum of per-iteration collect_time + train_time from the stats JSONL. The match is
approximate: the PPO baseline had the box to itself (20 workers / 30 cores) while each GRPO run
shares it (2 x 16 workers), so GRPO wall times are inflated by contention.
"""
import json
import os

LR = 'lambda_results'
PPO_FILE = f'{LR}/hero.pt.stats.jsonl'
# Only from-scratch runs belong in a wall-time-vs-PPO comparison. grpo_b is now a warm-started
# fork of grpo_a (the low-HP-penalty A/B), so its wall clock doesn't measure from-scratch speed --
# compare it to grpo_a on the iteration axis instead (see the plot), not here.
GRPO_FILES = {'grpo_a (from scratch, lr 1e-4)': f'{LR}/grpo_a.pt.stats.jsonl'}


def load(path, max_iter=None):
    seen = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            seen[r['iteration']] = r
    rows = [seen[i] for i in sorted(seen) if max_iter is None or i <= max_iter]
    cum = 0.0
    out = []
    for r in rows:
        cum += r.get('collect_time', 0) + r.get('train_time', 0)
        out.append({'iter': r['iteration'], 'wall_h': cum / 3600,
                    'win': r['win_rate'], 'floor': r['avg_floor'], 'entropy': r.get('entropy')})
    return out


def at_wall(rows, wall_h):
    """Last row at or before the given cumulative wall time (None if the run hasn't started)."""
    prev = None
    for r in rows:
        if r['wall_h'] > wall_h:
            break
        prev = r
    return prev


def main():
    ppo = load(PPO_FILE, max_iter=99)  # hero v1 = the from-scratch portion
    print(f"PPO baseline (hero v1): {len(ppo)} iters, {ppo[-1]['wall_h']:.1f}h total, "
          f"final win {ppo[-1]['win']:.3f} floor {ppo[-1]['floor']:.1f}\n")

    for name, path in GRPO_FILES.items():
        if not os.path.exists(path):
            print(f"{name}: no stats yet ({path})")
            continue
        g = load(path)
        latest = g[-1]
        p = at_wall(ppo, latest['wall_h'])
        print(f"=== {name} ===")
        print(f"  latest: iter {latest['iter']}  wall {latest['wall_h']:.2f}h  "
              f"win {latest['win']:.3f}  floor {latest['floor']:.2f}  ent {latest['entropy']:.3f}")
        if p is None:
            print("  (PPO baseline has no data this early)")
        else:
            print(f"  PPO at same wall time: iter {p['iter']}  wall {p['wall_h']:.2f}h  "
                  f"win {p['win']:.3f}  floor {p['floor']:.2f}")
            print(f"  delta: win {latest['win'] - p['win']:+.3f}  floor {latest['floor'] - p['floor']:+.2f}")
        # milestone table every 10 GRPO iters
        print(f"  {'iter':>5} {'wall_h':>7} {'win':>6} {'floor':>7} | {'PPO@wall iter':>13} {'win':>6} {'floor':>7}")
        for r in g:
            if r['iter'] % 10 == 0 or r['iter'] == g[-1]['iter']:
                p = at_wall(ppo, r['wall_h'])
                ps = f"{p['iter']:>13} {p['win']:>6.3f} {p['floor']:>7.2f}" if p else f"{'-':>13} {'-':>6} {'-':>7}"
                print(f"  {r['iter']:>5} {r['wall_h']:>7.2f} {r['win']:>6.3f} {r['floor']:>7.2f} | {ps}")
        print()


if __name__ == '__main__':
    main()
