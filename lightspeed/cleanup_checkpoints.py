#!/usr/bin/env python3
"""Prune PPO checkpoints, keeping the ones worth keeping.

Keep policy (per run prefix, e.g. heroe2.pt):
  weights  (<prefix>.iter_N)            : keep if N % KEEP_EVERY == 0, OR N is the latest
                                          saved iter for that run, OR (prefix, N) is IMPORTANT.
  optimizer(<prefix>.optimizer.iter_N)  : keep ONLY for the latest iter of each run + IMPORTANT.
                                          (Archival every-50 checkpoints keep weights only --
                                          they're for eval/reference, not resume; optimizer is 2x
                                          the size of the weights.)

DRY-RUN by default -- prints what it would delete. Pass --apply to actually delete.
Override the runs dir with RUNS=/path, KEEP_EVERY=N.
"""
import os, re, sys
from collections import defaultdict

RUNS = os.environ.get('RUNS', os.path.expanduser('~/sts/sts_rl/runs'))
KEEP_EVERY = int(os.environ.get('KEEP_EVERY', '50'))
APPLY = '--apply' in sys.argv
# (prefix, iter) pairs always kept with weights+optimizer, regardless of the every-N rule.
IMPORTANT = {('hero.pt', 130), ('heroe2.pt', 240)}


def human(n):
    n = float(n)
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def size(p):
    try:
        return os.path.getsize(os.path.join(RUNS, p))
    except OSError:
        return 0


wt = defaultdict(set)   # prefix -> {iters with a weights file}
opt = defaultdict(set)  # prefix -> {iters with an optimizer file}
for f in os.listdir(RUNS):
    m = re.match(r'(.+\.pt)\.optimizer\.iter_(\d+)$', f)
    if m:
        opt[m.group(1)].add(int(m.group(2)))
        continue
    m = re.match(r'(.+\.pt)\.iter_(\d+)$', f)
    if m:
        wt[m.group(1)].add(int(m.group(2)))

keep, delete = [], []
for prefix in sorted(set(wt) | set(opt)):
    latest = max(wt[prefix]) if wt[prefix] else (max(opt[prefix]) if opt[prefix] else None)
    for it in sorted(wt[prefix]):
        important = (prefix, it) in IMPORTANT
        keep_w = (it % KEEP_EVERY == 0) or (it == latest) or important
        (keep if keep_w else delete).append(f"{prefix}.iter_{it}")
    for it in sorted(opt[prefix]):
        keep_o = (it == latest) or ((prefix, it) in IMPORTANT)
        (keep if keep_o else delete).append(f"{prefix}.optimizer.iter_{it}")

# Per-prefix summary of kept iters (weights), for a quick sanity read.
kept_iters = defaultdict(list)
for p in keep:
    m = re.match(r'(.+\.pt)\.iter_(\d+)$', p)
    if m:
        kept_iters[m.group(1)].append(int(m.group(2)))

print(f"runs dir : {RUNS}")
print(f"keep-every: {KEEP_EVERY}   important: {sorted(IMPORTANT)}")
print(f"\nKEEP  {len(keep)} files, {human(sum(size(p) for p in keep))}")
for prefix in sorted(kept_iters):
    print(f"  {prefix:22s} iters {sorted(kept_iters[prefix])}")
print(f"\nDELETE {len(delete)} files, {human(sum(size(p) for p in delete))}")
for p in sorted(delete):
    print(f"  rm {p}  ({human(size(p))})")

if APPLY:
    freed = 0
    for p in delete:
        fp = os.path.join(RUNS, p)
        try:
            freed += os.path.getsize(fp)
            os.remove(fp)
        except OSError as e:
            print(f"  ! {p}: {e}")
    print(f"\nDELETED {len(delete)} files, freed {human(freed)}")
else:
    print("\n(dry-run — pass --apply to delete)")
