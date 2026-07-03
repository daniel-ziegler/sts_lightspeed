"""Pick potential-based reward-shaping coefficients (HP, #upgrades) from collected PPO data.

We extend the existing telescoping potential Phi(s) with shape(s) = c_hp*hp_frac + c_upg*nup.
A *sensible* coefficient makes shape(s) track the part of the value function explained by that
feature, so we regress the return-to-go on (hp_frac, nup) while controlling for floor (upgrades
accumulate with floor while return-to-go shrinks with floor -> naive correlation is confounded).

Data: runs/ppo_1h.pt.episodes/*.parquet. Each row is one non-battle decision with:
  obs.fixed_observation = [cur_hp, max_hp, gold, floorNum, boss, toSelect]
  obs.deck.upgrades      = per-card upgrade count
  return                 = GAE return-to-go (value target, in progress-reward units)
"""
import glob, os, re
import numpy as np
import pandas as pd

EPI_DIR = 'runs/ppo_1h.pt.episodes'

def iter_of(path):
    m = re.search(r'iter_(\d+)', path)
    return int(m.group(1)) if m else -1

# Use the later, more-converged iterations (most representative value structure).
files = sorted(glob.glob(os.path.join(EPI_DIR, '*.parquet')), key=iter_of)
files = files[-60:]
print(f"Loading {len(files)} parquet files: iters {iter_of(files[0])}..{iter_of(files[-1])}")

frames = []
for fi, f in enumerate(files):
    df = pd.read_parquet(f, columns=['obs.fixed_observation', 'obs.deck.upgrades',
                                      'return', 'final_floor', 'outcome', 'seed'])
    fo = np.stack(df['obs.fixed_observation'].to_numpy())  # (N,6)
    cur_hp = fo[:, 0].astype(np.float64)
    max_hp = fo[:, 1].astype(np.float64)
    floor  = fo[:, 3].astype(np.float64)
    nup = df['obs.deck.upgrades'].apply(lambda a: int((np.asarray(a) >= 1).sum())).to_numpy()
    g = pd.DataFrame({
        'file_id': fi,
        'seed': df['seed'].to_numpy(),
        'hp_frac': np.where(max_hp > 0, cur_hp / np.maximum(max_hp, 1), 0.0),
        'floor': floor,
        'nup': nup.astype(np.float64),
        'ret': df['return'].to_numpy().astype(np.float64),
        'final_floor': df['final_floor'].to_numpy().astype(np.float64),
        'outcome': df['outcome'].to_numpy().astype(np.float64),
    })
    frames.append(g)
D = pd.concat(frames, ignore_index=True)
D = D[D['hp_frac'] > 0]  # drop terminal/degenerate
print(f"\n{len(D):,} decision states. "
      f"hp_frac mean={D.hp_frac.mean():.3f} | nup mean={D.nup.mean():.2f} (max {int(D.nup.max())}) | "
      f"floor mean={D.floor.mean():.1f} | ret mean={D.ret.mean():.3f} std={D.ret.std():.3f}")

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print("\n=== Univariate Pearson correlations (confounded by floor) ===")
for tgt in ['ret', 'final_floor', 'outcome']:
    print(f"  vs {tgt:<11}: hp_frac r={pearson(D.hp_frac, D[tgt]):+.3f}   nup r={pearson(D.nup, D[tgt]):+.3f}")

def ols(X, y):
    """Return coefficients for [1, *cols]."""
    A = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return beta, r2

print("\n=== Multivariate OLS: ret ~ 1 + hp_frac + nup + floor + floor^2 ===")
floor2 = D.floor ** 2
beta, r2 = ols([D.hp_frac.to_numpy(), D.nup.to_numpy(), D.floor.to_numpy(), floor2.to_numpy()], D.ret.to_numpy())
names = ['intercept', 'hp_frac', 'nup', 'floor', 'floor^2']
for n, b in zip(names, beta):
    print(f"  {n:<10} {b:+.5f}")
print(f"  R^2 = {r2:.3f}")
c_hp_raw, c_upg_raw = beta[1], beta[2]
print(f"\n  >> floor-controlled marginal value: HP_frac coef = {c_hp_raw:+.4f} (per full bar), "
      f"upgrade coef = {c_upg_raw:+.4f} (per upgraded card), in return units")

# Standardized betas (the "based on correlation" view): effect of a 1-SD change.
print("\n=== Standardized partial effects (1-SD change in feature -> SD of ret) ===")
print(f"  hp_frac: {c_hp_raw * D.hp_frac.std() / D.ret.std():+.3f} sd   "
      f"nup: {c_upg_raw * D.nup.std() / D.ret.std():+.3f} sd")

# Offset K to neutralize the terminal clawback: mean shape at each game's LAST decision.
# Within a file, a game's rows are contiguous & ordered, so groupby(seed).last() is the last decision.
print("\n=== Offset K (mean feature at last decision of each game) ===")
last = D.sort_index().groupby(['file_id', 'seed']).last()
print(f"  E[hp_frac_end] = {last.hp_frac.mean():.3f}   E[nup_end] = {last.nup.mean():.2f}")
print(f"  (games sampled: {len(last):,}; mean end floor = {last.floor.mean():.1f})")

print("\nDone.")
