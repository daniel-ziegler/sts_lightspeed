"""Extended shaping-feature analysis: floor-controlled OLS of return-to-go on many candidate
features, to find which are worth shaping and at what coefficient. Same logic/caveats as
analyze_shaping_coeffs.py (floor controls the confound; note floor also MEDIATES some benefits,
so these slopes UNDER-weight features whose payoff is surviving to higher floors -> in practice
multiply by ~3-5x, as the upgrade dose-response showed).

Features (from obs): hp_frac, max_hp, gold, num_upgraded, deck_size, num_relics, num_potions.
fixed_observation = [cur_hp, max_hp, gold, floor, boss, toSelect]; potions empty-slot sentinel = 1.
"""
import glob, os, re
import numpy as np
import pandas as pd

EPI_DIR = 'runs/ppo_1h.pt.episodes'
iter_of = lambda p: int(re.search(r'iter_(\d+)', p).group(1))
files = sorted(glob.glob(os.path.join(EPI_DIR, '*.parquet')), key=iter_of)[-60:]
print(f"Loading {len(files)} files: iters {iter_of(files[0])}..{iter_of(files[-1])}")

frames = []
for f in files:
    df = pd.read_parquet(f, columns=['obs.fixed_observation', 'obs.deck.cards', 'obs.deck.upgrades',
                                      'obs.relics.relics', 'obs.potions', 'return'])
    fo = np.stack(df['obs.fixed_observation'].to_numpy())
    cur_hp, max_hp, gold, floor = fo[:, 0].astype(float), fo[:, 1].astype(float), fo[:, 2].astype(float), fo[:, 3].astype(float)
    frames.append(pd.DataFrame({
        'hp_frac': np.where(max_hp > 0, cur_hp / np.maximum(max_hp, 1), 0.0),
        'max_hp': max_hp,
        'gold': gold,
        'num_upgraded': df['obs.deck.upgrades'].apply(lambda a: int((np.asarray(a) >= 1).sum())).to_numpy().astype(float),
        'deck_size': df['obs.deck.cards'].apply(lambda a: len(a)).to_numpy().astype(float),
        'num_relics': df['obs.relics.relics'].apply(lambda a: len(a)).to_numpy().astype(float),
        'num_potions': df['obs.potions'].apply(lambda a: int((np.asarray(a) != 1).sum())).to_numpy().astype(float),
        'floor': floor,
        'ret': df['return'].to_numpy().astype(float),
    }))
D = pd.concat(frames, ignore_index=True)
D = D[D['hp_frac'] > 0]
print(f"{len(D):,} states\n")

feats = ['hp_frac', 'max_hp', 'gold', 'num_upgraded', 'deck_size', 'num_relics', 'num_potions']
print(f"{'feature':<13}{'mean':>9}{'std':>9}{'end_mean':>10}")
# end_mean = mean at each game's last decision (for offset = coef*end_mean); reuse last-row-per-block.
for fcol in feats:
    print(f"{fcol:<13}{D[fcol].mean():>9.3f}{D[fcol].std():>9.3f}")

y = D['ret'].to_numpy()
X = [D[c].to_numpy() for c in feats] + [D['floor'].to_numpy(), (D['floor']**2).to_numpy()]
A = np.column_stack([np.ones(len(y))] + X)
beta, *_ = np.linalg.lstsq(A, y, rcond=None)
yhat = A @ beta
r2 = 1 - ((y - yhat)**2).sum() / ((y - y.mean())**2).sum()
names = ['intercept'] + feats + ['floor', 'floor^2']

print(f"\n=== OLS ret ~ {' + '.join(feats)} + floor + floor^2   (R^2={r2:.3f}) ===")
print(f"{'term':<13}{'coef':>11}{'std_effect':>12}")
for n, b in zip(names, beta):
    if n in feats:
        se = b * D[n].std() / D['ret'].std()
        print(f"{n:<13}{b:>+11.5f}{se:>+12.3f}")
    else:
        print(f"{n:<13}{b:>+11.5f}")

print("\nstd_effect = SD of return per 1-SD feature change (the floor-controlled importance ranking).")
print("Coefs are in return units; multiply by ~3-5x for a shaping coef (mediator under-weighting).")
