"""Purely LINEAR value model (no interaction terms) on the same data/split as value_sl.py.

Tests how much of the held-out value EV is reachable by a model that is linear in features with
NO interactions: ridge regression on [summary scalars] and on [scalars + bag-of-cards + bag-of-relics]
(each card/relic contributes linearly; no cross terms). Compares to the transformer (~0.41) and the
semi-in-sample saved-value baseline (~0.45). Same seed-level train/val split as value_sl.
"""
import glob, os, re
import numpy as np, pandas as pd

EPI_DIR = 'runs/ppo_4ep.pt.episodes'
iter_of = lambda p: int(re.search(r'iter_(\d+)', p).group(1))


def load(max_rows=60000, seed=0):
    files = sorted(glob.glob(os.path.join(EPI_DIR, '*.parquet')), key=iter_of)
    files = files[-max(1, round(len(files) / 3)):]
    df = pd.concat([pd.read_parquet(f, columns=[
        'obs.fixed_observation', 'obs.deck.cards', 'obs.deck.upgrades',
        'obs.relics.relics', 'obs.potions', 'return', 'seed']) for f in files], ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df


def featurize(df, card_vocab=None, relic_vocab=None):
    fo = np.stack(df['obs.fixed_observation'].to_numpy()).astype(np.float64)
    cur_hp, max_hp, gold, floor = fo[:, 0], fo[:, 1], fo[:, 2], fo[:, 3]
    nup = df['obs.deck.upgrades'].apply(lambda a: int((np.asarray(a) >= 1).sum())).to_numpy(float)
    dsz = df['obs.deck.cards'].apply(lambda a: len(a)).to_numpy(float)
    nrel = df['obs.relics.relics'].apply(lambda a: len(a)).to_numpy(float)
    npot = df['obs.potions'].apply(lambda a: int((np.asarray(a) != 1).sum())).to_numpy(float)
    hp_frac = np.where(max_hp > 0, cur_hp / np.maximum(max_hp, 1), 0.0)
    scalars = np.column_stack([hp_frac, max_hp, gold, floor, floor**2, nup, dsz, nrel, npot])
    # bag-of-cards / bag-of-relics (linear, no interactions)
    if card_vocab is None:
        card_vocab = {c: i for i, c in enumerate(sorted({int(x) for a in df['obs.deck.cards'] for x in a}))}
    if relic_vocab is None:
        relic_vocab = {r: i for i, r in enumerate(sorted({int(x) for a in df['obs.relics.relics'] for x in a}))}
    bagc = np.zeros((len(df), len(card_vocab)))
    bagr = np.zeros((len(df), len(relic_vocab)))
    for i, (cs, rsx) in enumerate(zip(df['obs.deck.cards'], df['obs.relics.relics'])):
        for c in cs:
            j = card_vocab.get(int(c));  bagc[i, j] += 1 if j is not None else 0
        for r in rsx:
            j = relic_vocab.get(int(r));  bagr[i, j] += 1 if j is not None else 0
    return scalars, bagc, bagr, card_vocab, relic_vocab


def ridge_ev(Xtr, ytr, Xva, yva, lams=(1.0, 10.0, 100.0, 1000.0)):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-8] = 1.0
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr]); Xva = np.column_stack([np.ones(len(Xva)), Xva])
    ym = ytr.mean()
    best = (-1e9, None)
    A = Xtr.T @ Xtr
    for lam in lams:
        reg = lam * np.eye(A.shape[0]); reg[0, 0] = 0.0  # don't regularize intercept
        w = np.linalg.solve(A + reg, Xtr.T @ (ytr - ym))
        pred = Xva @ w + ym
        ev = 1.0 - np.var(yva - pred) / (np.var(yva) + 1e-12)
        if ev > best[0]:
            best = (ev, lam)
    return best


def main():
    df = load()
    seeds = df['seed'].unique().copy(); np.random.RandomState(0).shuffle(seeds)
    val_seeds = set(seeds[:int(len(seeds) * 0.15)].tolist())
    is_val = df['seed'].isin(val_seeds).to_numpy()
    tr, va = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
    ytr, yva = tr['return'].to_numpy(float), va['return'].to_numpy(float)
    print(f"train={len(tr):,} val={len(va):,} (seed split)  ret std={df['return'].std():.3f}")

    s_tr, bc_tr, br_tr, cv, rv = featurize(tr)
    s_va, bc_va, br_va, _, _ = featurize(va, cv, rv)
    print(f"features: {s_tr.shape[1]} scalars, {bc_tr.shape[1]} card-bag, {br_tr.shape[1]} relic-bag\n")

    ev1, l1 = ridge_ev(s_tr, ytr, s_va, yva)
    print(f"LINEAR (9 summary scalars only):        val EV = {ev1:.4f}  (ridge lam={l1})")
    Xtr2 = np.column_stack([s_tr, bc_tr, br_tr]); Xva2 = np.column_stack([s_va, bc_va, br_va])
    ev2, l2 = ridge_ev(Xtr2, ytr, Xva2, yva)
    print(f"LINEAR (scalars + bag-of-cards/relics): val EV = {ev2:.4f}  (ridge lam={l2})")
    print(f"\nfor reference: transformer SL ~0.40-0.41 held-out; saved-value baseline ~0.45 (semi-in-sample)")


if __name__ == '__main__':
    main()
