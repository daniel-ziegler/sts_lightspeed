"""Does the policy look ahead 2-3 path nodes? Identification by immediate-type matching.

Design: only pure path decisions where ALL offered options lead to the same immediate room
type (depth-1 signal matched away by construction), full 3-level subtrees (mapY <= 11).
Per-option features at exact depths (every edge advances one row, so depth k = row y+k):
counts of REST/ELITE/SHOP/EVENT among nodes exactly 2 (resp. 3) steps ahead, plus breadth
controls (|S2|, |S3|), out-degree, and x-centrality. Nested conditional logits
(controls -> +depth2 -> +depth3) with cluster-robust (by game seed) SEs and joint Wald
tests per depth block.
"""
import argparse
import glob

import numpy as np
import pandas as pd
import torch
import slaythespire as sts
from network import MAX_DECK_SIZE, MAX_FIXED_ACTIONS

PATHS_OFFSET = MAX_DECK_SIZE + 3 + sts.MAX_POTION_CAPACITY + MAX_FIXED_ACTIONS
T_REST, T_ELITE, T_SHOP, T_EVENT = (int(sts.Room.REST), int(sts.Room.ELITE),
                                    int(sts.Room.SHOP), int(sts.Room.EVENT))


def decision_features(r):
    """-> (features per option [n_opt, F], chosen k, seed) or None if not in the design."""
    xs = [int(v) for v in r['obs.map.xs']]
    ys = [int(v) for v in r['obs.map.ys']]
    rts = [int(v) for v in r['obs.map.roomTypes']]
    idx = {(x, y): j for j, (x, y) in enumerate(zip(xs, ys))}
    n = len(xs)
    succ = [[] for _ in range(n)]
    for j in range(n):
        for e in r['obs.map.pathXs'][j]:
            e = int(e)
            t = idx.get((e, ys[j] + 1))
            if e >= 0 and t is not None:
                succ[j].append(t)

    if int(r['obs.mapY']) > 10:  # option row = y+1 <= 11 ensures full depth-3 subtrees
        return None
    ydest = int(r['obs.mapY']) + 1
    opts = [idx.get((int(x), ydest)) for x in r['paths_offered']]
    k = int(r.chosen_idx) - PATHS_OFFSET
    if None in opts or len(opts) < 2 or not (0 <= k < len(opts)):
        return None
    if len({rts[o] for o in opts}) != 1:  # immediate-type-matched sets only
        return None

    feats = []
    for o in opts:
        s1 = set(succ[o])
        s2 = {t for j in s1 for t in succ[j]}
        s3 = {t for j in s2 for t in succ[j]}
        row = [len(s1), abs(xs[o] - 3)]
        for S in (s2, s3):
            row += [sum(1 for j in S if rts[j] == t) for t in (T_REST, T_ELITE, T_SHOP, T_EVENT)]
            row.append(len(S))
        feats.append(row)
    return np.array(feats, dtype=np.float64), k, int(r['seed'])


FEATURE_NAMES = (['outdeg', '|x-3|'] +
                 [f'd2_{t}' for t in ('rest', 'elite', 'shop', 'event')] + ['d2_size'] +
                 [f'd3_{t}' for t in ('rest', 'elite', 'shop', 'event')] + ['d3_size'])
BLOCKS = {'controls': [0, 1], 'depth2': [2, 3, 4, 5, 6], 'depth3': [7, 8, 9, 10, 11]}


def fit_clogit(X, mask, kk, cols, clusters):
    """Conditional logit on feature columns `cols`; -> (w, robust SE, cluster-robust cov, nll)."""
    Xc = X[..., cols]
    w = torch.zeros(len(cols), dtype=torch.float64, requires_grad=True)

    def nll_vec(wv):
        u = (Xc * wv).sum(-1).masked_fill(~mask, -1e9)
        return -(u.gather(1, kk[:, None]).squeeze(1) - torch.logsumexp(u, 1))

    opt = torch.optim.LBFGS([w], max_iter=400, tolerance_grad=1e-12)
    opt.step(lambda: (opt.zero_grad(), nll_vec(w).sum().backward(), nll_vec(w).sum().detach())[2])
    w_h = w.detach().requires_grad_(True)
    H = torch.autograd.functional.hessian(lambda v: nll_vec(v).sum(), w.detach())
    bread = torch.linalg.inv(H)
    # cluster-robust meat: per-decision scores summed within cluster, outer-product sum
    u = (Xc * w_h).sum(-1).masked_fill(~mask, -1e9)
    p = torch.softmax(u, dim=1)
    # score_i = x_chosen - E_p[x]  (gradient of per-decision loglik)
    x_chosen = Xc[torch.arange(len(kk)), kk]
    x_mean = (p.unsqueeze(-1) * Xc.masked_fill(~mask.unsqueeze(-1), 0)).sum(1)
    scores = (x_chosen - x_mean).detach()
    meat = torch.zeros(len(cols), len(cols), dtype=torch.float64)
    df = pd.DataFrame(scores.numpy())
    df['c'] = clusters
    for _, g in df.groupby('c'):
        s = torch.tensor(g.drop(columns='c').to_numpy().sum(0))
        meat += torch.outer(s, s)
    cov = bread @ meat @ bread
    return w.detach(), cov.diag().sqrt(), cov, float(nll_vec(w.detach()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', required=True)
    args = ap.parse_args()

    feats, ks, seeds = [], [], []
    for f in args.episodes:
        df = pd.read_parquet(f)
        for _, r in df[df.choice_type == 3].iterrows():
            out = decision_features(r)
            if out is not None:
                feats.append(out[0]); ks.append(out[1]); seeds.append(out[2])

    N = len(feats)
    MAXO = max(len(f) for f in feats)
    F = feats[0].shape[1]
    X = torch.zeros(N, MAXO, F, dtype=torch.float64)
    mask = torch.zeros(N, MAXO, dtype=torch.bool)
    kk = torch.tensor(ks)
    for i, f in enumerate(feats):
        X[i, :len(f)] = torch.tensor(f); mask[i, :len(f)] = True
    print(f"{N} immediate-type-matched path decisions (mapY<=10 origins), "
          f"{len(set(seeds))} games\n")

    full_cols = list(range(F))
    results = {}
    for mname, cols in (('M0 controls', BLOCKS['controls']),
                        ('M1 +depth2', BLOCKS['controls'] + BLOCKS['depth2']),
                        ('M2 +depth3', full_cols)):
        w, se, cov, nll = fit_clogit(X, mask, kk, cols, seeds)
        results[mname] = (cols, w, se, cov, nll)
        print(f"=== {mname}  (nll {nll:.1f}) ===")
        for c, wi, si in zip(cols, w, se):
            print(f"  {FEATURE_NAMES[c]:<10} {wi:>8.4f} ±{si:.4f}  z={wi/si:>6.2f}")
        print()

    # Joint Wald tests for each depth block in the FULL model (cluster-robust)
    cols, w, se, cov, _ = results['M2 +depth3']
    for bname in ('depth2', 'depth3'):
        bidx = [cols.index(c) for c in BLOCKS[bname]]
        wb = w[bidx]
        covb = cov[bidx][:, bidx]
        stat = float(wb @ torch.linalg.solve(covb, wb))
        from scipy import stats as st
        p = st.chi2.sf(stat, len(bidx))
        print(f"Joint Wald ({bname} block, cluster-robust): chi2({len(bidx)}) = {stat:.1f}, p = {p:.2g}")


if __name__ == '__main__':
    main()
