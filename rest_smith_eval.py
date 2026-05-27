#!/usr/bin/env python3
"""Evaluate a TRAINED policy's REST-vs-SMITH preference as a function of HP fraction.

For each campfire state (fixed actions offer both REST and SMITH), compute the policy's
P(REST | {REST,SMITH}) = softmax over just the REST & SMITH logits, binned by hp_frac.
A sensible policy rests more at low HP and smiths (upgrades) more at high HP. Also reports
the empirical choice rate (what the behaviour policy actually sampled).
"""
import argparse, glob
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import torch.utils.checkpoint  # network.py forward uses torch.utils.checkpoint.checkpoint
from network import NN, ModelHP, choice_space, collate_fn, move_to_device, SlayDataset, FixedAction, ActionType

REST = int(FixedAction.REST); SMITH = int(FixedAction.SMITH); FIXED = int(ActionType.FIXED)


def load(glob_pat, n):
    fs = sorted(glob.glob(glob_pat)); assert fs, glob_pat
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    fx = df[df['choice_type'] == FIXED].reset_index(drop=True)
    off = lambda fa: [int(d['action']) for d in fa]
    rs = fx[fx['fixed_actions'].apply(lambda fa: REST in off(fa) and SMITH in off(fa))].reset_index(drop=True)
    if len(rs) > n:
        rs = rs.sample(n, random_state=0).reset_index(drop=True)
    hp = np.array([r[0] / max(1, r[1]) for r in rs['obs.fixed_observation']], dtype=np.float32)
    jR = np.array([off(fa).index(REST) for fa in rs['fixed_actions']])
    jS = np.array([off(fa).index(SMITH) for fa in rs['fixed_actions']])
    dummy = collate_fn([SlayDataset(rs).__getitem__(0)])['choices']
    def seglen(k):
        v = dummy[k]['value']
        while isinstance(v, dict):
            v = next(iter(v.values()))
        return v.size(1)
    base = seglen('cards') + seglen('relics') + seglen('potions')
    posR, posS = base + jR, base + jS
    chosen = []
    for _, r in rs.iterrows():
        p = choice_space.ix_to_path(dummy, int(r['chosen_idx']))
        a = int(r['fixed_actions'][p[1]]['action']) if p[0] == 'fixed' else -1
        chosen.append(1 if a == REST else (0 if a == SMITH else -1))
    return rs, hp, posR, posS, np.array(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n', type=int, default=12000)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)

    rs, hp, posR, posS, chosen = load(args.episodes, args.n)
    print(f"campfire (REST&SMITH offered) states: {len(rs)}  hp mean {hp.mean():.2f}", flush=True)
    print(f"behaviour policy sampled: REST {np.mean(chosen==1):.1%}  SMITH {np.mean(chosen==0):.1%}  other {np.mean(chosen==-1):.1%}\n", flush=True)

    net = NN(ModelHP(use_value_head=True)).to(dev)
    state = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    try:
        net.load_state_dict(state)
    except Exception as e:
        print(f"strict load failed ({e}); retrying non-strict", flush=True)
        net.load_state_dict(state, strict=False)
    net.eval()
    ds = SlayDataset(rs)
    posR_t, posS_t = torch.tensor(posR), torch.tensor(posS)

    def p_rest(idx, chunk=128):
        if len(idx) == 0:
            return float('nan')
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), chunk):
                b = idx[s:s + chunk]
                batch = move_to_device(collate_fn([ds[int(i)] for i in b]), dev)
                logits = net(batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                ar = torch.arange(len(b), device=dev)
                two = torch.stack([logits[ar, posS_t[b].to(dev)], logits[ar, posR_t[b].to(dev)]], 1)  # [SMITH, REST]
                out.append(F.softmax(two, 1)[:, 1].cpu())
        return torch.cat(out).numpy()

    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01)
    print(f"{'hp bin':>10}{'n':>7}{'modelP(REST)':>14}{'emp REST%':>11}{'emp SMITH%':>12}  (of REST|SMITH)")
    all_idx = np.arange(len(rs))
    pr_all = p_rest(all_idx)
    for a_, b_ in zip(edges, edges[1:]):
        idx = np.where((hp >= a_) & (hp < b_))[0]
        if len(idx) == 0:
            print(f"  {a_:.1f}-{b_:.1f}{0:>7}"); continue
        ch = chosen[idx]; rs_sm = ch[ch >= 0]
        emp_rest = np.mean(rs_sm == 1) if len(rs_sm) else float('nan')
        print(f"  {a_:.1f}-{b_:.1f}{len(idx):>7}{np.nanmean(pr_all[idx]):>14.3f}{emp_rest:>11.1%}{(1-emp_rest):>12.1%}")

    lo, hi = np.where(hp < 0.5)[0], np.where(hp >= 0.9)[0]
    print(f"\nmodel P(REST): hp<0.5 = {np.nanmean(pr_all[lo]):.3f}   hp>=0.9 = {np.nanmean(pr_all[hi]):.3f}   "
          f"(want LOW >> HIGH; a smart policy rests when hurt, smiths when healthy)")


if __name__ == '__main__':
    main()
