"""How much does the policy / value head condition on the ascension token?

Counterfactual: take real decision states and override ONLY fixed_observation[6] (the ascension
input) to each level 0..20, holding the rest of the state (HP, deck, enemies, floor) fixed. The
spread of the value and the divergence of the action distribution across that override is the
network's DIRECT sensitivity to the ascension token (it cannot be a proxy via other features,
since only the token moved).

Value: should decrease monotonically with ascension (harder game -> lower expected return).
Policy: TV distance between the action distribution at A0 vs A20 on the same state; argmax-flip
rate. Both broken down so we can see where (if anywhere) ascension actually changes play.
"""
import argparse
import numpy as np, pandas as pd, torch
from lightspeed.network import NN, ModelHP, collate_fn, load_network_backward_compatible

ASC_IDX = 6


def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
    return load_network_backward_compatible(NN(ModelHP(use_value_head=True)), sd).to(device).eval()


def _to(v, device):
    if isinstance(v, dict):
        return {k: _to(x, device) for k, x in v.items()}
    return v.to(device)


def set_asc(rows, a):
    out = []
    for r in rows:
        d = dict(r)
        fo = list(d['obs.fixed_observation']); fo[ASC_IDX] = a
        d['obs.fixed_observation'] = fo
        out.append(d)
    return out


def fwd(net, rows, device, B=256):
    """Return (values[N], logits[N, slots]) over the rows."""
    vs, lg = [], []
    for i in range(0, len(rows), B):
        bt = _to(collate_fn(rows[i:i + B]), device)
        with torch.no_grad():
            out = net(bt)
        logits, values = out[0], out[1]
        vs.append(values.float().cpu().numpy()); lg.append(logits.float().cpu().numpy())
    return np.concatenate(vs), np.concatenate(lg)


def softmax_valid(logits):
    m = np.isfinite(logits)
    z = np.where(m, logits, -1e9)
    z = z - z.max(); e = np.exp(z) * m
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episodes', nargs='+', required=True)
    ap.add_argument('--n-states', type=int, default=3000)
    args = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = load_net(args.ckpt, device)

    rows = []
    for f in args.episodes:
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            d = r.to_dict()
            if len(d['obs.fixed_observation']) > ASC_IDX:
                rows.append(d)
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.choice(len(rows), size=min(args.n_states, len(rows)), replace=False)]
    ctype = np.array([int(r.get('choice_type', -1)) for r in rows])
    print(f"{len(rows)} states; choice_type counts {dict(zip(*np.unique(ctype, return_counts=True)))}\n")

    ascs = list(range(21))
    V = np.zeros((len(rows), 21))
    pol = {}  # a -> logits, for a in {0,10,20}
    for a in ascs:
        v, lg = fwd(net, set_asc(rows, a), device)
        V[:, a] = v
        if a in (0, 10, 20):
            pol[a] = lg

    # ---- VALUE responsiveness ----
    print("=== VALUE vs ascension (counterfactual, mean over states) ===")
    for a in (0, 5, 10, 15, 20):
        print(f"  A{a:<2} mean V {V[:, a].mean():+.3f}")
    drop = V[:, 0] - V[:, 20]
    aa = np.arange(21)
    corr = np.array([np.corrcoef(aa, V[i])[0, 1] for i in range(len(rows))])
    mono = np.mean([np.all(np.diff(V[i]) <= 1e-6) for i in range(len(rows))])
    print(f"  mean V(A0)-V(A20) = {drop.mean():+.3f}  (sd {drop.std():.3f}, "
          f"frac with V(A0)>V(A20): {(drop > 0).mean():.2f})")
    print(f"  per-state corr(V, ascension): mean {corr.mean():+.3f}   "
          f"strictly-monotone-decreasing frac {mono:.2f}")

    # ---- POLICY responsiveness ----
    def tv(la, lb):
        return np.array([0.5 * np.abs(softmax_valid(la[i]) - softmax_valid(lb[i])).sum()
                         for i in range(len(la))])
    tv_020 = tv(pol[0], pol[20]); tv_010 = tv(pol[0], pol[10]); tv_1020 = tv(pol[10], pol[20])
    flip = np.array([softmax_valid(pol[0][i]).argmax() != softmax_valid(pol[20][i]).argmax()
                     for i in range(len(rows))])
    print("\n=== POLICY shift under ascension override (TV distance, 0=identical 1=disjoint) ===")
    print(f"  A0->A20 mean TV {tv_020.mean():.3f}   A0->A10 {tv_010.mean():.3f}   "
          f"A10->A20 {tv_1020.mean():.3f}")
    print(f"  argmax action flips A0 vs A20: {flip.mean():.2%} of states")
    print("  by choice_type (A0->A20 TV / argmax-flip / n):")
    for c in sorted(set(ctype.tolist())):
        m = ctype == c
        if m.sum() >= 20:
            print(f"    type {c}: TV {tv_020[m].mean():.3f}  flip {flip[m].mean():.2%}  (n={m.sum()})")

    # scale reference: TV between two DIFFERENT random states' policies (how big is "big"?)
    perm = rng.permutation(len(rows))
    ref = tv(pol[0], pol[0][perm])
    print(f"  [scale ref] TV between different states' A0 policies: {ref.mean():.3f}")


if __name__ == '__main__':
    main()
