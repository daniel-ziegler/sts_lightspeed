"""Discriminator: does the pooled representation the value head reads still encode the deck?

The value head is a single Linear on a mean-pooled, RMSNorm'd token representation. We hook its
input (the pooled vector) and train held-out LINEAR ridge probes from it to deck-composition
targets (deck size, #strikes, #block, #attacks, #powers, #Perfected Strikes, has Corruption).

A linear probe is exactly as expressive as the value head. So:
  - high probe R2  => the representation linearly encodes the deck, but the value head doesn't
    use it (we showed dV is deck-flat) => value-target / credit problem, fixable policy-side.
  - low probe R2   => the learned representation has discarded deck detail => representational,
    needs an architecture change (e.g. offered-card -> deck attention).
"""
import argparse

import numpy as np
import pandas as pd
import torch

import slaythespire as sts
from lightspeed.network import NN, ModelHP, collate_fn, load_network_backward_compatible

CID = sts.CardId
STRIKE_KW = {int(getattr(CID, n)) for n in dir(CID) if n.isupper() and 'STRIKE' in n}
BLOCK_CARDS = {int(getattr(CID, n)) for n in ('DEFEND_RED', 'IRON_WAVE', 'SHRUG_IT_OFF', 'TRUE_GRIT',
               'GHOSTLY_ARMOR', 'FLAME_BARRIER', 'METALLICIZE', 'IMPERVIOUS', 'ENTRENCH', 'BARRICADE')
               if hasattr(CID, n)}


def _to(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: _to(x, device) for k, x in v.items()}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episodes', nargs='+', required=True)
    ap.add_argument('--n-states', type=int, default=6000)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = NN(ModelHP(use_value_head=True)).to(device)
    net = load_network_backward_compatible(net, torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()

    pooled_box = {}
    net.value_head.register_forward_hook(lambda m, inp, out: pooled_box.__setitem__('p', inp[0].detach()))

    rows = []
    for f in args.episodes:
        rows.extend(pd.read_parquet(f).to_dict('records'))
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.choice(len(rows), size=min(args.n_states, len(rows)), replace=False)]
    print(f"{len(rows)} states\n")

    reps, targets = [], []
    B = 256
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        with torch.no_grad():
            net(_to(collate_fn(chunk), device))
        reps.append(pooled_box['p'].cpu().numpy())
        for r in chunk:
            deck = [int(c) for c in r['obs.deck.cards']]
            targets.append([
                len(deck),
                sum(c in STRIKE_KW for c in deck),
                sum(c in BLOCK_CARDS for c in deck),
                sum(c == int(CID.PERFECTED_STRIKE) for c in deck),
                int(any(c == int(CID.CORRUPTION) for c in deck)),
                int(r['obs.fixed_observation'][3]),  # floor (easy control: should be high R2)
            ])
    X = np.concatenate(reps); Y = np.array(targets, dtype=float)
    names = ['deck_size', '#strikes', '#block', '#pstrikes', 'has_corruption', 'floor(ctrl)']

    # held-out ridge probe per target
    n = len(X); tr = slice(0, int(n * 0.8)); te = slice(int(n * 0.8), n)
    Xtr = np.column_stack([X[tr], np.ones(int(n * 0.8))])
    Xte = np.column_stack([X[te], np.ones(n - int(n * 0.8))])
    lam = 10.0
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    print("linear-probe held-out R2 from the pooled value representation:")
    for j, nm in enumerate(names):
        w = np.linalg.solve(A, Xtr.T @ Y[tr, j])
        pred = Xte @ w
        yte = Y[te, j]
        r2 = 1 - ((yte - pred) ** 2).sum() / (((yte - yte.mean()) ** 2).sum() + 1e-9)
        print(f"  {nm:16s} R2 {r2:+.3f}   (target mean {Y[:,j].mean():.2f}, sd {Y[:,j].std():.2f})")


if __name__ == '__main__':
    main()
