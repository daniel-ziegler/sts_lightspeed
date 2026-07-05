"""Probe whether heart1's value head conditions on deck composition.

Loads real states from episode parquets and, for diagnostic cards, computes the critic's
counterfactual marginal value
    dV(c | s) = V(deck + c) - V(deck)
by inserting card c into the deck obs and re-running the value head (everything else, incl. the
choice context, identical -> it differences out). Three reads:

  1. sanity  -- mean dV ranks good cards (Reaper) above junk (Flex).
  2. synergy -- regress dV(c) on the deck feature it should depend on (Perfected Strike vs
     #Strike-keyword cards; Body Slam vs #block/Defend cards) and the True Grit upgrade delta.
     Near-zero slope = the critic is deck-blind too.
  3. variance -- how much of dV(c)'s spread across decks a deck feature explains (R2). Flat=blind.

On-policy caveat: V predicts returns under the current (deck-blind) policy, so a flat slope
localizes "no value signal to drive deck-conditioning" but can't alone separate representational
incapacity from policy-induced. (Discriminator probe is a separate follow-up.)
"""
import argparse
import glob

import numpy as np
import pandas as pd
import torch

import slaythespire as sts
from silverbot.network import NN, ModelHP, collate_fn, load_network_backward_compatible

CID = sts.CardId
# Strike-keyword cards (Perfected Strike scales with cards whose NAME contains "Strike").
STRIKE_KW = {int(getattr(CID, n)) for n in dir(CID)
             if n.isupper() and 'STRIKE' in n and hasattr(CID, n)}
# Block/Defend-ish cards for the Body Slam synergy proxy.
BLOCK_CARDS = {int(getattr(CID, n)) for n in ('DEFEND_RED', 'IRON_WAVE', 'SHRUG_IT_OFF',
               'TRUE_GRIT', 'GHOSTLY_ARMOR', 'FLAME_BARRIER', 'METALLICIZE', 'IMPERVIOUS',
               'ENTRENCH', 'BARRICADE', 'BODY_SLAM') if hasattr(CID, n)}


def veval(net, rows, device):
    """Value head on a list of flattened obs/choice row-dicts -> np array of V."""
    out = []
    B = 256
    for i in range(0, len(rows), B):
        batch = collate_fn(rows[i:i + B])
        dev = {k: _to(v, device) for k, v in batch.items()}
        with torch.no_grad():
            res = net(dev)
        v = res[1] if isinstance(res, tuple) else res
        out.append(torch.as_tensor(v).flatten().cpu().numpy())
    return np.concatenate(out)


def _to(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: _to(x, device) for k, x in v.items()}
    return v


def with_card(row, cardid, up):
    r = dict(row)
    r['obs.deck.cards'] = list(row['obs.deck.cards']) + [int(cardid)]
    r['obs.deck.upgrades'] = list(row['obs.deck.upgrades']) + [int(up)]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episodes', nargs='+', required=True)
    ap.add_argument('--n-states', type=int, default=3000)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = NN(ModelHP(use_value_head=True)).to(device)
    net = load_network_backward_compatible(net, torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()

    # Sample real decision states (any screen) across the parquets.
    rows = []
    for f in args.episodes:
        df = pd.read_parquet(f)
        rows.extend(df.to_dict('records'))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rows), size=min(args.n_states, len(rows)), replace=False)
    rows = [rows[i] for i in idx]
    print(f"{len(rows)} real states sampled\n")

    # deck features per state
    def deck_ids(r):
        return [int(c) for c in r['obs.deck.cards']]
    n_strike = np.array([sum(1 for c in deck_ids(r) if c in STRIKE_KW) for r in rows])
    n_block = np.array([sum(1 for c in deck_ids(r) if c in BLOCK_CARDS) for r in rows])
    deck_sz = np.array([len(deck_ids(r)) for r in rows])

    base_v = veval(net, rows, device)

    def dV(cardid, up=0):
        return veval(net, [with_card(r, cardid, up) for r in rows], device) - base_v

    print("=== sanity: mean marginal value dV(card) over real decks ===")
    anchors = [('REAPER', 0), ('DEMON_FORM', 0), ('PERFECTED_STRIKE', 0),
               ('BODY_SLAM', 0), ('TRUE_GRIT', 0), ('TRUE_GRIT', 1), ('FLEX', 0)]
    dvs = {}
    for name, up in anchors:
        c = int(getattr(CID, name))
        d = dV(c, up)
        dvs[(name, up)] = d
        tag = name + ('+' if up else '')
        print(f"  {tag:18s} mean dV {d.mean():+.4f}   sd {d.std():.4f}   "
              f"[p10 {np.percentile(d,10):+.3f}, p90 {np.percentile(d,90):+.3f}]")

    def slope_r2(d, x):
        x = (x - x.mean()) / (x.std() + 1e-9)
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        pred = A @ coef
        ss = ((d - d.mean()) ** 2).sum()
        r2 = 1 - ((d - pred) ** 2).sum() / (ss + 1e-12)
        return coef[1], r2  # slope per SD of feature, R2

    # Control cards with NO strike/block synergy: their dV-vs-feature slope captures the
    # deck-quality confound (bad unthinned decks have lower marginal value for everything).
    # The synergy-specific signal is the SYNERGY card's slope MINUS the controls' mean slope.
    controls = {n: dV(int(getattr(CID, n))) for n in ('FLEX', 'DEMON_FORM', 'INFLAME')}

    def diff_slope(target_d, feat, label, frange):
        st, _ = slope_r2(target_d, feat)
        cs = np.mean([slope_r2(d, feat)[0] for d in controls.values()])
        print(f"  {label}: slope {st:+.4f}/SD, controls {cs:+.4f}/SD, "
              f"SYNERGY-specific {st - cs:+.4f}/SD  (feature range {frange})")

    print("\n=== synergy-conditioning (control-deconfounded) ===")
    diff_slope(dvs[('PERFECTED_STRIKE', 0)], n_strike, 'Perfected Strike vs #strike-cards',
               f"{n_strike.min()}-{n_strike.max()}")
    diff_slope(dvs[('BODY_SLAM', 0)], n_block, 'Body Slam        vs #block-cards ',
               f"{n_block.min()}-{n_block.max()}")
    up_delta = dvs[('TRUE_GRIT', 1)] - dvs[('TRUE_GRIT', 0)]
    print(f"  True Grit upgrade delta dV(+)-dV()  : {up_delta.mean():+.4f}  (sd {up_delta.std():.4f})  "
          f"[should be positive]")

    print("\n=== how much of dV varies with deck at all (sd across decks) ===")
    for (name, up), d in dvs.items():
        print(f"  {name + ('+' if up else ''):18s} sd(dV) {d.std():.4f}  (|mean| {abs(d.mean()):.4f})")


if __name__ == '__main__':
    main()
