"""How much do heart1 card-acquisition preferences vary by the act's upcoming boss?

Same conditional-logit-over-the-offered-slate framework as analyze_card_choices.py, but the
question is boss-conditioning. Each act fixes its boss at the start (visible on the map), so the
policy *could* steer picks toward it (block vs Hexaghost, artifact/debuff-light vs Time Eater,
single-target vs multi-body Donu&Deca, etc.).

Baseline utility for offered card c on a screen whose upcoming boss is b:
    u0[c] = base[c] + act_fe[act(b)]            (identity + global act effect)
Boss model adds a per-card, per-boss deviation:
    u1[c] = base[c] + act_fe[act(b)] + delta[c, b]
delta is ridge-penalized. Because every boss lives in exactly one act, act_fe absorbs the
act-average and delta[c,b] is the pick shift attributable to *which* boss within that act --
the clean "does knowing the boss (beyond the act) change the pick" signal. The held-out
log-loss gain u0->u1 is the headline magnitude; per-act delta spreads name the cards that move.

Only acts 1-3 (their bosses are a genuine 3-way draw). Act-4 boss is fixed, excluded.
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch

import slaythespire as sts

REWARDS = int(sts.ScreenState.REWARDS)
SHOP = int(sts.ScreenState.SHOP_ROOM)

# fixed_observation[4] is getBossEncoding(gc.boss) -- a 0..9 index, NOT the raw enum
# (see bindings/bindings-util.cpp). 9 == THE_HEART (act 4, fixed) is excluded.
BOSSES = {0: 'SlimeBoss', 1: 'Hexaghost', 2: 'Guardian',
          3: 'Champ', 4: 'Automaton', 5: 'Collector',
          6: 'TimeEater', 7: 'Donu&Deca', 8: 'AwakenedOne'}
BOSS_ACT = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
ACT_BOSSES = {1: [0, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8]}


def load(parquets):
    cols = ['screen_state', 'cards_offered.cards', 'chosen_idx', 'obs.fixed_observation']
    decs = []
    for fp in parquets:
        df = pd.read_parquet(fp, columns=cols)
        df = df[df.screen_state.isin([REWARDS, SHOP])]
        for _, r in df.iterrows():
            cards = [int(c) for c in r['cards_offered.cards']]
            if not cards:
                continue
            fo = list(r['obs.fixed_observation'])
            boss = int(fo[4])
            if boss not in BOSSES:
                continue
            n = len(cards)
            ci = int(r['chosen_idx'])
            decs.append(dict(cards=cards, chosen=(ci if ci < n else n), boss=boss))
    return decs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', required=True)
    ap.add_argument('--ridge', type=float, default=2.0, help='L2 on base per-card scores')
    ap.add_argument('--ridge-boss', type=float, default=5.0, help='L2 on per-card,per-boss deltas')
    ap.add_argument('--val-frac', type=float, default=0.2)
    args = ap.parse_args()

    decs = load(args.episodes)
    bc = Counter(d['boss'] for d in decs)
    print(f"{len(decs)} act1-3 card-acquisition decisions toward a known boss")
    for a in (1, 2, 3):
        parts = "  ".join(f"{BOSSES[b]} {bc.get(b,0)}" for b in ACT_BOSSES[a])
        print(f"  act{a}: {parts}")
    print()

    offered = Counter(c for d in decs for c in d['cards'])
    vocab = sorted(offered)
    cidx = {c: i for i, c in enumerate(vocab)}
    K = len(vocab)
    bvocab = sorted(BOSSES)            # 9 bosses
    bidx = {b: i for i, b in enumerate(bvocab)}
    B = len(bvocab)
    acts = sorted({BOSS_ACT[b] for b in bvocab})
    aidx = {a: i for i, a in enumerate(acts)}
    print(f"{K} distinct cards; base ridge {args.ridge}, boss-delta ridge {args.ridge_boss}\n")

    MAXALT = max(len(d['cards']) for d in decs) + 1
    N = len(decs)
    card_id = torch.full((N, MAXALT), -1, dtype=torch.long)
    mask = torch.zeros((N, MAXALT), dtype=torch.bool)
    chosen = torch.zeros(N, dtype=torch.long)
    boss_of = torch.zeros(N, dtype=torch.long)   # index into bvocab
    act_of = torch.zeros(N, dtype=torch.long)    # index into acts
    for i, d in enumerate(decs):
        for j, c in enumerate(d['cards']):
            card_id[i, j] = cidx[c]
            mask[i, j] = True
        mask[i, len(d['cards'])] = True
        chosen[i] = d['chosen']
        boss_of[i] = bidx[d['boss']]
        act_of[i] = aidx[BOSS_ACT[d['boss']]]

    rng = np.random.default_rng(0)
    perm = torch.tensor(rng.permutation(N))
    nval = int(N * args.val_frac)
    val, tr = perm[:nval], perm[nval:]

    def fit(use_boss):
        base = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        actfe = torch.zeros(len(acts), dtype=torch.float64, requires_grad=True)
        delta = torch.zeros(K, B, dtype=torch.float64, requires_grad=use_boss)
        params = [base, actfe] + ([delta] if use_boss else [])

        def utils(idx):
            cid = card_id[idx].clamp_min(0)
            is_card = (card_id[idx] >= 0)
            u = torch.where(is_card, base[cid], torch.zeros(1, dtype=torch.float64))
            u = u + actfe[act_of[idx]][:, None] * is_card
            if use_boss:
                bd = delta[cid, boss_of[idx][:, None]]   # (len,MAXALT)
                u = u + bd * is_card
            return u.masked_fill(~mask[idx], -1e9)

        def loss(idx):
            u = utils(idx)
            nll = -(u.gather(1, chosen[idx, None]).squeeze(1) - torch.logsumexp(u, 1)).mean()
            reg = args.ridge / len(idx) * (base ** 2).sum()
            if use_boss:
                reg = reg + args.ridge_boss / len(idx) * (delta ** 2).sum()
            return nll + reg

        opt = torch.optim.LBFGS(params, max_iter=400, tolerance_grad=1e-10, line_search_fn='strong_wolfe')
        opt.step(lambda: (opt.zero_grad(), loss(tr).backward(), loss(tr))[2])
        with torch.no_grad():
            u = utils(val)
            ll = (u.gather(1, chosen[val, None]).squeeze(1) - torch.logsumexp(u, 1)).mean().item()
            acc = (u.argmax(1) == chosen[val]).double().mean().item()
        return base.detach(), delta.detach(), ll, acc

    sizes = mask[val].sum(1).double()
    ll_null = (-torch.log(sizes)).mean().item()

    base0, _, ll0, acc0 = fit(False)
    base1, delta, ll1, acc1 = fit(True)

    print("=== identity + act (no boss) ===")
    print(f"  val accuracy {acc0:.3f}   val log-loss {-ll0:.4f}   McFadden R2 {1-ll0/ll_null:+.3f}")
    print("=== + per-card x upcoming-boss ===")
    print(f"  val accuracy {acc1:.3f}   val log-loss {-ll1:.4f}   "
          f"boss gain {ll1-ll0:+.4f} nats/decision  ({acc1-acc0:+.3f} acc)\n")

    def nm(c):
        return str(sts.CardId(c)).replace('CardId.', '')

    # Per act: cards whose score swings most across that act's three bosses.
    for a in (1, 2, 3):
        bs = [bidx[b] for b in ACT_BOSSES[a]]
        # only cards offered a meaningful number of times in this act
        seen = Counter()
        for d in decs:
            if BOSS_ACT[d['boss']] == a:
                for c in d['cards']:
                    seen[c] += 1
        rows = []
        for c in vocab:
            if seen.get(c, 0) < 300:
                continue
            ds = [delta[cidx[c], bi].item() for bi in bs]
            rows.append((max(ds) - min(ds), c, ds))
        rows.sort(reverse=True)
        print(f"act {a} -- cards most boss-sensitive (delta spread across "
              f"{'/'.join(BOSSES[b] for b in ACT_BOSSES[a])}):")
        hdr = "  " + " ".join(f"{BOSSES[b]:>11}" for b in ACT_BOSSES[a])
        print(f"  {'card':22s}{hdr}   spread")
        for spread, c, ds in rows[:12]:
            cells = " ".join(f"{x:>+11.2f}" for x in ds)
            print(f"  {nm(c):22s}  {cells}   {spread:.2f}")
        print()


if __name__ == '__main__':
    main()
