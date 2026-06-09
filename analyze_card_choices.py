"""Conditional logit on heart1 card-ACQUISITION choices (combat rewards + shop buys).

Each card-offering screen presents n cards plus a "no card" outcome (skip a reward; or at a
shop, buy a relic/potion/remove/leave). We model the pick as a McFadden conditional logit:

    u(card c) = ASC[c] + beta . context(c, state);   u(no-card) = 0  (reference)

ASC[c] is the per-card alternative-specific constant -- the net's baseline appetite for card c
when it's on offer, vs declining. context features probe responsiveness to the rest of the
deck / act / boss / hp / gold.

Three nested fits gauge how much of the choice is fixed preference vs context:
  M0  take-any-card-vs-none constant only
  M1  + per-card ASCs (card identity)
  M2  + context (deck count of the offered card, deck size, act, boss, hp, gold, pstrikes, upg)
McFadden pseudo-R2 and held-out accuracy/log-loss are reported for each; the M1->M2 lift is the
context responsiveness. We then dump the top/bottom ASCs (what it most/least wants) and the
deck-count coefficient (duplicate seeking/aversion), with Perfected Strike called out.

Removes/upgrades (CARD_SELECT screens) are excluded -- acquisition only. Shop per-card prices
aren't in the episode parquet, so the shop price confounder is only partly absorbed by gold +
a shop indicator (acceptable per spec).
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch

import slaythespire as sts

REWARDS = int(sts.ScreenState.REWARDS)
SHOP = int(sts.ScreenState.SHOP_ROOM)
PSTRIKE = int(sts.CardId.PERFECTED_STRIKE)


def act_of_floor(f):
    return 1 + (f >= 17) + (f >= 34) + (f >= 51)


def load(parquets):
    cols = ['screen_state', 'cards_offered.cards', 'cards_offered.upgrades', 'chosen_idx',
            'obs.deck.cards', 'obs.fixed_observation']
    decs = []
    for fp in parquets:
        df = pd.read_parquet(fp, columns=cols)
        df = df[df.screen_state.isin([REWARDS, SHOP])]
        for _, r in df.iterrows():
            cards = [int(c) for c in r['cards_offered.cards']]
            if not cards:
                continue
            ups = [int(u) for u in r['cards_offered.upgrades']]
            n = len(cards)
            ci = int(r['chosen_idx'])
            chosen = ci if ci < n else n  # index n == the "no-card" alternative
            deck = Counter(int(c) for c in r['obs.deck.cards'])
            fo = list(r['obs.fixed_observation'])
            floor = int(fo[3]); boss = int(fo[4]); gold = int(fo[2])
            hp_frac = (fo[0] / fo[1]) if fo[1] else 0.0
            decs.append(dict(cards=cards, ups=ups, chosen=chosen, deck=deck,
                             deck_size=sum(deck.values()), n_pstrike=deck.get(PSTRIKE, 0),
                             act=act_of_floor(floor), boss=boss, gold=gold, hp_frac=hp_frac,
                             is_shop=int(r['screen_state'] == SHOP)))
    return decs


# context feature builder for one card alternative
CTX_NAMES = ['deck_count', 'deck_size/10', 'hp_frac', 'gold/200', 'is_shop',
             'act2', 'act3', 'act4', 'n_pstrike_deck', 'is_upgraded']


def ctx_vec(cardid, up, d):
    a = d['act']
    return [d['deck'].get(cardid, 0), d['deck_size'] / 10.0, d['hp_frac'], d['gold'] / 200.0,
            d['is_shop'], int(a == 2), int(a == 3), int(a >= 4), d['n_pstrike'], up]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', required=True)
    ap.add_argument('--min-offers', type=int, default=40, help='min times a card is offered to get its own ASC')
    ap.add_argument('--val-frac', type=float, default=0.2)
    args = ap.parse_args()

    decs = load(args.episodes)
    print(f"{len(decs)} card-acquisition decisions "
          f"({sum(d['is_shop'] for d in decs)} shop, {sum(1-d['is_shop'] for d in decs)} reward)")
    take = sum(1 for d in decs if d['chosen'] < len(d['cards']))
    print(f"took a card in {take} ({take/len(decs):.1%}); skipped/other in the rest\n")

    # Card vocabulary for ASCs: cards offered >= min_offers get their own constant; rest share one.
    offered = Counter()
    for d in decs:
        for c in d['cards']:
            offered[c] += 1
    asc_cards = sorted([c for c, k in offered.items() if k >= args.min_offers])
    asc_idx = {c: i for i, c in enumerate(asc_cards)}
    OTHER = len(asc_cards)  # shared ASC bucket for rare cards
    n_asc = len(asc_cards) + 1
    print(f"{n_asc-1} cards with their own ASC (offered >= {args.min_offers}); rest pooled\n")

    # Build padded tensors: per decision, up to MAXALT alternatives (cards + the no-card option).
    MAXALT = max(len(d['cards']) for d in decs) + 1
    N = len(decs)
    F = len(CTX_NAMES)
    asc_id = torch.full((N, MAXALT), -1, dtype=torch.long)   # -1 -> no ASC (the no-card ref)
    ctx = torch.zeros((N, MAXALT, F), dtype=torch.float64)
    base = torch.zeros((N, MAXALT), dtype=torch.float64)     # 1 on card alternatives (take-any const)
    mask = torch.zeros((N, MAXALT), dtype=torch.bool)
    chosen = torch.zeros(N, dtype=torch.long)
    for i, d in enumerate(decs):
        for j, (c, u) in enumerate(zip(d['cards'], d['ups'])):
            asc_id[i, j] = asc_idx.get(c, OTHER)
            ctx[i, j] = torch.tensor(ctx_vec(c, u, d), dtype=torch.float64)
            base[i, j] = 1.0
            mask[i, j] = True
        mask[i, len(d['cards'])] = True   # the no-card reference alternative
        chosen[i] = d['chosen']

    # standardize continuous context columns (for conditioning; ASCs/dummies left as-is)
    cont = [0, 1, 2, 3, 8]  # deck_count, deck_size, hp, gold, n_pstrike
    flat = ctx[mask]
    for c in cont:
        col = flat[:, c]
        mu, sd = col.mean(), col.std().clamp_min(1e-6)
        ctx[..., c] = (ctx[..., c] - mu) / sd

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    nval = int(N * args.val_frac)
    val, tr = perm[:nval], perm[nval:]
    val = torch.tensor(val); tr = torch.tensor(tr)

    def fit(use_asc, use_ctx, iters=300):
        b0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)          # take-any const
        asc = torch.zeros(n_asc, dtype=torch.float64, requires_grad=use_asc)
        beta = torch.zeros(F, dtype=torch.float64, requires_grad=use_ctx)
        params = [b0] + ([asc] if use_asc else []) + ([beta] if use_ctx else [])

        def utils(idx):
            u = b0 * base[idx]
            if use_asc:
                aid = asc_id[idx].clamp_min(0)
                u = u + torch.where(asc_id[idx] >= 0, asc[aid], torch.zeros_like(u)) * base[idx]
            if use_ctx:
                u = u + (ctx[idx] * beta).sum(-1) * base[idx]
            return u.masked_fill(~mask[idx], -1e9)

        def nll(idx):
            u = utils(idx)
            return -(u.gather(1, chosen[idx, None]).squeeze(1) - torch.logsumexp(u, 1)).mean()

        opt = torch.optim.LBFGS(params, max_iter=iters, tolerance_grad=1e-10, line_search_fn='strong_wolfe')
        opt.step(lambda: (opt.zero_grad(), nll(tr).backward(), nll(tr))[2])

        with torch.no_grad():
            for name, idx in (('train', tr), ('val', val)):
                u = utils(idx)
                ll = (u.gather(1, chosen[idx, None]).squeeze(1) - torch.logsumexp(u, 1))
                acc = (u.argmax(1) == chosen[idx]).double().mean().item()
                if name == 'val':
                    val_ll = ll.mean().item(); val_acc = acc
        return b0.detach(), asc.detach(), beta.detach(), val_ll, val_acc

    # M0: take-any const only (null-ish model). McFadden baseline.
    _, _, _, ll0, acc0 = fit(False, False)
    _, asc1, _, ll1, acc1 = fit(True, False)
    b2, asc2, beta2, ll2, acc2 = fit(True, True)

    print("model        val_logloss  val_acc   McFaddenR2(vs M0)")
    for name, ll, acc in (('M0 const', ll0, acc0), ('M1 +card ASC', ll1, acc1), ('M2 +context', ll2, acc2)):
        r2 = 1 - (ll / ll0)
        print(f"  {name:14s} {-ll:8.4f}    {acc:.3f}     {r2:+.3f}")
    print(f"\nM1->M2 log-loss gain (context responsiveness): {ll2-ll1:+.4f} nats/decision")

    print("\ncontext coefficients (M2, standardized continuous):")
    for nm, b in sorted(zip(CTX_NAMES, beta2.tolist()), key=lambda x: -abs(x[1])):
        print(f"  {nm:16s} {b:+.3f}")

    print("\nmost-wanted cards (highest ASC = most taken when offered, M2):")
    order = sorted(range(len(asc_cards)), key=lambda i: -asc2[i].item())
    for i in order[:18]:
        c = asc_cards[i]
        print(f"  {str(sts.CardId(c)).replace('CardId.',''):22s} {asc2[i].item():+.2f}  (offered {offered[c]})")
    print("  ...")
    for i in order[-8:]:
        c = asc_cards[i]
        print(f"  {str(sts.CardId(c)).replace('CardId.',''):22s} {asc2[i].item():+.2f}  (offered {offered[c]})")

    if PSTRIKE in asc_idx:
        pr = order.index(asc_idx[PSTRIKE])
        print(f"\nPERFECTED_STRIKE ASC {asc2[asc_idx[PSTRIKE]].item():+.2f} "
              f"(rank {pr+1} of {len(asc_cards)}; offered {offered[PSTRIKE]})")


if __name__ == '__main__':
    main()
