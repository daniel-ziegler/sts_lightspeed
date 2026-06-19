"""Pure conditional logit on heart1 card-ACQUISITION choices (combat rewards + shop buys).

Primary model -- assumption-light, "fully captures the alternatives": each card-offering screen
presents a slate of cards plus a "no card" outcome (skip a reward; or at a shop, buy a
relic/potion, remove, or leave). Every distinct card gets ONE free score; "no card" is the
pinned reference (0). The pick is modeled as a softmax over the offered slate's scores and fit
by cross-entropy against the taken option:

    P(take card c | slate S) = exp(score[c]) / (1 + sum_{k in S} exp(score[k]))

No deck/act/boss/hp context is modeled. The competition between alternatives is handled by the
softmax (a card's score is deconfounded from whatever it was offered against), so score[c] is
the net's pure identity appetite for c when on offer, vs declining. A small ridge keeps
rarely-offered cards finite. The held-out accuracy / McFadden R2 of this identity-only model is
the headline: how much of card choice is a fixed ranking.

A secondary section adds deck/act/hp/gold context to the same fit to measure responsiveness
(how much the choice conditions on the rest of the deck) -- the deck_count coefficient is the
duplicate-aversion estimate.

Removes/upgrades (CARD_SELECT) are excluded -- acquisition only. Shop per-card prices aren't in
the parquet, so the shop price confounder is only partly absorbed (gold + shop flag, secondary
model only).
"""
import argparse

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
            n = len(cards)
            ci = int(r['chosen_idx'])
            chosen = ci if ci < n else n  # index n == the "no-card" reference alternative
            from collections import Counter
            deck = Counter(int(c) for c in r['obs.deck.cards'])
            fo = list(r['obs.fixed_observation'])
            decs.append(dict(cards=cards, ups=[int(u) for u in r['cards_offered.upgrades']],
                             chosen=chosen, deck=deck, deck_size=sum(deck.values()),
                             n_pstrike=deck.get(PSTRIKE, 0), act=act_of_floor(int(fo[3])),
                             gold=int(fo[2]), hp_frac=(fo[0] / fo[1]) if fo[1] else 0.0,
                             is_shop=int(r['screen_state'] == SHOP)))
    return decs


CTX_NAMES = ['deck_count', 'deck_size/10', 'hp_frac', 'gold/200', 'is_shop',
             'act2', 'act3', 'act4', 'n_pstrike_deck', 'is_upgraded']


def ctx_vec(cardid, up, d):
    a = d['act']
    return [d['deck'].get(cardid, 0), d['deck_size'] / 10.0, d['hp_frac'], d['gold'] / 200.0,
            d['is_shop'], int(a == 2), int(a == 3), int(a >= 4), d['n_pstrike'], up]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', required=True)
    ap.add_argument('--ridge', type=float, default=2.0, help='L2 on per-card scores (keeps rare cards finite)')
    ap.add_argument('--val-frac', type=float, default=0.2)
    args = ap.parse_args()

    decs = load(args.episodes)
    take = sum(1 for d in decs if d['chosen'] < len(d['cards']))
    print(f"{len(decs)} card-acquisition decisions "
          f"({sum(d['is_shop'] for d in decs)} shop, {sum(1 - d['is_shop'] for d in decs)} reward); "
          f"took a card {take/len(decs):.1%} of the time\n")

    # Vocabulary: EVERY distinct offered card gets its own score (no pooling).
    from collections import Counter
    offered = Counter(c for d in decs for c in d['cards'])
    cards_vocab = sorted(offered)
    cidx = {c: i for i, c in enumerate(cards_vocab)}
    K = len(cards_vocab)
    print(f"{K} distinct cards offered; each gets its own score (ridge {args.ridge})\n")

    MAXALT = max(len(d['cards']) for d in decs) + 1
    N, F = len(decs), len(CTX_NAMES)
    card_id = torch.full((N, MAXALT), -1, dtype=torch.long)  # -1 = no-card reference
    ctx = torch.zeros((N, MAXALT, F), dtype=torch.float64)
    mask = torch.zeros((N, MAXALT), dtype=torch.bool)
    chosen = torch.zeros(N, dtype=torch.long)
    for i, d in enumerate(decs):
        for j, (c, u) in enumerate(zip(d['cards'], d['ups'])):
            card_id[i, j] = cidx[c]
            ctx[i, j] = torch.tensor(ctx_vec(c, u, d), dtype=torch.float64)
            mask[i, j] = True
        mask[i, len(d['cards'])] = True  # no-card reference
        chosen[i] = d['chosen']
    is_card = card_id >= 0
    cont = [0, 1, 2, 3, 8]
    flat = ctx[mask & is_card]
    for c in cont:
        col = flat[:, c]
        ctx[..., c] = (ctx[..., c] - col.mean()) / col.std().clamp_min(1e-6)

    rng = np.random.default_rng(0)
    perm = torch.tensor(rng.permutation(N))
    nval = int(N * args.val_frac)
    val, tr = perm[:nval], perm[nval:]

    def run(use_ctx):
        score = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        beta = torch.zeros(F, dtype=torch.float64, requires_grad=use_ctx)
        params = [score] + ([beta] if use_ctx else [])

        def utils(idx):
            cid = card_id[idx].clamp_min(0)
            u = torch.where(card_id[idx] >= 0, score[cid], torch.zeros(1, dtype=torch.float64))
            if use_ctx:
                u = u + (ctx[idx] * beta).sum(-1) * is_card[idx]
            return u.masked_fill(~mask[idx], -1e9)

        def loss(idx):
            u = utils(idx)
            nll = -(u.gather(1, chosen[idx, None]).squeeze(1) - torch.logsumexp(u, 1)).mean()
            return nll + args.ridge / len(idx) * (score ** 2).sum()

        opt = torch.optim.LBFGS(params, max_iter=400, tolerance_grad=1e-10, line_search_fn='strong_wolfe')
        opt.step(lambda: (opt.zero_grad(), loss(tr).backward(), loss(tr))[2])
        with torch.no_grad():
            u = utils(val)
            ll = (u.gather(1, chosen[val, None]).squeeze(1) - torch.logsumexp(u, 1)).mean().item()
            acc = (u.argmax(1) == chosen[val]).double().mean().item()
        return score.detach(), beta.detach(), ll, acc

    # null: equal prob over the offered slate (each decision: 1/(n_cards+1))
    sizes = mask[val].sum(1).double()
    ll_null = (-torch.log(sizes)).mean().item()

    score, _, ll_pure, acc_pure = run(False)
    _, beta, ll_ctx, acc_ctx = run(True)

    print("=== PURE identity model (card scores only, no context) ===")
    print(f"  val accuracy {acc_pure:.3f}   val log-loss {-ll_pure:.4f}   "
          f"McFadden R2 (vs equal-prob) {1 - ll_pure/ll_null:+.3f}")
    print("=== + context (deck/act/hp/gold/upgrade) ===")
    print(f"  val accuracy {acc_ctx:.3f}   val log-loss {-ll_ctx:.4f}   "
          f"identity->+context gain {ll_ctx - ll_pure:+.4f} nats/decision\n")

    order = sorted(range(K), key=lambda i: -score[i].item())
    def show(i):
        c = cards_vocab[i]
        print(f"  {str(sts.CardId(c)).replace('CardId.',''):22s} {score[i].item():+.2f}  (offered {offered[c]})")
    print("most-wanted when offered (pure identity score, skip=0 reference):")
    for i in order[:20]:
        show(i)
    print("  ...least-wanted:")
    for i in order[-10:]:
        show(i)

    pr = order.index(cidx[PSTRIKE])
    print(f"\nPERFECTED_STRIKE score {score[cidx[PSTRIKE]].item():+.2f} "
          f"(rank {pr+1}/{K}; offered {offered[PSTRIKE]})")

    print("\ncontext coefficients (secondary fit; deck-responsiveness):")
    for nm, b in sorted(zip(CTX_NAMES, beta.tolist()), key=lambda x: -abs(x[1])):
        print(f"  {nm:16s} {b:+.3f}")


if __name__ == '__main__':
    main()
