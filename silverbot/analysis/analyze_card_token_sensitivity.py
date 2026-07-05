"""Which specific cards' pick probability moves most when only the boss / ascension token changes?

Per-card attribution of the counterfactual probes. The policy's choice logits put the offered
cards in the first slots (slot j == cards_offered[j], before CHOICE_PATHS_OFFSET), so
softmax(logits)[j] is P(pick offered card j). On real card-choice states we override ONE token
and watch each offered card's pick probability move:

  boss: override fixed_observation[4] to each of the state's act's 3 bosses; per offered card,
        within-state spread = max_b P - min_b P. Report cards by mean spread, with the per-boss
        mean P (in the card's dominant act) for direction.
  ascension: override fixed_observation[6] to A0 vs A20; per offered card, signed dP = P@A20 -
        P@A0. Report cards most boosted / most suppressed by high ascension.

Probabilities are in pick-probability units (0..1), directly comparable to the TV summaries.
"""
import argparse
from collections import defaultdict
import numpy as np, pandas as pd, torch
from silverbot.network import NN, ModelHP, collate_fn, load_network_backward_compatible
import slaythespire as sts

BOSS_IDX, ASC_IDX = 4, 6
ACT_BOSSES = {1: [0, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8]}
BOSS_ACT = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}
NAME = {0: 'SlimeBoss', 1: 'Hexaghost', 2: 'Guardian', 3: 'Champ', 4: 'Automaton',
        5: 'Collector', 6: 'TimeEater', 7: 'Donu&Deca', 8: 'AwakenedOne'}


def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
    return load_network_backward_compatible(NN(ModelHP(use_value_head=True)), sd).to(device).eval()


def _to(v, device):
    if isinstance(v, dict):
        return {k: _to(x, device) for k, x in v.items()}
    return v.to(device)


def override(rows, idx, valfn):
    out = []
    for r in rows:
        d = dict(r)
        fo = list(d['obs.fixed_observation']); fo[idx] = valfn(r)
        d['obs.fixed_observation'] = fo
        out.append(d)
    return out


def fwd(net, rows, device, B=256):
    lg = []
    for i in range(0, len(rows), B):
        bt = _to(collate_fn(rows[i:i + B]), device)
        with torch.no_grad():
            out = net(bt)
        lg.append(out[0].float().cpu().numpy())
    return np.concatenate(lg)


def softmax_valid(logits):
    m = np.isfinite(logits)
    z = np.where(m, logits, -1e9)
    z = z - z.max(); e = np.exp(z) * m
    return e / e.sum()


def nm(c):
    return str(sts.CardId(int(c))).replace('CardId.', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episodes', nargs='+', required=True)
    ap.add_argument('--n-states', type=int, default=8000)
    ap.add_argument('--min-offers', type=int, default=120)
    ap.add_argument('--act', type=int, default=0, help='restrict to this act (0 = all acts)')
    args = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = load_net(args.ckpt, device)

    rows = []
    for f in args.episodes:
        df = pd.read_parquet(f)
        df = df[df['cards_offered.cards'].apply(lambda x: len(x) > 1)]
        for _, r in df.iterrows():
            d = r.to_dict()
            fo = d['obs.fixed_observation']
            if len(fo) <= ASC_IDX:
                continue
            b = int(fo[BOSS_IDX])
            if b not in BOSS_ACT:
                continue
            if args.act and BOSS_ACT[b] != args.act:
                continue
            d['_act'] = BOSS_ACT[b]
            d['_cards'] = [int(c) for c in d['cards_offered.cards']]
            rows.append(d)
    rng = np.random.default_rng(0)
    if len(rows) > args.n_states:
        rows = [rows[i] for i in rng.choice(len(rows), size=args.n_states, replace=False)]
    print(f"{len(rows)} card-choice states ({'act '+str(args.act) if args.act else 'acts 1-3'}, "
          f">=2 cards)\n")

    # ---- BOSS: 3 forward passes, boss = act's k-th boss ----
    Pb = [np.stack([softmax_valid(lg) for lg in
                    fwd(net, override(rows, BOSS_IDX, lambda r, k=k: ACT_BOSSES[r['_act']][k]), device)])
          for k in range(3)]
    # ---- ASCENSION: A0 and A20 ----
    P0 = np.stack([softmax_valid(lg) for lg in fwd(net, override(rows, ASC_IDX, lambda r: 0), device)])
    P20 = np.stack([softmax_valid(lg) for lg in fwd(net, override(rows, ASC_IDX, lambda r: 20), device)])

    boss_spread = defaultdict(list)                     # cardid -> within-state max-min over 3 bosses
    boss_bymean = defaultdict(lambda: defaultdict(list))  # cardid -> {boss_enc -> [P]}
    asc_d = defaultdict(list)                            # cardid -> signed P@A20 - P@A0
    asc_p0 = defaultdict(list)
    for i, r in enumerate(rows):
        a = r['_act']
        for j, c in enumerate(r['_cards']):
            ps = [Pb[k][i, j] for k in range(3)]
            boss_spread[c].append(max(ps) - min(ps))
            for k in range(3):
                boss_bymean[c][ACT_BOSSES[a][k]].append(Pb[k][i, j])
            asc_d[c].append(P20[i, j] - P0[i, j])
            asc_p0[c].append(P0[i, j])

    def keep(c):
        return len(boss_spread[c]) >= args.min_offers

    print("================ BOSS: cards whose pick-prob swings most across the act's 3 bosses ================")
    print("(spread = mean over offered states of [max_boss P - min_boss P]; pick-prob units)\n")
    cards = [c for c in boss_spread if keep(c)]
    cards.sort(key=lambda c: -np.mean(boss_spread[c]))
    for c in cards[:20]:
        sp = np.mean(boss_spread[c])
        # dominant act for direction
        acts = [BOSS_ACT[b] for b in boss_bymean[c]]
        dom = max(set(acts), key=acts.count)
        bs = ACT_BOSSES[dom]
        means = {b: np.mean(boss_bymean[c][b]) for b in bs if boss_bymean[c][b]}
        dirn = "  ".join(f"{NAME[b]} {means[b]:.2f}" for b in bs if b in means)
        hi = max(means, key=means.get); lo = min(means, key=means.get)
        print(f"  {nm(c):20s} spread {sp:.3f}  (n={len(boss_spread[c])}, act{dom}: {dirn})"
              f"   most->{NAME[hi]} least->{NAME[lo]}")

    print("\n================ ASCENSION: cards most boosted / suppressed at A20 vs A0 ================")
    print("(dP = mean P(pick | A20) - P(pick | A0); + means high-ascension wants it MORE)\n")
    cards = [c for c in asc_d if keep(c)]
    cards.sort(key=lambda c: -np.mean(asc_d[c]))
    print("  MORE wanted at high ascension:")
    for c in cards[:12]:
        print(f"    {nm(c):20s} dP {np.mean(asc_d[c]):+.3f}  (A0 {np.mean(asc_p0[c]):.2f} -> "
              f"A20 {np.mean(asc_p0[c])+np.mean(asc_d[c]):.2f}, n={len(asc_d[c])})")
    print("  LESS wanted at high ascension:")
    for c in cards[-12:][::-1]:
        print(f"    {nm(c):20s} dP {np.mean(asc_d[c]):+.3f}  (A0 {np.mean(asc_p0[c]):.2f} -> "
              f"A20 {np.mean(asc_p0[c])+np.mean(asc_d[c]):.2f}, n={len(asc_d[c])})")


if __name__ == '__main__':
    main()
