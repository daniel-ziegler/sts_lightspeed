"""Counterfactual: how much does the policy's CARD choice move when only the upcoming boss changes?

Mirror of analyze_ascension_probe.py, but on fixed_observation[4] (the boss-encoding token) and
restricted to card-choice states (screens with an offered card slate). For each real card-choice
state we override ONLY the boss token to each of the three bosses its act can draw, holding HP,
deck, gold, floor, enemies fixed, and re-run the policy. The spread of the resulting
distribution over the offered cards is the network's DIRECT sensitivity to which boss is coming
(it can't be a proxy via other features -- only the token moved).

Boss encoding (bindings/bindings-util.cpp getBossEncoding):
  act1 0 SlimeBoss 1 Hexaghost 2 Guardian | act2 3 Champ 4 Automaton 5 Collector
  act3 6 TimeEater 7 Donu&Deca 8 AwakenedOne | 9 Heart (act4, excluded)

Headline = mean pairwise TV distance over a state's three same-act bosses (0 identical, 1
disjoint), with the same cross-state TV as a "how big is big" scale reference.
"""
import argparse
import numpy as np, pandas as pd, torch
from silverbot.network import NN, ModelHP, collate_fn, load_network_backward_compatible

BOSS_IDX = 4
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


def set_boss_slot(rows, slot):
    """Override each row's boss to the slot-th boss of its own act."""
    out = []
    for r in rows:
        d = dict(r)
        fo = list(d['obs.fixed_observation'])
        fo[BOSS_IDX] = ACT_BOSSES[r['_act']][slot]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episodes', nargs='+', required=True)
    ap.add_argument('--n-states', type=int, default=6000)
    args = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = load_net(args.ckpt, device)

    rows = []
    for f in args.episodes:
        df = pd.read_parquet(f)
        df = df[df['cards_offered.cards'].apply(lambda x: len(x) > 1)]  # need >=2 cards for a choice
        for _, r in df.iterrows():
            d = r.to_dict()
            fo = d['obs.fixed_observation']
            if len(fo) <= BOSS_IDX:
                continue
            b = int(fo[BOSS_IDX])
            if b not in BOSS_ACT:        # skip Heart / invalid
                continue
            d['_act'] = BOSS_ACT[b]
            d['_realboss'] = b
            rows.append(d)
    rng = np.random.default_rng(0)
    if len(rows) > args.n_states:
        rows = [rows[i] for i in rng.choice(len(rows), size=args.n_states, replace=False)]
    act = np.array([r['_act'] for r in rows])
    ctype = np.array([int(r.get('choice_type', -1)) for r in rows])
    print(f"{len(rows)} card-choice states (>=2 cards offered, acts 1-3)")
    print(f"  by act: {dict(zip(*np.unique(act, return_counts=True)))}")
    print(f"  by choice_type: {dict(zip(*np.unique(ctype, return_counts=True)))}\n")

    # three forward passes: slot k -> every row's boss set to its act's k-th boss
    P = [fwd(net, set_boss_slot(rows, k), device) for k in range(3)]
    sm = [np.stack([softmax_valid(P[k][i]) for i in range(len(rows))]) for k in range(3)]

    def tv(a, b):
        return 0.5 * np.abs(a - b).sum(1)

    tv01, tv02, tv12 = tv(sm[0], sm[1]), tv(sm[0], sm[2]), tv(sm[1], sm[2])
    pair = np.stack([tv01, tv02, tv12], 1)
    mean_pair = pair.mean(1)
    max_pair = pair.max(1)
    arg = np.stack([sm[k].argmax(1) for k in range(3)], 1)
    flip = np.array([len(set(arg[i])) > 1 for i in range(len(rows))])

    print("=== POLICY card-choice shift when only the upcoming boss changes (within-act, 3 bosses) ===")
    print(f"  mean pairwise TV {mean_pair.mean():.3f}   max pairwise TV {max_pair.mean():.3f}")
    print(f"  top-card pick changes across the 3 bosses: {flip.mean():.2%} of states")
    print("  by act (mean pairwise TV / top-card-flip / n):")
    for a in (1, 2, 3):
        m = act == a
        if m.sum():
            print(f"    act{a} [{'/'.join(NAME[b] for b in ACT_BOSSES[a])}]: "
                  f"TV {mean_pair[m].mean():.3f}  flip {flip[m].mean():.2%}  (n={m.sum()})")
    print("  by choice_type (mean pairwise TV / flip / n):")
    for c in sorted(set(ctype.tolist())):
        m = ctype == c
        if m.sum() >= 20:
            print(f"    type {c}: TV {mean_pair[m].mean():.3f}  flip {flip[m].mean():.2%}  (n={m.sum()})")

    # which specific boss pairing moves choices most, per act
    print("\n  most-divergent boss pair within each act (mean TV over states of that act):")
    for a in (1, 2, 3):
        m = act == a
        bs = ACT_BOSSES[a]
        pairs = [(0, 1), (0, 2), (1, 2)]
        vals = [(tv(sm[i][m], sm[j][m]).mean(), bs[i], bs[j]) for i, j in pairs]
        vals.sort(reverse=True)
        s = "  ".join(f"{NAME[bi]}vs{NAME[bj]} {v:.3f}" for v, bi, bj in vals)
        print(f"    act{a}: {s}")

    # scale reference: TV between two DIFFERENT random states' policies
    perm = rng.permutation(len(rows))
    ref = tv(sm[0], sm[0][perm])
    print(f"\n  [scale ref] TV between different states' policies (same boss): {ref.mean():.3f}")


if __name__ == '__main__':
    main()
