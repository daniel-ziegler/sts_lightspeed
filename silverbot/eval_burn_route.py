"""Direct A/B of the cone feature: on path-decision states where the burning elite (emerald key)
is reachable via some offered options but not others, how much policy probability does each
checkpoint put on the burning-reaching option(s)?

Both checkpoints load into the current (cone-feature) architecture: iter_850 predates the cone
params so they load zero-init (no-op -> pre-cone behavior); iter_895 has 45 iters of cone
training. Identical states, so the delta is purely the trained cone routing signal.
"""
import argparse
import numpy as np, pandas as pd, torch
from silverbot import network as nw
from silverbot.network import NN, ModelHP, collate_fn, load_network_backward_compatible, CHOICE_PATHS_OFFSET


def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
    return load_network_backward_compatible(NN(ModelHP(use_value_head=True)), sd).to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--episodes', nargs='+', required=True)
    args = ap.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rows, p_uniform, burn_sets = [], [], []
    for f in args.episodes:
        df = pd.read_parquet(f)
        for _, r in df[df.choice_type == 3].iterrows():
            d = r.to_dict()
            n = len(d['paths_offered'])
            if n < 2:
                continue
            *_, opt_burn = nw.map_dag_features(d)
            bidx = [j for j in range(n) if opt_burn[j]]
            if 0 < len(bidx) < n:                       # distinguishable: some reach, some don't
                rows.append(d); burn_sets.append(set(bidx)); p_uniform.append(len(bidx) / n)
    print(f"{len(rows)} burning-distinguishable path decisions; "
          f"uniform-policy burn-mass baseline {np.mean(p_uniform):.3f}\n")

    bt = collate_fn(rows)
    for ck in args.ckpts:
        net = load_net(ck, device)
        pm, hit = [], []
        B = 256
        for i in range(0, len(rows), B):
            sub = {k: _slice(v, i, i + B) for k, v in bt.items()}
            with torch.no_grad():
                logits = net({k: _to(v, device) for k, v in sub.items()})[0]
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for bi, j0 in enumerate(range(i, min(i + B, len(rows)))):
                n = len(rows[j0]['paths_offered'])
                pp = probs[bi, CHOICE_PATHS_OFFSET:CHOICE_PATHS_OFFSET + n]
                tot = pp.sum()
                if tot <= 0:
                    continue
                bmass = sum(pp[j] for j in burn_sets[j0]) / tot
                pm.append(bmass)
                hit.append(int(pp.argmax() in burn_sets[j0]))
        print(f"{ck.split('/')[-1]:24s} burn-prob mass {np.mean(pm):.3f}   "
              f"argmax picks burning {np.mean(hit):.3f}   (n={len(pm)})")


def _slice(v, a, b):
    if isinstance(v, dict):
        return {k: _slice(x, a, b) for k, x in v.items()}
    return v[a:b]


def _to(v, device):
    if isinstance(v, dict):
        return {k: _to(x, device) for k, x in v.items()}
    return v.to(device)


if __name__ == '__main__':
    main()
