"""SL probe: can the network learn map-dependent path policies from this obs encoding?

Trains a fresh net (RL architecture) on synthetic relabelings of real path decisions:
  - leftmost: pick the lowest-x option. Needs NO map parsing (x is in the option token
    itself) -- a pipeline/optimization control.
  - elite / rest / monster: pick the (lowest-x) option whose DESTINATION node is of the
    given type, on the subset of decisions where one is offered. Solvable only by
    resolving option x -> map node at (x, current_y + 1) -> room type, i.e. exactly the
    relational map-parsing the RL policy would need for "always take elites".

High probe accuracy => the encoding is learnable and the RL policy's indifference is a
choice/optimization outcome; near-chance accuracy => the map encoding is the bottleneck.
"""
import argparse
import glob
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # before torch (teardown race)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import slaythespire as sts
from network import NN, ModelHP, collate_fn, MAX_DECK_SIZE, MAX_FIXED_ACTIONS

PATHS_OFFSET = MAX_DECK_SIZE + 3 + sts.MAX_POTION_CAPACITY + MAX_FIXED_ACTIONS  # collate_fn padding


def index_batch(bt, idx):
    """Recursively index a collated batch structure with a 1-D index tensor."""
    if isinstance(bt, dict):
        return {k: index_batch(v, idx) for k, v in bt.items()}
    return bt[idx]


def to_device(bt, device):
    if isinstance(bt, dict):
        return {k: to_device(v, device) for k, v in bt.items()}
    return bt.to(device)


def load_path_rows(files):
    """All PATH decisions with resolvable destination types. Returns (rows, dest_types)."""
    rows, types = [], []
    for f in files:
        df = pd.read_parquet(f)
        for _, r in df[df.choice_type == 3].iterrows():
            node = {(int(x), int(y)): int(t) for x, y, t in
                    zip(r['obs.map.xs'], r['obs.map.ys'], r['obs.map.roomTypes'])}
            ydest = int(r['obs.mapY']) + 1
            tn = [sts.Room(node[(int(x), ydest)]).name if (int(x), ydest) in node else None
                  for x in r['paths_offered']]
            if None in tn or len(tn) < 2:
                continue
            rows.append(r)
            types.append(tn)
    return rows, types


def make_task(rows, types, task):
    """-> (row dicts with relabeled chosen_idx, labels k, chance acc) for the task subset."""
    out, ks = [], []
    chance = []
    for r, tn in zip(rows, types):
        xs = [int(x) for x in r['paths_offered']]
        if task == 'leftmost':
            k = int(np.argmin(xs))
        else:
            target = task.upper()
            cand = [i for i, t in enumerate(tn) if t == target]
            if not cand or len(cand) == len(tn):
                continue  # need the target present AND distinguishable
            k = min(cand, key=lambda i: xs[i])
        d = r.to_dict()
        d['chosen_idx'] = PATHS_OFFSET + k
        out.append(d)
        ks.append(PATHS_OFFSET + k)
        chance.append(1.0 / len(xs))
    return out, np.array(ks), float(np.mean(chance))


def run_task(task, rows, types, args, device):
    data, labels, chance = make_task(rows, types, task)
    if len(data) < 500:
        print(f"[{task}] only {len(data)} examples, skipping")
        return
    val = np.array([is_valid_seed(d['seed']) for d in data])
    print(f"\n=== task {task}: {len(data)} examples ({val.sum()} valid), "
          f"chance acc {chance:.3f} ===", flush=True)

    bt = collate_fn(data)
    y = torch.tensor(labels, dtype=torch.long)
    tr_idx = torch.tensor(np.where(~val)[0])
    va_idx = torch.tensor(np.where(val)[0])

    def valid_acc(net):
        hits = tot = 0
        with torch.no_grad():
            for i in range(0, len(va_idx), args.batch_size):
                idx = va_idx[i:i + args.batch_size]
                b = to_device(index_batch(bt, idx), device)
                hits += (net(b).argmax(-1) == y[idx].to(device)).sum().item()
                tot += len(idx)
        return hits / tot

    net = NN(ModelHP(use_value_head=False, dim=args.dim, n_layers=args.n_layers)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    for epoch in range(args.epochs):
        net.train()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        tot = n = 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            b = to_device(index_batch(bt, idx), device)
            logits = net(b)
            loss = F.cross_entropy(logits, y[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        net.eval()
        acc = valid_acc(net)
        print(f"[{task}] epoch {epoch + 1}: train loss {tot / n:.4f}, valid acc {acc:.4f}", flush=True)
    print(f"[{task}] FINAL valid acc {acc:.4f} (chance {chance:.3f})", flush=True)


def is_valid_seed(seed, frac=0.15):
    return (((int(seed) * 1327217885) & 0xFFFFFFFF) / 0xFFFFFFFF) < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes-glob', default='runs/ppo_hient.pt.episodes/*.parquet')
    ap.add_argument('--tasks', default='leftmost,elite,rest,monster')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--n-layers', type=int, default=4)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    files = sorted(glob.glob(args.episodes_glob))
    print(f"loading path decisions from {len(files)} files on {device}", flush=True)
    rows, types = load_path_rows(files)
    print(f"{len(rows)} path decisions with resolvable destinations", flush=True)
    for task in args.tasks.split(','):
        run_task(task, rows, types, args, device)


if __name__ == '__main__':
    main()
