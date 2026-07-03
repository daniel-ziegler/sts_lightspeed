"""Offline SL value-function fitting: how well can a net predict the GAE return-to-go from state?

Dataset = last third of the last local run (runs/ppo_4ep.pt.episodes). Target = the `return`
column (the PPO value target). Metric = explained variance on a HELD-OUT split, split by game
(seed) so no game leaks across train/val. Sweeps value-head architecture + optimizer hyperparams.

Question: the PPO run plateaued at value EV ~0.38. Is that an optimization/architecture ceiling
(=> a better-tuned value net beats it) or irreducible return variance (=> it doesn't)?
"""
import argparse, glob, os, re, json, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from lightspeed.network import NN, ModelHP, SlayDataset, collate_fn, move_to_device

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
iter_of = lambda p: int(re.search(r'iter_(\d+)', p).group(1))


def load_last_third(epi_dir, frac=1/3, max_rows=200000, seed=0):
    files = sorted(glob.glob(os.path.join(epi_dir, '*.parquet')), key=iter_of)
    k = max(1, int(round(len(files) * frac)))
    files = files[-k:]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df, iter_of(files[0]), iter_of(files[-1])


def split_by_seed(df, val_frac=0.15, seed=0):
    seeds = df['seed'].unique().copy()
    np.random.RandomState(seed).shuffle(seeds)
    val_seeds = set(seeds[:int(len(seeds) * val_frac)].tolist())
    is_val = df['seed'].isin(val_seeds).to_numpy()
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def value_collate(batch):
    collated = collate_fn(batch)
    collated['return'] = torch.tensor([float(b['return']) for b in batch], dtype=torch.float32)
    return collated


def expl_var(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1.0 - np.var(y - yhat) / (np.var(y) + 1e-12))


def run_config(cfg, train_df, val_df, ymean, ystd, log):
    torch.manual_seed(cfg.get('seed', 0))
    H = ModelHP(use_value_head=True, dim=cfg['dim'], n_layers=cfg['n_layers'],
                num_value_layers=cfg['num_value_layers'], value_fork_layer=cfg['value_fork_layer'])
    net = NN(H).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    tl = torch.utils.data.DataLoader(SlayDataset(train_df), batch_size=cfg['batch_size'],
                                     shuffle=True, collate_fn=value_collate, drop_last=True,
                                     num_workers=4, persistent_workers=True)
    vl = torch.utils.data.DataLoader(SlayDataset(val_df), batch_size=256,
                                     shuffle=False, collate_fn=value_collate,
                                     num_workers=2, persistent_workers=True)
    best_ev, best_ep, since = -1e9, -1, 0
    for ep in range(cfg['epochs']):
        net.train()
        for b in tl:
            b = move_to_device(b, DEVICE)
            tgt = (b['return'] - ymean) / ystd
            _, v = net(b)
            loss = F.mse_loss(v, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval(); ys, ps = [], []
        with torch.no_grad():
            for b in vl:
                b = move_to_device(b, DEVICE)
                _, v = net(b)
                ps.append(v.cpu().numpy() * ystd + ymean)
                ys.append(b['return'].cpu().numpy())
        e = expl_var(np.concatenate(ys), np.concatenate(ps))
        if e > best_ev:
            best_ev, best_ep, since = e, ep, 0
        else:
            since += 1
        log(f"    ep{ep:02d} val_EV={e:.4f}  (best {best_ev:.4f}@{best_ep})")
        if since >= cfg.get('patience', 6):
            break
    return best_ev, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epi-dir', default='runs/ppo_4ep.pt.episodes')
    ap.add_argument('--max-rows', type=int, default=200000)
    ap.add_argument('--out', default='runs/value_sl_results.jsonl')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    def log(*a):
        print(*a, flush=True)

    df, lo, hi = load_last_third(args.epi_dir, max_rows=args.max_rows)
    log(f"loaded {len(df):,} states from {args.epi_dir} iters {lo}..{hi}")
    base_ev = expl_var(df['return'].to_numpy(), df['value'].to_numpy())
    log(f"BASELINE (saved value head vs return, in-sample): EV={base_ev:.4f}")
    log(f"return: mean={df['return'].mean():.3f} std={df['return'].std():.3f}")

    train_df, val_df = split_by_seed(df, val_frac=0.15, seed=0)
    ymean, ystd = float(train_df['return'].mean()), float(train_df['return'].std() + 1e-8)
    base_ev_val = expl_var(val_df['return'].to_numpy(), val_df['value'].to_numpy())
    log(f"train={len(train_df):,} val={len(val_df):,} (split by seed) | baseline val EV={base_ev_val:.4f}\n")

    base = dict(dim=256, n_layers=4, num_value_layers=0, value_fork_layer=0,
                lr=3e-4, wd=1e-4, batch_size=128, epochs=25, patience=5, seed=0)
    if args.smoke:
        configs = [dict(base, epochs=2, name='smoke')]
    else:
        configs = [
            dict(base, name='base'),
            dict(base, lr=1e-4, name='lr1e-4'),
            dict(base, lr=5e-4, name='lr5e-4'),
            dict(base, lr=1e-3, name='lr1e-3'),
            dict(base, wd=0.0, name='wd0'),
            dict(base, wd=1e-3, name='wd1e-3'),
            dict(base, num_value_layers=2, name='vlayers2'),
            dict(base, num_value_layers=4, batch_size=96, name='vlayers4'),
            dict(base, num_value_layers=2, value_fork_layer=2, name='vlayers2_fork2'),
            dict(base, dim=384, batch_size=64, name='dim384'),
            dict(base, dim=384, num_value_layers=2, batch_size=64, name='dim384_vlayers2'),
            dict(base, n_layers=6, batch_size=64, name='nlayers6'),
            dict(base, n_layers=6, dim=384, batch_size=64, name='nlayers6_dim384'),
            dict(base, lr=1e-3, wd=1e-3, name='lr1e-3_wd1e-3'),
            dict(base, lr=5e-4, dim=384, num_value_layers=2, batch_size=64, name='combo_dim384_vl2_lr5e-4'),
        ]

    results = []
    with open(args.out, 'w') as f:
        for i, cfg in enumerate(configs):
            t0 = time.time()
            log(f"[{i+1}/{len(configs)}] {cfg['name']}: {cfg}")
            best_ev, best_ep = run_config(cfg, train_df, val_df, ymean, ystd, log)
            rec = dict(cfg, best_val_ev=best_ev, best_epoch=best_ep, secs=round(time.time()-t0, 1))
            results.append(rec)
            f.write(json.dumps(rec) + '\n'); f.flush()
            log(f"  -> best_val_EV={best_ev:.4f} @ep{best_ep}  ({rec['secs']}s)  | baseline {base_ev_val:.4f}\n")

    results.sort(key=lambda r: -r['best_val_ev'])
    log("=== RANKED (best val EV) ===")
    log(f"baseline (saved value head): {base_ev_val:.4f}")
    for r in results:
        log(f"  {r['best_val_ev']:.4f}  {r['name']}")


if __name__ == '__main__':
    main()
