"""Follow-up value-SL sweep: tiny transformers, higher weight decay, and an n_layers=0
reduce-to-linear cross-check. Proper 3-way seed split (train/val/test): early-stop on val,
report UNBIASED EV on the held-out test set (removes the best-epoch-on-val selection bias).
"""
import argparse, time, json
import numpy as np, torch, torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from network import NN, ModelHP, SlayDataset, move_to_device
from value_sl import load_last_third, value_collate, expl_var, DEVICE


def eval_ev(net, loader, ymean, ystd):
    net.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            b = move_to_device(b, DEVICE)
            out = net(b); v = out[1] if isinstance(out, tuple) else out
            ps.append(v.cpu().numpy() * ystd + ymean); ys.append(b['return'].cpu().numpy())
    return expl_var(np.concatenate(ys), np.concatenate(ps))


def run(cfg, tr, va, te, ymean, ystd, log):
    torch.manual_seed(0)
    if cfg.get('train_subset'):  # for the overfit-capacity check
        tr = tr.sample(min(cfg['train_subset'], len(tr)), random_state=0).reset_index(drop=True)
    H = ModelHP(use_value_head=True, dim=cfg['dim'], n_layers=cfg['n_layers'],
                num_value_layers=cfg.get('num_value_layers', 0), value_fork_layer=0)
    net = NN(H).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    tl = DataLoader(SlayDataset(tr), batch_size=cfg['batch_size'], shuffle=True,
                    collate_fn=value_collate, drop_last=True, num_workers=4, persistent_workers=True)
    vl = DataLoader(SlayDataset(va), batch_size=256, collate_fn=value_collate, num_workers=2, persistent_workers=True)
    tel = DataLoader(SlayDataset(te), batch_size=256, collate_fn=value_collate, num_workers=2, persistent_workers=True)
    # fixed train subsample to measure TRAIN EV (overfit check: does train EV -> ~1.0?)
    tr_eval = tr.sample(min(3000, len(tr)), random_state=1).reset_index(drop=True)
    trel = DataLoader(SlayDataset(tr_eval), batch_size=256, collate_fn=value_collate, num_workers=2, persistent_workers=True)
    best_val, best_test, best_ep, since, best_train = -1e9, None, -1, 0, None
    for ep in range(cfg['epochs']):
        net.train()
        for b in tl:
            b = move_to_device(b, DEVICE); tgt = (b['return'] - ymean) / ystd
            out = net(b); v = out[1] if isinstance(out, tuple) else out
            loss = F.mse_loss(v, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        tre = eval_ev(net, trel, ymean, ystd); ve = eval_ev(net, vl, ymean, ystd); tee = eval_ev(net, tel, ymean, ystd)
        if ve > best_val:
            best_val, best_test, best_ep, best_train, since = ve, tee, ep, tre, 0
        else:
            since += 1
        log(f"    ep{ep:02d} train={tre:.4f} val={ve:.4f} test={tee:.4f}  (best val {best_val:.4f}@{best_ep})")
        if since >= cfg.get('patience', 5):
            break
    return best_val, best_test, best_ep, best_train, tre  # final train EV = overfit level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epi-dir', default='runs/ppo_4ep.pt.episodes')
    ap.add_argument('--max-rows', type=int, default=60000)
    ap.add_argument('--out', default='runs/value_sl_followup.jsonl')
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)

    df, lo, hi = load_last_third(args.epi_dir, max_rows=args.max_rows)
    seeds = df['seed'].unique().copy(); np.random.RandomState(0).shuffle(seeds)
    n = len(seeds); te_s = set(seeds[:int(n*0.15)].tolist()); va_s = set(seeds[int(n*0.15):int(n*0.30)].tolist())
    te = df[df['seed'].isin(te_s)].reset_index(drop=True)
    va = df[df['seed'].isin(va_s)].reset_index(drop=True)
    tr = df[~df['seed'].isin(te_s | va_s)].reset_index(drop=True)
    ymean, ystd = float(tr['return'].mean()), float(tr['return'].std() + 1e-8)
    base_test = expl_var(te['return'].to_numpy(), te['value'].to_numpy())
    log(f"iters {lo}..{hi}  train={len(tr):,} val={len(va):,} test={len(te):,}  saved-value test EV={base_test:.4f} (semi-in-sample)\n")

    base = dict(dim=256, n_layers=4, lr=3e-4, wd=1e-4, batch_size=128, epochs=25, patience=5)
    configs = [
        # CAPACITY CHECK: small subset, no weight decay, no early stop -> train EV should -> ~1.0
        dict(base, name='overfit_small', wd=0.0, epochs=60, patience=999, train_subset=4000),
        # underfitting control: full data, base arch, ZERO weight decay (is even 1e-4 suppressing the fit?)
        dict(base, name='base_wd0', wd=0.0),
        dict(base, name='base_wd0_long', wd=0.0, epochs=40, patience=12),  # let train EV climb
        dict(base, name='nlayers0_linear', n_layers=0),
        dict(base, name='tiny_d64_L1', dim=64, n_layers=1, batch_size=256),
        dict(base, name='tiny_d64_L2', dim=64, n_layers=2, batch_size=256),
        dict(base, name='tiny_d128_L2', dim=128, n_layers=2, batch_size=128),
        dict(base, name='wd1e-2', wd=1e-2),
        dict(base, name='wd3e-2', wd=3e-2),
        dict(base, name='wd1e-1', wd=1e-1),
        dict(base, name='d64L2_wd1e-2', dim=64, n_layers=2, wd=1e-2, batch_size=256),
    ]
    results = []
    with open(args.out, 'w') as f:
        for i, cfg in enumerate(configs):
            t0 = time.time(); log(f"[{i+1}/{len(configs)}] {cfg['name']}: {cfg}")
            try:
                bv, bt, be, btr, final_tr = run(cfg, tr, va, te, ymean, ystd, log)
            except Exception as e:
                log(f"  CONFIG FAILED: {e}"); continue
            rec = dict(cfg, best_val=bv, best_test=bt, best_ep=be, train_at_best=btr,
                       final_train_ev=final_tr, secs=round(time.time()-t0, 1))
            results.append(rec); f.write(json.dumps(rec) + '\n'); f.flush()
            log(f"  -> test EV={bt:.4f} (val {bv:.4f}, train@best {btr:.4f}, final train {final_tr:.4f}) @ep{be}  ({rec['secs']}s)\n")
    results.sort(key=lambda r: -(r['best_test'] if r['best_test'] is not None else -9))
    log("=== RANKED by held-out TEST EV (train EV shows capacity/overfit) ===")
    log(f"saved-value baseline (test, semi-in-sample): {base_test:.4f}")
    for r in results:
        log(f"  test {r['best_test']:.4f}  | final-train {r['final_train_ev']:.4f}  {r['name']}")


if __name__ == '__main__':
    main()
