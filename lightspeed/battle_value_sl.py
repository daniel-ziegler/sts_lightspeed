"""Battle-outcome SL lab: does predicting a specific battle's ΔHP improve the value fit?

The gate experiment for BATTLE_OUTCOME_PLAN.md Phase 3. Value task = predict the GAE
return-to-go on heart1 episode states (held-out EV, split by game seed — value_sl.py's
protocol). Battle task = predict a battle's outcome from (state obs, encounter) on
gen_battle_outcomes.py shards, as 20-bucket CE (--buckets 20) or scaled-float MSE
(--buckets 0).

Modes:
  value      value-only baseline on episodes
  pretrain   battle-only training on train shards; saves trunk+head checkpoint
  probe      load a checkpoint, train ONLY the value head on episodes (--finetune: all params)
  multitask  joint: value batches + battle batches interleaved, total = value + coef*battle
  evalhead   distribution-level eval of a pretrained battle head on val shards
             (32-reroll groups -> empirical bucket histogram; reports CE, KL, accuracy)
"""
import argparse, glob, json, os, re, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

import slaythespire as sts
from lightspeed.battle_buckets import NUM_BUCKETS, bucket_midpoint_frac
from lightspeed.network import (NN, ModelHP, BattleOutcomeHead, SlayDataset, collate_fn,
                     move_to_device, load_network_backward_compatible)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
iter_of = lambda p: int(re.search(r'iter_(\d+)', p).group(1))

# Battle rows carry no pending decision; collate_fn needs >=1 valid choice token, so every
# row gets the same constant dummy fixed action (zero information).
_DUMMY_CHOICE = {'cards_offered.cards': [], 'cards_offered.upgrades': [], 'relics_offered': [],
                 'potions_offered': [], 'paths_offered': [],
                 'fixed_actions': [{'action': 0, 'gold': 0, 'card': 0, 'relic': 0, 'info': 0}],
                 'choice_type': 0, 'chosen_idx': 0, 'outcome': 0, 'return': 0.0}
FRAC_CLAMP = (-1.0, 0.6)


def expl_var(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1.0 - np.var(y - yhat) / (np.var(y) + 1e-12))


def load_episode_df(epi_dir, last_files=50, max_rows=200000, seed=0):
    files = sorted(glob.glob(os.path.join(epi_dir, '*.parquet')), key=iter_of)[-last_files:]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df, iter_of(files[0]), iter_of(files[-1])


def load_battle_df(shard_dir, max_rows=600000, seed=0):
    files = sorted(glob.glob(os.path.join(shard_dir, '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    for k, v in _DUMMY_CHOICE.items():
        df[k] = [v] * len(df) if isinstance(v, (list, dict)) else v
    return df


def split_by_col(df, col, val_frac=0.15, seed=0):
    vals = df[col].unique().copy()
    np.random.RandomState(seed).shuffle(vals)
    val_set = set(vals[:max(1, int(len(vals) * val_frac))].tolist())
    is_val = df[col].isin(val_set).to_numpy()
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def value_collate(batch):
    c = collate_fn(batch)
    c['return'] = torch.tensor([float(b['return']) for b in batch], dtype=torch.float32)
    return c


def battle_collate(batch):
    c = collate_fn(batch)
    c['battle_encounter'] = torch.tensor([int(b['encounter']) for b in batch], dtype=torch.long)
    c['battle_bucket'] = torch.tensor([int(b['bucket']) for b in batch], dtype=torch.long)
    c['battle_frac'] = torch.tensor(
        [min(max(float(b['hp_frac_delta']), FRAC_CLAMP[0]), FRAC_CLAMP[1]) for b in batch],
        dtype=torch.float32)
    return c


def make_loader(df, collate, batch_size, shuffle, workers=4):
    return torch.utils.data.DataLoader(
        SlayDataset(df), batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
        drop_last=shuffle, num_workers=workers, persistent_workers=workers > 0)


def battle_loss_and_metrics(head, pooled, b, buckets):
    if buckets > 0:
        logits = head(pooled, b['battle_encounter'])
        loss = F.cross_entropy(logits, b['battle_bucket'])
        acc = (logits.argmax(-1) == b['battle_bucket']).float().mean().item()
        return loss, {'battle_ce': loss.item(), 'battle_acc': acc}
    pred = head(pooled, b['battle_encounter']).squeeze(-1)
    loss = F.mse_loss(pred, b['battle_frac'])
    mae = (pred - b['battle_frac']).abs().mean().item()
    return loss, {'battle_mse': loss.item(), 'battle_mae': mae}


def eval_value(net, vl, ymean, ystd):
    net.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in vl:
            b = move_to_device(b, DEVICE)
            _, v = net(b)
            ps.append(v.cpu().numpy() * ystd + ymean); ys.append(b['return'].cpu().numpy())
    return expl_var(np.concatenate(ys), np.concatenate(ps))


def eval_battle(net, head, vl, buckets):
    net.eval(); head.eval()
    agg, n = {}, 0
    with torch.no_grad():
        for b in vl:
            b = move_to_device(b, DEVICE)
            _, _, pooled = net(b, return_pooled=True)
            _, m = battle_loss_and_metrics(head, pooled, b, buckets)
            k = len(b['battle_encounter'])
            for key, v in m.items():
                agg[key] = agg.get(key, 0.0) + v * k
            n += k
    return {k: v / n for k, v in agg.items()}


def build_net(args, ckpt=None):
    torch.manual_seed(args.seed)
    net = NN(ModelHP(use_value_head=True, dim=args.dim, n_layers=args.n_layers)).to(DEVICE)
    head = BattleOutcomeHead(args.dim, args.buckets if args.buckets > 0 else 1).to(DEVICE)
    if ckpt:
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        net = load_network_backward_compatible(net, state['net'])
        if 'head' in state and state.get('buckets') == args.buckets:
            head.load_state_dict(state['head'])
    return net, head


def log_factory(out):
    f = open(out, 'a') if out else None
    def log(msg, rec=None):
        print(msg, flush=True)
        if f and rec is not None:
            f.write(json.dumps(rec) + '\n'); f.flush()
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['value', 'pretrain', 'probe', 'multitask', 'evalhead'])
    ap.add_argument('--epi-dir', default='runs/heart1.pt.episodes')
    ap.add_argument('--battle-dir', default='battle_data/train')
    ap.add_argument('--battle-val-dir', default='battle_data/val')
    ap.add_argument('--ckpt', default=None, help='checkpoint in (probe/evalhead) ')
    ap.add_argument('--ckpt-out', default=None, help='checkpoint out (pretrain/multitask)')
    ap.add_argument('--buckets', type=int, default=20, help='20=bucket CE head, 0=float MSE head')
    ap.add_argument('--coef', type=float, default=1.0, help='battle-loss weight (multitask)')
    ap.add_argument('--finetune', action='store_true', help='probe: train all params, not just value head')
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--last-files', type=int, default=50)
    ap.add_argument('--max-rows', type=int, default=200000)
    ap.add_argument('--battle-max-rows', type=int, default=600000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--name', default=None)
    ap.add_argument('--out', default='battle_value_sl_results.jsonl')
    args = ap.parse_args()
    name = args.name or f'{args.mode}_b{args.buckets}'
    log = log_factory(args.out)
    t0 = time.time()
    torch.set_float32_matmul_precision('high')

    # ---------- data ----------
    need_epi = args.mode in ('value', 'probe', 'multitask')
    need_battle = args.mode in ('pretrain', 'multitask', 'evalhead')
    if need_epi:
        edf, lo, hi = load_episode_df(args.epi_dir, args.last_files, args.max_rows, args.seed)
        etr, eva = split_by_col(edf, 'seed', seed=0)
        ymean, ystd = float(etr['return'].mean()), float(etr['return'].std() + 1e-8)
        base_ev = expl_var(eva['return'].to_numpy(), eva['value'].to_numpy())
        log(f'[{name}] episodes iters {lo}..{hi}: train {len(etr):,} val {len(eva):,} | '
            f'return mean {ymean:.3f} std {ystd:.3f} | PPO-value baseline val EV {base_ev:.4f}')
    if need_battle:
        bdir = args.battle_val_dir if args.mode == 'evalhead' else args.battle_dir
        bdf = load_battle_df(bdir, args.battle_max_rows, args.seed)
        if args.mode == 'evalhead':
            log(f'[{name}] battle val rows {len(bdf):,} '
                f'({bdf.state_id.nunique()} states)')
        else:
            btr, bva = split_by_col(bdf, 'game_seed', seed=0)
            log(f'[{name}] battle rows: train {len(btr):,} val {len(bva):,} | '
                f'death {bdf.died.mean():.3f} bucket-mode {bdf.bucket.mode()[0]}')

    # ---------- modes ----------
    if args.mode == 'evalhead':
        net, head = build_net(args, args.ckpt)
        evalhead(net, head, bdf, args, log, name)
        return

    net, head = build_net(args, args.ckpt)

    if args.mode == 'value' or args.mode == 'probe':
        if args.mode == 'probe' and not args.finetune:
            for p in net.parameters():
                p.requires_grad = False
            for p in net.value_head.parameters():
                p.requires_grad = True
            params = list(net.value_head.parameters())
            lr = args.lr or 1e-3
        else:
            params = list(net.parameters())
            lr = args.lr or (1e-4 if args.mode == 'probe' else 3e-4)
        opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr, weight_decay=args.wd)
        tl = make_loader(etr, value_collate, args.batch_size, True)
        vl = make_loader(eva, value_collate, 256, False, workers=2)
        best, best_ep, since = -1e9, -1, 0
        for ep in range(args.epochs):
            net.train()
            for b in tl:
                b = move_to_device(b, DEVICE)
                _, v = net(b)
                loss = F.mse_loss(v, (b['return'] - ymean) / ystd)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            ev = eval_value(net, vl, ymean, ystd)
            if ev > best:
                best, best_ep, since = ev, ep, 0
            else:
                since += 1
            log(f'  [{name}] ep{ep:02d} val_EV={ev:.4f} (best {best:.4f}@{best_ep})')
            if since >= args.patience:
                break
        log(f'[{name}] DONE best val EV {best:.4f} @ep{best_ep} ({time.time()-t0:.0f}s)',
            rec=dict(name=name, mode=args.mode, buckets=args.buckets, ckpt=args.ckpt,
                     finetune=args.finetune, lr=lr, best_val_ev=best, best_epoch=best_ep,
                     baseline_val_ev=base_ev, secs=round(time.time()-t0)))

    elif args.mode == 'pretrain':
        opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()),
                                lr=args.lr or 3e-4, weight_decay=args.wd)
        tl = make_loader(btr, battle_collate, args.batch_size, True)
        vl = make_loader(bva, battle_collate, 256, False, workers=2)
        key = 'battle_ce' if args.buckets > 0 else 'battle_mse'
        best, best_ep, since = 1e9, -1, 0
        for ep in range(args.epochs):
            net.train(); head.train()
            for b in tl:
                b = move_to_device(b, DEVICE)
                _, _, pooled = net(b, return_pooled=True)
                loss, _ = battle_loss_and_metrics(head, pooled, b, args.buckets)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(head.parameters()), 5.0)
                opt.step()
            m = eval_battle(net, head, vl, args.buckets)
            if m[key] < best:
                best, best_ep, since = m[key], ep, 0
                if args.ckpt_out:
                    torch.save({'net': net.state_dict(), 'head': head.state_dict(),
                                'buckets': args.buckets}, args.ckpt_out)
            else:
                since += 1
            log(f'  [{name}] ep{ep:02d} ' + ' '.join(f'{k}={v:.4f}' for k, v in m.items())
                + f' (best {best:.4f}@{best_ep})')
            if since >= args.patience:
                break
        log(f'[{name}] DONE best {key} {best:.4f} @ep{best_ep} -> {args.ckpt_out} '
            f'({time.time()-t0:.0f}s)',
            rec=dict(name=name, mode='pretrain', buckets=args.buckets, best=best, metric=key,
                     best_epoch=best_ep, ckpt_out=args.ckpt_out, secs=round(time.time()-t0)))

    elif args.mode == 'multitask':
        opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()),
                                lr=args.lr or 3e-4, weight_decay=args.wd)
        tl = make_loader(etr, value_collate, args.batch_size, True)
        btl = make_loader(btr, battle_collate, args.batch_size, True)
        vl = make_loader(eva, value_collate, 256, False, workers=2)
        bvl = make_loader(bva, battle_collate, 256, False, workers=2)
        best, best_ep, since = -1e9, -1, 0
        for ep in range(args.epochs):
            net.train(); head.train()
            bit = iter(btl)
            for b in tl:
                try:
                    bb = next(bit)
                except StopIteration:
                    bit = iter(btl); bb = next(bit)
                b, bb = move_to_device(b, DEVICE), move_to_device(bb, DEVICE)
                _, v = net(b)
                vloss = F.mse_loss(v, (b['return'] - ymean) / ystd)
                _, _, pooled = net(bb, return_pooled=True)
                bloss, _ = battle_loss_and_metrics(head, pooled, bb, args.buckets)
                loss = vloss + args.coef * bloss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(head.parameters()), 5.0)
                opt.step()
            ev = eval_value(net, vl, ymean, ystd)
            bm = eval_battle(net, head, bvl, args.buckets)
            if ev > best:
                best, best_ep, since = ev, ep, 0
                if args.ckpt_out:
                    torch.save({'net': net.state_dict(), 'head': head.state_dict(),
                                'buckets': args.buckets}, args.ckpt_out)
            else:
                since += 1
            log(f'  [{name}] ep{ep:02d} val_EV={ev:.4f} '
                + ' '.join(f'{k}={v:.4f}' for k, v in bm.items()) + f' (best {best:.4f}@{best_ep})')
            if since >= args.patience:
                break
        log(f'[{name}] DONE best val EV {best:.4f} @ep{best_ep} ({time.time()-t0:.0f}s)',
            rec=dict(name=name, mode='multitask', buckets=args.buckets, coef=args.coef,
                     best_val_ev=best, best_epoch=best_ep, baseline_val_ev=base_ev,
                     secs=round(time.time()-t0)))


def evalhead(net, head, bdf, args, log, name):
    """Distribution-level eval: group val rows by (state_id, encounter) (32 rerolls of the
    same sim), compare predicted bucket distribution to the empirical histogram. Reports
    mean CE, the empirical entropy floor H(emp), their gap KL(emp||pred), and single-row
    accuracy. Float heads report MSE/MAE of the prediction vs the group-mean frac."""
    groups = bdf.groupby(['state_id', 'encounter'])
    reps = bdf.loc[[idx[0] for idx in groups.indices.values()]]  # one obs row per group
    rl = make_loader(reps, battle_collate, 256, False, workers=2)
    net.eval(); head.eval()
    preds = []
    with torch.no_grad():
        for b in rl:
            b = move_to_device(b, DEVICE)
            _, _, pooled = net(b, return_pooled=True)
            out = head(pooled, b['battle_encounter'])
            preds.append(out.cpu().numpy())
    preds = np.concatenate(preds)
    rec = dict(name=name, mode='evalhead', buckets=args.buckets, ckpt=args.ckpt,
               n_groups=len(groups))
    if args.buckets > 0:
        ces, ents, accs = [], [], []
        for (key, idx), logit in zip(groups.indices.items(), preds):
            buckets = bdf.loc[idx, 'bucket'].to_numpy()
            emp = np.bincount(buckets, minlength=NUM_BUCKETS) / len(buckets)
            logp = logit - np.log(np.exp(logit - logit.max()).sum()) - logit.max()
            ces.append(-(emp * logp).sum())
            nz = emp[emp > 0]
            ents.append(-(nz * np.log(nz)).sum())
            accs.append(emp[logit.argmax()])
        rec.update(mean_ce=float(np.mean(ces)), emp_entropy=float(np.mean(ents)),
                   mean_kl=float(np.mean(ces) - np.mean(ents)), exp_acc=float(np.mean(accs)))
    else:
        mses, maes = [], []
        for (key, idx), pred in zip(groups.indices.items(), preds):
            mean_frac = bdf.loc[idx, 'hp_frac_delta'].clip(*FRAC_CLAMP).mean()
            mses.append((pred[0] - mean_frac) ** 2); maes.append(abs(pred[0] - mean_frac))
        rec.update(mean_mse=float(np.mean(mses)), mean_mae=float(np.mean(maes)))
    log(f'[{name}] EVALHEAD ' + ' '.join(f'{k}={v}' for k, v in rec.items() if k != 'name'),
        rec=rec)


if __name__ == '__main__':
    main()
