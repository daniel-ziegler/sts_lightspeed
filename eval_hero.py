"""Eval the hero checkpoint on fresh seeds with high-MCTS-sim playout.

Reuses run_episode for the rollout machinery; we don't need PPO training, just outcomes.
Shaping coefs are zeroed (don't matter for outcome). Defaults match the user's spec:
seeds 1_000_000..1_000_099 (well outside any training seed range), mcts_simulations=10000,
4 collect workers.
"""
import argparse, csv, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
# Must precede `import torch`: disables TorchInductor's compile-worker subprocess pool whose
# pipe-reader thread races with torch teardown and intermittently segfaults (rl_train sets this
# only when it is the main module; here torch is imported before rl_train).
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import torch

import slaythespire as sts
from network import NN, ModelHP, load_network_backward_compatible
from rl_train import (
    NNService, TrainConfig, run_episode, compute_progress_reward,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='path to e.g. runs/hero.pt.iter_130')
    ap.add_argument('--n-games', type=int, default=100)
    ap.add_argument('--seed-start', type=int, default=1_000_000)
    ap.add_argument('--mcts-sims', type=int, default=10_000)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--battle-timeout', type=float, default=300.0,
                    help='per-battle MCTS wall-clock budget (s); raise for high sim counts')
    ap.add_argument('--out-csv', default='runs/eval_hero.csv')
    ap.add_argument('--boss-widening', choices=['on', 'off'], default='on',
                    help="off forces boss fights to the general widening (A/B control arm)")
    ap.add_argument('--battle-csv', default=None,
                    help='also write one row per battle (boss analysis) to this path')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    print(f"loading {args.ckpt} ...", flush=True)
    hp = ModelHP(use_value_head=True, dim=256, n_layers=4)
    net = NN(hp).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    net = load_network_backward_compatible(net, state)
    net.eval()

    # TrainConfig with eval knobs. Search exploration/widening are left unset so battles use the
    # engine's jointly-tuned SearchAgent defaults (knobs + eval weights are a coupled set).
    # Shaping coefs zero -> reward (and the rewards we ignore) match plain victory/progress
    # signal. inf_batch_size defaults are fine.
    config = TrainConfig(
        mcts_simulations=args.mcts_sims,
        # 'off' pins boss widening to the general tuned values (engine default is the boss-gated set)
        mcts_boss_widening_c=None if args.boss_widening == 'on' else 4.6,
        mcts_boss_widening_alpha=None if args.boss_widening == 'on' else 0.37,
        log_battle_outcomes=args.battle_csv is not None,
        shaping_hp_coef=0.0, shaping_upg_coef=0.0,
        shaping_offset=0.0, shaping_relic_coef=0.0, shaping_maxhp_coef=0.0,
        num_workers=args.num_workers,
        num_games_per_step=args.n_games,
        battle_timeout=args.battle_timeout,
    )
    service = NNService(net, batch_size=config.inf_batch_size,
                        batch_size_factor=config.inf_batch_size_factor,
                        torch_compile_mode='no')
    # update_weights one-shot in case service expects it
    service.update_weights(net)

    seeds = list(range(args.seed_start, args.seed_start + args.n_games))

    # Crash-resumable: results are appended to the CSV one row per completed game and flushed
    # immediately, so a crash only loses in-flight games. On restart we skip seeds already present.
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    rows = []
    done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv, newline='') as fin:
            for r in csv.DictReader(fin):
                rows.append((int(r['seed']), int(r['won']), int(r['floor'])))
                done.add(int(r['seed']))
    todo = [s for s in seeds if s not in done]
    fout = open(args.out_csv, 'a', newline='')
    writer = csv.writer(fout)
    if not done:
        writer.writerow(['seed', 'won', 'floor']); fout.flush()

    # Boss encounter ids (MonsterEncounter enum order; see constants/MonsterEncounters.h)
    BOSS_ENCOUNTERS = {18, 19, 20, 37, 38, 39, 52, 53, 54, 56}
    bwriter = bfout = None
    if args.battle_csv:
        bexists = os.path.exists(args.battle_csv)
        bfout = open(args.battle_csv, 'a', newline='')
        bwriter = csv.writer(bfout)
        if not bexists:
            bwriter.writerow(['seed', 'battle_idx', 'floor', 'act', 'is_boss',
                              'post_hp', 'potions', 'won_game', 'final_floor']); bfout.flush()

    t0 = time.time()
    print(f"running {len(todo)} games (skipping {len(done)} already done) | "
          f"mcts_sims={args.mcts_sims} | workers={args.num_workers}", flush=True)
    # Battle executor is shared (matching rl_train pattern: one battle thread per main worker)
    with ThreadPoolExecutor(max_workers=args.num_workers) as battle_executor:
        with ThreadPoolExecutor(max_workers=args.num_workers) as main_executor:
            futs = {
                main_executor.submit(run_episode, s, service,
                                     compute_progress_reward, battle_executor, config): s
                for s in todo
            }
            for f in as_completed(futs):
                s = futs[f]
                try:
                    traj = f.result()
                    won = int(traj.final_metrics.outcome == sts.GameOutcome.PLAYER_VICTORY)
                    floor = int(traj.final_metrics.floor_num)
                except Exception as e:
                    print(f"  seed {s}: FAILED {e}", flush=True)
                    won, floor = -1, -1
                rows.append((s, won, floor))
                writer.writerow((s, won, floor)); fout.flush()
                if bwriter is not None and won != -1:
                    for i, snap in enumerate(traj.battle_log):
                        bwriter.writerow((s, i, snap.floor, snap.act,
                                          int(snap.encounter in BOSS_ENCOUNTERS),
                                          snap.cur_hp, snap.potion_count, won, floor))
                    bfout.flush()
                n = len(rows)
                wn = sum(1 for _, w, _ in rows if w == 1)
                err = sum(1 for _, w, _ in rows if w == -1)
                print(f"  {n:4d}/{len(seeds)}  seed={s} won={won} floor={floor} "
                      f"running_win={wn/max(1,n-err):.3f} ({wn}/{n-err})  "
                      f"elapsed={time.time()-t0:.0f}s",
                      flush=True)
    fout.close()
    if bfout is not None:
        bfout.close()

    rows_clean = [r for r in rows if r[1] != -1]
    n = len(rows_clean)
    wins = sum(r[1] for r in rows_clean)
    win_rate = wins / max(1, n)
    floors = [r[2] for r in rows_clean]
    mean_floor = sum(floors) / max(1, n)
    # Wilson-ish std for a binary proportion
    import math
    sd = math.sqrt(win_rate * (1 - win_rate) / max(1, n))
    print()
    print(f"=== EVAL RESULT ===")
    print(f"  games:     {n} (failed: {len(rows)-n})")
    print(f"  win_rate:  {win_rate:.4f} ± {sd:.4f}  ({wins}/{n})")
    print(f"  avg_floor: {mean_floor:.2f}")
    print(f"  out_csv:   {args.out_csv}")
    print(f"  total:     {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
