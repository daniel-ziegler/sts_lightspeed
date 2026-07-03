"""Eval the hero checkpoint on fresh seeds with high-MCTS-sim playout.

Reuses run_episode for the rollout machinery; we don't need PPO training, just outcomes.
Shaping coefs are zeroed (don't matter for outcome). Defaults match the user's spec:
seeds 1_000_000..1_000_099 (well outside any training seed range), mcts_simulations=10000,
4 collect workers.
"""
import argparse, csv, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
# Must precede `import torch`: disables TorchInductor's compile-worker subprocess pool whose
# pipe-reader thread races with torch teardown and intermittently segfaults (rl_train sets this
# only when it is the main module; here torch is imported before rl_train).
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import torch

import slaythespire as sts
from lightspeed.network import NN, ModelHP, load_network_backward_compatible
from lightspeed.rl_train import (
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
    ap.add_argument('--out', default=None,
                    help='write rich per-game JSONL here (scalars + individual keys + final deck + '
                         'relics) instead of the flat CSV; crash-resumable by seed')
    ap.add_argument('--temperature', type=float, default=1.0,
                    help='NN overworld-pick softmax temperature; <=0 is greedy/argmax')
    ap.add_argument('--boss-widening', choices=['on', 'off'], default='on',
                    help="off forces boss fights to the general widening (A/B control arm)")
    ap.add_argument('--boss-widening-c', type=float, default=None,
                    help='explicit boss DPW widening C (with --boss-widening-alpha); overrides on/off')
    ap.add_argument('--boss-widening-alpha', type=float, default=None)
    ap.add_argument('--exploration', type=float, default=None,
                    help='override search exploration (deployment gate for tuned knob candidates; '
                         'rides on the engine-default eval weights)')
    ap.add_argument('--widening-c', type=float, default=None)
    ap.add_argument('--widening-alpha', type=float, default=None)
    ap.add_argument('--legacy-config', action='store_true',
                    help='use the pre-tuning coupled search config (exploration 4.24, widening '
                         '1.0/0.5 incl. boss, old eval weights) instead of the engine defaults')
    ap.add_argument('--battle-csv', default=None,
                    help='also write one row per battle (boss analysis) to this path')
    ap.add_argument('--randomize-paths', action='store_true',
                    help='intervention arm: uniform-random path choices (prices the routing policy)')
    ap.add_argument('--ascension', type=int, default=0,
                    help='play every game at this ascension level')
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
    # The pre-tuning coupled era: engine-default knobs of the old engine + old eval weights,
    # no boss specialization. Overrides --boss-widening (the legacy era has a single widening).
    legacy = dict(
        mcts_exploration=3 * 2 ** 0.5,
        mcts_widening_c=1.0, mcts_widening_alpha=0.5,
        mcts_boss_widening_c=1.0, mcts_boss_widening_alpha=0.5,
        mcts_win_bonus=100.0, mcts_potion_weight=10.0, mcts_victory_turn_penalty=0.01,
        mcts_monster_damage_weight=10.0, mcts_alive_weight=1.0,
        mcts_energy_waste_weight=0.2, mcts_draw_weight=0.03, mcts_turn_survival_weight=0.2,
    ) if args.legacy_config else {}
    if args.exploration is not None:
        # general-knob candidate: boss widening pinned to the same values (no boss gate yet
        # for the honest engine)
        legacy = dict(mcts_exploration=args.exploration,
                      mcts_widening_c=args.widening_c, mcts_widening_alpha=args.widening_alpha,
                      mcts_boss_widening_c=args.widening_c,
                      mcts_boss_widening_alpha=args.widening_alpha)
    elif not args.legacy_config and args.boss_widening_c is not None:
        # explicit boss widening candidate (per-battle tuning winner under deployment gate)
        legacy = dict(mcts_boss_widening_c=args.boss_widening_c,
                      mcts_boss_widening_alpha=args.boss_widening_alpha)
    elif not args.legacy_config and args.boss_widening == 'off':
        # pin boss widening to the general tuned values (engine default is the boss-gated set)
        legacy = dict(mcts_boss_widening_c=4.6, mcts_boss_widening_alpha=0.37)
    config = TrainConfig(
        mcts_simulations=args.mcts_sims,
        log_battle_outcomes=args.battle_csv is not None,
        randomize_path_choices=args.randomize_paths,
        fixed_ascension=args.ascension,
        **legacy,
        shaping_hp_coef=0.0, shaping_upg_coef=0.0,
        shaping_offset=0.0, shaping_relic_coef=0.0, shaping_maxhp_coef=0.0,
        num_workers=args.num_workers,
        num_games_per_step=args.n_games,
        battle_timeout=args.battle_timeout,
        sampling_temperature=args.temperature,
    )
    service = NNService(net, batch_size=config.inf_batch_size,
                        batch_size_factor=config.inf_batch_size_factor,
                        torch_compile_mode='no')
    # update_weights one-shot in case service expects it
    service.update_weights(net)

    seeds = list(range(args.seed_start, args.seed_start + args.n_games))

    # Output format: rich JSONL (--out) carries individual keys + final deck + relics per game;
    # the flat CSV (--out-csv) is the scalar fallback. Both are crash-resumable: one record per
    # completed game, flushed immediately, and on restart we skip seeds already present.
    use_jsonl = args.out is not None
    out_path = args.out if use_jsonl else args.out_csv
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    rows = []          # (seed, won, floor, keys) for the running summary, both formats
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline='') as fin:
            if use_jsonl:
                for line in fin:
                    if line.strip():
                        d = json.loads(line)
                        rows.append((d['seed'], d['won'], d['floor'], d.get('keys', -1)))
                        done.add(d['seed'])
            else:
                for r in csv.DictReader(fin):
                    rows.append((int(r['seed']), int(r['won']), int(r['floor']), int(r.get('keys', -1))))
                    done.add(int(r['seed']))
    todo = [s for s in seeds if s not in done]
    fout = open(out_path, 'a', newline='')
    writer = None
    if not use_jsonl:
        writer = csv.writer(fout)
        if not done:
            writer.writerow(['seed', 'won', 'floor', 'keys']); fout.flush()

    def card_rec(c):
        return {'id': int(c.id), 'name': str(c.id).split('.')[-1], 'upgraded': bool(c.upgraded)}

    def relic_rec(r):
        # traj.final_relics holds Relic objects (with .id RelicId and .data counter), not RelicIds.
        return {'id': int(r.id), 'name': str(r.id).split('.')[-1], 'data': int(r.data)}

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
                rec = None
                try:
                    traj = f.result()
                    m = traj.final_metrics
                    won = int(m.outcome == sts.GameOutcome.PLAYER_VICTORY)
                    floor = int(m.floor_num)
                    keys = int(m.num_keys)
                    if use_jsonl:
                        rec = {
                            'seed': s, 'won': won, 'floor': floor, 'act': int(m.act), 'keys': keys,
                            'red_key': bool(m.red_key), 'green_key': bool(m.green_key),
                            'blue_key': bool(m.blue_key),
                            'deck': [card_rec(c) for c in traj.final_deck],
                            'relics': [relic_rec(r) for r in traj.final_relics],
                        }
                except Exception as e:
                    print(f"  seed {s}: FAILED {e}", flush=True)
                    won, floor, keys = -1, -1, -1
                    if use_jsonl:
                        rec = {'seed': s, 'won': -1, 'floor': -1, 'act': -1, 'keys': -1,
                               'red_key': False, 'green_key': False, 'blue_key': False,
                               'deck': [], 'relics': []}
                rows.append((s, won, floor, keys))
                if use_jsonl:
                    fout.write(json.dumps(rec) + '\n'); fout.flush()
                else:
                    writer.writerow((s, won, floor, keys)); fout.flush()
                if bwriter is not None and won != -1:
                    for i, snap in enumerate(traj.battle_log):
                        bwriter.writerow((s, i, snap.floor, snap.act,
                                          int(snap.encounter in BOSS_ENCOUNTERS),
                                          snap.cur_hp, snap.potion_count, won, floor))
                    bfout.flush()
                n = len(rows)
                wn = sum(1 for _, w, _, _ in rows if w == 1)
                hk = sum(1 for _, w, _, k in rows if w == 1 and k == 3)
                err = sum(1 for _, w, _, _ in rows if w == -1)
                print(f"  {n:4d}/{len(seeds)}  seed={s} won={won} floor={floor} keys={keys} "
                      f"running_win={wn/max(1,n-err):.3f} heart_kill={hk/max(1,n-err):.3f} "
                      f"({hk}/{n-err})  elapsed={time.time()-t0:.0f}s",
                      flush=True)
    fout.close()
    if bfout is not None:
        bfout.close()

    rows_clean = [r for r in rows if r[1] != -1]
    n = len(rows_clean)
    wins = sum(r[1] for r in rows_clean)
    win_rate = wins / max(1, n)
    heart_kills = sum(1 for r in rows_clean if r[1] == 1 and r[3] == 3)
    heart_kill_rate = heart_kills / max(1, n)
    floors = [r[2] for r in rows_clean]
    mean_floor = sum(floors) / max(1, n)
    # Wilson-ish std for a binary proportion
    import math
    sd = math.sqrt(win_rate * (1 - win_rate) / max(1, n))
    hk_sd = math.sqrt(heart_kill_rate * (1 - heart_kill_rate) / max(1, n))
    print()
    print(f"=== EVAL RESULT ===")
    print(f"  games:      {n} (failed: {len(rows)-n})")
    print(f"  win_rate:   {win_rate:.4f} ± {sd:.4f}  ({wins}/{n})")
    print(f"  heart_kill: {heart_kill_rate:.4f} ± {hk_sd:.4f}  ({heart_kills}/{n})  [won AND keys==3]")
    print(f"  avg_floor:  {mean_floor:.2f}")
    print(f"  out:       {out_path}")
    print(f"  total:     {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
