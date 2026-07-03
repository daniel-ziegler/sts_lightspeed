"""Collect pre-boss-battle states at production strength, in gen_states format.

Games are driven by run_episode (the training/eval harness: NN out-of-combat + MCTS battles at
the engine's tuned defaults), with TrainConfig.record_boss_states capturing the replayable mixed
action stream up to each boss battle. Records are written to {out_prefix}_act{1,2,3}.txt as they
arrive (crash-resumable: seeds already present in the act files are skipped).
"""
import argparse, os, time
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

import slaythespire as sts
from lightspeed.network import NN, ModelHP, load_network_backward_compatible
from lightspeed.rl_train import NNService, TrainConfig, run_episode, compute_progress_reward


def act_for_floor(f):
    return 1 if f <= 16 else 2 if f <= 33 else 3


def format_record(seed, prefix, ascension=0):
    parts = [str(int(sts.CharacterClass.IRONCLAD)), f"{seed:x}", str(ascension), str(len(prefix))]
    parts.extend(f"{b & 0xFFFFFFFF:x}" for b in prefix)
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n-games', type=int, default=1000)
    ap.add_argument('--seed-start', type=int, default=7_000_000)
    ap.add_argument('--mcts-sims', type=int, default=1000)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--per-act', type=int, default=700, help='stop collecting an act at this many records')
    ap.add_argument('--out-prefix', required=True)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    net = NN(ModelHP(use_value_head=True, dim=256, n_layers=4)).to(device)
    net = load_network_backward_compatible(net, torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()

    config = TrainConfig(
        mcts_simulations=args.mcts_sims,
        record_boss_states=True,
        shaping_hp_coef=0.0, shaping_upg_coef=0.0, shaping_offset=0.0,
        shaping_relic_coef=0.0, shaping_maxhp_coef=0.0,
        num_workers=args.num_workers,
        battle_timeout=120.0,
    )
    service = NNService(net, batch_size=config.inf_batch_size,
                        batch_size_factor=config.inf_batch_size_factor, torch_compile_mode='no')
    service.update_weights(net)

    files, counts, done = {}, {}, set()
    for act in (1, 2, 3):
        path = f"{args.out_prefix}_act{act}.txt"
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.split(None, 2)
                    if len(parts) >= 2:
                        done.add(int(parts[1], 16))
            counts[act] = sum(1 for _ in open(path))
        else:
            counts[act] = 0
        files[act] = open(path, 'a')

    seeds = [s for s in range(args.seed_start, args.seed_start + args.n_games) if s not in done]
    print(f"collecting from {len(seeds)} games (skipping {len(done)} done) | "
          f"counts {counts} | target {args.per_act}/act", flush=True)

    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as battle_executor:
        with ThreadPoolExecutor(max_workers=args.num_workers) as main_executor:
            futs = {main_executor.submit(run_episode, s, service, compute_progress_reward,
                                         battle_executor, config): s for s in seeds}
            for fut in as_completed(futs):
                s = futs[fut]
                n += 1
                try:
                    traj = fut.result()
                except Exception as e:
                    print(f"  seed {s} FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                for floor, prefix in traj.boss_state_records:
                    act = act_for_floor(floor)
                    if counts[act] < args.per_act:
                        files[act].write(format_record(s, prefix) + "\n")
                        files[act].flush()
                        counts[act] += 1
                if n % 25 == 0:
                    print(f"  {n}/{len(seeds)} games  counts={counts}  elapsed={time.time()-t0:.0f}s", flush=True)
                if all(c >= args.per_act for c in counts.values()):
                    for f2 in futs:
                        f2.cancel()
                    break
    for f in files.values():
        f.close()
    service.stop()
    print(f"DONE: {counts} in {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
