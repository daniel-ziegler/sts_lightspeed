"""Replay one GRPO group from a checkpoint and print the playthroughs.

Runs `group_size` games from the SAME map seed with distinct sampling seeds (exactly how GRPO
collects a group), computes the RLOO advantages, and prints the group summary plus the best and
worst members' full playthroughs. Lets you read how a policy plays without touching training.
"""
import argparse

import torch

from network import NN, ModelHP, load_network_backward_compatible
from rl_train import NNService, TrainConfig, collect_experience, compute_progress_reward
from algorithms import GRPOAlgorithm, CollectionJob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='e.g. runs/grpo_a.pt.iter_80')
    ap.add_argument('--map-seed', type=int, default=2_000_000,
                    help='game/map seed shared by all group members')
    ap.add_argument('--group-size', type=int, default=4)
    ap.add_argument('--mcts-sims', type=int, default=1000)
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hp = ModelHP(use_value_head=False, dim=256, n_layers=4)
    net = NN(hp).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    net = load_network_backward_compatible(net, state)
    net.eval()

    # Same shaping/MCTS knobs as the GRPO training runs so returns/advantages match what
    # training would compute for this group.
    config = TrainConfig(
        algo='grpo', group_size=args.group_size,
        mcts_simulations=args.mcts_sims,
        mcts_exploration=6.57, mcts_widening_c=3.14, mcts_widening_alpha=0.97,
        shaping_upg_coef=0.035, shaping_offset=0.307,
        num_workers=args.num_workers, num_games_per_step=args.group_size,
        battle_timeout=120,
    )
    service = NNService(net, batch_size=config.inf_batch_size,
                        batch_size_factor=config.inf_batch_size_factor, torch_compile_mode='no')
    service.update_weights(net)

    jobs = [CollectionJob(game_seed=args.map_seed, sample_seed=args.map_seed * 64 + m, group_id=0)
            for m in range(args.group_size)]
    print(f"Replaying group: map seed {args.map_seed}, {args.group_size} members, "
          f"{args.mcts_sims} MCTS sims, ckpt {args.ckpt}", flush=True)
    trajectories = collect_experience(config, service, compute_progress_reward, jobs)
    service.stop()

    returns_by_idx = {i: float(sum(t.rewards)) for i, t in enumerate(trajectories) if t.experiences}
    GRPOAlgorithm._print_group_debug(trajectories, {0: list(returns_by_idx)}, returns_by_idx)


if __name__ == '__main__':
    main()
