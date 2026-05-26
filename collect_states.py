# %%
# Collect pre-battle GameContexts reached by the trained NN policy, in the
# gen_states text format so ./test show_states / eval_states can load them.
#
# Drives games from Python like playouts.run_game (NN for out-of-combat choices,
# agent.playout_battle for battles) and records the MIXED action stream
# (out-of-combat GameAction.bits + in-battle search::Action.bits, in order).
# A record's prefix ends exactly when a BATTLE screen is reached.
from __future__ import annotations

import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm.auto import tqdm

import slaythespire as sts
from network import NN, ModelHP, load_network_backward_compatible
from playouts import (
    NNService,
    construct_choice,
    pick_card_with_net,
)


def load_policy_net(model_path, torch_compile_mode="default"):
    """Load a PPO checkpoint for inference.

    PPO single-network checkpoints are trained with use_value_head=True
    (ppo_train.py), so the net must be constructed the same way for the
    state dict to match. The value head is unused at inference; only the
    policy logits drive out-of-combat action selection.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = NN(ModelHP(use_value_head=True)).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    net = load_network_backward_compatible(net, state)
    if torch_compile_mode != "no":
        net = torch.compile(net, mode=torch_compile_mode)
    net.eval()
    return net


def collect_game(seed: int, net: NNService | None, sim_count: int,
                 k: int, temperature: float):
    """Play one game with the NN policy; return up to k pre-battle records.

    Each record is (seed, prefix_bits_list, floor_at_battle). The prefix is the
    mixed action stream up to (but not including) the chosen battle's in-battle
    actions.
    """
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    rng = random.Random(seed)

    agent = sts.Agent()
    agent.simulation_count_base = sim_count
    agent.verbosity_level = 0
    agent.exploration_parameter = 6.57
    agent.chance_widening_c = 3.14
    agent.chance_widening_alpha = 0.97
    agent.record_actions = True

    stream: list[int] = []          # mixed GameAction / search::Action bits, in order
    battle_starts: list[int] = []   # len(stream) at the moment each battle began
    battle_floors: list[int] = []   # floor_num when each battle began

    while gc.outcome == sts.GameOutcome.UNDECIDED:
        if gc.screen_state == sts.ScreenState.BATTLE:
            battle_starts.append(len(stream))
            battle_floors.append(gc.floor_num)
            pre = len(agent.game_action_history)
            agent.playout_battle(gc)
            # in-battle search::Action bits, in the order they were taken
            stream.extend(int(b) for b in agent.game_action_history[pre:])
        else:
            obs = sts.getNNRepresentation(gc)
            actions = sts.GameAction.getAllActionsInState(gc)
            choice = construct_choice(gc, obs, actions)

            if net is not None and choice is not None and len(actions) > 1:
                action, _ = pick_card_with_net(
                    net, choice, actions, temperature=temperature, rng=rng)
            else:
                action = agent.pick_gameaction(gc)

            assert action.isValidAction(gc), \
                f"Invalid action: {action.getDesc(gc)} (seed {seed})"
            stream.append(int(action.bits))
            action.execute(gc)

    if not battle_starts:
        return []

    # Pick up to k distinct random battle entries from this game.
    n = min(k, len(battle_starts))
    idxs = rng.sample(range(len(battle_starts)), n)
    records = []
    for i in idxs:
        prefix_len = battle_starts[i]
        records.append((seed, stream[:prefix_len], battle_floors[i]))
    return records


def format_record(seed: int, prefix: list[int], ascension: int = 0) -> str:
    """One gen_states line: charInt seed_hex ascension prefixLen action0_hex ..."""
    parts = [str(int(sts.CharacterClass.IRONCLAD)), f"{seed:x}",
             str(ascension), str(len(prefix))]
    parts.extend(f"{b & 0xFFFFFFFF:x}" for b in prefix)
    return " ".join(parts)


def main(args):
    torch.set_float32_matmul_precision("high")

    if args.checkpoint in ("", "-", "<simple>"):
        net = None
        service = None
        print("No checkpoint: using heuristic out-of-combat policy")
    else:
        model = load_policy_net(args.checkpoint, torch_compile_mode=args.torch_compile)
        service = NNService(
            model,
            batch_size=args.batch_size,
            batch_size_factor=min(min(8, args.batch_size),
                                  max(1, (args.num_threads + 1) // 2)),
            torch_compile_mode=args.torch_compile,
        )
        net = service
        print(f"Loaded checkpoint {args.checkpoint}")

    records: list[tuple[int, list[int], int]] = []
    lock = threading.Lock()
    floors: list[int] = []

    seeds = range(args.start_seed, args.start_seed + args.num_games)
    pbar = tqdm(total=args.num_records, desc="records")

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {
            executor.submit(collect_game, s, net, args.mcts_simulations,
                            args.k, args.temperature): s
            for s in seeds
        }
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                game_records = fut.result()
            except Exception as e:
                print(f"seed {seed} failed: {type(e).__name__}: {e}")
                continue
            with lock:
                if len(records) >= args.num_records:
                    continue
                for rec in game_records:
                    records.append(rec)
                    floors.append(rec[2])
                    pbar.update(1)
                    if len(records) >= args.num_records:
                        break
            if len(records) >= args.num_records:
                # cancel remaining work; we have enough
                for f in futures:
                    f.cancel()
                break

    pbar.close()
    if service is not None:
        service.stop()

    records = records[:args.num_records]
    with open(args.out, "w") as f:
        for seed, prefix, _floor in records:
            f.write(format_record(seed, prefix) + "\n")

    print(f"\nWrote {len(records)} records to {args.out}")
    if floors:
        floors_used = floors[:len(records)]
        floors_sorted = sorted(floors_used)
        n = len(floors_sorted)
        print(f"floor stats: min={floors_sorted[0]} "
              f"median={floors_sorted[n // 2]} max={floors_sorted[-1]} "
              f"mean={sum(floors_sorted) / n:.1f}")
        # crude histogram by act boundaries
        buckets = {"act1 (1-16)": 0, "act2 (17-33)": 0,
                   "act3 (34-50)": 0, "act4 (51+)": 0}
        for fl in floors_used:
            if fl <= 16:
                buckets["act1 (1-16)"] += 1
            elif fl <= 33:
                buckets["act2 (17-33)"] += 1
            elif fl <= 50:
                buckets["act3 (34-50)"] += 1
            else:
                buckets["act4 (51+)"] += 1
        for k_, v in buckets.items():
            print(f"  {k_}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect NN-policy pre-battle states in gen_states format")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to NN checkpoint (.pt). '-' for heuristic.")
    parser.add_argument("--out", type=str, default="states_policy2000.txt")
    parser.add_argument("--num-records", type=int, default=2000,
                        help="Stop once this many records are collected")
    parser.add_argument("--num-games", type=int, default=4000,
                        help="Max games (seed range size) to attempt")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=3,
                        help="Max records emitted per game")
    parser.add_argument("--mcts-simulations", type=int, default=800,
                        help="agent.simulation_count_base for battles")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--torch-compile", type=str, default="default",
                        choices=["no", "default", "reduce-overhead",
                                 "max-autotune"])
    args = parser.parse_args()
    main(args)

# %%
