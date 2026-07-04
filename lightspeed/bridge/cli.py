"""Entry point: policy/net loading and the comm.py command line (run_agent_cli)."""

import os
import sys
import traceback
import json
import argparse
import itertools

from spirecomm.spire.character import PlayerClass
from spirecomm.communication.coordinator import Coordinator
from lightspeed import RUNS_DIR
from lightspeed.bridge.actions import test_basic_conversion
from lightspeed.bridge.agent import STSLightspeedAgent




DEFAULT_CKPT = os.path.join(RUNS_DIR, "heart1.pt")


def load_policy_service(ckpt_path, device=None):
    """Load the heart1 policy checkpoint into an NNService for inference.

    Imports torch/network/playouts lazily (kept out of module import so the conversion tests and
    offline replay tooling load without torch). Mirrors eval_hero.py's loader: single net, value
    head, default ModelHP (the architecture rl_train.py used for the heart1 run)."""
    import torch
    from lightspeed.network import NN, ModelHP, load_network_backward_compatible
    from lightspeed.playouts import NNService

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Determinism: the combat MCTS is already seeded (BattleSearcher uses bc.seed+floor) and runs in
    # pure C++, but the policy net's CUDA forward is not -- cuBLAS/cuDNN can return slightly different
    # floats across processes, occasionally flipping an argmax on a near-tie in an out-of-combat
    # decision, which cascades into a divergent run. Pin every RNG and force deterministic GPU kernels
    # so a given seed replays identically (needed to reproduce a specific loss/crash for debugging).
    # cuBLAS determinism additionally requires CUBLAS_WORKSPACE_CONFIG; set it here (read lazily when
    # the cuBLAS handle is first created, which is after this) so it survives the mod's config rewrite.
    # warn_only keeps a rare kernel-less op from aborting a live run.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")
    hp = ModelHP(use_value_head=True, dim=256, n_layers=4)
    net = NN(hp).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    net = load_network_backward_compatible(net, state)
    net.eval()
    # torch_compile_mode='no': avoid compile/cudagraph latency+warmup for live single-request play.
    service = NNService(net, batch_size=8, max_wait_time=0.005, torch_compile_mode="no")
    service.update_weights(net)
    print(f"[net] loaded heart1 policy from {ckpt_path} on {device}", file=sys.stderr)
    return service


def run_agent_cli():
    """
    Main CLI entry point for running the AI agent.
    """
    parser = argparse.ArgumentParser(description="STS Lightspeed AI Agent for CommunicationMod")
    parser.add_argument("--character", "-c", 
                       choices=["ironclad", "silent", "defect"],
                       default="ironclad",
                       help="Character class to play")
    parser.add_argument("--games", "-g", type=int, default=1,
                       help="Number of games to play (0 for infinite)")
    parser.add_argument("--test", action="store_true",
                       help="Run conversion tests instead of playing")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                       help="heart1 policy checkpoint for out-of-combat decisions")
    # Each knob defaults from an env var when present. ModTheSpire/CommunicationMod re-normalizes
    # config.properties at startup and drops appended CLI flags, but preserves env vars set via the
    # command's `/usr/bin/env VAR=val ...` prefix (like STS_COMM_CAPTURE) -- so run_live.sh passes
    # these as env vars, and explicit CLI flags still override for manual invocations.
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("STS_TEMPERATURE", 0.0)),
                       help="Network action-sampling temperature (0 = greedy/argmax)")
    parser.add_argument("--seed", default=os.environ.get("STS_START_SEED") or None,
                       help="Start runs on this exact base-35 StS seed string (e.g. 54FYPZX13RLTT) "
                            "to replay a specific game")
    parser.add_argument("--ascension", type=int, default=int(os.environ.get("STS_ASCENSION", 0)),
                       help="Ascension level to start new runs on (0-20)")
    parser.add_argument("--sims", type=int, default=int(os.environ.get("STS_SIMS", 1000)),
                       help="Combat MCTS simulations per decision (simulation_count_base)")
    parser.add_argument("--watch", action="store_true",
                       default=("STS_WATCH_PRE_MS" in os.environ or "STS_WATCH_POST_MS" in os.environ
                                or "STS_WATCH_REWARD_MS" in os.environ),
                       help="Enable watch mode (also auto-enabled by setting either watch delay / its "
                            "env var): at each net decision pause, move the cursor onto the intended "
                            "pick, pause, then commit. Off = full speed.")
    parser.add_argument("--watch-pre-ms", type=int, default=int(os.environ.get("STS_WATCH_PRE_MS", 0)),
                       help="Watch mode: ms to wait BEFORE moving the cursor to the pick (default 0).")
    parser.add_argument("--watch-post-ms", type=int, default=int(os.environ.get("STS_WATCH_POST_MS", 0)),
                       help="Watch mode: ms to wait AFTER moving the cursor, before committing (default 0).")
    parser.add_argument("--watch-reward-ms", type=int, default=int(os.environ.get("STS_WATCH_REWARD_MS", 100)),
                       help="Watch mode: shorter pause between combat-reward list claims, so the "
                            "no-decision pickups tick through instead of using the full decision "
                            "pacing (default 100).")

    args = parser.parse_args()
    
    if args.test:
        print("Testing spirecomm to GameContext converter...")
        success = test_basic_conversion()
        sys.exit(0 if success else 1)
    
    # Map character name to enum
    class_mapping = {
        "ironclad": PlayerClass.IRONCLAD,
        "silent": PlayerClass.THE_SILENT, 
        "defect": PlayerClass.DEFECT,
    }
    chosen_class = class_mapping[args.character]
    
    print(f"Starting STS Lightspeed Agent for {args.character.title()}", file=sys.stderr)

    net = load_policy_service(args.ckpt)

    # Create agent and coordinator
    agent = STSLightspeedAgent(chosen_class, net=net, temperature=args.temperature,
                               start_seed=args.seed, ascension=args.ascension, sims=args.sims,
                               watch=args.watch, watch_pre_ms=args.watch_pre_ms,
                               watch_post_ms=args.watch_post_ms, watch_reward_ms=args.watch_reward_ms)
    coordinator = Coordinator()
    agent.coordinator = coordinator  # lets the agent capture raw decision states for replay

    # Register callbacks
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    
    # Play games. Always play the chosen character every game -- the policy net is
    # character-specific (heart1 is Ironclad), so cycling classes would run it off-distribution.
    games_played = 0
    character_classes = itertools.repeat(chosen_class)

    for current_class in character_classes:
        if args.games > 0 and games_played >= args.games:
            break
            
        agent.change_class(current_class)
        print(f"Playing game {games_played + 1} as {current_class.name}", file=sys.stderr)
        
        try:
            # Pass --seed through so play_one_game's StartGameAction uses it (it defaults to a random
            # seed otherwise). With a seed set, every game replays the same run -- intended for
            # deterministic crash repro (--seed <s> --games 1).
            result = coordinator.play_one_game(current_class, ascension_level=args.ascension,
                                               seed=args.seed)
            games_played += 1
            # Split a victory into a heart kill (reached act 4) vs an act-3-only win -- a heart-run
            # agent's act-3 wins mean it failed to collect the keys, so they are NOT heart wins.
            max_act = getattr(coordinator, "last_game_max_act", 0)
            max_floor = getattr(coordinator, "last_game_max_floor", 0)
            kind = "heart" if (result and max_act >= 4) else "act3" if result else "loss"
            print(f"Game {games_played} completed with result: {result} "
                  f"(max_act={max_act} max_floor={max_floor} kind={kind})", file=sys.stderr)
        except KeyboardInterrupt:
            print("Interrupted by user", file=sys.stderr)
            break
        except Exception as e:
            print(f"Game error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Dump the last raw communication state so a hang/crash is debuggable after the fact -- the
            # errlog otherwise keeps only [step]/[net] summaries, not the raw screen (choice_list,
            # available_commands, ready_for_command, prices) needed to root-cause e.g. a wedged shop buy.
            try:
                raw = getattr(coordinator, "last_raw_communication_state", None)
                if raw is not None:
                    dump = os.path.join(RUNS_DIR, "game_error_states.jsonl")
                    with open(dump, "a") as f:
                        f.write(json.dumps({"error": str(e), "raw": raw}) + "\n")
                    print(f"[dump] last raw state -> {dump}", file=sys.stderr)
            except Exception as de:
                print(f"[dump] failed to write last raw state: {de}", file=sys.stderr)
            break


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run test
        print("Testing spirecomm to GameContext converter...")
        test_basic_conversion()
    else:
        # Arguments provided - run CLI
        run_agent_cli()
