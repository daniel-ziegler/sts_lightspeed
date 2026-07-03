"""Game watch mode: let the NN + MCTS agent play a full run while you watch.

Mirrors the play logic of rl_train.run_episode -- the path eval_hero uses (MCTS for battles,
the network for >1-option overworld screens, the C++ heuristic for the rest) -- but plays each
in-battle action one at a time and pauses after every decision so a human can follow along.

Battles step at --battle-delay seconds per action (default 0.3); out-of-combat choices
(card rewards, map, shop, events, ...) pause for --choice-delay seconds (default 1.0).

Usage:
    python watch_game.py --model-path nets/honest1.pt --seed 42
    python watch_game.py                       # heuristic agent, no network
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import slaythespire as sts

from network import choice_space
from playouts import (
    NNService,
    construct_choice,
    load_net,
    path_to_action_and_desc,
    take_free_rewards,
)


# --- formatting helpers -------------------------------------------------------

def _statuses(get, names):
    """Collect "name N" for each non-zero (status-enum, label) pair."""
    out = []
    for status, label in names:
        try:
            v = get(status)
        except Exception:
            v = 0
        if v:
            out.append(f"{label} {v}")
    return out


_PLAYER_STATUSES = [
    (sts.PlayerStatus.WEAK, "Weak"),
    (sts.PlayerStatus.VULNERABLE, "Vuln"),
    (sts.PlayerStatus.FRAIL, "Frail"),
    (sts.PlayerStatus.ARTIFACT, "Artifact"),
]

_MONSTER_STATUSES = [
    (sts.MonsterStatus.VULNERABLE, "Vuln"),
    (sts.MonsterStatus.WEAK, "Weak"),
]


def fmt_player(bc) -> str:
    p = bc.player
    parts = [f"HP {p.curHp}/{p.maxHp}"]
    if p.block:
        parts.append(f"Blk {p.block}")
    parts.append(f"E {p.energy}")
    if p.strength:
        parts.append(f"Str {p.strength}")
    if p.dexterity:
        parts.append(f"Dex {p.dexterity}")
    parts += _statuses(p.getStatus, _PLAYER_STATUSES)
    return "Player: " + " | ".join(parts)


def fmt_monster(m) -> str:
    s = f"{m.getName()} {m.curHp}/{m.maxHp}"
    extra = []
    if m.block:
        extra.append(f"Blk {m.block}")
    if m.strength:
        extra.append(f"Str {m.strength}")
    if m.vulnerable:
        extra.append(f"Vuln {m.vulnerable}")
    if m.weak:
        extra.append(f"Weak {m.weak}")
    if m.poison:
        extra.append(f"Poison {m.poison}")
    if extra:
        s += " (" + ", ".join(extra) + ")"
    return s


def fmt_monsters(bc) -> str:
    alive = [bc.monsters[i] for i in range(len(bc.monsters)) if bc.monsters[i].isAlive()]
    if not alive:
        return "Monsters: (none)"
    return "Monsters: " + " ; ".join(fmt_monster(m) for m in alive)


def fmt_board(bc) -> str:
    return f"  {fmt_player(bc)}\n  {fmt_monsters(bc)}"


def clean_action_desc(desc: str) -> str:
    """Tidy the engine's raw Action::printDesc string for display."""
    return " ".join(desc.replace("{", "").replace("}", "").split())


# --- battle: step the search one action at a time -----------------------------

def watch_battle(gc, agent, battle_delay: float):
    bc = gc.create_battle_context()
    print(f"\n=== BATTLE (floor {gc.floor_num}, act {gc.act}): {bc.encounter} ===")
    print(fmt_board(bc), flush=True)
    time.sleep(battle_delay)

    step = 0
    while bc.outcome == sts.BattleOutcome.UNDECIDED:
        searcher = sts.BattleSearcher(bc)
        sims = agent.configure_searcher(searcher, bc)
        searcher.search(sims)
        if not searcher.get_root_edges():
            break
        action = searcher.get_best_action()
        desc = clean_action_desc(action.print_desc(bc))
        action.execute(bc)

        step += 1
        print(f"\n[{step}] >> {desc}")
        print(fmt_board(bc), flush=True)
        time.sleep(battle_delay)

    result = {
        sts.BattleOutcome.PLAYER_VICTORY: "VICTORY",
        sts.BattleOutcome.PLAYER_LOSS: "DEFEAT",
    }.get(bc.outcome, str(bc.outcome))
    print(f"--- battle over: {result} ---", flush=True)
    gc.sync_from_battle_context(bc)


# --- overworld: a single non-battle decision ----------------------------------

def watch_choice(gc, agent, service, rng, temperature: float, choice_delay: float):
    # Mirrors run_episode (rl_train.py): the network only decides screens with >1 representable
    # option; single-/no-option screens defer to the C++ heuristic pick_gameaction. Gold/relic/
    # potion rewards aren't in the net's choice space (construct_choice encodes only the card
    # portion of a REWARDS screen, plus SKIP -- and SKIP forfeits the whole screen), so sweep the
    # free rewards up front and let the net decide only the card.
    swept = take_free_rewards(gc)
    if swept:
        print(f"  (auto-took {len(swept)} free reward(s): gold/relics)", flush=True)

    obs = sts.getNNRepresentation(gc)
    actions = sts.GameAction.getAllActionsInState(gc)
    choice = construct_choice(gc, obs, actions)

    print(f"\n--- {str(gc.screen_state).split('.')[-1]} "
          f"(floor {gc.floor_num}, act {gc.act}, HP {gc.cur_hp}/{gc.max_hp}, gold {gc.gold}) ---")

    total = 0 if choice is None else (
        len(choice.cards_offered) + len(choice.relics_offered) + len(choice.potions_offered) +
        len(choice.fixed_actions) + len(choice.paths_offered))

    if service is not None and choice is not None and total > 1:
        batch_tensors, output = service.get_logits(choice)
        logits = output[0] if isinstance(output, tuple) else output
        logits_tensor = torch.tensor(logits)
        if temperature != 1.0:
            logits_tensor = logits_tensor / temperature
        probs = np.exp(F.log_softmax(logits_tensor, dim=0).numpy())
        chosen_idx = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])
        path = choice_space.ix_to_path(batch_tensors['choices'], chosen_idx)
        action, desc = path_to_action_and_desc(choice, path, gc)
        print(f"  chose [{path[0]}] {desc}", flush=True)
    else:
        action = agent.pick_gameaction(gc)
        print(f"  chose {action.getDesc(gc)}", flush=True)

    action.execute(gc)
    time.sleep(choice_delay)


# --- driver -------------------------------------------------------------------

def watch_game(seed, character, ascension, service, mcts_sims,
               temperature, battle_delay, choice_delay):
    gc = sts.GameContext(character, seed, ascension)

    agent = sts.Agent()
    agent.simulation_count_base = mcts_sims
    agent.verbosity_level = 0  # we do our own printing
    rng = random.Random(seed)

    print(f"Watching seed {seed} ({str(character).split('.')[-1]}, ascension {ascension}, "
          f"{mcts_sims} sims/decision)")

    while gc.outcome == sts.GameOutcome.UNDECIDED:
        if gc.screen_state == sts.ScreenState.BATTLE:
            watch_battle(gc, agent, battle_delay)
        else:
            watch_choice(gc, agent, service, rng, temperature, choice_delay)

    outcome = "WON" if gc.outcome == sts.GameOutcome.PLAYER_VICTORY else "LOST"
    print(f"\n========================================")
    print(f"Game over: {outcome} on floor {gc.floor_num} "
          f"(HP {gc.cur_hp}/{gc.max_hp})")
    print(f"========================================", flush=True)
    return gc.outcome, gc.floor_num


def main():
    parser = argparse.ArgumentParser(description="Watch the NN + MCTS agent play a full game.")
    parser.add_argument("--model-path", type=str, default="-",
                        help="path to network checkpoint; '-' uses the heuristic agent (no NN)")
    parser.add_argument("--seed", type=int, default=0, help="game seed")
    parser.add_argument("--character", type=str, default="IRONCLAD",
                        help="character class (IRONCLAD, SILENT, DEFECT, WATCHER)")
    parser.add_argument("--ascension", type=int, default=0, help="ascension level")
    parser.add_argument("--mcts-simulations", type=int, default=1000,
                        help="MCTS rollouts per in-battle decision")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Boltzmann temperature for network overworld choices")
    parser.add_argument("--battle-delay", type=float, default=0.3,
                        help="seconds to pause after each in-battle action")
    parser.add_argument("--choice-delay", type=float, default=1.0,
                        help="seconds to pause after each out-of-combat choice")
    parser.add_argument("--torch-compile", type=str, default="no",
                        choices=["no", "default", "reduce-overhead", "max-autotune"],
                        help="torch compile mode for the network")
    parser.add_argument("--value-head", action=argparse.BooleanOptionalAction, default=True,
                        help="build the network with a value head (heart1 / PPO single-net "
                             "checkpoints need this; pass --no-value-head for policy-only nets)")
    args = parser.parse_args()

    character = getattr(sts.CharacterClass, args.character.upper())

    service = None
    if args.model_path not in ("-", "", "<simple>"):
        net = load_net(args.model_path, torch_compile_mode=args.torch_compile,
                       use_value_head=args.value_head)
        service = NNService(net, batch_size=1, batch_size_factor=1,
                            torch_compile_mode=args.torch_compile)
        print(f"Loaded network from {args.model_path}")
    else:
        print("No model: using the built-in heuristic agent for overworld choices")

    try:
        watch_game(
            seed=args.seed,
            character=character,
            ascension=args.ascension,
            service=service,
            mcts_sims=args.mcts_simulations,
            temperature=args.temperature,
            battle_delay=args.battle_delay,
            choice_delay=args.choice_delay,
        )
    finally:
        if service is not None:
            service.stop()


if __name__ == "__main__":
    main()
