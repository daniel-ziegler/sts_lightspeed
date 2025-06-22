# %%
from __future__ import annotations

import random
from enum import IntEnum, auto
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Thread, Event, Timer
from typing import NamedTuple, Optional, List
import time
import threading
import argparse

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch import nn
import torch.nn.functional as F

from network import NN, ActionType, FixedAction, ModelHP, collate_fn, process_batch, output_to_cpu, action_logit_space, move_to_device
from inputs import Path
import slaythespire as sts

# %%
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# %%
@dataclass
class Choice:
    """A set of possible actions"""
    obs: sts.NNRepresentation

    # ActionType.CARD
    cards_offered: list[sts.Card]
    card_actions: list[sts.GameAction]

    # ActionType.PATH
    paths_offered: list[int]  # room ids (indices in NNMapRepresentation vectors)
    path_actions: list[sts.GameAction]

    # ActionType.RELIC
    relics_offered: list[sts.RelicId]
    relic_actions: list[sts.GameAction]

    # ActionType.POTION
    potions_offered: list[sts.Potion]
    potion_actions: list[sts.GameAction]

    # ActionType.FIXED
    fixed_actions: list[FixedAction]  # Actions like SKIP
    fixed_actions_list: list[sts.GameAction]

    def as_dict(self):
        # Extract card IDs and upgrades from the Card objects
        all_card_ids = []
        all_upgrades = []
        
        for card in self.cards_offered:
            all_card_ids.append(int(card.id))
            all_upgrades.append(card.upgrade_count)
        
        return dict(
            obs=self.obs.as_dict(),
            cards_offered=dict(
                cards=np.array(all_card_ids, dtype=np.int32),
                upgrades=np.array(all_upgrades, dtype=np.int32),
            ),
            relics_offered=np.array(self.relics_offered, dtype=np.int32),
            potions_offered=np.array(self.potions_offered, dtype=np.int32),
            fixed_actions=np.array(self.fixed_actions if self.fixed_actions else [], dtype=np.int32),
            paths_offered=np.array(self.paths_offered, dtype=np.int32),
        )


def process_choice(net: NN, choice: Choice) -> np.ndarray:
    """
    Process a single Choice through the neural network.
    Returns the flat logits array for this choice.
    """
    # Create a minimal batch with just this choice
    batch = [{
        **choice.as_dict(),
        'chosen_idx': 0,  # Dummy value, not used
        'outcome': 0.0,   # Dummy value, not used
    }]

    # Use existing collate_fn to create tensors
    batch_tensors = collate_fn(batch)
    
    # Process through network
    output = process_batch(batch_tensors, net)
    
    # Convert to CPU and return first (and only) item
    return output_to_cpu(output, batch_tensors)[0]


@dataclass
class Decision:
    """A Choice and what was chosen from it"""
    choice: Choice

    choice_type: ActionType  # which choice_type was chosen
    chosen_idx: int  # idx in arr/ays corresponding to choice_type

    def as_dict(self):
        return {
            **self.choice.as_dict(),
            'chosen_idx': self.chosen_idx,
            'choice_type': self.choice_type,
        }

def load_net(model_path, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = NN(ModelHP())
    net = net.to(device)
    net = torch.compile(net, mode="reduce-overhead")
    
    if model_path is not None:
        state = torch.load(model_path, map_location=device, weights_only=True)
        net.load_state_dict(state)
    net.eval()
    
    return net

class NNRequest(NamedTuple):
    choice: Choice
    response_queue: Queue

class NNService:
    """Background service that batches neural network inference requests"""
    def __init__(self, net: NN, batch_size=32, max_wait_time=0.01, batch_size_factor=8):
        self.net = net
        # Round batch_size up to nearest multiple of batch_size_factor
        self.batch_size = ((batch_size + batch_size_factor - 1) // batch_size_factor) * batch_size_factor
        self.batch_size_factor = batch_size_factor
        self.max_wait_time = max_wait_time
        self.request_queue = Queue()
        self.shutdown_event = Event()
        self.thread = Thread(target=self._process_requests, daemon=True)
        self.thread.start()
    
    def _process_requests(self):
        while not self.shutdown_event.is_set():
            # Collect requests
            requests = []
            try:
                # Get at least one request
                requests.append(self.request_queue.get(timeout=0.1))
                
                # Try to get more requests up to next multiple of batch_size_factor
                start_time = time.time()
                target_size = ((len(requests) + self.batch_size_factor - 1) 
                             // self.batch_size_factor) * self.batch_size_factor
                target_size = min(target_size, self.batch_size)
                
                while len(requests) < target_size and time.time() - start_time < self.max_wait_time:
                    try:
                        requests.append(self.request_queue.get_nowait())
                    except Empty:
                        break

                unpadded_len = len(requests)
                
                # Pad batch to multiple of batch_size_factor if needed
                if len(requests) < target_size:
                    target_size = ((len(requests) + self.batch_size_factor - 1) 
                                 // self.batch_size_factor) * self.batch_size_factor
                    # Duplicate last request to pad batch
                    while len(requests) < target_size:
                        requests.append(requests[-1])
                
                # Process batch
                choices = [req.choice for req in requests]
                
                # Create batch
                batch = [{
                    **flatten_dict(choice.as_dict()),
                    'choice_type': 0,  # Dummy value
                    'chosen_idx': 0,  # Dummy value
                    'outcome': 0.0,   # Dummy value
                } for choice in choices]
                
                # Process through network
                batch_tensors = collate_fn(batch)
                with torch.no_grad():
                    output = process_batch(batch_tensors, self.net)
                    responses = output_to_cpu(output, batch_tensors)
                
                # Send responses as (batch_tensors, output) pairs
                # output can be logits only, or (logits, values) tuple
                if isinstance(responses, tuple):
                    # Handle (logits, values) from value head
                    logits, values = responses
                    for i, req in enumerate(requests[:unpadded_len]):
                        req.response_queue.put((batch_tensors, (logits[i], values[i])))
                else:
                    # Handle logits only
                    for i, req in enumerate(requests[:unpadded_len]):
                        req.response_queue.put((batch_tensors, responses[i]))
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in NN service: {type(e)} {e}")
                # Send error response to all waiting requests
                for req in requests:
                    req.response_queue.put(e)
    
    def get_logits(self, choice: Choice) -> tuple[dict, np.ndarray]:
        """Get (batch_tensors, logits) from the network. Thread-safe."""
        response_queue = Queue()
        self.request_queue.put(NNRequest(choice, response_queue))
        response = response_queue.get()
        
        if isinstance(response, Exception):
            raise response
        
        return response
    
    def stop(self):
        """Stop the service and wait for it to finish"""
        self.shutdown_event.set()
        self.thread.join()

def get_card_probs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities"""
    # Just pretend they're softmax logits (even though they're really sigmoid logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def entropy(probs: np.ndarray) -> float:
    """Calculate entropy of a probability distribution"""
    probs /= np.sum(probs)
    return -np.sum(probs * np.log(np.maximum(probs, 1e-20)))

def get_boltzmann_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Convert probabilities to Boltzmann distribution"""
    logits = np.log(np.maximum(probs, 1e-20)) / temperature
    logits = logits - np.max(logits)  # Subtract max for numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def sample_boltzmann(probs: np.ndarray, temperature: float, rng: random.Random = None) -> int:
    """Sample an index using Boltzmann distribution"""
    softmax_probs = get_boltzmann_probs(probs, temperature)
    if rng is None:
        return int(np.random.choice(len(probs), p=softmax_probs))
    return int(rng.choices(range(len(probs)), weights=softmax_probs, k=1)[0])

def pick_card_with_net(service: NNService, choice: Choice, actions: list[sts.GameAction], 
                      temperature: float = 1.0, stats: ChoiceStats = None, rng: random.Random = None) -> tuple[sts.GameAction, Path]:
    """Use neural network to pick a card/relic from the choices using Boltzmann sampling"""
    collated_input, output = service.get_logits(choice)
    
    # Handle both single logits and (logits, values) tuple
    if isinstance(output, tuple):
        logits, values = output
    else:
        logits = output
        values = None
    
    assert logits.size > 0, logits.shape
    
    # Convert logits to probabilities
    probs = get_card_probs(logits)
    
    if stats is not None:
        stats.add_choice(probs, temperature)
        
    chosen_idx = sample_boltzmann(probs, temperature, rng)
    
    # Convert flat index back to semantic path using action_logit_space
    path = action_logit_space.ix_to_path(collated_input['choices'], chosen_idx)
    
    if path[0] == 'cards':
        # path is ['cards', card_index]
        card_index = path[1]
        if card_index >= len(choice.card_actions) or card_index < 0:
            print(f"Chosen index: {chosen_idx} from logits {logits}")
            print(f"{collated_input['choices']=}")
            raise ValueError(f"Chosen index {chosen_idx} out of bounds for {path}")
        return choice.card_actions[card_index], path
    
    elif path[0] == 'relics':
        # path is ['relics', relic_index]
        relic_index = path[1]
        return choice.relic_actions[relic_index], path
    
    elif path[0] == 'potions':
        # path is ['potions', potion_index]
        potion_index = path[1]
        return choice.potion_actions[potion_index], path
    
    elif path[0] == 'fixed':
        # path is ['fixed', action_index]
        action_index = path[1]
        return choice.fixed_actions_list[action_index], path
    
    raise ValueError(f"Could not find action for index {chosen_idx}")

def construct_choice(gc: sts.GameContext, obs: sts.NNRepresentation, actions: list[sts.GameAction]) -> Choice:
    """Construct a Choice object from the current game state and available actions."""
    cards_offered = []
    card_actions = []
    relics_offered = []
    relic_actions = []
    potions_offered = []
    potion_actions = []
    fixed_actions = []
    fixed_actions_list = []
    paths_offered = []
    path_actions = []
    
    # Build from available game actions, maintaining correspondence
    if gc.screen_state == sts.ScreenState.REWARDS:
        for action in actions:
            if action.rewards_action_type == sts.RewardsActionType.CARD:
                # handle singing bowl
                if action.idx2 == 5:
                    fixed_actions.append(FixedAction.SINGING_BOWL)
                    fixed_actions_list.append(action)
                else:
                    cards_offered.append(gc.screen_state_info.rewards_container.cards[action.idx1][action.idx2])
                    card_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.POTION:
                potions_offered.append(gc.screen_state_info.rewards_container.potions[action.idx1])
                potion_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.SKIP:
                fixed_actions.append(FixedAction.SKIP)
                fixed_actions_list.append(action)
                
    elif gc.screen_state == sts.ScreenState.SHOP_ROOM:
        # Shop cards are now returned as [card_set] where card_set contains all shop cards
        all_shop_relics = gc.screen_state_info.shop.relics
        all_shop_potions = gc.screen_state_info.shop.potions
        for action in actions:
            assert action.isValidAction(gc), f"Invalid shop action: {action.getDesc(gc)}"
            if action.rewards_action_type == sts.RewardsActionType.CARD:
                cards_offered.append(gc.screen_state_info.shop.cards[action.idx2])
                card_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.RELIC:
                relics_offered.append(all_shop_relics[action.idx1])
                relic_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.POTION:
                potions_offered.append(all_shop_potions[action.idx1])
                potion_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.SKIP:
                fixed_actions.append(FixedAction.SKIP)
                fixed_actions_list.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.CARD_REMOVE:
                fixed_actions.append(FixedAction.REMOVE)
                fixed_actions_list.append(action)
                
    elif gc.screen_state == sts.ScreenState.BOSS_RELIC_REWARDS:
        all_boss_relics = gc.screen_state_info.boss_relics
        for action in actions:
            if action.rewards_action_type == sts.RewardsActionType.RELIC:
                relics_offered.append(all_boss_relics[action.idx1])
                relic_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.SKIP:
                fixed_actions.append(FixedAction.SKIP)
                fixed_actions_list.append(action)
            else:
                raise ValueError(f"Invalid boss relic reward action: {action.getDesc(gc)}")
        
    elif gc.screen_state == sts.ScreenState.MAP_SCREEN:
        def xy_to_roomid(x, y):
            roomids = [i for i in range(len(obs.map.xs)) if (y == 15 or obs.map.xs[i] == x) and obs.map.ys[i] == y]
            try:
                roomid, = roomids
            except ValueError:
                print(x, y, obs.map.xs, obs.map.ys)
                raise
            return roomid
        for action in actions:
            paths_offered.append(xy_to_roomid(action.idx1, gc.cur_map_node_y+1))
            path_actions.append(action)

    return Choice(obs, cards_offered=cards_offered, card_actions=card_actions,
                  paths_offered=paths_offered, path_actions=path_actions,
                  fixed_actions=fixed_actions, fixed_actions_list=fixed_actions_list, 
                  relics_offered=relics_offered, relic_actions=relic_actions,
                  potions_offered=potions_offered, potion_actions=potion_actions)

def run_game(seed: int, net: NN = None, temperature: float = 1.0, verbose: bool = False, stats: ChoiceStats = None):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    # Create seeded RNG instance for this game
    rng = random.Random(seed)

    agent = sts.Agent()
    agent.simulation_count_base = 1000
    choices: list[Decision] = []

    # Create an event to signal timeout
    timeout_event = threading.Event()
    
    def timeout_handler():
        timeout_event.set()
        print(f"Warning: Battle simulation taking too long for seed {seed}")

    while gc.outcome == sts.GameOutcome.UNDECIDED:
        try:
            if gc.screen_state == sts.ScreenState.BATTLE:
                if verbose:
                    print(gc.deck)
                
                # Start a timer before battle simulation
                timer = Timer(30.0, timeout_handler)
                timer.start()
                
                try:
                    agent.playout_battle(gc)
                finally:
                    timer.cancel()
                    
                # Check if we hit the timeout
                if timeout_event.is_set():
                    print(f"Seed {seed} did finish")
                    timeout_event.clear()
                    
                obs = sts.getNNRepresentation(gc)
            else:
                obs = sts.getNNRepresentation(gc)
                actions = sts.GameAction.getAllActionsInState(gc)

                choice = construct_choice(gc, obs, actions)
                # Pick action using either network or agent
                if net is not None and gc.screen_state in (sts.ScreenState.REWARDS, sts.ScreenState.SHOP_ROOM, sts.ScreenState.BOSS_RELIC_REWARDS):
                    assert choice.cards_offered or choice.paths_offered or choice.relics_offered or choice.potions_offered or choice.fixed_actions, (gc.screen_state, actions, gc.screen_state_info.boss_relics)
                    action, action_path = pick_card_with_net(net, choice, actions, temperature=temperature, stats=stats, rng=rng)
                    
                    # Use path information to determine choice_type and chosen_idx
                    if action_path[0] == 'cards':
                        choice_type = ActionType.CARD
                        chosen_idx = action_path[1]
                    elif action_path[0] == 'relics':
                        choice_type = ActionType.RELIC
                        chosen_idx = action_path[1]
                    elif action_path[0] == 'potions':
                        choice_type = ActionType.POTION
                        chosen_idx = action_path[1]
                    elif action_path[0] == 'fixed':
                        choice_type = ActionType.FIXED
                        chosen_idx = action_path[1]
                    else:
                        choice_type = ActionType.INVALID
                        chosen_idx = -1
                else:
                    action = agent.pick_gameaction(gc)
                    
                    # For non-network actions (like map choices), use the old logic
                    choice_type = ActionType.INVALID
                    chosen_idx = -1
                    
                    if gc.screen_state == sts.ScreenState.MAP_SCREEN:
                        choice_type = ActionType.PATH
                        chosen_idx, = [ix for ix, a in enumerate(actions) if a.idx1 == action.idx1]
                
                if action not in actions:
                    print(gc)
                    print("chose", action.getDesc(gc))
                    print("options:")
                    for a in actions:
                        print(a.getDesc(gc))
                    raise ValueError("chosen action not in list of actions")

                # Record choice if valid
                if choice_type != ActionType.INVALID:
                    choices.append(Decision(choice, choice_type=choice_type, chosen_idx=chosen_idx))
                    
                if verbose:
                    print(action.getDesc(gc))
                assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                action.execute(gc)
        except Exception:
            raise

    print(gc.outcome, gc.floor_num)
    return (choices, gc.outcome, gc.floor_num)

def run_game_data(seed: int, net: NN = None, temperature: float = 1.0, stats: ChoiceStats = None):
    try:
        choices, outcome, final_floor = run_game(seed, net=net, temperature=temperature, verbose=False, stats=stats)
    except Exception as e:
        print(f"Error in run_game_data for seed {seed}: {e}")
        raise

    df = pd.DataFrame([{
        **flatten_dict(c.choice.as_dict()),
        'choice_type': c.choice_type,
        'chosen_idx': c.chosen_idx,
    } for c in choices])
    df["outcome"] = {sts.GameOutcome.PLAYER_LOSS: 0, sts.GameOutcome.PLAYER_VICTORY: 1}[outcome]
    df["seed"] = seed
    df["final_floor"] = final_floor
    return df

class ChoiceStats:
    def __init__(self):
        self.entropies = []
        self.n_options = []
        self.boltzmann_entropies = []
        
    def add_choice(self, probs: np.ndarray, temperature: float):
        """Record statistics for a choice"""
        # Get raw entropy
        self.entropies.append(entropy(probs))
        
        # Count valid options
        self.n_options.append(np.sum(probs != float('-inf')))
        
        # Get Boltzmann entropy
        boltz_probs = get_boltzmann_probs(probs, temperature)
        self.boltzmann_entropies.append(entropy(boltz_probs))
    
    def plot_stats(self):
        import matplotlib.pyplot as plt
        
        # Raw entropy histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.entropies, bins=50, label='Raw')
        plt.hist(self.boltzmann_entropies, bins=50, alpha=0.5, label='After Boltzmann')
        plt.xlabel('Entropy')
        plt.ylabel('Count')
        plt.title(f'Distribution of Choice Entropies\n' +
                 f'Raw mean={np.mean(self.entropies):.3f}, ' +
                 f'Boltzmann mean={np.mean(self.boltzmann_entropies):.3f}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Entropy vs number of options scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(self.n_options, self.entropies, alpha=0.1, label='Raw')
        plt.scatter(self.n_options, self.boltzmann_entropies, alpha=0.1, label='After Boltzmann')
        plt.xlabel('Number of Options')
        plt.ylabel('Entropy')
        plt.title('Entropy vs Number of Options')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"\nChoice Statistics:")
        print(f"Total choices: {len(self.entropies)}")
        print(f"Raw entropy: mean={np.mean(self.entropies):.3f}, median={np.median(self.entropies):.3f}")
        print(f"Boltzmann entropy: mean={np.mean(self.boltzmann_entropies):.3f}, median={np.median(self.boltzmann_entropies):.3f}")
        print(f"Options: mean={np.mean(self.n_options):.1f}, median={np.median(self.n_options):.1f}")

def main(args):
    torch.set_float32_matmul_precision('high')
    
    if args.model_path in ("", "-"):
        model_path = None
    else:
        model_path = args.model_path
    # Load neural network and start service
    net = load_net(model_path)
    service = NNService(
        net,
        batch_size=args.batch_size,
        batch_size_factor=min(min(8, args.batch_size), (args.num_threads + 1) // 2),
    )
    print(f"Loaded neural network from {args.model_path}")

    stats = ChoiceStats()
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [
            executor.submit(run_game_data, s, service, args.temperature, stats) 
            for s in range(args.start_seed, args.start_seed + args.num_games)
        ]
        df = pd.concat([
            future.result()
            for future
            in tqdm(
                as_completed(futures),
                total=args.num_games,
                mininterval=5,
                maxinterval=60,
                miniters=args.num_threads,
                smoothing=0.1,
            )
        ])

    service.stop()

    # Shuffle the DataFrame
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if not args.no_save:
        df_path = f"rollouts_v3_{args.start_seed}_{args.start_seed+args.num_games}.parquet"
        df.to_parquet(df_path, engine="pyarrow")
        print(f"Saved to {df_path}")
    
    # Calculate and print winrate
    n_unique_seeds = df['seed'].nunique()
    n_wins = df.groupby('seed')['outcome'].last().sum()
    winrate = n_wins / n_unique_seeds
    print(f"\nResults from {n_unique_seeds} games:")
    print(f"Wins: {n_wins}")
    print(f"Winrate: {winrate:.1%}")
    
    # Plot choice statistics
    if args.plots:
        stats.plot_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Slay the Spire simulations with neural network guidance')
    parser.add_argument('--model-path', type=str, default="-",
                        help='Path to the neural network model file')
    parser.add_argument('--num-threads', type=int, default=4,
                        help='Number of parallel threads to use')
    parser.add_argument('--start-seed', type=int, default=0,
                        help='Starting seed for simulations')
    parser.add_argument('--num-games', type=int, default=1000,
                        help='Number of games to simulate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for neural network inference')
    parser.add_argument('--plots', action='store_true',
                        help='Plot statistics')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving results to parquet file')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for Boltzmann sampling (default: 1.0)')
    
    args = parser.parse_args()
    main(args)

## %%

# %%
