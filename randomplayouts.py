# %%
from __future__ import annotations

import sys
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

import pickle
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch import nn
import torch.nn.functional as F

from network import NN, ActionType, FixedAction, ModelHP, collate_fn, process_batch
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
    cards_offered: list[sts.NNCardRepresentation]

    # ActionType.PATH
    paths_offered: list[int]  # room ids (indices in NNMapRepresentation vectors)

    # ActionType.FIXED
    fixed_actions: list[FixedAction]  # Actions like SKIP

    def as_dict(self):
        return dict(
            obs=self.obs.as_dict(),
            cards_offered=dict(
                cards=(
                    np.concatenate([s.cards for s in self.cards_offered], axis=0, dtype=np.int32)
                    if self.cards_offered
                    else np.array([], dtype=np.int32)
                ),
                upgrades=(
                    np.concatenate([s.upgrades for s in self.cards_offered], axis=0, dtype=np.int32)
                    if self.cards_offered
                    else np.array([], dtype=np.int32)
                ),
            ),
            fixed_actions=np.array(self.fixed_actions if self.fixed_actions else [], dtype=np.int32),
            paths_offered=np.array(self.paths_offered, dtype=np.int32),
        )


def process_choice(net: NN, choice: Choice) -> dict:
    """
    Process a single Choice through the neural network.
    Returns the network output dictionary for this choice.
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
    
    return output


def get_choice_winprobs(net: NN, choice: Choice) -> dict[str, np.ndarray]:
    """
    Get win probabilities for each option in a Choice.
    Returns dict with 'card_probs' and 'fixed_probs' arrays.
    """
    with torch.no_grad():
        output = process_choice(net, choice)
        card_probs = torch.sigmoid(output['card_logits'][0]).cpu().numpy()
        fixed_probs = torch.sigmoid(output['fixed_logits'][0]).cpu().numpy()
        
        # Mask invalid entries
        n_valid = sum(len(s.cards) for s in choice.cards_offered)
        card_probs[n_valid:] = float('-inf')  # Mask invalid card choices
        
        # Mask invalid fixed actions
        n_fixed = len(choice.fixed_actions)
        fixed_probs[n_fixed:] = float('-inf')  # Mask invalid fixed actions
    
    return {
        'card_probs': card_probs,
        'fixed_probs': fixed_probs,
    }
@dataclass
class ChoiceOutcome:
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
                
                # Send responses
                for i, req in enumerate(requests[:unpadded_len]):
                    card_logits = output['card_logits'][i].cpu().numpy()
                    fixed_logits = output['fixed_logits'][i].cpu().numpy()
                    
                    # Get number of valid options
                    n_valid_cards = len(batch[i]['cards_offered.cards'])
                    n_fixed = len(req.choice.fixed_actions)
                    
                    # Trim logits to valid lengths
                    req.response_queue.put({
                        'card_logits': card_logits[:n_valid_cards],
                        'fixed_logits': fixed_logits[:n_fixed],
                    })
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in NN service: {e}")
                # Send error response to all waiting requests
                for req in requests:
                    req.response_queue.put(e)
    
    def get_logits(self, choice: Choice) -> dict[str, np.ndarray]:
        """Get raw logits from the network. Thread-safe."""
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
                      temperature: float = 0.01, stats: ChoiceStats = None, rng: random.Random = None) -> sts.GameAction:
    """Use neural network to pick a card from the choices using Boltzmann sampling"""
    logits = service.get_logits(choice)
    
    # Combine logits for sampling
    all_logits = np.concatenate([
        logits['card_logits'],
        logits['fixed_logits']
    ])
    probs = get_card_probs(all_logits)
    
    if stats is not None:
        stats.add_choice(probs, temperature)
        
    chosen_idx = sample_boltzmann(probs, temperature, rng)
    
    # Find the GameAction that corresponds to this card index
    total = 0
    for which_set, card_set in enumerate(choice.cards_offered):
        if chosen_idx < total + len(card_set.cards):
            which_card = chosen_idx - total
            # Find matching action in actions list
            for action in actions:
                if (action.idx1 == which_set and 
                    action.idx2 == which_card):
                    return action
            break
        total += len(card_set.cards)
    
    # If we get here, check if it's a fixed action
    if choice.fixed_actions and chosen_idx == total:
        if FixedAction.SKIP in choice.fixed_actions:
            # Find the skip action
            for action in actions:
                if action.rewards_action_type == sts.RewardsActionType.SKIP:
                    return action
    
    # Fallback to random choice if something went wrong
    print(f"Warning: Could not find action for card index {chosen_idx} (set {which_set}, card {which_card})")
    return rng.choice(actions) if rng else random.choice(actions)

def run_game(seed: int, net: NN = None, temperature: float = 0.01, verbose: bool = False, stats: ChoiceStats = None):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    # Create seeded RNG instance for this game
    rng = random.Random(seed)

    agent = sts.Agent()
    agent.simulation_count_base = 1000
    choices: list[ChoiceOutcome] = []

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
                cards_offered: list[sts.NNCardRepresentation] = []
                paths_offered: list[int] = []
                actions = sts.GameAction.getAllActionsInState(gc)
                
                # Use network for card choices if available
                if net is not None and gc.screen_state == sts.ScreenState.REWARDS:
                    cards_offered = gc.screen_state_info.rewards_container.cards
                    # Check if skip is allowed
                    fixed_actions = []
                    if any(a.rewards_action_type == sts.RewardsActionType.SKIP for a in actions):
                        fixed_actions.append(FixedAction.SKIP)
                    
                    if cards_offered:
                        choice = Choice(obs, cards_offered=cards_offered, paths_offered=[], 
                                     fixed_actions=fixed_actions)
                        action = pick_card_with_net(net, choice, actions, temperature=temperature, stats=stats, rng=rng)
                    else:
                        action = agent.pick_gameaction(gc)
                else:
                    action = agent.pick_gameaction(gc)
                
                if action not in actions:
                    print(gc)
                    print("chose", action.getDesc(gc))
                    print("options:")
                    for a in actions:
                        print(a.getDesc(gc))
                    raise ValueError("chosen action not in list of actions")
                
                if gc.screen_state == sts.ScreenState.REWARDS:
                    cards_offered = gc.screen_state_info.rewards_container.cards
                    which_set, which_card = action.idx1, action.idx2
                    
                    # Determine choice type and index based on the action taken
                    if action.rewards_action_type == sts.RewardsActionType.SKIP:
                        choice_type = ActionType.FIXED
                        chosen_idx = 0
                    elif cards_offered and which_card < len(cards_offered[which_set].cards):
                        choice_type = ActionType.CARD
                        chosen_idx = sum([len(s.cards) for s in cards_offered[:which_set]]) + which_card
                    else:
                        choice_type = ActionType.INVALID
                        chosen_idx = -1

                elif gc.screen_state == sts.ScreenState.MAP_SCREEN:
                    def xy_to_roomid(x, y):
                        roomids = [i for i in range(len(obs.map.xs)) if (y == 15 or obs.map.xs[i] == x) and obs.map.ys[i] == y]
                        try:
                            roomid, = roomids
                        except ValueError:
                            print(x, y, obs.map.xs, obs.map.ys)
                            raise
                        return roomid
                    paths_offered = [xy_to_roomid(a.idx1, gc.cur_map_node_y+1) for a in actions]
                    choice_type = ActionType.PATH
                    chosen_idx, = [ix for ix, a in enumerate(actions) if a.idx1 == action.idx1]
                else:
                    choice_type = ActionType.INVALID
                    chosen_idx = -1
                if choice_type != ActionType.INVALID:
                    # Create Choice with all available options
                    fixed_actions = []
                    if gc.screen_state == sts.ScreenState.REWARDS:
                        if any(a.rewards_action_type == sts.RewardsActionType.SKIP for a in actions):
                            fixed_actions.append(FixedAction.SKIP)
                    
                    choice = Choice(obs, cards_offered=cards_offered, paths_offered=paths_offered, 
                                  fixed_actions=fixed_actions)
                    choices.append(ChoiceOutcome(choice, choice_type=choice_type, chosen_idx=chosen_idx))
                if verbose:
                    print(action.getDesc(gc))
                action.execute(gc)
        except Exception:
            raise

    print(gc.outcome, gc.floor_num)
    return (choices, gc.outcome)

def run_game_data(seed: int, net: NN = None, temperature: float = 0.01, stats: ChoiceStats = None):
    choices, outcome = run_game(seed, net=net, temperature=temperature, verbose=False, stats=stats)
    df = pd.DataFrame([{
        **flatten_dict(c.choice.as_dict()),
        'choice_type': c.choice_type,
        'chosen_idx': c.chosen_idx,
    } for c in choices])
    df["outcome"] = {sts.GameOutcome.PLAYER_LOSS: 0, sts.GameOutcome.PLAYER_VICTORY: 1}[outcome]
    df["seed"] = seed
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
    if not args.no_plots:
        stats.plot_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Slay the Spire simulations with neural network guidance')
    parser.add_argument('--model-path', type=str, default="net.outcome.pt",
                        help='Path to the neural network model file')
    parser.add_argument('--num-threads', type=int, default=30,
                        help='Number of parallel threads to use')
    parser.add_argument('--start-seed', type=int, default=200_000,
                        help='Starting seed for simulations')
    parser.add_argument('--num-games', type=int, default=50_000,
                        help='Number of games to simulate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for neural network inference')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plotting of statistics')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving results to parquet file')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Temperature for Boltzmann sampling (default: 0.05)')
    
    args = parser.parse_args()
    main(args)

## %%

# %%
