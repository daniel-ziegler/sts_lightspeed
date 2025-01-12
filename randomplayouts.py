# %%
from __future__ import annotations

import sys
import random
from enum import IntEnum, auto
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Thread, Event
from typing import NamedTuple
import time

import pickle
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch import nn
import torch.nn.functional as F

from network import NN, ModelHP, collate_fn, process_batch
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
class ActionType(IntEnum):
    INVALID = auto()
    CARD = auto()
    PATH = auto()
    EVENT_OPTION = auto()

# %%
@dataclass
class Choice:
    """A set of possible actions"""
    obs: sts.NNRepresentation

    # ActionType.CARD
    cards_offered: list[sts.NNCardRepresentation]

    # ActionType.PATH
    paths_offered: list[int]  # room ids (indices in NNMapRepresentation vectors)

    choice_type: ActionType

    def as_dict(self):
        return dict(
            obs=self.obs.as_dict(),
            cards_offered=dict(
                # TODO preserve 2D structure
                cards=(
                    np.concatenate([s.cards for s in self.cards_offered], axis=0, dtype=np.int32)
                    if self.cards_offered
                    else np.array([], dtype=np.int32)
                ),
                upgrades=(
                    np.concatenate([s.upgrades for s in self.cards_offered], axis=0, dtype=np.int32)
                    if self.cards_offered
                    else np.array([], dtype=np.int32)
                )
            ),
            paths_offered=np.array(self.paths_offered, dtype=np.int32),
            choice_type=self.choice_type,
        )


def process_choice(net: NN, choice: Choice) -> dict:
    """
    Process a single Choice through the neural network.
    Returns the network output dictionary for this choice.
    """
    # Create a minimal batch with just this choice
    batch = [{
        'deck': np.array(choice.obs.deck.cards, dtype=np.int32),
        'deck_upgrades': np.array(choice.obs.deck.upgrades, dtype=np.int32),
        'choices': (
            np.concatenate([s.cards for s in choice.cards_offered], axis=0, dtype=np.int32)
            if choice.cards_offered
            else np.array([], dtype=np.int32)
        ),
        'choice_upgrades': (
            np.concatenate([s.upgrades for s in choice.cards_offered], axis=0, dtype=np.int32)
            if choice.cards_offered
            else np.array([], dtype=np.int32)
        ),
        'fixed_obs': np.array(choice.obs.fixed_observation, dtype=np.int32),
        'chosen_idx': 0,  # Dummy value, not used
        'outcome': 0.0,   # Dummy value, not used
    }]

    # Use existing collate_fn to create tensors
    batch_tensors = collate_fn(batch)
    
    # Process through network
    output = process_batch(batch_tensors, net)
    
    return output


def get_choice_winprobs(net: NN, choice: Choice) -> np.ndarray:
    """
    Get win probabilities for each option in a Choice.
    Returns numpy array of probabilities.
    """
    if choice.choice_type != ActionType.CARD:
        raise ValueError("Only card choices are supported currently")
        
    with torch.no_grad():
        output = process_choice(net, choice)
        probs = torch.sigmoid(output['card_choice_winprob_logits'][0]).cpu().numpy()
        
        # Mask invalid entries
        n_valid = sum(len(s.cards) for s in choice.cards_offered)
        probs[n_valid:] = float('-inf')
    
    return probs
@dataclass
class ChoiceOutcome:
    """A Choice and what was chosen from it"""
    choice: Choice
    chosen_idx: int  # idx in arr/ays corresponding to choice_type

    def as_dict(self):
        return {
            **self.choice.as_dict(),
            'chosen_idx': self.chosen_idx,
        }

def load_net(device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = NN(ModelHP())
    net = net.to(device)
    net = torch.compile(net, mode="reduce-overhead")
    
    state = torch.load("net.outcome.pt", map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    
    return net

class NNRequest(NamedTuple):
    choice: Choice
    response_queue: Queue

class NNService:
    """Background service that batches neural network inference requests"""
    def __init__(self, net: NN, batch_size=32, max_wait_time=0.001):
        self.net = net
        self.batch_size = batch_size
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
                
                # Try to get more requests up to batch_size or max_wait_time
                start_time = time.time()
                while len(requests) < self.batch_size and time.time() - start_time < self.max_wait_time:
                    try:
                        requests.append(self.request_queue.get_nowait())
                    except Empty:
                        break
                
                # Process batch
                choices = [req.choice for req in requests]
                
                # Create batch
                batch = [{
                    'deck': np.array(choice.obs.deck.cards, dtype=np.int32),
                    'deck_upgrades': np.array(choice.obs.deck.upgrades, dtype=np.int32),
                    'choices': (
                        np.concatenate([s.cards for s in choice.cards_offered], axis=0, dtype=np.int32)
                        if choice.cards_offered
                        else np.array([], dtype=np.int32)
                    ),
                    'choice_upgrades': (
                        np.concatenate([s.upgrades for s in choice.cards_offered], axis=0, dtype=np.int32)
                        if choice.cards_offered
                        else np.array([], dtype=np.int32)
                    ),
                    'fixed_obs': np.array(choice.obs.fixed_observation, dtype=np.int32),
                    'chosen_idx': 0,  # Dummy value
                    'outcome': 0.0,   # Dummy value
                } for choice in choices]
                
                # Process through network
                batch_tensors = collate_fn(batch)
                with torch.no_grad():
                    output = process_batch(batch_tensors, self.net)
                
                # Send responses
                for i, req in enumerate(requests):
                    logits = output['card_choice_winprob_logits'][i].cpu().numpy()
                    # Only return valid logits
                    n_valid = len(batch[i]['choices'])
                    logits = logits[:n_valid]  # Slice to only valid choices
                    req.response_queue.put(logits)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in NN service: {e}")
                # Send error response to all waiting requests
                for req in requests:
                    req.response_queue.put(e)
    
    def get_logits(self, choice: Choice) -> np.ndarray:
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
    return -np.sum(probs * np.log(probs))

def get_boltzmann_probs(probs: np.ndarray, temperature: float = 0.01) -> np.ndarray:
    """Convert probabilities to Boltzmann distribution"""
    logits = np.log(np.maximum(probs, 1e-20)) / temperature
    logits = logits - np.max(logits)  # Subtract max for numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def sample_boltzmann(probs: np.ndarray, temperature: float = 0.01) -> int:
    """Sample an index using Boltzmann distribution"""
    softmax_probs = get_boltzmann_probs(probs, temperature)
    return int(np.random.choice(len(probs), p=softmax_probs))

def pick_card_with_net(service: NNService, choice: Choice, actions: list[sts.GameAction], gc: sts.GameContext, stats: ChoiceStats = None) -> sts.GameAction:
    """Use neural network to pick a card from the choices using Boltzmann sampling"""
    if choice.choice_type != ActionType.CARD:
        raise ValueError("Only card choices are supported")
        
    logits = service.get_logits(choice)
    probs = get_card_probs(logits)
    
    if stats is not None:
        stats.add_choice(probs)
        
    chosen_idx = sample_boltzmann(probs)
    
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
            # If we get here, something went wrong
            print(f"Warning: Could not find action for card index {chosen_idx} (set {which_set}, card {which_card})")
            break
        total += len(card_set.cards)
    
    # Fallback to random choice if something went wrong
    return random.choice(actions)

def random_playout(seed: int, net: NN = None, verbose: bool = False, stats: ChoiceStats = None):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)

    agent = sts.Agent()
    agent.simulation_count_base = 1000
    choices: list[ChoiceOutcome] = []

    while gc.outcome == sts.GameOutcome.UNDECIDED:
        try:
            if gc.screen_state == sts.ScreenState.BATTLE:
                if verbose:
                    print(gc.deck)
                agent.playout_battle(gc)
                obs = sts.getNNRepresentation(gc)
            else:
                obs = sts.getNNRepresentation(gc)
                cards_offered: list[sts.NNCardRepresentation] = []
                paths_offered: list[int] = []
                actions = sts.GameAction.getAllActionsInState(gc)
                
                # Use network for card choices if available
                if net is not None and gc.screen_state == sts.ScreenState.REWARDS:
                    cards_offered = gc.screen_state_info.rewards_container.cards
                    if cards_offered:
                        choice = Choice(obs, cards_offered=cards_offered, paths_offered=[], choice_type=ActionType.CARD)
                        action = pick_card_with_net(net, choice, actions, gc, stats)
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
                    if cards_offered and which_card < len(cards_offered[which_set].cards):
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
                    choice = Choice(obs, cards_offered=cards_offered, paths_offered=paths_offered, choice_type=choice_type)
                    choices.append(ChoiceOutcome(choice, chosen_idx))
                if verbose:
                    print(action.getDesc(gc))
                action.execute(gc)
        except Exception:
            raise

    print(gc.outcome, gc.floor_num)
    return (choices, gc.outcome)

def random_playout_data(seed: int, net: NN = None, stats: ChoiceStats = None):
    choices, outcome = random_playout(seed, net=net, verbose=False, stats=stats)
    df = pd.DataFrame([flatten_dict(c.as_dict()) for c in choices])
    df["outcome"] = {sts.GameOutcome.PLAYER_LOSS: 0, sts.GameOutcome.PLAYER_VICTORY: 1}[outcome]
    df["seed"] = seed
    return df

class ChoiceStats:
    def __init__(self):
        self.entropies = []
        self.n_options = []
        self.boltzmann_entropies = []
        
    def add_choice(self, probs: np.ndarray):
        """Record statistics for a choice"""
        # Get raw entropy
        self.entropies.append(entropy(probs))
        
        # Count valid options
        self.n_options.append(np.sum(probs != float('-inf')))
        
        # Get Boltzmann entropy
        boltz_probs = get_boltzmann_probs(probs)
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

# %%
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    num_threads = 4
    start_seed = 0
    num_playouts = 100  #10_000
    
    # Load neural network and start service
    net = load_net()
    service = NNService(net, batch_size=32)
    print("Loaded neural network and started service")

    stats = ChoiceStats()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(random_playout_data, s, service, stats) for s in range(start_seed, start_seed+num_playouts)]
        df = pd.concat([
            future.result()
            for future
            in tqdm(
                as_completed(futures),
                total=num_playouts,
                mininterval=5,
                maxinterval=60,
                miniters=num_threads,
                smoothing=0.1,
            )
        ])

    service.stop()
    
    # Calculate and print winrate
    n_unique_seeds = df['seed'].nunique()
    n_wins = df.groupby('seed')['outcome'].last().sum()
    winrate = n_wins / n_unique_seeds
    print(f"\nResults from {n_unique_seeds} games:")
    print(f"Wins: {n_wins}")
    print(f"Winrate: {winrate:.1%}")
    
    # Plot choice statistics
    stats.plot_stats()

    df.to_parquet(f"rollouts{start_seed}_{start_seed+num_playouts}.net.parquet", engine="pyarrow")
## %%
#
