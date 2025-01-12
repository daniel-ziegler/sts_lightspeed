# %%
import sys
import random
from enum import IntEnum, auto
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def pick_card_with_net(net: NN, choice: Choice, actions: list[sts.GameAction], gc: sts.GameContext) -> sts.GameAction:
    """Use neural network to pick a card from the choices"""
    probs = get_choice_winprobs(net, choice)
    chosen_idx = int(np.argmax(probs))
    
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

def random_playout(seed: int, net: NN = None, verbose: bool = False):
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
                        action = pick_card_with_net(net, choice, actions, gc)
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

def random_playout_data(seed: int, net: NN = None):
    choices, outcome = random_playout(seed, net=net, verbose=False)
    df = pd.DataFrame([flatten_dict(c.as_dict()) for c in choices])
    df["outcome"] = {sts.GameOutcome.PLAYER_LOSS: 0, sts.GameOutcome.PLAYER_VICTORY: 1}[outcome]
    df["seed"] = seed
    return df

# %%
if __name__ == "__main__":
    num_threads = 1
    start_seed = 0
    num_playouts = 100 # _000
    
    # Load neural network
    net = load_net()
    print("Loaded neural network")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(random_playout_data, s, net) for s in range(start_seed, start_seed+num_playouts)]
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

    # Calculate and print winrate
    n_unique_seeds = df['seed'].nunique()
    n_wins = df.groupby('seed')['outcome'].last().sum()
    winrate = n_wins / n_unique_seeds
    print(f"\nResults from {n_unique_seeds} games:")
    print(f"Wins: {n_wins}")
    print(f"Winrate: {winrate:.1%}")

    df.to_parquet(f"rollouts{start_seed}_{start_seed+num_playouts}.net.parquet", engine="pyarrow")
## %%
#
