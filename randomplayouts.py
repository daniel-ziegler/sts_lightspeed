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
    """A set of possible actions and a choice among them"""
    obs: sts.NNRepresentation

    # ActionType.CARD
    cards_offered: list[sts.NNCardRepresentation]

    # ActionType.PATH
    paths_offered: list[int]  # room ids (indices in NNMapRepresentation vectors). TODO maybe switch to x,y

    chosen_type: ActionType
    chosen_idx: int  # idx in arrays corresponding to chosen_type

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
            chosen_type=self.chosen_type,
            chosen_idx=self.chosen_idx,
        )

# %%

def random_playout(seed: int, verbose: bool = False):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)

    agent = sts.Agent()
    agent.simulation_count_base = 1000
    choices: list[Choice] = []

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
                        chosen_type = ActionType.CARD
                        chosen_idx = sum([len(s.cards) for s in cards_offered[:which_set]]) + which_card
                    else:
                        # TODO empty choices, singing bowl
                        chosen_type = ActionType.INVALID
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
                    chosen_type = ActionType.PATH
                    chosen_idx, = [ix for ix, a in enumerate(actions) if a.idx1 == action.idx1]
                else:
                    chosen_type = ActionType.INVALID
                    chosen_idx = -1
                if chosen_type != ActionType.INVALID:
                    choices.append(Choice(obs, cards_offered=cards_offered, paths_offered=paths_offered, chosen_type=chosen_type, chosen_idx=chosen_idx))
                if verbose:
                    print(action.getDesc(gc))
                action.execute(gc)
        except Exception:
            raise

    print(gc.outcome, gc.floor_num)
    return (choices, gc.outcome)
# %%
def random_playout_data(seed: int):
    choices, outcome = random_playout(seed, verbose=False)
    df = pd.DataFrame([flatten_dict(c.as_dict()) for c in choices])
    df["outcome"] = {sts.GameOutcome.PLAYER_LOSS: 0, sts.GameOutcome.PLAYER_VICTORY: 1}[outcome]
    df["seed"] = seed
    return df

# %%
if __name__ == "__main__":
    num_threads = 30
    start_seed = 0
    num_playouts = 100_000

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(random_playout_data, s) for s in range(start_seed, start_seed+num_playouts)]
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

    df.to_parquet(f"rollouts{start_seed}_{start_seed+num_playouts}.parquet", engine="pyarrow")
## %%
#
