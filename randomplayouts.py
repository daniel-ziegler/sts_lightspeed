# %%
import sys
import random
from enum import IntEnum, auto
from dataclasses import dataclass, asdict
import pickle
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
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
@dataclass
class Choice:
    obs: sts.NNRepresentation
    cards_offered: list[sts.NNCardRepresentation]
    paths_offered: list[int]  # room ids (indices in NNMapRepresentation vectors)
    actions: list[sts.GameAction]
    choice: int  # idx in actions

    def as_dict(self):
        return dict(
            obs=self.obs.as_dict(),
            actions=np.array([a.bits for a in self.actions], dtype=np.int32),
            cards_offered=dict(
                # TODO preserve 2d structure, doesn't work for pyarrow
                cards=np.concatenate([o.cards for o in self.cards_offered], axis=0, dtype=np.int32) if self.cards_offered else np.array([], dtype=np.int32),
                upgrades=np.concatenate([o.upgrades for o in self.cards_offered], axis=0, dtype=np.int32) if self.cards_offered else np.array([], dtype=np.int32),
            ),
            paths_offered=np.array(self.paths_offered, dtype=np.int32),
            choice=self.choice,
        )

# %%
seed = 777
verbose = True

# %%

def random_playout(seed: int, verbose: bool = False):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)

    agent = sts.Agent()
    agent.simulation_count_base = 10000
    choices: list[Choice] = []

    while gc.outcome == sts.GameOutcome.UNDECIDED:
        try:
            if gc.screen_state == sts.ScreenState.BATTLE:
                agent.playout_battle(gc)
                obs = sts.getNNRepresentation(gc)
            else:
                obs = sts.getNNRepresentation(gc)
                cards_offered: list[sts.NNCardRepresentation] = []
                paths_offered: list[int] = []
                actions = sts.GameAction.getAllActionsInState(gc)
                if gc.screen_state == sts.ScreenState.REWARDS:
                    cards_offered = gc.screen_state_info.rewards_container.cards
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
                chosen = agent.pick_gameaction(gc)
                try:
                    choice = actions.index(chosen)
                except ValueError:
                    print(gc)
                    print("chose", chosen.getDesc(gc))
                    print("options:")
                    for a in actions:
                        print(a.getDesc(gc))
                    raise
                choices.append(Choice(obs, cards_offered=cards_offered, paths_offered=paths_offered, actions=actions, choice=choice))
                if verbose:
                    print(chosen.getDesc(gc))
                chosen.execute(gc)
        except Exception:
            break

    print(gc.outcome, gc.floor_num)
    return (choices, gc.outcome)
# %%
def random_playout_data(seed: int):
    choices, outcome = random_playout(seed)
    df = pd.DataFrame([flatten_dict(c.as_dict()) for c in choices])
    df["outcome"] = {sts.GameOutcome.PLAYER_LOSS: 0, sts.GameOutcome.PLAYER_VICTORY: 1}[outcome]
    df["seed"] = seed
    return df

# %%

from concurrent.futures import ThreadPoolExecutor, as_completed

num_threads = 4
start_seed = 6400
num_playouts = 3200

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(random_playout_data, s) for s in range(start_seed, start_seed+num_playouts)]
    df = pd.concat([future.result() for future in tqdm(as_completed(futures), total=num_playouts)])

df.to_parquet("rollouts.parquet", engine="pyarrow")
## %%
#
