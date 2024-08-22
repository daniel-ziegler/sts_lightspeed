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
                cards=np.concatenate([o.cards for o in self.cards_offered], axis=0) if self.cards_offered else np.array([], dtype=np.uint16),
                upgrades=np.concatenate([o.upgrades for o in self.cards_offered], axis=0) if self.cards_offered else np.array([], dtype=np.int32),
            ),
            paths_offered=np.array(self.paths_offered, dtype=np.int32),
        )

# %%

def random_playout(seed: int, verbose: bool = False):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)

    agent = sts.Agent()
    agent.simulation_count_base = 10000
    choices: list[Choice] = []

    while gc.outcome == sts.GameOutcome.UNDECIDED:
        if gc.screen_state == sts.ScreenState.BATTLE:
            agent.playout_battle(gc)
            obs = sts.getNNRepresentation(gc)
        else:
            chosen = None
            actions = sts.GameAction.getAllActionsInState(gc)
            obs = sts.getNNRepresentation(gc)
            cards_offered: list[sts.NNCardRepresentation] = []
            paths_offered: list[int] = []
            if len(actions) == 1:
                chosen, = actions
            elif gc.screen_state == sts.ScreenState.REWARDS:
                cards_offered = gc.screen_state_info.rewards_container.cards
                for a in actions:
                    if a.rewards_action_type in (
                        sts.RewardsActionType.GOLD, sts.RewardsActionType.POTION, sts.RewardsActionType.RELIC,
                    ):
                        chosen = a
                        break
            elif gc.screen_state == sts.ScreenState.EVENT_SCREEN:
                # TODO neow, events
                chosen = random.choice(actions)
            elif gc.screen_state == sts.ScreenState.REST_ROOM:
                if gc.cur_hp < 30:
                    chosen = actions[0]
            elif gc.screen_state == sts.ScreenState.MAP_SCREEN:
                def xy_to_roomid(x, y):
                    roomid, = [i for i in range(len(obs.map.xs)) if obs.map.xs[i] == x and obs.map.ys[i] == y]
                    return roomid
                paths_offered = [xy_to_roomid(a.idx1, gc.cur_map_node_y+1) for a in actions]
            if chosen is None:
                choice = random.randrange(len(actions))
                choices.append(Choice(obs, cards_offered=cards_offered, paths_offered=paths_offered, actions=actions, choice=choice))
                chosen = actions[choice]
            if verbose:
                print(chosen.getDesc(gc))
            chosen.execute(gc)

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

num_threads = 8
num_playouts = 320

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(random_playout_data, s) for s in range(num_playouts)]
    df = pd.concat([future.result() for future in tqdm(as_completed(futures), total=num_playouts)])

df.to_parquet("rollouts.parquet", engine="pyarrow")
# %%
