# %%
import sys
import random
from enum import IntEnum, auto
from dataclasses import dataclass
import pickle

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import slaythespire as sts

# %%
# todo pyarrow everything
@dataclass
class Choice:
    obs: sts.NNRepresentation
    cards_offered: list[sts.NNCardRepresentation]
    paths_offered: list[int]  # room ids (indices in NNMapRepresentation vectors)
    actions: list[sts.GameAction]
    choice: int  # idx in actions

# %%

def random_playout(seed: int):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)

    agent = sts.Agent()
    agent.simulation_count_base = 5000
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
            elif gc.screen_state == sts.ScreenState.MAP_SCREEN:
                def xy_to_roomid(x, y):
                    roomids = [i for i in range(len(obs.map.xs)) if obs.map.xs[i] == x and obs.map.ys[i] == y]
                    if len(roomids) != 1:
                        print(x, y, obs.map.xs, obs.map.edge_starts)
                    roomid, = roomids
                    return roomid
                paths_offered = [xy_to_roomid(a.idx1, gc.cur_map_node_y+1) for a in actions]
            if chosen is None:
                choice = random.randrange(len(actions))
                choices.append(Choice(obs, cards_offered=cards_offered, paths_offered=paths_offered, actions=actions, choice=choice))
                chosen = actions[choice]
            print(chosen.getDesc(gc))
            chosen.execute(gc)

    return (choices, gc.outcome)
# %%
choices, outcome = random_playout(777)
print(outcome)

# %%
