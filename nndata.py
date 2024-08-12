# %%
from math import e
import sys
import random
import torch

import slaythespire as sts

seed = 777
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
print(gc.map)
nnmap = gc.map.get_nn_rep()

agent = sts.Agent()
agent.simulation_count_base = 5000

# %%
while gc.outcome == sts.GameOutcome.UNDECIDED:
    if gc.screen_state == sts.ScreenState.BATTLE:
        agent.playout_battle(gc) 
        rep = sts.getNNRepresentation(gc)
        print(gc)
    else:
        chosen = None
        actions = sts.GameAction.getAllActionsInState(gc)
        if len(actions) == 1:
            chosen, = actions
        elif gc.screen_state == sts.ScreenState.REWARDS:
            for a in actions:
                if a.rewards_action_type in (sts.RewardsActionType.GOLD, sts.RewardsActionType.POTION, sts.RewardsActionType.RELIC):
                    chosen = a
                    break
        else:
            chosen = random.choice(actions)
        if chosen is not None:
            print(chosen.getDesc(gc))
            chosen.execute(gc)
        else:
            break

# %%
for a in actions:
    print(a.getDesc(gc))

gc.screen_state

# %%
gc.screen_state_info.rewards_container.cards[0].cards

# %%
actions[0].execute(gc)
# %%
