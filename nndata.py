# %%
import sys
import random

import slaythespire as sts

seed = 777
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
print(gc.map)
nnmap = gc.map.get_nn_rep()

print("MAXIMUMS:", sts.getFixedObservationMaximums())

agent = sts.Agent()
agent.simulation_count_base = 5000

# %%
nnmap.room_types

# %%
list(zip(nnmap.edge_starts, nnmap.edge_ends))

# %%
i = 0
while gc.outcome == sts.GameOutcome.UNDECIDED and i < 10:
    if gc.screen_state == sts.ScreenState.BATTLE:
        agent.playout_battle(gc) 
        rep = sts.getNNRepresentation(gc)
        print(rep.deck.cards)
    else:
        actions = sts.GameAction.getAllActionsInState(gc)
        action = random.choice(actions)
        print(action.getDesc(gc))
        action.execute(gc)
    i += 1

print(gc.outcome)
# %%
