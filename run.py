import sys

import slaythespire as sts

seed = len(sys.argv) > 1 and int(sys.argv[1]) or 0
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
print(gc.map)

nni = sts.getNNInterface()
print("MAXIMUMS:", nni.getObservationMaximums())

agent = sts.Agent()
agent.simulation_count_base = 5000

while gc.outcome == sts.GameOutcome.UNDECIDED:
    if gc.screen_state == sts.ScreenState.BATTLE:
        agent.playout_battle(gc) 
        print(nni.getObservation(gc))
    else:
        actions = sts.GameAction.getAllActionsInState(gc)
        action = actions[0]
        print(action.getDesc(gc))
        action.execute(gc)

print(gc.outcome)