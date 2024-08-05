import slaythespire as sts

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 777, 0)
print(gc.map)
agent = sts.Agent()
agent.simulation_count_base = 5000

while gc.outcome == sts.GameOutcome.UNDECIDED:
    if gc.screen_state == sts.ScreenState.BATTLE:
        agent.playout_battle(gc) 
        print(gc)
    else:
        actions = sts.GameAction.getAllActionsInState(gc)
        action = actions[0]
        print(action.getDesc(gc))
        action.execute(gc)

