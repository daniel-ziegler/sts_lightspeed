//
// Created by keega on 9/19/2021.
//

#ifndef STS_LIGHTSPEED_SEARCHAGENT_H
#define STS_LIGHTSPEED_SEARCHAGENT_H

#include "game/GameContext.h"
#include "sim/search/Action.h"
#include "game/GameAction.h"

#include <memory>
#include <random>
#include <cmath>

namespace sts::search {

    class BattleSearcher;

    struct SearchAgent {
        std::int64_t simulationCountTotal = 0;
        std::vector<int> gameActionHistory;

        bool recordActions = false;
        std::vector<int> battleStartIndices;

        int stepCount = 0;
        bool paused = false;
        bool pauseOnCardReward = false;

        bool printActions = false;
        int verbosityLevel = 1; // 0=quiet, 1=concise, 2=full

        int simulationCountBase = 50000;
        double bossSimulationMultiplier = 3;
        int stepsNoSolution = 5;
        int stepsWithSolution = 15;

        double explorationParameter = 3 * std::sqrt(2.0);
        double chanceWideningC = 1.0;
        double chanceWideningAlpha = 0.5;

        std::default_random_engine rng;

        // public interface
        void playout(GameContext &gc);

        // private methods
        void playoutBattle(BattleContext &bc);

        void takeAction(GameContext &gc, GameAction a);
        void takeAction(BattleContext &bc, Action a);

        void stepThroughSolution(BattleContext &bc, std::vector<search::Action> &actions);
        void stepThroughSearchTree(BattleContext &bc, const search::BattleSearcher &s);
        
        void printConciseAction(const BattleContext &bc, const Action &action);

        void stepOutOfCombatPolicy(GameContext &gc);
        GameAction pickOutOfCombatAction(const GameContext &gc);
        GameAction pickCardSelectAction(const GameContext &gc);
        GameAction pickEventAction(const GameContext &gc);
        GameAction pickRandomAction(const GameContext &gc);
        GameAction pickRewardsAction(const GameContext &gc);
        GameAction pickWeightedCardRewardAction(const GameContext &gc);
    };

}


#endif //STS_LIGHTSPEED_SEARCHAGENT_H
