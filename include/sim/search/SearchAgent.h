//
// Created by keega on 9/19/2021.
//

#ifndef STS_LIGHTSPEED_SEARCHAGENT_H
#define STS_LIGHTSPEED_SEARCHAGENT_H

#include "game/GameContext.h"
#include "sim/search/Action.h"
#include "game/GameAction.h"
#include "sim/search/BattleSearcher.h"

#include <memory>
#include <random>
#include <cmath>

#include <atomic>
namespace sts::search { extern std::atomic<long> g_rerootExact, g_rerootPermuted, g_rerootMiss; }
namespace sts::search {

    // Post-battle snapshot for studying which per-battle features predict full-game wins.
    struct BattleSnapshot {
        int floor, act, curHp, maxHp, potionCount, deckSize, encounter;
    };

    struct SearchAgent {
        std::int64_t simulationCountTotal = 0;
        search::SearchStats searchStats;   // summed over this agent's battles
        std::vector<int> gameActionHistory;

        bool recordActions = false;
        std::vector<int> battleStartIndices;

        bool logBattleOutcomes = false;
        std::vector<BattleSnapshot> battleLog;

        int stepCount = 0;
        bool paused = false;
        bool pauseOnCardReward = false;

        bool printActions = false;
        int verbosityLevel = 1; // 0=quiet, 1=concise, 2=full

        int simulationCountBase = 50000;
        double bossSimulationMultiplier = 3;
        std::int64_t searchTimeMicros = 0;  // >0: search by wall-clock budget (us) instead of rollout count
        int stepsNoSolution = 5;
        int stepsWithSolution = 15;

        // Tuned for the honest (canonical CardPile) engine, whose draw chance nodes give value
        // estimates much noisier than the old clairvoyant engine's: the search needs far more
        // exploration before committing. Deployment-validated on 1000-seed paired eval_hero
        // blocks: exploration 9.9 -> 54-58%, 18.5 -> 60.7%, 25 -> 62.5%, 35 -> 56.3% (peak
        // bracketed near 25). Widening from the coarse honest tune's top region; widening
        // sensitivity at deployment is much weaker than exploration's.
        double explorationParameter = 25.0;
        double explorationParameterChance = 25.0;  // UCB constant for stochastic edges (chance-node children)
        double chanceWideningC = 3.7028;
        double chanceWideningAlpha = 0.52389;
        EvalWeights evalWeights;

        // Boss-specific chance-node widening, applied via isBossEncounter in playoutBattle.
        // Defaults equal the general widening (no specialization yet for the honest engine;
        // the clairvoyant-era boss studies are not transferable).
        double bossChanceWideningC = 3.7028;
        double bossChanceWideningAlpha = 0.52389;

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
