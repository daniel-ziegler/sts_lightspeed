//
// Created by gamerpuppy on 6/25/2021.
//

#ifndef STS_LIGHTSPEED_CONSOLESIMULATOR_H
#define STS_LIGHTSPEED_CONSOLESIMULATOR_H

#include <cstdint>
#include <iostream>

#include "constants/CharacterClasses.h"
#include "game/GameContext.h"
#include "game/GameAction.h"

#include "sim/BattleSimulator.h"
#include "sim/search/SearchAgent.h"

namespace sts {

    class BattleSimulator;

    struct SimulatorContext {
        // state
        int line = 0;
        bool quitCommandGiven = false;
        bool tookAction = false;
        bool failedTest = false;

        // settings
        bool printFirstLine = true;
        bool skipTests = false;
        bool printLogActions = true;
        bool printInput = true;
        bool printPrompts = true;
        bool quitOnTestFailed = true;
        
        // auto-battle settings
        bool autoBattleMode = false;
        int autoBattleVerbosity = 1; // 0=quiet, 1=concise, 2=full
        int autoBattleSimCount = 5000; // MCTS simulationCountBase for auto-fought battles
    };

    struct ConsoleSimulator {
        // settings

        // state
        GameContext *gc = nullptr;
        BattleSimulator battleSim;
        search::SearchAgent autoBattleAgent;

        ConsoleSimulator() = default;
        void setupGame(std::uint64_t seed, CharacterClass c, int ascension);
        void setupGameFromSaveFile(const SaveFile &save);
        void setupGameFromContext(const GameContext &src);  // adopt a prebuilt state (e.g. a replayed pre-battle state)
        void reset();

        void play(std::istream &is, std::ostream &os, SimulatorContext &c);

        void getInputLoop(std::istream &is, std::ostream &os, SimulatorContext &c);
        void handleInputLine(const std::string &line, std::ostream &os, SimulatorContext &c);
        void handleAutoBattle(std::ostream &os, SimulatorContext &c);

        void doPrintCommand(std::ostream &os, const std::string &cmd);
        void doSetCommand(const std::string &cmd, SimulatorContext &c);

        void printActions(std::ostream &os) const;
        GameAction parseAction(const std::string &action);

        void printBossRelicRewardsActions(std::ostream &os) const;
        GameAction parseBossRelicRewardsAction(const std::string &action);

        void printRestRoomActions(std::ostream &os) const;
        GameAction parseRestRoomAction(const std::string &action);

        void printShopRoomActions(std::ostream &os) const;
        GameAction parseShopRoomAction(const std::string &action);

        void printTreasureRoomActions(std::ostream &os) const;
        GameAction parseTreasureRoomAction(const std::string &action);

        void printMapScreenActions(std::ostream &os) const;
        GameAction parseMapScreenAction(const std::string &action);

        void printRewardsScreenActions(std::ostream &os) const;
        GameAction parseRewardScreenAction(const std::string &action);

        void printCardSelectScreenActions(std::ostream &os) const;
        GameAction parseCardSelectScreenAction(const std::string &action);

        void printEventActions(std::ostream &) const;
        GameAction parseEventAction(const std::string &action);
    };

}

#endif //STS_LIGHTSPEED_CONSOLESIMULATOR_H
