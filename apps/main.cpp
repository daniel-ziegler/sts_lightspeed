#include <iostream>
#include <string>

#include "data_structure/fixed_list.h"
#include "constants/CardPools.h"
#include "constants/MonsterEncounters.h"
#include "game/Game.h"
#include "game/Map.h"
#include "game/Neow.h"
#include "combat/BattleContext.h"
#include "sim/ConsoleSimulator.h"
#include "sim/StateReplay.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
using namespace sts;

namespace {

void printUsage() {
    std::cout <<
        "usage:\n"
        "  main                         interactive: prompts 'seed character(I/S/D/W) ascensionLevel'\n"
        "  main list <stateFile> [n]    list recorded pre-battle states (floor / encounter / hp)\n"
        "  main replay <stateFile> <i>  teleport into recorded pre-battle state #i and play it\n";
}

// One-line summary of a replayed pre-battle GameContext.
void printStateSummary(std::ostream &os, int idx, const GameContext &gc) {
    os << "[" << idx << "] floor " << gc.floorNum
       << " act " << gc.act
       << " asc " << gc.ascension
       << " | " << monsterEncounterStrings[static_cast<int>(gc.info.encounter)]
       << " | hp " << gc.curHp << "/" << gc.maxHp
       << " | gold " << gc.gold << "\n";
}

int listStates(const std::string &stateFile, int limit) {
    const std::vector<GameStateRecord> records = loadStateRecords(stateFile, limit);
    int shown = 0;
    for (std::size_t i = 0; i < records.size(); ++i) {
        try {
            const GameContext gc = replayToPreBattle(records[i]);
            printStateSummary(std::cout, static_cast<int>(i), gc);
            ++shown;
        } catch (const std::exception &e) {
            std::cout << "[" << i << "] UNREPLAYABLE: " << e.what() << "\n";
        }
    }
    std::cout << shown << "/" << records.size() << " states replayable\n";
    return 0;
}

int replayState(const std::string &stateFile, int index) {
    const std::vector<GameStateRecord> records = loadStateRecords(stateFile, 0);
    if (index < 0 || index >= static_cast<int>(records.size())) {
        std::cerr << "index " << index << " out of range (" << records.size() << " records)\n";
        return 1;
    }
    GameContext gc = replayToPreBattle(records[index]);
    std::cout << "Teleported to state #" << index << ":\n";
    printStateSummary(std::cout, index, gc);
    std::cout << "You are now playing this battle. Good luck.\n\n";

    ConsoleSimulator sim;
    sim.setupGameFromContext(gc);
    SimulatorContext simCtx;
    sim.play(std::cin, std::cout, simCtx);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc >= 2) {
        const std::string mode = argv[1];
        if (mode == "list" && argc >= 3) {
            const int limit = argc >= 4 ? std::stoi(argv[3]) : 0;
            return listStates(argv[2], limit);
        }
        if (mode == "replay" && argc >= 4) {
            return replayState(argv[2], std::stoi(argv[3]));
        }
        printUsage();
        return 1;
    }

    while (!std::cin.eof()) {
        std::cout << "usage: seed character(I/S/D/W) ascensionLevel" << std::endl;

        SimulatorContext simCtx;
        ConsoleSimulator simulator;
        simulator.play(std::cin, std::cout, simCtx);
    }

    return 0;
}

#pragma clang diagnostic pop
