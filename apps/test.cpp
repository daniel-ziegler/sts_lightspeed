//
// Created by gamerpuppy on 7/8/2021.
//

#include <iostream>
#include <chrono>
#include <cstdint>
#include <thread>
#include <memory>
#include <mutex>
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <string>
#include <cmath>

#include "data_structure/fixed_list.h"
#include "constants/Cards.h"
#include "constants/Events.h"
#include "constants/CardPools.h"
#include "game/Game.h"
#include "game/Map.h"
#include "game/Neow.h"
#include "game/SaveFile.h"
#include "combat/BattleContext.h"
#include "sim/ConsoleSimulator.h"
#include "sim/PrintHelpers.h"
#include "sim/RandomAgent.h"
#include "sim/search/SearchAgent.h"
#include "sim/search/SimpleAgent.h"

#include "sim/search/BattleSearcher.h"
#include "combat/Actions.h"
#include "constants/MonsterEncounters.h"

using namespace sts;

void printSizes() {
    std::cout << "sizeof Map:" << sizeof(Map) << '\n';
    std::cout << "sizeof Player: " << sizeof(Player) << '\n';
    std::cout << "sizeof Monster: " << sizeof(Monster) << '\n';
    std::cout << "sizeof MonsterGroup : " << sizeof(MonsterGroup) << '\n';
    std::cout << "sizeof CardInstance: " << sizeof(CardInstance) << '\n';
    std::cout << "sizeof CardManager : " << sizeof(CardManager) << '\n';
    std::cout << "sizeof Action : " << sizeof(Action) << '\n';
    std::cout << "sizeof ActionQueue<40> : " << sizeof(ActionQueue<40>) << '\n';
    std::cout << "sizeof BattleContext: " << sizeof(BattleContext) << '\n';

    std::cout << "sizeof GameContext: " << sizeof(GameContext) << '\n';
    std::cout << "sizeof Deck: " << sizeof(Deck) << '\n';
    std::cout << "sizeof Card: " << sizeof(Card) << '\n';
    std::cout << "sizeof SelectScreenCard: " << sizeof(SelectScreenCard) << '\n';
}

void playFromSaveFile(const std::string &fname, const std::string &actionFile) {
    CharacterClass cc;
    switch (tolower(fname[0])) {
        case 'i':
            cc = sts::CharacterClass::IRONCLAD;
        default:
            cc = sts::CharacterClass::IRONCLAD;
    }

    SaveFile saveFile = SaveFile::loadFromPath(fname, cc);

    ConsoleSimulator sim;
    sim.setupGameFromSaveFile(saveFile);
    SimulatorContext simContext;
    simContext.quitOnTestFailed = false;



    std::ifstream actionListInputStream(actionFile);

    sim.play(actionListInputStream, std::cout, simContext);
    actionListInputStream.close();

//    simContext.printFirstLine = true;
    simContext.quitCommandGiven = false;
    sim.play(std::cin, std::cout, simContext);
}

void replayActionFile(const GameContext &startState, const std::string &fname) {
    std::ifstream ifs(fname);
    GameContext gc(startState);
    BattleContext bc;


    bool inBattle = false;

    int lineNum = 0;
    std::uint32_t actionBits;
    while (true) {
        if (inBattle) {
            if (bc.outcome != sts::Outcome::UNDECIDED) {
                bc.exitBattle(gc);
                inBattle = false;

            } else {
                ++lineNum;
                ifs >> std::hex >> actionBits;
                search::Action a(actionBits);
                a.printDesc(std::cout, bc) << std::endl;
                a.execute(bc);
            }

        } else {
            if (gc.outcome != GameOutcome::UNDECIDED) {
                break;
            }
            if (gc.screenState == sts::ScreenState::BATTLE) {
                bc = {};
                bc.init(gc);
                inBattle = true;

            } else {
                ++lineNum;
                ifs >> std::hex >> actionBits;
                GameAction a(actionBits);
                a.printDesc(std::cout, gc) << std::endl;
                a.execute(gc);
            }
        }
    }
}

struct AgentMtInfo {
    std::mutex m;

    std::uint64_t curSeed;
    std::uint64_t seedStart;
    std::uint64_t seedEnd;

    std::int64_t winCount = 0;
    std::int64_t lossCount = 0;
    std::int64_t floorSum = 0;
    std::int64_t totalSimulations = 0;
};

static int g_searchAscension = 0;
static int g_simulationCount = 5;
static int g_print_level = 0;
static double g_explorationParameter = 3 * std::sqrt(2.0);
static double g_chanceWideningC = 1.0;
static double g_chanceWideningAlpha = 0.5;
static search::EvalWeights g_evalWeights;

void agentMtRunner(AgentMtInfo *info) {
    std::uint64_t seed;
    {
        std::scoped_lock lock(info->m);
        seed = info->curSeed++;
    }

    while(true) {
        if (seed >= info->seedEnd) {
            break;
        }

        GameContext gc(CharacterClass::IRONCLAD, seed, g_searchAscension);
        search::SearchAgent agent;
        agent.simulationCountBase = g_simulationCount;
        agent.explorationParameter = g_explorationParameter;
        agent.chanceWideningC = g_chanceWideningC;
        agent.chanceWideningAlpha = g_chanceWideningAlpha;
        agent.evalWeights = g_evalWeights;
        agent.rng = std::default_random_engine(gc.seed);

        agent.printActions = g_print_level & 0x1;
        agent.verbosityLevel = (g_print_level & 0x2) ? 2 : 0;

        agent.playout(gc);

        printOutcome(std::cout, gc);

        {
            std::scoped_lock lock(info->m);
            info->floorSum += gc.floorNum;
            if (gc.outcome == sts::GameOutcome::PLAYER_VICTORY) {
                ++info->winCount;
            } else {
                ++info->lossCount;
            }
            info->totalSimulations += agent.simulationCountTotal;

            seed = info->curSeed++;
        }
    }
}

void agentMt(int threadCount, std::uint64_t startSeed, int playoutCount) {
    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<std::unique_ptr<std::thread>> threads;

    AgentMtInfo info;
    info.curSeed = startSeed;
    info.seedStart = startSeed;
    info.seedEnd = startSeed + playoutCount;


    if (threadCount == 1) { // doing this for more consistency when benchmarking
        agentMtRunner(&info);

    } else {
        for (int tid = 0; tid < threadCount; ++tid) {
            threads.emplace_back(new std::thread(agentMtRunner, &info));
        }
    }

    for (int tid = 0; tid < threadCount; ++tid) {
        if (threadCount > 1) {
            threads[tid]->join();
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(endTime-startTime).count();

    std::cout << "w/l: (" << info.winCount  << ", " << info.lossCount << ")"
        << " percentWin: " << static_cast<double>(info.winCount) / playoutCount * 100 << "%"
        << " avgFloorReached: " << static_cast<double>(info.floorSum) / playoutCount << '\n'
        << " totalSimulations: " << info.totalSimulations
        << " avgPerFloor: " << (double)info.totalSimulations/info.floorSum << '\n';

    std::cout << "threads: " << threadCount
              << " playoutCount: " << playoutCount
              << " depth: " << g_simulationCount
        << " asc: " << g_searchAscension
        << " elapsed: " << duration
        << std::endl;
}

int mcts(int argc, const char *argv[]) {
    const auto saveFilePath = argv[2];
    const auto simulationCount = std::stoll(argv[3]);

    SaveFile saveFile = SaveFile::loadFromPath(saveFilePath, sts::CharacterClass::IRONCLAD);
    GameContext gc;
    gc.initFromSave(saveFile);

    std::cout << SeedHelper::getString(gc.seed) << std::endl;

    BattleContext bc = BattleContext();
    bc.init(gc);

    search::BattleSearcher searcher(bc);

    auto startTime = std::chrono::high_resolution_clock::now();
    searcher.search(simulationCount);
    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(endTime-startTime).count();

    std::cout << "steps: " << simulationCount << " search time: " << duration << "s\n";

    try {
        auto bestAction = searcher.getBestAction();
        std::cout << "best action: ";
        bestAction.printDesc(std::cout, bc) << '\n';
        bestAction.execute(bc);
    } catch (const std::runtime_error& e) {
        std::cout << "No actions available: " << e.what() << std::endl;
        return 0;
    }

    std::cout << "player hp after action: " << bc.player.curHp << '\n';

    searcher.printSearchTree(std::cout, 3);

    std::cout.flush();
    return 0;
}

// ---- Pre-battle state collection + hyperparameter evaluation ----

struct GameStateRecord {
    int charInt;
    std::uint64_t seed;
    int ascension;
    std::vector<std::uint32_t> actions;
};

// Reconstruct the GameContext sitting right at the start of the target battle by
// replaying a recorded mixed GameAction/search::Action stream (no search). Mirrors
// replayActionFile, but the prefix ends exactly when the battle screen is reached.
static GameContext loadPreBattleState(const GameStateRecord &rec) {
    GameContext gc(static_cast<CharacterClass>(rec.charInt), rec.seed, rec.ascension);
    BattleContext bc;
    bool inBattle = false;
    std::size_t idx = 0;

    while (idx < rec.actions.size()) {
        if (inBattle) {
            if (bc.outcome != sts::Outcome::UNDECIDED) {
                bc.exitBattle(gc);
                inBattle = false;
            } else {
                search::Action(rec.actions[idx++]).execute(bc);
            }
        } else {
            if (gc.outcome != GameOutcome::UNDECIDED) {
                throw std::runtime_error("loadPreBattleState: game ended before consuming prefix");
            }
            if (gc.screenState == ScreenState::BATTLE) {
                bc = {};
                bc.init(gc);
                inBattle = true;
            } else {
                GameAction(rec.actions[idx++]).execute(gc);
            }
        }
    }

    if (inBattle) {
        throw std::runtime_error("loadPreBattleState: prefix ended mid-battle");
    }
    if (gc.screenState != ScreenState::BATTLE) {
        throw std::runtime_error("loadPreBattleState: prefix did not end at a battle");
    }
    return gc;
}

struct GenStatesInfo {
    std::mutex m;
    std::uint64_t curSeed = 0;
    std::uint64_t seedEnd = 0;
    int simBudget = 0;
    int ascension = 0;
    std::ofstream *out = nullptr;
    int recordsWritten = 0;
};

static void genStatesRunner(GenStatesInfo *info) {
    while (true) {
        std::uint64_t seed;
        {
            std::scoped_lock lock(info->m);
            if (info->curSeed >= info->seedEnd) {
                break;
            }
            seed = info->curSeed++;
        }

        GameContext gc(CharacterClass::IRONCLAD, seed, info->ascension);
        search::SearchAgent agent;
        agent.simulationCountBase = info->simBudget;
        agent.recordActions = true;
        agent.verbosityLevel = 0;
        agent.rng = std::default_random_engine(gc.seed);

        agent.playout(gc);

        if (agent.battleStartIndices.empty()) {
            continue;
        }

        std::mt19937_64 pickRng(gc.seed);
        std::uniform_int_distribution<std::size_t> dist(0, agent.battleStartIndices.size() - 1);
        const int prefixLen = agent.battleStartIndices[dist(pickRng)];

        std::ostringstream line;
        line << static_cast<int>(CharacterClass::IRONCLAD) << ' '
             << std::hex << seed << std::dec << ' '
             << info->ascension << ' '
             << prefixLen;
        for (int i = 0; i < prefixLen; ++i) {
            line << ' ' << std::hex << static_cast<std::uint32_t>(agent.gameActionHistory[i]) << std::dec;
        }
        line << '\n';

        {
            std::scoped_lock lock(info->m);
            (*info->out) << line.str();
            ++info->recordsWritten;
            std::cout << "\rrecords: " << info->recordsWritten << std::flush;
        }
    }
}

static int genStates(int argc, const char *argv[]) {
    const int threadCount = std::stoi(argv[2]);
    const std::uint64_t startSeed = std::stoull(argv[3]);
    const int gameCount = std::stoi(argv[4]);
    const int simBudget = std::stoi(argv[5]);
    const int ascension = std::stoi(argv[6]);
    const std::string outFile = argv[7];

    std::ofstream out(outFile);
    if (!out) {
        throw std::runtime_error("gen_states: cannot open output file: " + outFile);
    }

    GenStatesInfo info;
    info.curSeed = startSeed;
    info.seedEnd = startSeed + gameCount;
    info.simBudget = simBudget;
    info.ascension = ascension;
    info.out = &out;

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<std::unique_ptr<std::thread>> threads;
    if (threadCount == 1) {
        genStatesRunner(&info);
    } else {
        for (int t = 0; t < threadCount; ++t) {
            threads.emplace_back(new std::thread(genStatesRunner, &info));
        }
        for (auto &th : threads) {
            th->join();
        }
    }
    out.flush();

    auto endTime = std::chrono::high_resolution_clock::now();
    const double duration = std::chrono::duration<double>(endTime - startTime).count();
    std::cout << "\ngen_states: wrote " << info.recordsWritten << " records in "
              << duration << "s" << std::endl;
    return 0;
}

static std::vector<GameStateRecord> loadRecords(const std::string &path, int limit) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("cannot open state file: " + path);
    }
    std::vector<GameStateRecord> records;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        GameStateRecord rec;
        int n = 0;
        iss >> rec.charInt;
        iss >> std::hex >> rec.seed;
        iss >> std::dec >> rec.ascension >> n;
        rec.actions.resize(n);
        iss >> std::hex;
        for (int i = 0; i < n; ++i) {
            iss >> rec.actions[i];
        }
        records.push_back(std::move(rec));
        if (limit > 0 && static_cast<int>(records.size()) >= limit) {
            break;
        }
    }
    return records;
}

struct EvalStatesInfo {
    std::mutex m;
    const std::vector<GameStateRecord> *records = nullptr;
    std::size_t next = 0;

    double explorationParameter = 3 * std::sqrt(2.0);
    double chanceWideningC = 1.0;
    double chanceWideningAlpha = 0.5;
    search::EvalWeights evalWeights;
    int simBudget = 0;

    double scoreSum = 0;
    int winCount = 0;
    double hpSum = 0;
    int n = 0;
};

static void evalStatesRunner(EvalStatesInfo *info) {
    while (true) {
        std::size_t idx;
        {
            std::scoped_lock lock(info->m);
            if (info->next >= info->records->size()) {
                break;
            }
            idx = info->next++;
        }

        GameContext gc = loadPreBattleState((*info->records)[idx]);
        BattleContext bc;
        bc.init(gc);

        search::SearchAgent agent;
        agent.simulationCountBase = info->simBudget;
        agent.explorationParameter = info->explorationParameter;
        agent.chanceWideningC = info->chanceWideningC;
        agent.chanceWideningAlpha = info->chanceWideningAlpha;
        agent.evalWeights = info->evalWeights;
        agent.verbosityLevel = 0;
        agent.rng = std::default_random_engine(gc.seed);

        agent.playoutBattle(bc);

        const bool dead = (bc.outcome == sts::Outcome::PLAYER_LOSS);
        const double score = dead ? -200.0 : (bc.player.curHp + 10.0 * bc.potionCount);

        {
            std::scoped_lock lock(info->m);
            info->scoreSum += score;
            if (!dead) {
                ++info->winCount;
                info->hpSum += bc.player.curHp;
            }
            ++info->n;
        }
    }
}

// eval_states <threadCount> <stateFile> <simBudget> <stateLimit> [param=value ...]
// params: exploration, wideningC, wideningAlpha, winBonus, potionWeight, victoryTurnPenalty,
//         monsterDamage, aliveWeight, energyWaste, drawWeight, turnSurvival (unset -> default).
static int evalStates(int argc, const char *argv[]) {
    const int threadCount = std::stoi(argv[2]);
    const std::string stateFile = argv[3];
    const int simBudget = std::stoi(argv[4]);
    const int stateLimit = std::stoi(argv[5]);

    EvalStatesInfo info;
    info.simBudget = simBudget;
    for (int i = 6; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto eq = arg.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("eval_states: expected key=value, got: " + arg);
        }
        const std::string key = arg.substr(0, eq);
        const double val = std::stod(arg.substr(eq + 1));
        if (key == "exploration") info.explorationParameter = val;
        else if (key == "wideningC") info.chanceWideningC = val;
        else if (key == "wideningAlpha") info.chanceWideningAlpha = val;
        else if (key == "winBonus") info.evalWeights.winBonus = val;
        else if (key == "potionWeight") info.evalWeights.potionWeight = val;
        else if (key == "victoryTurnPenalty") info.evalWeights.victoryTurnPenalty = val;
        else if (key == "monsterDamage") info.evalWeights.monsterDamageWeight = val;
        else if (key == "aliveWeight") info.evalWeights.aliveWeight = val;
        else if (key == "energyWaste") info.evalWeights.energyWasteWeight = val;
        else if (key == "drawWeight") info.evalWeights.drawWeight = val;
        else if (key == "turnSurvival") info.evalWeights.turnSurvivalWeight = val;
        else throw std::runtime_error("eval_states: unknown param: " + key);
    }

    const std::vector<GameStateRecord> records = loadRecords(stateFile, stateLimit);
    info.records = &records;

    std::vector<std::unique_ptr<std::thread>> threads;
    if (threadCount == 1) {
        evalStatesRunner(&info);
    } else {
        for (int t = 0; t < threadCount; ++t) {
            threads.emplace_back(new std::thread(evalStatesRunner, &info));
        }
        for (auto &th : threads) {
            th->join();
        }
    }

    const double meanScore = info.n > 0 ? info.scoreSum / info.n : 0.0;
    const double winRate = info.n > 0 ? static_cast<double>(info.winCount) / info.n : 0.0;
    const double avgHp = info.winCount > 0 ? info.hpSum / info.winCount : 0.0;
    std::cout << "SCORE " << meanScore << ' ' << winRate << ' ' << avgHp << ' ' << info.n << std::endl;
    return 0;
}

static int showStates(int argc, const char *argv[]) {
    const std::string stateFile = argv[2];
    const int count = std::stoi(argv[3]);
    const std::vector<GameStateRecord> records = loadRecords(stateFile, count);
    for (std::size_t i = 0; i < records.size(); ++i) {
        const GameContext gc = loadPreBattleState(records[i]);
        std::cout << "===== state " << i << "  seed=" << records[i].seed
                  << "  (prefix " << records[i].actions.size() << " actions) =====\n";
        std::cout << "  floor " << gc.floorNum << "  act " << gc.act
                  << "  asc " << gc.ascension
                  << "  | upcoming: " << monsterEncounterStrings[static_cast<int>(gc.info.encounter)] << "\n";
        std::cout << "  hp " << gc.curHp << "/" << gc.maxHp
                  << "  | gold " << gc.gold
                  << "  | potions " << gc.potionCount << "/" << gc.potionCapacity << "\n";
        std::cout << "  relics " << gc.relics << "\n";
        std::cout << "  deck " << gc.deck << "\n";
    }
    return 0;
}

// Parse one "key=value" arg into the global search-knob / eval-weight config.
static void applyGlobalParam(const std::string &arg) {
    const auto eq = arg.find('=');
    if (eq == std::string::npos) throw std::runtime_error("expected key=value, got: " + arg);
    const std::string key = arg.substr(0, eq);
    const double val = std::stod(arg.substr(eq + 1));
    if (key == "exploration") g_explorationParameter = val;
    else if (key == "wideningC") g_chanceWideningC = val;
    else if (key == "wideningAlpha") g_chanceWideningAlpha = val;
    else if (key == "winBonus") g_evalWeights.winBonus = val;
    else if (key == "potionWeight") g_evalWeights.potionWeight = val;
    else if (key == "victoryTurnPenalty") g_evalWeights.victoryTurnPenalty = val;
    else if (key == "monsterDamage") g_evalWeights.monsterDamageWeight = val;
    else if (key == "aliveWeight") g_evalWeights.aliveWeight = val;
    else if (key == "energyWaste") g_evalWeights.energyWasteWeight = val;
    else if (key == "drawWeight") g_evalWeights.drawWeight = val;
    else if (key == "turnSurvival") g_evalWeights.turnSurvivalWeight = val;
    else throw std::runtime_error("unknown param: " + key);
}

struct DumpInfo {
    std::mutex m;
    std::uint64_t curSeed = 0, seedEnd = 0;
    int simBudget = 0, ascension = 0;
    std::ofstream *out = nullptr;
    int gamesDone = 0;
};

static void dumpBattleOutcomesRunner(DumpInfo *info) {
    while (true) {
        std::uint64_t seed;
        {
            std::scoped_lock lock(info->m);
            if (info->curSeed >= info->seedEnd) break;
            seed = info->curSeed++;
        }
        GameContext gc(CharacterClass::IRONCLAD, seed, info->ascension);
        search::SearchAgent agent;
        agent.simulationCountBase = info->simBudget;
        agent.explorationParameter = g_explorationParameter;
        agent.chanceWideningC = g_chanceWideningC;
        agent.chanceWideningAlpha = g_chanceWideningAlpha;
        agent.evalWeights = g_evalWeights;
        agent.logBattleOutcomes = true;
        agent.verbosityLevel = 0;
        agent.rng = std::default_random_engine(gc.seed);

        agent.playout(gc);

        const int won = (gc.outcome == sts::GameOutcome::PLAYER_VICTORY) ? 1 : 0;
        const int finalFloor = gc.floorNum;
        std::ostringstream line;
        for (std::size_t b = 0; b < agent.battleLog.size(); ++b) {
            const auto &s = agent.battleLog[b];
            line << seed << ',' << b << ',' << s.floor << ',' << s.act << ',' << s.curHp << ','
                 << s.maxHp << ',' << s.potionCount << ',' << s.deckSize << ',' << s.encounter << ','
                 << won << ',' << finalFloor << '\n';
        }
        {
            std::scoped_lock lock(info->m);
            (*info->out) << line.str();
            ++info->gamesDone;
        }
    }
}

// dump_battle_outcomes <threads> <startSeed> <ngames> <simBudget> <ascension> <outFile> [param=value ...]
// Per battle in each full game: post-battle features + whether the game was eventually won.
static int dumpBattleOutcomes(int argc, const char *argv[]) {
    const int threadCount = std::stoi(argv[2]);
    const std::uint64_t startSeed = std::stoull(argv[3]);
    const int gameCount = std::stoi(argv[4]);
    const int simBudget = std::stoi(argv[5]);
    const int ascension = std::stoi(argv[6]);
    const std::string outFile = argv[7];
    for (int i = 8; i < argc; ++i) {
        applyGlobalParam(argv[i]);
    }

    std::ofstream out(outFile);
    if (!out) throw std::runtime_error("dump_battle_outcomes: cannot open " + outFile);
    out << "seed,battle_idx,floor,act,curHp,maxHp,potions,deckSize,encounter,game_won,final_floor\n";

    DumpInfo info;
    info.curSeed = startSeed;
    info.seedEnd = startSeed + gameCount;
    info.simBudget = simBudget;
    info.ascension = ascension;
    info.out = &out;

    std::vector<std::unique_ptr<std::thread>> threads;
    if (threadCount == 1) {
        dumpBattleOutcomesRunner(&info);
    } else {
        for (int t = 0; t < threadCount; ++t) threads.emplace_back(new std::thread(dumpBattleOutcomesRunner, &info));
        for (auto &th : threads) th->join();
    }
    out.flush();
    std::cout << "dump_battle_outcomes: " << info.gamesDone << " games written to " << outFile << std::endl;
    return 0;
}

int main(int argc, const char* argv[]) {

    if (argc < 2) {
        std::cout << "incorrect arguments" << std::endl;
        return 0;
    }

    const std::string command(argv[1]);

    if (command == "replay") {
        const std::uint64_t seed = std::stoull(argv[2]);
        const int ascension = std::stoi(argv[3]);
        const std::string actionFile(argv[4]);
        replayActionFile(GameContext(sts::CharacterClass::IRONCLAD, seed, ascension), actionFile);

    } else if (command == "save") {
        playFromSaveFile(argv[2], argv[3]);

    } if (command == "agent_mt") { // actually doing tree search now
        const int threadCount(std::stoi(argv[2]));
        const int depthArg = std::stoi(argv[3]);
        const int ascensionIn = std::stoi(argv[4]);
        const std::uint64_t startSeedLong(std::stoull(argv[5]));
        const int playoutCount(std::stoi(argv[6]));
        const int printLevel = std::stoi(argv[7]);
        g_print_level = printLevel;
        g_searchAscension = ascensionIn;
        g_simulationCount = depthArg;

        agentMt(threadCount, startSeedLong, playoutCount);

    } if (command == "simple_agent_mt") { // actually doing tree search now
        const int threadCount(std::stoi(argv[2]));
        const std::uint64_t startSeedLong(std::stoull(argv[3]));
        const int playoutCount(std::stoi(argv[4]));

        bool print = false;
        if (argc > 5) {
            print = true;
        }

        search::SimpleAgent::runAgentsMt(threadCount, startSeedLong, playoutCount, print);

    } else if (command == "json") {
        const std::string saveFilePath(argv[2]);
        const std::string jsonOutPath(argv[3]);
        std::ofstream outFileStream(jsonOutPath);
        outFileStream << SaveFile::getJsonFromSaveFile(saveFilePath);
        outFileStream.close();

    } else if (command == "json_to_save") {
        const std::string jsonInPath(argv[2]);
        const std::string saveFileOutPath(argv[3]);

        std::ifstream jsonIfStream(jsonInPath);
        SaveFile::writeJsonToSaveFile(jsonIfStream, saveFileOutPath);

    }  else if (command == "scum_searcher") {
        const std::uint64_t startSeedLong(std::stoull(argv[2]));
        const int playoutCount(std::stoi(argv[3]));

        for (std::uint64_t seed = startSeedLong; seed < startSeedLong+playoutCount; ++seed) {
//            playRandom4(startSeedLong);
        }

    } else if (command == "mcts_save") {
        return mcts(argc, argv);
    } else if (command == "gen_states") {
        return genStates(argc, argv);
    } else if (command == "eval_states") {
        return evalStates(argc, argv);
    } else if (command == "show_states") {
        return showStates(argc, argv);
    } else if (command == "dump_battle_outcomes") {
        return dumpBattleOutcomes(argc, argv);
    } else if (command == "winrate_mt") {
        // winrate_mt <threadCount> <startSeed> <playoutCount> <simBudget> <ascension> [param=value ...]
        // params: exploration, wideningC, wideningAlpha, winBonus, potionWeight, victoryTurnPenalty,
        //         monsterDamage, aliveWeight, energyWaste, drawWeight, turnSurvival.
        const int threadCount = std::stoi(argv[2]);
        const std::uint64_t startSeed = std::stoull(argv[3]);
        const int playoutCount = std::stoi(argv[4]);
        g_simulationCount = std::stoi(argv[5]);
        g_searchAscension = std::stoi(argv[6]);
        g_print_level = 0;
        for (int i = 7; i < argc; ++i) {
            applyGlobalParam(argv[i]);
        }
        agentMt(threadCount, startSeed, playoutCount);
    } else if (command == "playground") {
        std::vector<CardInstance> cards;
        cards.push_back(CardInstance(CardId::DEFEND_RED));
        Action a = Actions::MakeTempCardsInHand(cards);
        std::cout << a << '\n';
    }

    //    printSizes();
//    std::cout << SeedHelper::getString(77) << '\n';
//    playRandom();
//    std::cout << getSeedWithGuardian();
//    replayActionList(argv[1]);

    return 0;
}