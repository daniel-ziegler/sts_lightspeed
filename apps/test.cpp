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
    search::SearchStats stats;
};

static int g_searchAscension = 0;
static int g_simulationCount = 5;
static int g_print_level = 0;
static double g_explorationParameter = 25.0;   // honest-engine tuned default (see SearchAgent.h)
static double g_chanceWideningC = 3.7028;
static double g_chanceWideningAlpha = 0.52389;
static double g_bossChanceWideningC = 3.7028;     // boss-specific widening; defaults = general
static double g_bossChanceWideningAlpha = 0.52389;
static std::int64_t g_searchTimeMicros = 0;    // >0: search by time budget (us) instead of rollout count
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
        agent.bossChanceWideningC = g_bossChanceWideningC;
        agent.bossChanceWideningAlpha = g_bossChanceWideningAlpha;
        agent.searchTimeMicros = g_searchTimeMicros;
        agent.evalWeights = g_evalWeights;
        agent.rng = std::default_random_engine(gc.seed);

        agent.printActions = g_print_level & 0x1;
        agent.verbosityLevel = (g_print_level & 0x2) ? 2 : 0;

        agent.playout(gc);

        printOutcome(std::cout, gc);

        {
            std::scoped_lock lock(info->m);
            info->stats.add(agent.searchStats);
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

    const auto &st = info.stats;
    std::cout << "STATS steps=" << st.steps
              << " nodesCreated=" << st.nodesCreated
              << " detTrans=" << st.detTranspositions
              << " chanceSampled=" << st.chanceOutcomesSampled
              << " sibReuse=" << st.chanceSiblingReuse
              << " chanceTrans=" << st.chanceTranspositions
              << " avgDepth=" << (st.steps ? (double)st.depthSum/st.steps : 0)
              << " avgChanceDepth=" << (st.steps ? (double)st.chanceDepthSum/st.steps : 0) << '\n';
    std::cout << "threads: " << threadCount
              << " playoutCount: " << playoutCount
              << " depth: " << g_simulationCount
        << " asc: " << g_searchAscension
        << " elapsed: " << duration
        << std::endl;
}

// Distribution checks for the canonical draw-pile representation: deferred randomness must
// induce the same draw-sequence distributions the legacy concrete-shuffle engine produced.
// For each seed: navigate to the first battle, then run per-trial scenarios from a fresh
// CardManager::init with an independent rng. Reports per-position chi^2 vs the exact uniform
// expectation plus deterministic invariants (innates first, Headbutt top exact).
namespace drawdist {

    // chi^2 acceptance threshold: mean df + 6 sigma (df, 2df) — loose single-test bound, we
    // only want to catch gross non-uniformity, not borderline noise.
    bool chi2Ok(double chi2, int df) {
        return chi2 <= df + 6.0 * std::sqrt(2.0 * df) + 1.0;
    }

    struct Tally {
        // counts[drawPosition][uniqueId]
        std::vector<std::vector<int>> counts;
        void record(int pos, int uid) {
            if (pos >= static_cast<int>(counts.size())) counts.resize(pos + 1);
            auto &row = counts[pos];
            if (uid >= static_cast<int>(row.size())) row.resize(uid + 1, 0);
            ++row[uid];
        }
    };

    // chi^2 of one position's empirical distribution vs uniform over `candidates` uids.
    double posChi2(const std::vector<int> &row, const std::vector<int> &candidates, int trials) {
        const double expect = static_cast<double>(trials) / candidates.size();
        double chi2 = 0;
        for (int uid : candidates) {
            const int obs = uid < static_cast<int>(row.size()) ? row[uid] : 0;
            chi2 += (obs - expect) * (obs - expect) / expect;
        }
        return chi2;
    }

    void run(std::uint64_t startSeed, int numSeeds, int trials) {
        int failures = 0;
        for (std::uint64_t seed = startSeed; seed < startSeed + numSeeds; ++seed) {
            GameContext gc(CharacterClass::IRONCLAD, seed, 0);
            search::SimpleAgent nav;
            while (gc.outcome == GameOutcome::UNDECIDED && gc.screenState != ScreenState::BATTLE) {
                nav.stepOutOfCombat(gc);
            }
            if (gc.outcome != GameOutcome::UNDECIDED) {
                continue;
            }
            BattleContext bc0;
            bc0.init(gc);

            // identify innates/bottled (they form the known top at init)
            std::vector<int> innateUids, normalUids;
            for (int deckIdx = 0; deckIdx < gc.deck.size(); ++deckIdx) {
                bool isBottled = std::find(gc.deck.bottleIdxs.begin(), gc.deck.bottleIdxs.end(), deckIdx)
                        != gc.deck.bottleIdxs.end();
                if (gc.deck.cards[deckIdx].isInnate() || isBottled) {
                    innateUids.push_back(deckIdx);
                } else {
                    normalUids.push_back(deckIdx);
                }
            }
            const int deckSize = gc.deck.size();
            const int innateCount = static_cast<int>(innateUids.size());

            Tally initTally, reshuffleTally;
            Tally woundTallies[2];  // [markerCount-1]: wound draw-position histograms
            Tally bareTally;        // shuffle-in with no prior knowledge (top-promotion path)
            bool headbuttExact = true, innatesFirst = true, woundNeverFirst = true;
            bool bottomExact = true;  // Forethought: bottom card drawn dead last
            int bottomWoundBelow = 0;  // shuffle-in landing below a known bottom (drawn after it)

            for (int t = 0; t < trials; ++t) {
                // scenario A: draw out the full pile from battle init
                BattleContext bc = bc0;
                bc.rng = Random(seed * 1000003ULL + t);
                bc.cards.clear();
                bc.cards.init(gc, bc);
                for (int pos = 0; pos < deckSize; ++pos) {
                    const auto c = bc.cards.popFromDrawPile(bc.rng);
                    initTally.record(pos, c.uniqueId);
                    if (pos < innateCount &&
                        std::find(innateUids.begin(), innateUids.end(), c.uniqueId) == innateUids.end()) {
                        innatesFirst = false;
                    }
                    bc.cards.moveToDiscardPile(c);
                }

                // scenario B: reshuffle the (sorted) discard back in, draw out again
                bc.cards.moveDiscardPileIntoToDrawPile(bc.rng);
                for (int pos = 0; pos < deckSize; ++pos) {
                    const auto c = bc.cards.popFromDrawPile(bc.rng);
                    reshuffleTally.record(pos, c.uniqueId);
                    bc.cards.moveToDiscardPile(c);
                }

                // scenarios C/D: known top of 1 or 2 cards (Headbutt-style) + shuffle-in.
                // Wound draw position must be uniform over [markerCount? no: 1, pileSize-1] —
                // exactly the legacy distribution (it can land between known cards but never
                // above the top one).
                for (int markerCount = 1; markerCount <= 2; ++markerCount) {
                    bc.cards.moveDiscardPileIntoToDrawPile(bc.rng);
                    std::int16_t topUid = -1;
                    for (int m = 0; m < markerCount; ++m) {
                        CardInstance marker(CardId::ANGER);
                        topUid = static_cast<std::int16_t>(deckSize + m);
                        marker.uniqueId = topUid;
                        bc.cards.moveToDrawPileTop(marker);
                    }
                    CardInstance wound(CardId::WOUND);
                    wound.uniqueId = static_cast<std::int16_t>(deckSize + 2);
                    bc.cards.shuffleIntoDrawPile(bc.rng, wound);
                    const int pileN = bc.cards.drawPile.size();
                    for (int pos = 0; pos < pileN; ++pos) {
                        const auto c = bc.cards.popFromDrawPile(bc.rng);
                        if (pos == 0 && c.uniqueId != topUid) {
                            headbuttExact = false;  // known top must pop first, deterministically
                        }
                        if (c.uniqueId == wound.uniqueId) {
                            woundTallies[markerCount - 1].record(pos, 0);
                            if (pos == 0) {
                                woundNeverFirst = false;  // legacy gaps exclude the very top
                            }
                        } else if (c.uniqueId < deckSize) {
                            bc.cards.moveToDiscardPile(c);  // markers/wound stay out of later rounds
                        }
                    }
                }

                // scenario F: shuffle-in with no prior order knowledge. Legacy never inserts at
                // the very top, so the wound is never the next draw; its position must be
                // uniform over [1, deck].
                {
                    bc.cards.moveDiscardPileIntoToDrawPile(bc.rng);
                    CardInstance wound(CardId::WOUND);
                    wound.uniqueId = static_cast<std::int16_t>(deckSize + 2);
                    bc.cards.shuffleIntoDrawPile(bc.rng, wound);
                    const int pileN = bc.cards.drawPile.size();
                    for (int pos = 0; pos < pileN; ++pos) {
                        const auto c = bc.cards.popFromDrawPile(bc.rng);
                        if (c.uniqueId == wound.uniqueId) {
                            bareTally.record(pos, 0);
                            if (pos == 0) {
                                woundNeverFirst = false;
                            }
                        } else if (c.uniqueId < deckSize) {
                            bc.cards.moveToDiscardPile(c);
                        }
                    }
                }

                // scenario E: known bottom (Forethought) + shuffle-in. The bottom marker is
                // drawn dead last, except when the shuffled-in wound lands below it (legacy
                // gap 0, prob 1/N) — then the wound is last and the marker second-to-last.
                {
                    bc.cards.moveDiscardPileIntoToDrawPile(bc.rng);
                    CardInstance marker(CardId::ANGER);
                    marker.uniqueId = static_cast<std::int16_t>(deckSize);
                    bc.cards.moveToDrawPileBottom(marker);
                    CardInstance wound(CardId::WOUND);
                    wound.uniqueId = static_cast<std::int16_t>(deckSize + 2);
                    bc.cards.shuffleIntoDrawPile(bc.rng, wound);
                    const int pileN = bc.cards.drawPile.size();
                    int markerPos = -1, woundPos = -1;
                    for (int pos = 0; pos < pileN; ++pos) {
                        const auto c = bc.cards.popFromDrawPile(bc.rng);
                        if (c.uniqueId == marker.uniqueId) markerPos = pos;
                        if (c.uniqueId == wound.uniqueId) woundPos = pos;
                        if (c.uniqueId < deckSize) {
                            bc.cards.moveToDiscardPile(c);
                        }
                    }
                    if (woundPos == pileN - 1) {
                        ++bottomWoundBelow;
                        if (markerPos != pileN - 2) bottomExact = false;
                    } else if (markerPos != pileN - 1) {
                        bottomExact = false;
                    }
                }
            }

            // evaluate: positions [innateCount, deckSize) uniform over normals (positions
            // [0, innateCount) are covered by the innatesFirst invariant)
            double maxChi2 = 0;
            const int df = static_cast<int>(normalUids.size()) - 1;
            for (int pos = innateCount; pos < deckSize; ++pos) {
                maxChi2 = std::max(maxChi2, posChi2(initTally.counts[pos], normalUids, trials));
            }
            double maxChi2Reshuffle = 0;
            std::vector<int> allUids(deckSize);
            for (int i = 0; i < deckSize; ++i) allUids[i] = i;
            for (int pos = 0; pos < deckSize; ++pos) {
                maxChi2Reshuffle = std::max(maxChi2Reshuffle,
                                            posChi2(reshuffleTally.counts[pos], allUids, trials));
            }

            // wound position must be uniform over [1, pileN-1] (legacy-exact), pileN = deck +
            // markers + wound
            double woundChi2[2];
            int woundDf[2];
            for (int m = 0; m < 2; ++m) {
                const int pileN = deckSize + (m + 1) + 1;
                const double expect = static_cast<double>(trials) / (pileN - 1);
                double chi2 = 0;
                for (int pos = 1; pos < pileN; ++pos) {
                    int obs = 0;
                    if (pos < static_cast<int>(woundTallies[m].counts.size())
                        && !woundTallies[m].counts[pos].empty()) {
                        obs = woundTallies[m].counts[pos][0];
                    }
                    chi2 += (obs - expect) * (obs - expect) / expect;
                }
                woundChi2[m] = chi2;
                woundDf[m] = pileN - 2;
            }

            const bool initOk = df < 1 || chi2Ok(maxChi2, df);
            const bool reshufOk = chi2Ok(maxChi2Reshuffle, deckSize - 1);
            const bool woundOk = chi2Ok(woundChi2[0], woundDf[0]) && chi2Ok(woundChi2[1], woundDf[1]);
            // scenario F: wound uniform over [1, deck] (pile = deck + wound; promoted top first)
            double bareChi2 = 0;
            {
                const double expect = static_cast<double>(trials) / deckSize;
                for (int pos = 1; pos <= deckSize; ++pos) {
                    int obs = 0;
                    if (pos < static_cast<int>(bareTally.counts.size()) && !bareTally.counts[pos].empty()) {
                        obs = bareTally.counts[pos][0];
                    }
                    bareChi2 += (obs - expect) * (obs - expect) / expect;
                }
            }
            const bool bareOk = chi2Ok(bareChi2, deckSize - 1);
            // wound-below-bottom = legacy gap 0 of the deck+1 pre-insert gaps: 5-sigma binomial
            const double pBelow = 1.0 / (deckSize + 1);
            const double belowSig = std::sqrt(trials * pBelow * (1 - pBelow));
            const bool bottomFreqOk = std::abs(bottomWoundBelow - trials * pBelow) < 5 * belowSig + 1;
            const bool ok = initOk && reshufOk && woundOk && bareOk && headbuttExact && innatesFirst
                    && woundNeverFirst && bottomExact && bottomFreqOk;
            if (!ok) ++failures;

            std::cout << "seed " << seed
                      << " deck=" << deckSize << " innate=" << innateCount
                      << " | initMaxChi2=" << maxChi2 << " (df " << df << (initOk ? " OK" : " FAIL") << ")"
                      << " reshufMaxChi2=" << maxChi2Reshuffle << " (df " << deckSize - 1 << (reshufOk ? " OK" : " FAIL") << ")"
                      << " woundChi2={" << woundChi2[0] << "," << woundChi2[1] << "}" << (woundOk ? " OK" : " FAIL")
                      << " bareChi2=" << bareChi2 << (bareOk ? " OK" : " FAIL")
                      << " innatesFirst=" << (innatesFirst ? "OK" : "FAIL")
                      << " headbuttTop=" << (headbuttExact ? "OK" : "FAIL")
                      << " woundNeverFirst=" << (woundNeverFirst ? "OK" : "FAIL")
                      << " bottomLast=" << (bottomExact ? "OK" : "FAIL")
                      << " woundBelowBottom=" << bottomWoundBelow << "/" << trials
                      << (bottomFreqOk ? " OK" : " FAIL")
                      << std::endl;
        }
        std::cout << (failures == 0 ? "ALL OK" : "FAILURES: " + std::to_string(failures)) << std::endl;
    }

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
    search::SearchStats stats;
    std::mutex m;
    const std::vector<GameStateRecord> *records = nullptr;
    std::size_t next = 0;

    double explorationParameter = 9.9;   // tuned default
    double chanceWideningC = 4.6;         // tuned default
    double chanceWideningAlpha = 0.37;    // tuned default
    std::int64_t searchTimeMicros = 0;    // >0: search by time budget (us) instead of rollout count
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
        // eval_states applies its widening args uniformly (incl. boss fights) so tuning on a
        // boss-only set controls boss-state behavior; overrides SearchAgent's baked-in boss default.
        agent.bossChanceWideningC = info->chanceWideningC;
        agent.bossChanceWideningAlpha = info->chanceWideningAlpha;
        agent.searchTimeMicros = info->searchTimeMicros;
        agent.evalWeights = info->evalWeights;
        agent.verbosityLevel = 0;
        agent.rng = std::default_random_engine(gc.seed);

        agent.playoutBattle(bc);

        const bool dead = (bc.outcome == sts::Outcome::PLAYER_LOSS);
        // postBattleHealedHp: boss victories are scored on post-act-transition-heal HP, matching
        // what the player actually carries forward.
        const double score = dead ? -200.0 : (bc.postBattleHealedHp() + 10.0 * bc.potionCount);

        {
            std::scoped_lock lock(info->m);
            info->stats.add(agent.searchStats);
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
// params: exploration, wideningC, wideningAlpha, time, winBonus, potionWeight, victoryTurnPenalty,
//         monsterDamage, aliveWeight, energyWaste, drawWeight, turnSurvival (unset -> default).
//   time=<us>: search by a per-decision wall-clock budget (microseconds) instead of <simBudget> rollouts.
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
        else if (key == "time") info.searchTimeMicros = static_cast<std::int64_t>(val);
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
    const auto &st = info.stats;
    std::cout << "STATS steps=" << st.steps
              << " nodesCreated=" << st.nodesCreated
              << " detTrans=" << st.detTranspositions
              << " chanceSampled=" << st.chanceOutcomesSampled
              << " sibReuse=" << st.chanceSiblingReuse
              << " chanceTrans=" << st.chanceTranspositions
              << " avgDepth=" << (st.steps ? (double)st.depthSum/st.steps : 0)
              << " avgChanceDepth=" << (st.steps ? (double)st.chanceDepthSum/st.steps : 0) << '\n';
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
    else if (key == "bossWideningC") g_bossChanceWideningC = val;
    else if (key == "bossWideningAlpha") g_bossChanceWideningAlpha = val;
    else if (key == "time") g_searchTimeMicros = static_cast<std::int64_t>(val);
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
        agent.bossChanceWideningC = g_bossChanceWideningC;
        agent.bossChanceWideningAlpha = g_bossChanceWideningAlpha;
        agent.searchTimeMicros = g_searchTimeMicros;
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
    } else if (command == "verify_draw_dist") {
        // verify_draw_dist <startSeed> <numSeeds> <trialsPerSeed>
        drawdist::run(std::stoull(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]));
    } else if (command == "playground") {
        Action a = Actions::MakeTempCardInHand(CardInstance(CardId::DEFEND_RED), 1);
        std::cout << a << '\n';
    }

    //    printSizes();
//    std::cout << SeedHelper::getString(77) << '\n';
//    playRandom();
//    std::cout << getSeedWithGuardian();
//    replayActionList(argv[1]);

    return 0;
}