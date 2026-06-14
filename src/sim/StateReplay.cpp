//
// Created for replaying recorded pre-battle states (gen_states format).
//

#include "sim/StateReplay.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "combat/BattleContext.h"
#include "game/GameAction.h"
#include "sim/search/Action.h"

namespace sts {

    std::vector<GameStateRecord> loadStateRecords(const std::string &path, int limit) {
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

    GameContext replayToPreBattle(const GameStateRecord &rec) {
        GameContext gc(static_cast<CharacterClass>(rec.charInt), rec.seed, rec.ascension);
        BattleContext bc;
        bool inBattle = false;
        std::size_t idx = 0;

        while (idx < rec.actions.size()) {
            if (inBattle) {
                if (bc.outcome != Outcome::UNDECIDED) {
                    bc.exitBattle(gc);
                    inBattle = false;
                } else {
                    // Engine changes can make a recorded run play out differently (e.g. Runic
                    // Dome battles draw differently since deferred rolls); validate so divergence
                    // throws instead of executing a stale action on a mismatched state.
                    const search::Action a(rec.actions[idx++]);
                    if (!a.isValidAction(bc)) {
                        throw std::runtime_error("replayToPreBattle: battle action diverged from prefix");
                    }
                    a.execute(bc);
                }
            } else {
                if (gc.outcome != GameOutcome::UNDECIDED) {
                    throw std::runtime_error("replayToPreBattle: game ended before consuming prefix");
                }
                if (gc.screenState == ScreenState::BATTLE) {
                    bc = {};
                    bc.init(gc);
                    inBattle = true;
                } else {
                    const GameAction a(rec.actions[idx++]);
                    if (!a.isValidAction(gc)) {
                        throw std::runtime_error("replayToPreBattle: game action diverged from prefix");
                    }
                    a.execute(gc);
                }
            }
        }

        if (inBattle) {
            throw std::runtime_error("replayToPreBattle: prefix ended mid-battle");
        }
        if (gc.screenState != ScreenState::BATTLE) {
            throw std::runtime_error("replayToPreBattle: prefix did not end at a battle");
        }
        return gc;
    }

}
