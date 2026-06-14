//
// Reconstructing a recorded pre-battle GameContext from a gen_states action prefix.
//
// A gen_states line is "charInt seed_hex ascension prefixLen action0_hex ..." where the prefix
// is the mixed out-of-combat GameAction / in-battle search::Action bit stream that reaches (but
// does not enter) a target battle. collect_states.py emits this format; loadStateRecords parses
// it and replayToPreBattle replays it deterministically (no search) to the pre-battle state.
//

#ifndef STS_LIGHTSPEED_STATEREPLAY_H
#define STS_LIGHTSPEED_STATEREPLAY_H

#include <cstdint>
#include <string>
#include <vector>

#include "game/GameContext.h"

namespace sts {

    struct GameStateRecord {
        int charInt;
        std::uint64_t seed;
        int ascension;
        std::vector<std::uint32_t> actions;
    };

    // Parse gen_states lines from `path` (up to `limit`, or all when limit <= 0).
    std::vector<GameStateRecord> loadStateRecords(const std::string &path, int limit);

    // Replay `rec`'s prefix to the GameContext sitting exactly at the start of the target battle
    // (screenState == BATTLE, battle not yet initialized). Throws if the recording diverges from
    // the current engine (an action is no longer legal) or does not end at a battle.
    GameContext replayToPreBattle(const GameStateRecord &rec);

}

#endif //STS_LIGHTSPEED_STATEREPLAY_H
