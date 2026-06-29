//
// Created by keega on 9/16/2021.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <pybind11/detail/internals.h>

#include <sstream>
#include <algorithm>

#include "constants/MonsterEncounters.h"
#include "constants/Potions.h"
#include "constants/Events.h"
#include "constants/Cards.h"
#include "constants/CardPools.h"
#include "constants/Relics.h"
#include "constants/MonsterIds.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/MonsterMoves.h"
#include "combat/BattleContext.h"
#include "combat/InputState.h"
#include "combat/Player.h"
#include "combat/Monster.h"
#include "combat/MonsterGroup.h"
#include "combat/CardManager.h"
#include "combat/CardInstance.h"
#include "combat/CardSelectInfo.h"
#include "sim/ConsoleSimulator.h"
#include "sim/search/SearchAgent.h"
#include "sim/search/BattleSearcher.h"
#include "sim/search/Action.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "game/GameAction.h"
#include "game/Neow.h"

#include "slaythespire.h"


using namespace sts;
using namespace pybind11::literals;

pybind11::dict py::NNCardsRepresentation::as_dict() const {
    return pybind11::dict("cards"_a=cards,
                          "upgrades"_a=upgrades);
}

pybind11::dict py::NNRelicsRepresentation::as_dict() const {
    return pybind11::dict("relics"_a=relics,
                          "relic_counters"_a=relicCounters);
}

pybind11::dict py::NNMapRepresentation::as_dict() const {
    return pybind11::dict("xs"_a=xs,
                          "ys"_a=ys,
                          "roomTypes"_a=roomTypes,
                          "pathXs"_a=pathXs,
                          "burningEliteX"_a=burningEliteX,
                          "burningEliteY"_a=burningEliteY);
}

pybind11::dict py::NNRepresentation::as_dict() const {
    return pybind11::dict("fixed_observation"_a=fixedObservation,
                        "deck"_a=deck.as_dict(),
                        "relics"_a=relics.as_dict(),
                        "potions"_a=potions,
                        "map"_a=map.as_dict(),
                        "mapX"_a=mapX,
                        "mapY"_a=mapY);
}

// Bind a field as a read/write property whose getter returns BY VALUE (a snapshot),
// rather than by reference. pybind11's def_readwrite uses return_value_policy::reference_internal,
// which for bound enum types (py::enum_) returns a Python object aliasing the live C++ field, so a
// stored Python read silently tracks later mutation. Returning by value copies the enum, avoiding
// that aliasing bug. Use this for enum-typed fields; struct/class fields intentionally keep reference
// semantics (live navigation) and scalars are already copied.
template <class C, class T>
void def_value(pybind11::class_<C> &cls, const char *name, T C::*m) {
    cls.def_property(name,
        [m](const C &o) { return o.*m; },              // by-value -> snapshot, no alias
        [m](C &o, T v) { o.*m = v; });
}

PYBIND11_MODULE(slaythespire, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("play", &sts::py::play, "play Slay the Spire Console");
    m.def("potion_requires_target", &sts::potionRequiresTarget,
          "whether a potion needs a target monster (true only for Fear/Fire/Poison/Weak); the "
          "authoritative source vs spirecomm's requires_target, which mis-flags AOE potions like "
          "Explosive");
    m.def("get_seed_str", &SeedHelper::getString, "gets the integral representation of seed string used in the game ui");
    m.def("get_seed_long", &SeedHelper::getLong, "gets the seed string representation of an integral seed");

    m.def("getFixedObservation", &py::getFixedObservation, "get observation array given a GameContext");
    m.def("getFixedObservationMaximums", &py::getFixedObservationMaximums, "get the defined maximum values of the observation space");
    m.def("getNNRepresentation", &py::getNNRepresentation, "get the neural network representation of a GameContext");
    m.def("get_card_pool", [](CharacterClass cc, CardType type, CardRarity rarity) {
        std::vector<CardId> out;
        const int n = TypeRarityCardPool::getPoolSize(cc, type, rarity);
        for (int i = 0; i < n; ++i) {
            out.push_back(TypeRarityCardPool::getCardFromPool(cc, type, rarity, i));
        }
        return out;
    }, "the engine's obtainable-card pool for a class/type/rarity (what card rewards draw from)");
    m.def("get_colorless_card_pool", []() {
        return std::vector<CardId>(srcColorlessCardPool, srcColorlessCardPool + srcColorlessCardPoolSize);
    }, "the engine's colorless card pool");

    pybind11::class_<search::EvalWeights>(m, "EvalWeights")
        .def(pybind11::init<>())
        .def_readwrite("win_bonus", &search::EvalWeights::winBonus)
        .def_readwrite("potion_weight", &search::EvalWeights::potionWeight)
        .def_readwrite("victory_turn_penalty", &search::EvalWeights::victoryTurnPenalty)
        .def_readwrite("monster_damage_weight", &search::EvalWeights::monsterDamageWeight)
        .def_readwrite("alive_weight", &search::EvalWeights::aliveWeight)
        .def_readwrite("energy_waste_weight", &search::EvalWeights::energyWasteWeight)
        .def_readwrite("draw_weight", &search::EvalWeights::drawWeight)
        .def_readwrite("turn_survival_weight", &search::EvalWeights::turnSurvivalWeight)
        .def_readwrite("gold_loss_weight", &search::EvalWeights::goldLossWeight, "penalty per gold permanently lost to an escaped Looter/Mugger")
        .def_readwrite("max_hp_weight", &search::EvalWeights::maxHpWeight, "bonus per max HP gained vs the search root (Feed, Darkstone)")
        .def_readwrite("parasite_penalty", &search::EvalWeights::parasitePenalty, "flat penalty when Writhing Mass's implant will add a Parasite");

    pybind11::class_<search::BattleSnapshot>(m, "BattleSnapshot")
        .def_readonly("floor", &search::BattleSnapshot::floor)
        .def_readonly("act", &search::BattleSnapshot::act)
        .def_readonly("cur_hp", &search::BattleSnapshot::curHp)
        .def_readonly("max_hp", &search::BattleSnapshot::maxHp)
        .def_readonly("potion_count", &search::BattleSnapshot::potionCount)
        .def_readonly("deck_size", &search::BattleSnapshot::deckSize)
        .def_readonly("encounter", &search::BattleSnapshot::encounter);

    pybind11::class_<search::SearchAgent> agent(m, "Agent");
    agent.def(pybind11::init<>());
    agent.def_readwrite("simulation_count_base", &search::SearchAgent::simulationCountBase, "number of simulations the agent uses for monte carlo tree search each turn")
        .def_readwrite("boss_simulation_multiplier", &search::SearchAgent::bossSimulationMultiplier, "bonus multiplier to the simulation count for boss fights")
        .def_readwrite("pause_on_card_reward", &search::SearchAgent::pauseOnCardReward, "causes the agent to pause so as to cede control to the user when it encounters a card reward choice")
        .def_readwrite("verbosity_level", &search::SearchAgent::verbosityLevel, "verbosity level: 0=quiet, 1=concise, 2=full")
        .def_readwrite("exploration_parameter", &search::SearchAgent::explorationParameter, "MCTS UCB exploration constant used per battle")
        .def_readwrite("exploration_parameter_chance", &search::SearchAgent::explorationParameterChance, "MCTS UCB exploration constant for stochastic edges (chance-node children)")
        .def_readwrite("chance_widening_c", &search::SearchAgent::chanceWideningC, "double progressive widening C for chance nodes")
        .def_readwrite("chance_widening_alpha", &search::SearchAgent::chanceWideningAlpha, "double progressive widening alpha for chance nodes")
        .def_readwrite("boss_chance_widening_c", &search::SearchAgent::bossChanceWideningC, "double progressive widening C for chance nodes in boss fights")
        .def_readwrite("boss_chance_widening_alpha", &search::SearchAgent::bossChanceWideningAlpha, "double progressive widening alpha for chance nodes in boss fights")
        .def_readwrite("end_turn_widening_c", &search::SearchAgent::endTurnWideningC, "double progressive widening C for END_TURN chance nodes")
        .def_readwrite("end_turn_widening_alpha", &search::SearchAgent::endTurnWideningAlpha, "double progressive widening alpha for END_TURN chance nodes")
        .def_readwrite("eval_weights", &search::SearchAgent::evalWeights, "evaluateEndState weights backed up by the battle search")
        .def_readwrite("log_battle_outcomes", &search::SearchAgent::logBattleOutcomes, "record a BattleSnapshot after each battle into battle_log")
        .def_readonly("battle_log", &search::SearchAgent::battleLog, "post-battle snapshots (floor/act/hp/potions/deck/encounter), one per battle")
        .def_readwrite("record_actions", &search::SearchAgent::recordActions, "record taken actions into game_action_history")
        .def_readwrite("game_action_history", &search::SearchAgent::gameActionHistory, "bits of actions taken (in-battle search::Actions via playout_battle)")
        .def("pick_gameaction", &search::SearchAgent::pickOutOfCombatAction)
        .def("playout_battle", [](search::SearchAgent &agent, GameContext &gc,
                                  std::optional<MonsterEncounter> encounter) {
            pybind11::gil_scoped_release release;
            BattleContext bc;
            // None = use the encounter the game rolled (gc.info.encounter); otherwise
            // force the given encounter against this state.
            if (encounter.has_value() && *encounter != MonsterEncounter::INVALID) {
                bc.init(gc, *encounter);
            } else {
                bc.init(gc);
            }

            const auto battleEncounter = bc.encounter;
            agent.playoutBattle(bc);
            bc.exitBattle(gc);
            // Mirror SearchAgent::playout's per-battle snapshot so python-driven games
            // (run_episode) populate battle_log too.
            if (agent.logBattleOutcomes) {
                agent.battleLog.push_back({gc.floorNum, gc.act, gc.curHp, gc.maxHp, gc.potionCount,
                                           static_cast<int>(gc.deck.size()),
                                           static_cast<int>(battleEncounter)});
            }
        }, "gc"_a, "encounter"_a = pybind11::none(),
           "playout a battle; optionally force a specific encounter instead of the rolled one")
        .def("configure_searcher", &search::SearchAgent::configureSearcher, "searcher"_a, "bc"_a,
             "apply this agent's tuned battle-search knobs to a BattleSearcher and return the "
             "matching per-decision simulation count (for single-step search, e.g. the comm bridge)")
        .def("playout", &search::SearchAgent::playout);

    // ActionType enum binding
    pybind11::enum_<search::ActionType>(m, "ActionType")
        .value("CARD", search::ActionType::CARD)
        .value("POTION", search::ActionType::POTION)
        .value("SINGLE_CARD_SELECT", search::ActionType::SINGLE_CARD_SELECT)
        .value("MULTI_CARD_SELECT", search::ActionType::MULTI_CARD_SELECT)
        .value("END_TURN", search::ActionType::END_TURN);

    // InputState enum binding  
    pybind11::enum_<InputState>(m, "InputState")
        .value("EXECUTING_ACTIONS", InputState::EXECUTING_ACTIONS)
        .value("PLAYER_NORMAL", InputState::PLAYER_NORMAL)
        .value("CARD_SELECT", InputState::CARD_SELECT)
        .value("CHOOSE_STANCE_ACTION", InputState::CHOOSE_STANCE_ACTION);

    // MonsterMoveId enum binding (complete enum)
    pybind11::enum_<MonsterMoveId>(m, "MonsterMoveId")
        .value("INVALID", MonsterMoveId::INVALID)
        .value("GENERIC_ESCAPE_MOVE", MonsterMoveId::GENERIC_ESCAPE_MOVE)
        .value("ACID_SLIME_L_CORROSIVE_SPIT", MonsterMoveId::ACID_SLIME_L_CORROSIVE_SPIT)
        .value("ACID_SLIME_L_LICK", MonsterMoveId::ACID_SLIME_L_LICK)
        .value("ACID_SLIME_L_TACKLE", MonsterMoveId::ACID_SLIME_L_TACKLE)
        .value("ACID_SLIME_L_SPLIT", MonsterMoveId::ACID_SLIME_L_SPLIT)
        .value("ACID_SLIME_M_CORROSIVE_SPIT", MonsterMoveId::ACID_SLIME_M_CORROSIVE_SPIT)
        .value("ACID_SLIME_M_LICK", MonsterMoveId::ACID_SLIME_M_LICK)
        .value("ACID_SLIME_M_TACKLE", MonsterMoveId::ACID_SLIME_M_TACKLE)
        .value("ACID_SLIME_S_LICK", MonsterMoveId::ACID_SLIME_S_LICK)
        .value("ACID_SLIME_S_TACKLE", MonsterMoveId::ACID_SLIME_S_TACKLE)
        .value("AWAKENED_ONE_SLASH", MonsterMoveId::AWAKENED_ONE_SLASH)
        .value("AWAKENED_ONE_SOUL_STRIKE", MonsterMoveId::AWAKENED_ONE_SOUL_STRIKE)
        .value("AWAKENED_ONE_REBIRTH", MonsterMoveId::AWAKENED_ONE_REBIRTH)
        .value("AWAKENED_ONE_DARK_ECHO", MonsterMoveId::AWAKENED_ONE_DARK_ECHO)
        .value("AWAKENED_ONE_SLUDGE", MonsterMoveId::AWAKENED_ONE_SLUDGE)
        .value("AWAKENED_ONE_TACKLE", MonsterMoveId::AWAKENED_ONE_TACKLE)
        .value("BEAR_BEAR_HUG", MonsterMoveId::BEAR_BEAR_HUG)
        .value("BEAR_LUNGE", MonsterMoveId::BEAR_LUNGE)
        .value("BEAR_MAUL", MonsterMoveId::BEAR_MAUL)
        .value("BLUE_SLAVER_STAB", MonsterMoveId::BLUE_SLAVER_STAB)
        .value("BLUE_SLAVER_RAKE", MonsterMoveId::BLUE_SLAVER_RAKE)
        .value("BOOK_OF_STABBING_MULTI_STAB", MonsterMoveId::BOOK_OF_STABBING_MULTI_STAB)
        .value("BOOK_OF_STABBING_SINGLE_STAB", MonsterMoveId::BOOK_OF_STABBING_SINGLE_STAB)
        .value("BRONZE_AUTOMATON_BOOST", MonsterMoveId::BRONZE_AUTOMATON_BOOST)
        .value("BRONZE_AUTOMATON_FLAIL", MonsterMoveId::BRONZE_AUTOMATON_FLAIL)
        .value("BRONZE_AUTOMATON_HYPER_BEAM", MonsterMoveId::BRONZE_AUTOMATON_HYPER_BEAM)
        .value("BRONZE_AUTOMATON_SPAWN_ORBS", MonsterMoveId::BRONZE_AUTOMATON_SPAWN_ORBS)
        .value("BRONZE_AUTOMATON_STUNNED", MonsterMoveId::BRONZE_AUTOMATON_STUNNED)
        .value("BRONZE_ORB_BEAM", MonsterMoveId::BRONZE_ORB_BEAM)
        .value("BRONZE_ORB_STASIS", MonsterMoveId::BRONZE_ORB_STASIS)
        .value("BRONZE_ORB_SUPPORT_BEAM", MonsterMoveId::BRONZE_ORB_SUPPORT_BEAM)
        .value("BYRD_CAW", MonsterMoveId::BYRD_CAW)
        .value("BYRD_FLY", MonsterMoveId::BYRD_FLY)
        .value("BYRD_HEADBUTT", MonsterMoveId::BYRD_HEADBUTT)
        .value("BYRD_PECK", MonsterMoveId::BYRD_PECK)
        .value("BYRD_STUNNED", MonsterMoveId::BYRD_STUNNED)
        .value("BYRD_SWOOP", MonsterMoveId::BYRD_SWOOP)
        .value("CENTURION_SLASH", MonsterMoveId::CENTURION_SLASH)
        .value("CENTURION_FURY", MonsterMoveId::CENTURION_FURY)
        .value("CENTURION_DEFEND", MonsterMoveId::CENTURION_DEFEND)
        .value("CHOSEN_POKE", MonsterMoveId::CHOSEN_POKE)
        .value("CHOSEN_ZAP", MonsterMoveId::CHOSEN_ZAP)
        .value("CHOSEN_DEBILITATE", MonsterMoveId::CHOSEN_DEBILITATE)
        .value("CHOSEN_DRAIN", MonsterMoveId::CHOSEN_DRAIN)
        .value("CHOSEN_HEX", MonsterMoveId::CHOSEN_HEX)
        .value("CORRUPT_HEART_DEBILITATE", MonsterMoveId::CORRUPT_HEART_DEBILITATE)
        .value("CORRUPT_HEART_BLOOD_SHOTS", MonsterMoveId::CORRUPT_HEART_BLOOD_SHOTS)
        .value("CORRUPT_HEART_ECHO", MonsterMoveId::CORRUPT_HEART_ECHO)
        .value("CORRUPT_HEART_BUFF", MonsterMoveId::CORRUPT_HEART_BUFF)
        .value("CULTIST_INCANTATION", MonsterMoveId::CULTIST_INCANTATION)
        .value("CULTIST_DARK_STRIKE", MonsterMoveId::CULTIST_DARK_STRIKE)
        .value("DAGGER_STAB", MonsterMoveId::DAGGER_STAB)
        .value("DAGGER_EXPLODE", MonsterMoveId::DAGGER_EXPLODE)
        .value("DARKLING_NIP", MonsterMoveId::DARKLING_NIP)
        .value("DARKLING_CHOMP", MonsterMoveId::DARKLING_CHOMP)
        .value("DARKLING_HARDEN", MonsterMoveId::DARKLING_HARDEN)
        .value("DARKLING_REINCARNATE", MonsterMoveId::DARKLING_REINCARNATE)
        .value("DARKLING_REGROW", MonsterMoveId::DARKLING_REGROW)
        .value("DECA_SQUARE_OF_PROTECTION", MonsterMoveId::DECA_SQUARE_OF_PROTECTION)
        .value("DECA_BEAM", MonsterMoveId::DECA_BEAM)
        .value("DONU_CIRCLE_OF_POWER", MonsterMoveId::DONU_CIRCLE_OF_POWER)
        .value("DONU_BEAM", MonsterMoveId::DONU_BEAM)
        .value("EXPLODER_SLAM", MonsterMoveId::EXPLODER_SLAM)
        .value("EXPLODER_EXPLODE", MonsterMoveId::EXPLODER_EXPLODE)
        .value("FAT_GREMLIN_SMASH", MonsterMoveId::FAT_GREMLIN_SMASH)
        .value("FUNGI_BEAST_BITE", MonsterMoveId::FUNGI_BEAST_BITE)
        .value("FUNGI_BEAST_GROW", MonsterMoveId::FUNGI_BEAST_GROW)
        .value("GIANT_HEAD_COUNT", MonsterMoveId::GIANT_HEAD_COUNT)
        .value("GIANT_HEAD_GLARE", MonsterMoveId::GIANT_HEAD_GLARE)
        .value("GIANT_HEAD_IT_IS_TIME", MonsterMoveId::GIANT_HEAD_IT_IS_TIME)
        .value("GREEN_LOUSE_BITE", MonsterMoveId::GREEN_LOUSE_BITE)
        .value("GREEN_LOUSE_SPIT_WEB", MonsterMoveId::GREEN_LOUSE_SPIT_WEB)
        .value("GREMLIN_LEADER_ENCOURAGE", MonsterMoveId::GREMLIN_LEADER_ENCOURAGE)
        .value("GREMLIN_LEADER_RALLY", MonsterMoveId::GREMLIN_LEADER_RALLY)
        .value("GREMLIN_LEADER_STAB", MonsterMoveId::GREMLIN_LEADER_STAB)
        .value("GREMLIN_NOB_BELLOW", MonsterMoveId::GREMLIN_NOB_BELLOW)
        .value("GREMLIN_NOB_RUSH", MonsterMoveId::GREMLIN_NOB_RUSH)
        .value("GREMLIN_NOB_SKULL_BASH", MonsterMoveId::GREMLIN_NOB_SKULL_BASH)
        .value("GREMLIN_WIZARD_CHARGING", MonsterMoveId::GREMLIN_WIZARD_CHARGING)
        .value("GREMLIN_WIZARD_ULTIMATE_BLAST", MonsterMoveId::GREMLIN_WIZARD_ULTIMATE_BLAST)
        .value("HEXAGHOST_ACTIVATE", MonsterMoveId::HEXAGHOST_ACTIVATE)
        .value("HEXAGHOST_DIVIDER", MonsterMoveId::HEXAGHOST_DIVIDER)
        .value("HEXAGHOST_INFERNO", MonsterMoveId::HEXAGHOST_INFERNO)
        .value("HEXAGHOST_SEAR", MonsterMoveId::HEXAGHOST_SEAR)
        .value("HEXAGHOST_TACKLE", MonsterMoveId::HEXAGHOST_TACKLE)
        .value("HEXAGHOST_INFLAME", MonsterMoveId::HEXAGHOST_INFLAME)
        .value("JAW_WORM_CHOMP", MonsterMoveId::JAW_WORM_CHOMP)
        .value("JAW_WORM_THRASH", MonsterMoveId::JAW_WORM_THRASH)
        .value("JAW_WORM_BELLOW", MonsterMoveId::JAW_WORM_BELLOW)
        .value("LAGAVULIN_ATTACK", MonsterMoveId::LAGAVULIN_ATTACK)
        .value("LAGAVULIN_SIPHON_SOUL", MonsterMoveId::LAGAVULIN_SIPHON_SOUL)
        .value("LAGAVULIN_SLEEP", MonsterMoveId::LAGAVULIN_SLEEP)
        .value("LOOTER_MUG", MonsterMoveId::LOOTER_MUG)
        .value("LOOTER_LUNGE", MonsterMoveId::LOOTER_LUNGE)
        .value("LOOTER_SMOKE_BOMB", MonsterMoveId::LOOTER_SMOKE_BOMB)
        .value("LOOTER_ESCAPE", MonsterMoveId::LOOTER_ESCAPE)
        .value("MAD_GREMLIN_SCRATCH", MonsterMoveId::MAD_GREMLIN_SCRATCH)
        .value("MUGGER_MUG", MonsterMoveId::MUGGER_MUG)
        .value("MUGGER_LUNGE", MonsterMoveId::MUGGER_LUNGE)
        .value("MUGGER_SMOKE_BOMB", MonsterMoveId::MUGGER_SMOKE_BOMB)
        .value("MUGGER_ESCAPE", MonsterMoveId::MUGGER_ESCAPE)
        .value("MYSTIC_HEAL", MonsterMoveId::MYSTIC_HEAL)
        .value("MYSTIC_BUFF", MonsterMoveId::MYSTIC_BUFF)
        .value("MYSTIC_ATTACK_DEBUFF", MonsterMoveId::MYSTIC_ATTACK_DEBUFF)
        .value("NEMESIS_DEBUFF", MonsterMoveId::NEMESIS_DEBUFF)
        .value("NEMESIS_ATTACK", MonsterMoveId::NEMESIS_ATTACK)
        .value("NEMESIS_SCYTHE", MonsterMoveId::NEMESIS_SCYTHE)
        .value("ORB_WALKER_LASER", MonsterMoveId::ORB_WALKER_LASER)
        .value("ORB_WALKER_CLAW", MonsterMoveId::ORB_WALKER_CLAW)
        .value("POINTY_ATTACK", MonsterMoveId::POINTY_ATTACK)
        .value("RED_LOUSE_BITE", MonsterMoveId::RED_LOUSE_BITE)
        .value("RED_LOUSE_GROW", MonsterMoveId::RED_LOUSE_GROW)
        .value("RED_SLAVER_STAB", MonsterMoveId::RED_SLAVER_STAB)
        .value("RED_SLAVER_SCRAPE", MonsterMoveId::RED_SLAVER_SCRAPE)
        .value("RED_SLAVER_ENTANGLE", MonsterMoveId::RED_SLAVER_ENTANGLE)
        .value("REPTOMANCER_SUMMON", MonsterMoveId::REPTOMANCER_SUMMON)
        .value("REPTOMANCER_SNAKE_STRIKE", MonsterMoveId::REPTOMANCER_SNAKE_STRIKE)
        .value("REPTOMANCER_BIG_BITE", MonsterMoveId::REPTOMANCER_BIG_BITE)
        .value("REPULSOR_BASH", MonsterMoveId::REPULSOR_BASH)
        .value("REPULSOR_REPULSE", MonsterMoveId::REPULSOR_REPULSE)
        .value("ROMEO_MOCK", MonsterMoveId::ROMEO_MOCK)
        .value("ROMEO_AGONIZING_SLASH", MonsterMoveId::ROMEO_AGONIZING_SLASH)
        .value("ROMEO_CROSS_SLASH", MonsterMoveId::ROMEO_CROSS_SLASH)
        .value("SENTRY_BEAM", MonsterMoveId::SENTRY_BEAM)
        .value("SENTRY_BOLT", MonsterMoveId::SENTRY_BOLT)
        .value("SHELLED_PARASITE_DOUBLE_STRIKE", MonsterMoveId::SHELLED_PARASITE_DOUBLE_STRIKE)
        .value("SHELLED_PARASITE_FELL", MonsterMoveId::SHELLED_PARASITE_FELL)
        .value("SHELLED_PARASITE_STUNNED", MonsterMoveId::SHELLED_PARASITE_STUNNED)
        .value("SHELLED_PARASITE_SUCK", MonsterMoveId::SHELLED_PARASITE_SUCK)
        .value("SHIELD_GREMLIN_PROTECT", MonsterMoveId::SHIELD_GREMLIN_PROTECT)
        .value("SHIELD_GREMLIN_SHIELD_BASH", MonsterMoveId::SHIELD_GREMLIN_SHIELD_BASH)
        .value("SLIME_BOSS_GOOP_SPRAY", MonsterMoveId::SLIME_BOSS_GOOP_SPRAY)
        .value("SLIME_BOSS_PREPARING", MonsterMoveId::SLIME_BOSS_PREPARING)
        .value("SLIME_BOSS_SLAM", MonsterMoveId::SLIME_BOSS_SLAM)
        .value("SLIME_BOSS_SPLIT", MonsterMoveId::SLIME_BOSS_SPLIT)
        .value("SNAKE_PLANT_CHOMP", MonsterMoveId::SNAKE_PLANT_CHOMP)
        .value("SNAKE_PLANT_ENFEEBLING_SPORES", MonsterMoveId::SNAKE_PLANT_ENFEEBLING_SPORES)
        .value("SNEAKY_GREMLIN_PUNCTURE", MonsterMoveId::SNEAKY_GREMLIN_PUNCTURE)
        .value("SNECKO_PERPLEXING_GLARE", MonsterMoveId::SNECKO_PERPLEXING_GLARE)
        .value("SNECKO_TAIL_WHIP", MonsterMoveId::SNECKO_TAIL_WHIP)
        .value("SNECKO_BITE", MonsterMoveId::SNECKO_BITE)
        .value("SPHERIC_GUARDIAN_SLAM", MonsterMoveId::SPHERIC_GUARDIAN_SLAM)
        .value("SPHERIC_GUARDIAN_ACTIVATE", MonsterMoveId::SPHERIC_GUARDIAN_ACTIVATE)
        .value("SPHERIC_GUARDIAN_HARDEN", MonsterMoveId::SPHERIC_GUARDIAN_HARDEN)
        .value("SPHERIC_GUARDIAN_ATTACK_DEBUFF", MonsterMoveId::SPHERIC_GUARDIAN_ATTACK_DEBUFF)
        .value("SPIKER_CUT", MonsterMoveId::SPIKER_CUT)
        .value("SPIKER_SPIKE", MonsterMoveId::SPIKER_SPIKE)
        .value("SPIKE_SLIME_L_FLAME_TACKLE", MonsterMoveId::SPIKE_SLIME_L_FLAME_TACKLE)
        .value("SPIKE_SLIME_L_LICK", MonsterMoveId::SPIKE_SLIME_L_LICK)
        .value("SPIKE_SLIME_L_SPLIT", MonsterMoveId::SPIKE_SLIME_L_SPLIT)
        .value("SPIKE_SLIME_M_FLAME_TACKLE", MonsterMoveId::SPIKE_SLIME_M_FLAME_TACKLE)
        .value("SPIKE_SLIME_M_LICK", MonsterMoveId::SPIKE_SLIME_M_LICK)
        .value("SPIKE_SLIME_S_TACKLE", MonsterMoveId::SPIKE_SLIME_S_TACKLE)
        .value("SPIRE_GROWTH_QUICK_TACKLE", MonsterMoveId::SPIRE_GROWTH_QUICK_TACKLE)
        .value("SPIRE_GROWTH_SMASH", MonsterMoveId::SPIRE_GROWTH_SMASH)
        .value("SPIRE_GROWTH_CONSTRICT", MonsterMoveId::SPIRE_GROWTH_CONSTRICT)
        .value("SPIRE_SHIELD_BASH", MonsterMoveId::SPIRE_SHIELD_BASH)
        .value("SPIRE_SHIELD_FORTIFY", MonsterMoveId::SPIRE_SHIELD_FORTIFY)
        .value("SPIRE_SHIELD_SMASH", MonsterMoveId::SPIRE_SHIELD_SMASH)
        .value("SPIRE_SPEAR_BURN_STRIKE", MonsterMoveId::SPIRE_SPEAR_BURN_STRIKE)
        .value("SPIRE_SPEAR_PIERCER", MonsterMoveId::SPIRE_SPEAR_PIERCER)
        .value("SPIRE_SPEAR_SKEWER", MonsterMoveId::SPIRE_SPEAR_SKEWER)
        .value("TASKMASTER_SCOURING_WHIP", MonsterMoveId::TASKMASTER_SCOURING_WHIP)
        .value("TORCH_HEAD_TACKLE", MonsterMoveId::TORCH_HEAD_TACKLE)
        .value("THE_CHAMP_DEFENSIVE_STANCE", MonsterMoveId::THE_CHAMP_DEFENSIVE_STANCE)
        .value("THE_CHAMP_FACE_SLAP", MonsterMoveId::THE_CHAMP_FACE_SLAP)
        .value("THE_CHAMP_TAUNT", MonsterMoveId::THE_CHAMP_TAUNT)
        .value("THE_CHAMP_HEAVY_SLASH", MonsterMoveId::THE_CHAMP_HEAVY_SLASH)
        .value("THE_CHAMP_GLOAT", MonsterMoveId::THE_CHAMP_GLOAT)
        .value("THE_CHAMP_EXECUTE", MonsterMoveId::THE_CHAMP_EXECUTE)
        .value("THE_CHAMP_ANGER", MonsterMoveId::THE_CHAMP_ANGER)
        .value("THE_COLLECTOR_BUFF", MonsterMoveId::THE_COLLECTOR_BUFF)
        .value("THE_COLLECTOR_FIREBALL", MonsterMoveId::THE_COLLECTOR_FIREBALL)
        .value("THE_COLLECTOR_MEGA_DEBUFF", MonsterMoveId::THE_COLLECTOR_MEGA_DEBUFF)
        .value("THE_COLLECTOR_SPAWN", MonsterMoveId::THE_COLLECTOR_SPAWN)
        .value("THE_GUARDIAN_CHARGING_UP", MonsterMoveId::THE_GUARDIAN_CHARGING_UP)
        .value("THE_GUARDIAN_FIERCE_BASH", MonsterMoveId::THE_GUARDIAN_FIERCE_BASH)
        .value("THE_GUARDIAN_VENT_STEAM", MonsterMoveId::THE_GUARDIAN_VENT_STEAM)
        .value("THE_GUARDIAN_WHIRLWIND", MonsterMoveId::THE_GUARDIAN_WHIRLWIND)
        .value("THE_GUARDIAN_DEFENSIVE_MODE", MonsterMoveId::THE_GUARDIAN_DEFENSIVE_MODE)
        .value("THE_GUARDIAN_ROLL_ATTACK", MonsterMoveId::THE_GUARDIAN_ROLL_ATTACK)
        .value("THE_GUARDIAN_TWIN_SLAM", MonsterMoveId::THE_GUARDIAN_TWIN_SLAM)
        .value("THE_MAW_ROAR", MonsterMoveId::THE_MAW_ROAR)
        .value("THE_MAW_DROOL", MonsterMoveId::THE_MAW_DROOL)
        .value("THE_MAW_SLAM", MonsterMoveId::THE_MAW_SLAM)
        .value("THE_MAW_NOM", MonsterMoveId::THE_MAW_NOM)
        .value("TIME_EATER_REVERBERATE", MonsterMoveId::TIME_EATER_REVERBERATE)
        .value("TIME_EATER_HEAD_SLAM", MonsterMoveId::TIME_EATER_HEAD_SLAM)
        .value("TIME_EATER_RIPPLE", MonsterMoveId::TIME_EATER_RIPPLE)
        .value("TIME_EATER_HASTE", MonsterMoveId::TIME_EATER_HASTE)
        .value("TRANSIENT_ATTACK", MonsterMoveId::TRANSIENT_ATTACK)
        .value("WRITHING_MASS_IMPLANT", MonsterMoveId::WRITHING_MASS_IMPLANT)
        .value("WRITHING_MASS_FLAIL", MonsterMoveId::WRITHING_MASS_FLAIL)
        .value("WRITHING_MASS_WITHER", MonsterMoveId::WRITHING_MASS_WITHER)
        .value("WRITHING_MASS_MULTI_STRIKE", MonsterMoveId::WRITHING_MASS_MULTI_STRIKE)
        .value("WRITHING_MASS_STRONG_STRIKE", MonsterMoveId::WRITHING_MASS_STRONG_STRIKE);

    // Action class binding
    pybind11::class_<search::Action> action(m, "Action");
    action.def(pybind11::init<>())
        .def(pybind11::init<std::uint32_t>())
        .def(pybind11::init<search::ActionType>())
        .def(pybind11::init<search::ActionType, int>())
        .def(pybind11::init<search::ActionType, int, int>())
        .def("get_action_type", &search::Action::getActionType)
        .def("get_source_idx", &search::Action::getSourceIdx)
        .def("get_target_idx", &search::Action::getTargetIdx)
        .def("get_select_idx", &search::Action::getSelectIdx)
        .def("get_selected_idxs", [](const search::Action &a) {
            const auto sel = a.getSelectedIdxs();
            return std::vector<int>(sel.begin(), sel.end());
        })
        .def("is_valid_action", &search::Action::isValidAction)
        .def("print_desc", [](const search::Action &action, const BattleContext &bc) {
            std::ostringstream oss;
            action.printDesc(oss, bc);
            return oss.str();
        })
        .def("execute", &search::Action::execute);

    // BattleSearcher Node and Edge bindings
    pybind11::class_<search::BattleSearcher::Node>(m, "BattleSearcherNode")
        .def_readonly("simulation_count", &search::BattleSearcher::Node::simulationCount)
        .def_readonly("evaluation_sum", &search::BattleSearcher::Node::evaluationSum);

    pybind11::class_<search::BattleSearcher::Edge>(m, "BattleSearcherEdge")
        .def_readonly("action", &search::BattleSearcher::Edge::action)
        .def_readonly("node", &search::BattleSearcher::Edge::node);

    // BattleSearcher class binding
    pybind11::class_<search::BattleSearcher> battleSearcher(m, "BattleSearcher");
    battleSearcher.def(pybind11::init<const BattleContext&>())
        .def(pybind11::init<const BattleContext&, search::EvalFnc>())
        .def("search", &search::BattleSearcher::search)
        .def("step", &search::BattleSearcher::step)
        .def("get_best_action", &search::BattleSearcher::getBestAction)
        .def("get_root_edges", &search::BattleSearcher::getRootEdges, pybind11::return_value_policy::reference_internal)
        .def_readwrite("exploration_parameter", &search::BattleSearcher::explorationParameter);

    pybind11::class_<GameContext> gameContext(m, "GameContext");
    gameContext.def(pybind11::init<CharacterClass, std::int64_t, int, bool>(),
            pybind11::arg("character_class"), pybind11::arg("seed"),
            pybind11::arg("ascension"), pybind11::arg("neow_mini_blessing") = false)
        .def("get_card_reward", &sts::py::getCardReward, "return the current card reward list")
        .def_property_readonly("encounter", [](const GameContext &gc) { return gc.info.encounter; })
        .def_property_readonly("deck",
               [](const GameContext &gc) { return std::vector(gc.deck.cards.begin(), gc.deck.cards.end());},
               "returns a copy of the list of cards in the deck"
        )
        .def("obtain_card",
             [](GameContext &gc, Card card) { gc.deck.obtain(gc, card); },
             "add a card to the deck"
        )
        .def("obtain_relic", &GameContext::obtainRelic, "add a relic to the player")
        .def("set_relic_value", [](GameContext &gc, RelicId relic, int value) {
            if (gc.relics.has(relic)) {
                gc.relics.getRelicValueRef(relic) = value;
            }
        }, "relic"_a, "value"_a,
           "set a held relic's stored value/counter to match the live game (e.g. Girya's lift "
           "count, which gates whether LIFT is offered at a campfire)")
        .def("setup_event", &GameContext::setupEvent,
             "initialize the current event (gc.cur_event) into EVENT_SCREEN state, populating the "
             "event-specific info fields getValidEventSelectBits/the NN reads; consumes RNG for "
             "events with randomized setup")
        .def("remove_card",
            [](GameContext &gc, int idx) {
                if (idx < 0 || idx >= gc.deck.size()) {
                    std::cerr << "invalid remove deck remove idx" << std::endl;
                    return;
                }
                gc.deck.remove(gc, idx);
            },
             "remove a card at a idx in the deck"
        )
        .def("obtain_potion", &GameContext::obtainPotion, "add a potion to the player")
        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        .def_readwrite("screen_state_info", &GameContext::info)
        .def("__repr__", [](const GameContext &gc) {
            std::ostringstream oss;
            oss << "<" << gc << ">";
            return oss.str();
        }, "returns a string representation of the GameContext");

    gameContext
        .def_readwrite("act", &GameContext::act)
        .def_readwrite("floor_num", &GameContext::floorNum)
        .def_readwrite("ascension", &GameContext::ascension)
        .def_readwrite("skip_battles", &GameContext::skipBattles)

        .def_readwrite("seed", &GameContext::seed)
        .def_readwrite("map", &GameContext::map)
        .def_readwrite("cur_map_node_x", &GameContext::curMapNodeX)
        .def_readwrite("cur_map_node_y", &GameContext::curMapNodeY)

        .def_readwrite("cur_hp", &GameContext::curHp)
        .def_readwrite("max_hp", &GameContext::maxHp)
        .def_readwrite("gold", &GameContext::gold)

        .def_readwrite("blue_key", &GameContext::blueKey)
        .def_readwrite("green_key", &GameContext::greenKey)
        .def_readwrite("red_key", &GameContext::redKey)

        .def_readwrite("card_rarity_factor", &GameContext::cardRarityFactor)
        .def_readwrite("potion_chance", &GameContext::potionChance)
        .def_readwrite("monster_chance", &GameContext::monsterChance)
        .def_readwrite("shop_chance", &GameContext::shopChance)
        .def_readwrite("treasure_chance", &GameContext::treasureChance)

        .def_readwrite("shop_remove_count", &GameContext::shopRemoveCount)
        .def_readwrite("speedrun_pace", &GameContext::speedrunPace)
        .def_readwrite("note_for_yourself_card", &GameContext::noteForYourselfCard)
        .def_readwrite("potion_capacity", &GameContext::potionCapacity)
        .def("create_battle_context", [](GameContext &gc) -> BattleContext* {
            BattleContext *bc = new BattleContext();
            bc->init(gc);
            return bc;
        }, pybind11::return_value_policy::take_ownership, "create a new BattleContext initialized from this GameContext")
        .def("empty_battle_context", [](GameContext &gc) -> BattleContext* {
            BattleContext *bc = new BattleContext();
            bc->init_empty(gc);
            return bc;
        }, pybind11::return_value_policy::take_ownership, "create an empty BattleContext initialized from this GameContext")
        .def("sync_from_battle_context", [](GameContext &gc, BattleContext &bc) {
            bc.exitBattle(gc);
        }, "sync changes from BattleContext back to GameContext")
        .def("clear_deck", [](GameContext &gc) {
            gc.deck.cards.clear();
        }, "clear all cards from the deck")
        .def("copy", [](const GameContext &gc) { return GameContext(gc); },
             "value copy for snapshot/mutate/simulate workflows. The Map is shared (shared_ptr), "
             "not deep-copied -- fine for battle sims, which never mutate the map. Vary `seed` on "
             "the copy to reroll battle randomness (battle + search rng derive from seed+floor).");

    // Enum-typed fields bound by value (snapshot) to avoid reference_internal aliasing.
    def_value(gameContext, "outcome", &GameContext::outcome);
    def_value(gameContext, "screen_state", &GameContext::screenState);
    def_value(gameContext, "cur_room", &GameContext::curRoom);
    def_value(gameContext, "cur_event", &GameContext::curEvent);
    def_value(gameContext, "boss", &GameContext::boss);

    pybind11::class_<GameAction> gameAction(m, "GameAction");
    gameAction.def("getAllActionsInState", &GameAction::getAllActionsInState);
    gameAction.def(pybind11::init<std::uint32_t>());  // from bits
    gameAction.def_readonly("bits", &GameAction::bits);
    gameAction.def_property_readonly("idx1", &GameAction::getIdx1);
    gameAction.def_property_readonly("idx2", &GameAction::getIdx2);
    gameAction.def_property_readonly("idx3", &GameAction::getIdx3);
    gameAction.def("execute", &GameAction::execute);
    gameAction.def("getDesc", [](const GameAction &ga, const GameContext &gc) {
        std::ostringstream oss;
        ga.printDesc(oss, gc);
        return oss.str();
    });
    gameAction.def("isValidAction", [](const GameAction &ga, const GameContext &gc) {
        return ga.isValidAction(gc);
    });
    gameAction.def_property_readonly("rewards_action_type", [](const GameAction &ga) {
        return ga.getRewardsActionType();
    });
    gameAction.def("__repr__", [](const GameAction &ga) {
        std::ostringstream oss;
        oss << "<GameAction " << ga.bits << ">";
        return oss.str();
    });
    gameAction.def("__eq__", [](const GameAction &self, const GameAction &other) {
        return self.bits == other.bits;
    })
    .def("__hash__", [](const GameAction &ga) {
        return std::hash<std::uint32_t>{}(ga.bits);
    });

    pybind11::class_<Rewards> rewards(m, "Rewards");
    rewards.def_property_readonly("gold", [](const Rewards &r) {
        return std::vector<int>(r.gold.begin(), r.gold.begin() + r.goldRewardCount);
    });
    rewards.def_property_readonly("cards", [](const Rewards &r) {
        // filter out invalid
        std::vector<std::vector<Card>> ret;
        for (int i = 0; i < r.cardRewardCount; ++i) {
            const auto &cardReward = r.cardRewards[i];
            std::vector<Card> cards;
            for (int j = 0; j < cardReward.size(); ++j) {
                if (cardReward[j] != CardId::INVALID) {
                    cards.push_back(cardReward[j]);
                }
            }
            ret.push_back(cards);
        }
        return ret;
    });
    rewards.def_property_readonly("relics", [](const Rewards &r) {
        return std::vector<RelicId>(r.relics.begin(), r.relics.begin() + r.relicCount);
    });
    rewards.def_property_readonly("potions", [](const Rewards &r) {
        return std::vector<Potion>(r.potions.begin(), r.potions.begin() + r.potionCount);
    });
    rewards.def_readwrite("emerald_key", &Rewards::emeraldKey);
    rewards.def_readwrite("sapphire_key", &Rewards::sapphireKey);
    // Mutators for injecting an observed reward screen (e.g. from CommunicationMod) so that
    // getAllActionsInState() enumerates the real offered rewards. The read-only `cards`/
    // `relics`/`potions` properties above return copies, so these methods are the only way to
    // populate the container from Python.
    rewards.def("clear", &Rewards::clear, "remove all rewards");
    rewards.def("add_gold", &Rewards::addGold, "add a gold reward of the given amount");
    rewards.def("add_relic", &Rewards::addRelic, "add a relic reward");
    rewards.def("add_potion", &Rewards::addPotion, "add a potion reward");
    rewards.def("add_card_reward", [](Rewards &r, const std::vector<Card> &cards) {
        CardReward reward;
        for (const auto &c : cards) {
            reward.push_back(c);
        }
        r.addCardReward(reward);
    }, "add a card-reward group (list of Cards the player chooses one of)");

    pybind11::class_<ScreenStateInfo> screenStateInfo(m, "ScreenStateInfo");
        screenStateInfo
        .def_property_readonly("boss_relics", [](const ScreenStateInfo &s) {
            return std::vector<RelicId>(s.bossRelics, s.bossRelics+3);
        })
        // The boss_relics getter above returns a copy, so add a setter for injecting the three
        // offered boss relics (idx 0-2) when reconstructing a BOSS_RELIC_REWARDS screen.
        .def("set_boss_relic", [](ScreenStateInfo &s, int idx, RelicId relic) {
            if (idx < 0 || idx >= 3) throw std::out_of_range("boss relic idx must be 0-2");
            s.bossRelics[idx] = relic;
        }, "idx"_a, "relic"_a)
        .def_property_readonly("shop", [](const ScreenStateInfo& info) -> const Shop& {
            return info.shop;
        })
        .def_property_readonly("to_select_cards", [](const ScreenStateInfo& info) {
            std::vector<Card> cards;
            for (const auto& select_card : info.toSelectCards) {
                cards.push_back(select_card.card);
            }
            return cards;
        })
        .def_property_readonly("have_selected_cards", [](const ScreenStateInfo& info) {
            std::vector<Card> cards;
            for (const auto& select_card : info.haveSelectedCards) {
                cards.push_back(select_card.card);
            }
            return cards;
        })
        // Mutators for the card-select screen: the getters above return copies, so reconstructing a
        // live grid/hand-select needs explicit setters. Order is preserved, so to_select_cards[i]
        // lines up with the live screen's card i for translating the net's pick back to a command.
        .def("clear_to_select_cards", [](ScreenStateInfo& info) { info.toSelectCards.clear(); })
        .def("add_to_select_card", [](ScreenStateInfo& info, const Card& c, int deckIdx) {
            info.toSelectCards.push_back(SelectScreenCard(c, deckIdx));
        }, "card"_a, "deck_idx"_a = -1)
        .def("clear_have_selected_cards", [](ScreenStateInfo& info) { info.haveSelectedCards.clear(); })
        .def("add_have_selected_card", [](ScreenStateInfo& info, const Card& c, int deckIdx) {
            info.haveSelectedCards.push_back(SelectScreenCard(c, deckIdx));
        }, "card"_a, "deck_idx"_a = -1)
        .def_property("to_select_count",
            [](const ScreenStateInfo& info) { return info.toSelectCount; },
            [](ScreenStateInfo& info, int n) { info.toSelectCount = n; })
        .def_readwrite("rewards_container", &ScreenStateInfo::rewardsContainer)
        .def_readwrite("event_data", &ScreenStateInfo::eventData)
        .def_readwrite("neowRewards", &ScreenStateInfo::neowRewards)
        .def_readwrite("hpAmount0", &ScreenStateInfo::hpAmount0)
        .def_readwrite("hpAmount1", &ScreenStateInfo::hpAmount1)
        .def_readwrite("hpAmount2", &ScreenStateInfo::hpAmount2)
        .def_readwrite("goldLoss", &ScreenStateInfo::goldLoss)
        .def_readwrite("gold", &ScreenStateInfo::gold)
        .def_readwrite("cardIdx", &ScreenStateInfo::cardIdx)
        .def_readwrite("potionIdx", &ScreenStateInfo::potionIdx)
        .def_readwrite("relicIdx0", &ScreenStateInfo::relicIdx0)
        .def_readwrite("relicIdx1", &ScreenStateInfo::relicIdx1)
        .def_readwrite("skillCardDeckIdx", &ScreenStateInfo::skillCardDeckIdx)
        .def_readwrite("powerCardDeckIdx", &ScreenStateInfo::powerCardDeckIdx)
        .def_readwrite("attackCardDeckIdx", &ScreenStateInfo::attackCardDeckIdx);

    def_value(screenStateInfo, "encounter", &ScreenStateInfo::encounter);
    def_value(screenStateInfo, "select_screen_type", &ScreenStateInfo::selectScreenType);

    pybind11::class_<Shop>(m, "Shop")
        .def_property_readonly("prices", [](const Shop& s) {
            return std::vector<int>(s.prices, s.prices + 13);
        })
        .def_property_readonly("remove_cost", [](const Shop& s) -> std::optional<int> {
            return s.removeCost == -1 ? std::nullopt : std::make_optional(s.removeCost);
        })
        .def_property_readonly("cards", [](const Shop& s) {
            std::vector<Card> cards;
            for (int i = 0; i < 7; ++i) {
                if (s.cards[i] != CardId::INVALID) {
                    cards.push_back(s.cards[i]);
                }
            }
            return cards;
        })
        .def_property_readonly("potions", [](const Shop& s) {
            return std::vector<Potion>(s.potions, s.potions + 3);
        })
        .def_property_readonly("relics", [](const Shop& s) {
            return std::vector<RelicId>(s.relics, s.relics + 3);
        })
        // Mutators: the getters above return copies, so reconstructing a live shop into the
        // ScreenStateInfo needs explicit setters. Prices are laid out cards[0..6], relics[7..9],
        // potions[10..12]; -1 marks an empty/sold slot (matches the engine's own convention).
        .def("clear", [](Shop& s) {
            for (int i = 0; i < 13; ++i) s.prices[i] = -1;
            for (int i = 0; i < 7; ++i) s.cards[i] = Card(CardId::INVALID);
            for (int i = 0; i < 3; ++i) s.relics[i] = RelicId::INVALID;
            for (int i = 0; i < 3; ++i) s.potions[i] = Potion::EMPTY_POTION_SLOT;
            s.removeCost = -1;
        })
        .def("set_card", [](Shop& s, int idx, const Card& c, int price) {
            if (idx < 0 || idx >= 7) throw std::out_of_range("shop card idx out of range [0,7)");
            s.cards[idx] = c;
            s.prices[idx] = price;
        }, "idx"_a, "card"_a, "price"_a)
        .def("set_relic", [](Shop& s, int idx, RelicId relic, int price) {
            if (idx < 0 || idx >= 3) throw std::out_of_range("shop relic idx out of range [0,3)");
            s.relics[idx] = relic;
            s.prices[7 + idx] = price;
        }, "idx"_a, "relic"_a, "price"_a)
        .def("set_potion", [](Shop& s, int idx, Potion potion, int price) {
            if (idx < 0 || idx >= 3) throw std::out_of_range("shop potion idx out of range [0,3)");
            s.potions[idx] = potion;
            s.prices[10 + idx] = price;
        }, "idx"_a, "potion"_a, "price"_a)
        .def("set_remove_cost", [](Shop& s, int cost) { s.removeCost = cost; }, "cost"_a);

    pybind11::class_<RelicInstance> relic(m, "Relic");
    relic.def(pybind11::init<>())
        .def(pybind11::init<RelicId, int>())
        .def_readwrite("data", &RelicInstance::data);
    def_value(relic, "id", &RelicInstance::id);

    pybind11::class_<Map, std::shared_ptr<Map>> map(m, "SpireMap");
    map.def(pybind11::init<std::uint64_t, int,int,bool>());
    map.def_static("act4", &Map::act4Map);
    map.def("get_room_type", &sts::py::getRoomType);
    map.def("has_edge", &sts::py::hasEdge);
    map.def("edges", [](const Map &m, int x, int y) {
        std::vector<int> ret;
        for (int i = 0; i < m.getNode(x,y).edgeCount; ++i) {
            ret.push_back(m.getNode(x,y).edges[i]);
        }
        return ret;
    });
    map.def("get_nn_rep", &sts::py::getNNMapRepresentation);
    map.def("__repr__", [](const Map &m) {
        return m.toString(MonsterEncounter::INVALID);
    });

    pybind11::class_<sts::py::NNCardsRepresentation> nn_cards_rep(m, "NNCardRepresentation");
    nn_cards_rep
        .def_readwrite("cards", &sts::py::NNCardsRepresentation::cards)
        .def_readwrite("upgrades", &sts::py::NNCardsRepresentation::upgrades)
        .def("as_dict", &sts::py::NNCardsRepresentation::as_dict);

    pybind11::class_<sts::py::NNRelicsRepresentation> nn_relics_rep(m, "NNRelicRepresentation");
    nn_relics_rep
        .def_readwrite("relics", &sts::py::NNRelicsRepresentation::relics)
        .def_readwrite("relic_counters", &sts::py::NNRelicsRepresentation::relicCounters)
        .def("as_dict", &sts::py::NNRelicsRepresentation::as_dict);

    pybind11::class_<sts::py::NNMapRepresentation> nn_map_rep(m, "NNMapRepresentation");
    nn_map_rep
        .def_readwrite("xs", &sts::py::NNMapRepresentation::xs)
        .def_readwrite("ys", &sts::py::NNMapRepresentation::ys)
        .def_readwrite("room_types", &sts::py::NNMapRepresentation::roomTypes)
        .def_readwrite("path_xs", &sts::py::NNMapRepresentation::pathXs)
        .def("as_dict", &sts::py::NNMapRepresentation::as_dict);

    pybind11::class_<sts::py::NNRepresentation> nn_rep(m, "NNRepresentation");
    nn_rep
        .def_readwrite("fixed_observation", &sts::py::NNRepresentation::fixedObservation)
        .def_readwrite("deck", &sts::py::NNRepresentation::deck)
        .def_readwrite("relics", &sts::py::NNRepresentation::relics)
        .def_readwrite("potions", &sts::py::NNRepresentation::potions)
        .def_readwrite("map", &sts::py::NNRepresentation::map)
        .def_readwrite("mapX", &sts::py::NNRepresentation::mapX)
        .def_readwrite("mapY", &sts::py::NNRepresentation::mapY)
        .def("as_dict", &sts::py::NNRepresentation::as_dict);

    pybind11::class_<Card> card(m, "Card");
    card.def(pybind11::init<CardId, int>())
        .def("__repr__", [](const Card &c) {
            std::string s("<");
            s += c.getName();
            if (c.isUpgraded()) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW) {
                    s += std::to_string(c.getUpgraded());
                }
            }
            return s += ">";
        }, "returns a string representation of a Card")
        .def("upgrade", &Card::upgrade)
        .def_readwrite("misc", &Card::misc, "value internal to the simulator used for things like ritual dagger damage");

    card.def_property_readonly("id", &Card::getId)
        .def_property_readonly("upgraded", &Card::isUpgraded)
        .def_property_readonly("upgrade_count", &Card::getUpgraded)
        .def_property_readonly("innate", &Card::isInnate)
        .def_property_readonly("transformable", &Card::canTransform)
        .def_property_readonly("upgradable", &Card::canUpgrade)
        .def_property_readonly("is_strikeCard", &Card::isStrikeCard)
        .def_property_readonly("is_starter_strike_or_defend", &Card::isStarterStrikeOrDefend)
        .def_property_readonly("rarity", &Card::getRarity)
        .def_property_readonly("type", &Card::getType);

    // Battle outcome enum (BattleContext::Outcome); named BattleOutcome to avoid clashing with
    // the overworld GameOutcome enum. Lets python drive a battle action-by-action and detect when
    // it ends (e.g. the watch-mode game viewer).
    pybind11::enum_<Outcome>(m, "BattleOutcome")
        .value("UNDECIDED", Outcome::UNDECIDED)
        .value("PLAYER_VICTORY", Outcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", Outcome::PLAYER_LOSS);

    // Battle Context bindings
    pybind11::class_<BattleContext> battleContext(m, "BattleContext");
    battleContext.def_readwrite("turn", &BattleContext::turn)
        .def_readwrite("potionCount", &BattleContext::potionCount)
        .def_readwrite("intents_hidden", &BattleContext::intentsHidden)
        .def_property_readonly("player", [](BattleContext &bc) -> Player& {
            return bc.player; 
        }, pybind11::return_value_policy::reference_internal)
        .def_property_readonly("monsters", [](BattleContext &bc) -> MonsterGroup& { 
            return bc.monsters; 
        }, pybind11::return_value_policy::reference_internal)
        .def_property_readonly("cards", [](BattleContext &bc) -> CardManager& { 
            return bc.cards; 
        }, pybind11::return_value_policy::reference_internal)
        .def("__str__", [](const BattleContext &bc) {
            std::ostringstream oss;
            oss << bc;
            return oss.str();
        })
        .def("copy", [](const BattleContext &bc) { return bc; },
            "deep copy of this BattleContext (value type), so the persistent bc can be advanced "
            "independently of the reconstruction the shadow keeps")
        .def("register_relics_from", &BattleContext::registerRelicsFrom, "gc"_a,
            "copy the player's relic-ownership bits from a GameContext without firing atBattleStart "
            "effects (for reconstructing a mid-combat state)")
        .def("open_card_select", &BattleContext::openSimpleCardSelectScreen, "task"_a, "count"_a,
            "put the bc into the CARD_SELECT input state for the given task (sets cardSelectTask + "
            "pickCount), so the search resolves an in-combat card-select from the reconstructed piles")
        .def("open_discovery_select", [](BattleContext &bc, std::vector<CardId> cards, int copyCount,
                bool setCostToZero) {
            std::array<CardId, 3> arr { CardId::INVALID, CardId::INVALID, CardId::INVALID };
            for (size_t i = 0; i < cards.size() && i < 3; ++i) {
                arr[i] = cards[i];
            }
            bc.openDiscoveryScreen(arr, copyCount, setCostToZero);
        }, "cards"_a, "copy_count"_a = 1, "set_cost_to_zero"_a = true,
            "put the bc into a DISCOVERY card-select over the given generated cards (Discovery / "
            "Attack-Skill-Power Potion / etc.), so the search picks which to add to hand")
        .def("get_card_base_damage", &BattleContext::getCardBaseDamage, "card"_a,
            "engine's base attack damage for a card in this state (Perfect Strike strikeCount bonus, "
            "Body Slam block, etc.); -1 for non-attacks. Compare to the live card's base_damage.")
        .def("get_card_damage_display", &BattleContext::getCardDamageDisplay, "card"_a,
            "engine's in-hand displayed damage for a card (base + player-side modifiers, no target "
            "vulnerable); -1 for non-attacks. Compare to the live card's damage field.")
        .def("open_codex_select", [](BattleContext &bc, std::vector<CardId> cards) {
            // Nilry's Codex: choose 1 of 3 offered cards. Distinct task from DISCOVERY (no
            // cost-to-zero; shuffled into the draw pile, not made free this turn). Inject the live
            // offered cards so the search picks among the real options.
            bc.inputState = InputState::CARD_SELECT;
            bc.cardSelectInfo.cardSelectTask = CardSelectTask::CODEX;
            std::array<CardId, 3> arr { CardId::INVALID, CardId::INVALID, CardId::INVALID };
            for (size_t i = 0; i < cards.size() && i < 3; ++i) {
                arr[i] = cards[i];
            }
            bc.cardSelectInfo.codexCards() = arr;
        }, "cards"_a,
            "put the bc into a CODEX (Nilry's Codex) card-select over the given offered cards");

    def_value(battleContext, "input_state", &BattleContext::inputState);
    def_value(battleContext, "encounter", &BattleContext::encounter);
    def_value(battleContext, "outcome", &BattleContext::outcome);

    // Player bindings
    pybind11::class_<Player> player(m, "Player");
    player.def_readwrite("energy", &Player::energy)
        .def_readwrite("curHp", &Player::curHp)
        .def_readwrite("maxHp", &Player::maxHp)
        .def_readwrite("block", &Player::block)
        .def_readwrite("energyPerTurn", &Player::energyPerTurn)
        .def_readwrite("orbSlots", &Player::orbSlots)
        .def_readwrite("artifact", &Player::artifact)
        .def_readwrite("dexterity", &Player::dexterity)
        .def_readwrite("focus", &Player::focus)
        .def_readwrite("strength", &Player::strength)
        .def_readwrite("gold", &Player::gold)
        .def_readwrite("cardDrawPerTurn", &Player::cardDrawPerTurn)
        .def_readwrite("cardsPlayedThisTurn", &Player::cardsPlayedThisTurn)
        .def_readwrite("attacksPlayedThisTurn", &Player::attacksPlayedThisTurn)
        .def_readwrite("skillsPlayedThisTurn", &Player::skillsPlayedThisTurn)
        .def_readwrite("cardsDiscardedThisTurn", &Player::cardsDiscardedThisTurn)
        // Per-combat relic counters (every-Nth-card/attack/turn relics). Writable so a converted
        // mid-fight state restores the live progress toward the next trigger (e.g. Pen Nib's
        // double-damage attack, Nunchaku's bonus energy) instead of restarting from zero.
        .def_readwrite("happyFlowerCounter", &Player::happyFlowerCounter)
        .def_readwrite("incenseBurnerCounter", &Player::incenseBurnerCounter)
        .def_readwrite("inkBottleCounter", &Player::inkBottleCounter)
        .def_readwrite("nunchakuCounter", &Player::nunchakuCounter)
        .def_readwrite("penNibCounter", &Player::penNibCounter)
        .def_readwrite("sundialCounter", &Player::sundialCounter)
        // The Bomb countdown slots: bombN explodes for that much damage to all enemies N end-of-turns
        // from now (bomb1 fires this end of turn, then slots shift down). Writable so a converted
        // mid-fight state restores in-flight bombs at their correct countdown instead of dropping them.
        .def_readwrite("bomb1", &Player::bomb1)
        .def_readwrite("bomb2", &Player::bomb2)
        .def_readwrite("bomb3", &Player::bomb3)
        .def("hasStatus", [](const Player &p, PlayerStatus s) -> bool {
            return p.hasStatusRuntime(s);
        })
        .def("getStatus", [](const Player &p, PlayerStatus s) -> int {
            return p.getStatusRuntime(s);
        })
        .def("buff", [](Player &p, PlayerStatus s, int amount) {
            p.buff(s, amount);
        }, pybind11::arg("status"), pybind11::arg("amount") = 1)
        .def("debuff", [](Player &p, PlayerStatus s, int amount, bool isSourceMonster) {
            p.debuff(s, amount, isSourceMonster);
        }, pybind11::arg("status"), pybind11::arg("amount"), pybind11::arg("isSourceMonster") = true)
        .def("hasRelic", [](const Player &p, RelicId r) -> bool {
            return p.hasRelicRuntime(r);
        })
        .def("setHasRelic", [](Player &p, RelicId r, bool value) {
            // Runtime mirror of Player::setHasRelic<r> (compile-time template) -- flips the combat
            // relic bit. Reconstruction uses this to clear a spent one-shot relic (Lizard Tail used,
            // Omamori out of charges) that register_relics_from would otherwise leave marked present.
            const int idx = static_cast<int>(r);
            if (value) {
                if (idx < 64) p.relicBits0 |= 1ULL << idx;
                else          p.relicBits1 |= 1ULL << (idx - 64);
            } else {
                if (idx < 64) p.relicBits0 &= ~(1ULL << idx);
                else          p.relicBits1 &= ~(1ULL << (idx - 64));
            }
        }, "relic"_a, "value"_a)
        .def("gainBlock", [](Player &p, BattleContext &bc, int amount) {
            p.gainBlock(bc, amount);
        })
        .def("gainEnergy", &Player::gainEnergy)
        .def("useEnergy", &Player::useEnergy)
        .def("heal", &Player::heal)
        .def("increaseMaxHp", &Player::increaseMaxHp);

    def_value(player, "stance", &Player::stance);

    // Monster bindings
    pybind11::class_<Monster> monster(m, "Monster");
    monster.def_readwrite("curHp", &Monster::curHp)
        .def_readwrite("maxHp", &Monster::maxHp)
        .def_readwrite("block", &Monster::block)
        .def_readwrite("halfDead", &Monster::halfDead)
        .def_readwrite("idx", &Monster::idx)
        .def_property("moveHistory", 
            [](Monster &m) { return std::array<int, 2>{static_cast<int>(m.moveHistory[0]), static_cast<int>(m.moveHistory[1])}; },
            [](Monster &m, const std::array<int, 2> &arr) { 
                m.moveHistory[0] = static_cast<MonsterMoveId>(arr[0]); 
                m.moveHistory[1] = static_cast<MonsterMoveId>(arr[1]); 
            })
        .def_readonly("pending_move_rolls", &Monster::pendingMoveRolls)
        .def_readwrite("artifact", &Monster::artifact)
        .def_readwrite("strength", &Monster::strength)
        .def_readwrite("vulnerable", &Monster::vulnerable)
        .def_readwrite("weak", &Monster::weak)
        .def_readwrite("poison", &Monster::poison)
        .def_readwrite("regen", &Monster::regen)
        .def_readwrite("metallicize", &Monster::metallicize)
        .def_readwrite("platedArmor", &Monster::platedArmor)
        // Hidden move-state the live game can't observe but the search's takeTurn rollouts read.
        // miscInfo holds move-specific data the engine rolls at battle start / sets mid-move (e.g.
        // Louse bite damage, Orb Walker laser, Hexaghost Divider per-hit damage, Darkling Nip
        // damage); left at 0 by a mid-fight reconstruction, attacks that read it simulate as 0
        // damage so the search never blocks them. uniquePower0/1 are per-monster counters
        // (Hexaghost Sear cycle, Awakened One phase, etc.).
        .def_readwrite("miscInfo", &Monster::miscInfo)
        .def_readwrite("uniquePower0", &Monster::uniquePower0)
        .def_readwrite("uniquePower1", &Monster::uniquePower1)
        .def("getName", &Monster::getName)
        .def("get_move_base_damage", [](const Monster &m, const BattleContext &bc) {
            // (per-hit base damage, hit count) the engine predicts for this monster's current move,
            // before strength/vulnerable. attackCount 0 => the move is not an attack. Used to
            // sanity-check a reconstruction against the live game's displayed intent damage.
            auto d = m.getMoveBaseDamage(bc);
            return std::make_pair(d.damage, d.attackCount);
        }, "bc"_a)
        .def("hasStatus", [](const Monster &m, MonsterStatus s) -> bool {
            return m.hasStatusInternal(s);
        })
        .def("getStatus", [](const Monster &m, MonsterStatus s) -> int {
            return m.getStatusInternal(s);
        })
        .def("buff", [](Monster &m, MonsterStatus s, int amount) {
            m.buff(s, amount);
        }, pybind11::arg("status"), pybind11::arg("amount") = 1)
        .def("addDebuff", [](Monster &m, MonsterStatus s, int amount, bool isSourceMonster) {
            m.addDebuff(s, amount, isSourceMonster);
        }, pybind11::arg("status"), pybind11::arg("amount"), pybind11::arg("isSourceMonster") = true)
        .def("isAlive", &Monster::isAlive)
        .def("isTargetable", &Monster::isTargetable)
        .def("isDying", &Monster::isDying)
        .def("isEscaping", &Monster::isEscaping)
        .def("addBlock", &Monster::addBlock)
        .def("heal", &Monster::heal)
        .def("rollMove", &Monster::rollMove, "bc"_a,
            "roll this monster's next move via its own AI (keys on moveHistory). Used to reconstruct "
            "an intent the live game hasn't committed yet -- e.g. a flying Byrd between turns whose "
            "intent CommunicationMod reports as NONE with no move_id.")
        .def("setMove", [](Monster &m, int moveId) {
            m.setMove(static_cast<MonsterMoveId>(moveId));
        }, "moveId"_a,
            "Force this monster's current move, shifting moveHistory (history[1]=history[0]; "
            "history[0]=moveId). The observed-move primitive for the persistent-bc bridge: keeps "
            "moveHistory-driven selection correct for the 61/65 monsters that don't read selection-time "
            "miscInfo. The 4 that do (Champ/Darkling/Book of Stabbing/Gremlin Wizard) need miscInfo "
            "reconciled separately -- see PERSISTENT_BC_PLAN.md.")
        .def("commit_observed_move", [](Monster &m, int moveId) {
            m.setMove(static_cast<MonsterMoveId>(moveId));
            m.cancelPendingMove();
        }, "moveId"_a,
            "setMove + cancelPendingMove: force the current move to an OBSERVED value AND discard any "
            "deferred (Runic Dome) roll, so END_TURN replays the real move instead of re-rolling a "
            "hidden guess from bc.rng. Used by the ET shadow to inject each monster's actual move "
            "(from the next turn's last_move_id) before advancing the prediction.");

    def_value(monster, "id", &Monster::id);

    // MonsterGroup bindings
    pybind11::class_<MonsterGroup> monsterGroup(m, "MonsterGroup");
    monsterGroup.def_readwrite("monsterCount", &MonsterGroup::monsterCount)
        .def_readwrite("monstersAlive", &MonsterGroup::monstersAlive)
        .def("__getitem__", [](MonsterGroup &mg, int idx) -> Monster& {
            if (idx < 0 || idx >= mg.monsterCount) throw pybind11::index_error();
            return mg.arr[idx];
        }, pybind11::return_value_policy::reference_internal)
        .def("__len__", [](const MonsterGroup &mg) { return mg.monsterCount; })
        .def("createMonster", &MonsterGroup::createMonster)
        .def("skipMonsterSlot", &MonsterGroup::skipMonsterSlot)
        .def("getAliveCount", &MonsterGroup::getAliveCount)
        .def("getTargetableCount", &MonsterGroup::getTargetableCount)
        .def("getFirstTargetable", &MonsterGroup::getFirstTargetable)
        .def("areMonstersBasicallyDead", &MonsterGroup::areMonstersBasicallyDead)
        .def("__repr__", [](const MonsterGroup &mg) {
            std::string s = "<MonsterGroup[" + std::to_string(mg.monsterCount) + "]: ";
            for (int i = 0; i < mg.monsterCount; ++i) {
                if (i > 0) s += ", ";
                s += mg.arr[i].getName();
                s += "(";
                s += std::to_string(mg.arr[i].curHp);
                s += "/";
                s += std::to_string(mg.arr[i].maxHp);
                s += ")";
                if (mg.arr[i].halfDead) s += " [DEAD]";
            }
            return s + ">";
        });

    // CardInstance bindings
    pybind11::class_<CardInstance> cardInstance(m, "CardInstance");
    cardInstance.def(pybind11::init<>())
        .def(pybind11::init<CardId, bool>(), pybind11::arg("id"), pybind11::arg("upgraded") = false)
        .def(pybind11::init<const Card&>())
        .def_property_readonly("id", &CardInstance::getId)
        .def_property_readonly("upgrade_count", &CardInstance::getUpgradeCount)
        .def_readwrite("uniqueId", &CardInstance::uniqueId)
        .def_readwrite("upgraded", &CardInstance::upgraded)
        .def_readwrite("specialData", &CardInstance::specialData)
        .def_readwrite("cost", &CardInstance::cost)
        .def_readwrite("costForTurn", &CardInstance::costForTurn)
        .def_readwrite("freeToPlayOnce", &CardInstance::freeToPlayOnce)
        .def_readwrite("retain", &CardInstance::retain)
        .def("getName", &CardInstance::getName)
        .def("getType", &CardInstance::getType)
        .def("isUpgraded", &CardInstance::isUpgraded)
        .def("canUpgrade", &CardInstance::canUpgrade)
        .def("isEthereal", &CardInstance::isEthereal)
        .def("isStrikeCard", &CardInstance::isStrikeCard)
        .def("doesExhaust", &CardInstance::doesExhaust)
        .def("requiresTarget", &CardInstance::requiresTarget)
        .def("isXCost", &CardInstance::isXCost)
        .def("isBloodCard", &CardInstance::isBloodCard)
        .def("upgrade", &CardInstance::upgrade)
        .def("canUse", &CardInstance::canUse)
        .def("canUseOnAnyTarget", &CardInstance::canUseOnAnyTarget)
        .def("__repr__", [](const CardInstance &c) {
            std::string s = "<";
            s += c.getName();
            if (c.upgraded) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW && c.specialData > 1) {
                    s += std::to_string(c.specialData);
                }
            }
            s += " [" + std::to_string(c.cost) + "]";
            if (c.uniqueId != -1) {
                s += " #" + std::to_string(c.uniqueId);
            }
            return s + ">";
        });

    def_value(cardInstance, "id", &CardInstance::id);

    // CardManager bindings
    pybind11::class_<CardManager> cardManager(m, "CardManager");
    cardManager.def_readwrite("cardsInHand", &CardManager::cardsInHand)
        .def_readwrite("strikeCount", &CardManager::strikeCount,
            "count of in-combat Strike-named cards (hand+draw+discard); Perfect Strike's bonus reads it")
        .def_readwrite("next_unique_card_id", &CardManager::nextUniqueCardId,
            "the counter the engine assigns to each new in-battle card; reconstructing a state must "
            "give every card a distinct id (cards are tracked through hand/queue/piles by uniqueId) "
            "and leave this past the highest assigned id so generated cards don't collide")
        .def_property_readonly("hand", [](CardManager &cm) {
            return std::vector<CardInstance>(cm.hand.begin(), cm.hand.begin() + cm.cardsInHand);
        })
        .def_property_readonly("drawPile", [](CardManager &cm) {
            return std::vector<CardInstance>(cm.drawPile.begin(), cm.drawPile.end());
        })
        .def_property_readonly("discardPile", [](CardManager &cm) {
            return std::vector<CardInstance>(cm.discardPile.begin(), cm.discardPile.end());
        })
        .def_property_readonly("exhaustPile", [](CardManager &cm) {
            return std::vector<CardInstance>(cm.exhaustPile.begin(), cm.exhaustPile.end());
        })
        .def("notify_add_card_to_combat", &CardManager::notifyAddCardToCombat,
            "register a card as having entered combat (maintains strikeCount for Perfect Strike). "
            "Native deck-load calls this for every card; a reconstruction must too, or strikeCount "
            "stays 0 (and goes negative as strikes are moved to exhaust) so Perfect Strike loses its "
            "per-Strike bonus damage")
        .def("moveToHand", &CardManager::moveToHand)
        .def("moveToDiscardPile", &CardManager::moveToDiscardPile)
        .def("moveToExhaustPile", &CardManager::moveToExhaustPile)
        .def("moveToDrawPileTop", &CardManager::moveToDrawPileTop)
        .def("moveToDrawPileUnknown", &CardManager::moveToDrawPileUnknown)
        .def("removeFromDrawPile", &CardManager::removeFromDrawPile, "match"_a,
             "remove the first draw-pile card matching by id + upgrade; returns it (INVALID if none)")
        .def("set_stasis_card", [](CardManager &cm, int slot, const CardInstance &c) {
                cm.stasisCards[slot] = c;
            }, "slot"_a, "card"_a,
            "store a Bronze Orb's in-stasis card (slot = min(orbMonsterIdx, 1)). A reconstructed "
            "Bronze Automaton fight whose orbs already used Stasis must set this, or the engine's "
            "returnStasisCard asserts on the orb's death (the stolen card is missing from every pile)")
        .def("removeFromHandAtIdx", &CardManager::removeFromHandAtIdx)
        .def("draw", &CardManager::draw)
        .def("clear", &CardManager::clear);

    auto &internals = pybind11::detail::get_internals();
    auto pybind11_metaclass = pybind11::reinterpret_borrow<pybind11::object>((PyObject*)internals.default_metaclass);
    auto standard_metaclass = pybind11::reinterpret_borrow<pybind11::object>((PyObject *)&PyType_Type);
    pybind11::dict attributes;
    attributes["__len__"] = pybind11::cpp_function(
        [](pybind11::object cls) {
            return pybind11::len(cls.attr("__entries"));
        }
        , pybind11::is_method(pybind11::none())
        );
    auto enum_metaclass = standard_metaclass(std::string("pybind11_ext_enum")
        , pybind11::make_tuple(pybind11_metaclass)
        , attributes);


    pybind11::enum_<GameOutcome> gameOutcome(m, "GameOutcome", pybind11::metaclass(enum_metaclass));
    gameOutcome.value("UNDECIDED", GameOutcome::UNDECIDED)
        .value("PLAYER_VICTORY", GameOutcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", GameOutcome::PLAYER_LOSS);

    pybind11::enum_<ScreenState> screenState(m, "ScreenState", pybind11::metaclass(enum_metaclass));
    screenState.value("INVALID", ScreenState::INVALID)
        .value("EVENT_SCREEN", ScreenState::EVENT_SCREEN)
        .value("REWARDS", ScreenState::REWARDS)
        .value("BOSS_RELIC_REWARDS", ScreenState::BOSS_RELIC_REWARDS)
        .value("CARD_SELECT", ScreenState::CARD_SELECT)
        .value("MAP_SCREEN", ScreenState::MAP_SCREEN)
        .value("TREASURE_ROOM", ScreenState::TREASURE_ROOM)
        .value("REST_ROOM", ScreenState::REST_ROOM)
        .value("SHOP_ROOM", ScreenState::SHOP_ROOM)
        .value("BATTLE", ScreenState::BATTLE);

    pybind11::enum_<Event> eventEnum(m, "Event", pybind11::metaclass(enum_metaclass));
    eventEnum.value("INVALID", Event::INVALID)
        .value("MONSTER", Event::MONSTER)
        .value("REST", Event::REST)
        .value("SHOP", Event::SHOP)
        .value("TREASURE", Event::TREASURE)
        .value("NEOW", Event::NEOW)
        .value("OMINOUS_FORGE", Event::OMINOUS_FORGE)
        .value("PLEADING_VAGRANT", Event::PLEADING_VAGRANT)
        .value("ANCIENT_WRITING", Event::ANCIENT_WRITING)
        .value("OLD_BEGGAR", Event::OLD_BEGGAR)
        .value("BIG_FISH", Event::BIG_FISH)
        .value("BONFIRE_SPIRITS", Event::BONFIRE_SPIRITS)
        .value("COLOSSEUM", Event::COLOSSEUM)
        .value("CURSED_TOME", Event::CURSED_TOME)
        .value("DEAD_ADVENTURER", Event::DEAD_ADVENTURER)
        .value("DESIGNER_IN_SPIRE", Event::DESIGNER_IN_SPIRE)
        .value("AUGMENTER", Event::AUGMENTER)
        .value("DUPLICATOR", Event::DUPLICATOR)
        .value("FACE_TRADER", Event::FACE_TRADER)
        .value("FALLING", Event::FALLING)
        .value("FORGOTTEN_ALTAR", Event::FORGOTTEN_ALTAR)
        .value("THE_DIVINE_FOUNTAIN", Event::THE_DIVINE_FOUNTAIN)
        .value("GHOSTS", Event::GHOSTS)
        .value("GOLDEN_IDOL", Event::GOLDEN_IDOL)
        .value("GOLDEN_SHRINE", Event::GOLDEN_SHRINE)
        .value("WING_STATUE", Event::WING_STATUE)
        .value("KNOWING_SKULL", Event::KNOWING_SKULL)
        .value("LAB", Event::LAB)
        .value("THE_SSSSSERPENT", Event::THE_SSSSSERPENT)
        .value("LIVING_WALL", Event::LIVING_WALL)
        .value("MASKED_BANDITS", Event::MASKED_BANDITS)
        .value("MATCH_AND_KEEP", Event::MATCH_AND_KEEP)
        .value("MINDBLOOM", Event::MINDBLOOM)
        .value("HYPNOTIZING_COLORED_MUSHROOMS", Event::HYPNOTIZING_COLORED_MUSHROOMS)
        .value("MYSTERIOUS_SPHERE", Event::MYSTERIOUS_SPHERE)
        .value("THE_NEST", Event::THE_NEST)
        .value("NLOTH", Event::NLOTH)
        .value("NOTE_FOR_YOURSELF", Event::NOTE_FOR_YOURSELF)
        .value("PURIFIER", Event::PURIFIER)
        .value("SCRAP_OOZE", Event::SCRAP_OOZE)
        .value("SECRET_PORTAL", Event::SECRET_PORTAL)
        .value("SENSORY_STONE", Event::SENSORY_STONE)
        .value("SHINING_LIGHT", Event::SHINING_LIGHT)
        .value("THE_CLERIC", Event::THE_CLERIC)
        .value("THE_JOUST", Event::THE_JOUST)
        .value("THE_LIBRARY", Event::THE_LIBRARY)
        .value("THE_MAUSOLEUM", Event::THE_MAUSOLEUM)
        .value("THE_MOAI_HEAD", Event::THE_MOAI_HEAD)
        .value("THE_WOMAN_IN_BLUE", Event::THE_WOMAN_IN_BLUE)
        .value("TOMB_OF_LORD_RED_MASK", Event::TOMB_OF_LORD_RED_MASK)
        .value("TRANSMORGRIFIER", Event::TRANSMORGRIFIER)
        .value("UPGRADE_SHRINE", Event::UPGRADE_SHRINE)
        .value("VAMPIRES", Event::VAMPIRES)
        .value("WE_MEET_AGAIN", Event::WE_MEET_AGAIN)
        .value("WHEEL_OF_CHANGE", Event::WHEEL_OF_CHANGE)
        .value("WINDING_HALLS", Event::WINDING_HALLS)
        .value("WORLD_OF_GOOP", Event::WORLD_OF_GOOP);

    pybind11::enum_<Neow::Bonus>(m, "NeowBonus")
        .value("THREE_CARDS", Neow::Bonus::THREE_CARDS)
        .value("ONE_RANDOM_RARE_CARD", Neow::Bonus::ONE_RANDOM_RARE_CARD)
        .value("REMOVE_CARD", Neow::Bonus::REMOVE_CARD)
        .value("UPGRADE_CARD", Neow::Bonus::UPGRADE_CARD)
        .value("TRANSFORM_CARD", Neow::Bonus::TRANSFORM_CARD)
        .value("RANDOM_COLORLESS", Neow::Bonus::RANDOM_COLORLESS)
        .value("THREE_SMALL_POTIONS", Neow::Bonus::THREE_SMALL_POTIONS)
        .value("RANDOM_COMMON_RELIC", Neow::Bonus::RANDOM_COMMON_RELIC)
        .value("TEN_PERCENT_HP_BONUS", Neow::Bonus::TEN_PERCENT_HP_BONUS)
        .value("THREE_ENEMY_KILL", Neow::Bonus::THREE_ENEMY_KILL)
        .value("HUNDRED_GOLD", Neow::Bonus::HUNDRED_GOLD)
        .value("RANDOM_COLORLESS_2", Neow::Bonus::RANDOM_COLORLESS_2)
        .value("REMOVE_TWO", Neow::Bonus::REMOVE_TWO)
        .value("ONE_RARE_RELIC", Neow::Bonus::ONE_RARE_RELIC)
        .value("THREE_RARE_CARDS", Neow::Bonus::THREE_RARE_CARDS)
        .value("TWO_FIFTY_GOLD", Neow::Bonus::TWO_FIFTY_GOLD)
        .value("TRANSFORM_TWO_CARDS", Neow::Bonus::TRANSFORM_TWO_CARDS)
        .value("TWENTY_PERCENT_HP_BONUS", Neow::Bonus::TWENTY_PERCENT_HP_BONUS)
        .value("BOSS_RELIC", Neow::Bonus::BOSS_RELIC)
        .value("INVALID", Neow::Bonus::INVALID);

    pybind11::enum_<Neow::Drawback>(m, "NeowDrawback")
        .value("INVALID", Neow::Drawback::INVALID)
        .value("NONE", Neow::Drawback::NONE)
        .value("TEN_PERCENT_HP_LOSS", Neow::Drawback::TEN_PERCENT_HP_LOSS)
        .value("NO_GOLD", Neow::Drawback::NO_GOLD)
        .value("CURSE", Neow::Drawback::CURSE)
        .value("PERCENT_DAMAGE", Neow::Drawback::PERCENT_DAMAGE)
        .value("LOSE_STARTER_RELIC", Neow::Drawback::LOSE_STARTER_RELIC);

    pybind11::class_<Neow::Option> neowOption(m, "NeowOption");
    def_value(neowOption, "r", &Neow::Option::r);
    def_value(neowOption, "d", &Neow::Option::d);

    pybind11::enum_<CardSelectScreenType> cardSelectScreenType(m, "CardSelectScreenType", pybind11::metaclass(enum_metaclass));
    cardSelectScreenType.value("INVALID", CardSelectScreenType::INVALID)
        .value("TRANSFORM", CardSelectScreenType::TRANSFORM)
        .value("TRANSFORM_UPGRADE", CardSelectScreenType::TRANSFORM_UPGRADE)
        .value("UPGRADE", CardSelectScreenType::UPGRADE)
        .value("REMOVE", CardSelectScreenType::REMOVE)
        .value("DUPLICATE", CardSelectScreenType::DUPLICATE)
        .value("OBTAIN", CardSelectScreenType::OBTAIN)
        .value("BOTTLE", CardSelectScreenType::BOTTLE)
        .value("BONFIRE_SPIRITS", CardSelectScreenType::BONFIRE_SPIRITS);

    pybind11::enum_<CardSelectTask> cardSelectTask(m, "CardSelectTask", pybind11::metaclass(enum_metaclass));
    cardSelectTask.value("INVALID", CardSelectTask::INVALID)
        .value("ARMAMENTS", CardSelectTask::ARMAMENTS)
        .value("CODEX", CardSelectTask::CODEX)
        .value("DISCOVERY", CardSelectTask::DISCOVERY)
        .value("DUAL_WIELD", CardSelectTask::DUAL_WIELD)
        .value("EXHAUST_ONE", CardSelectTask::EXHAUST_ONE)
        .value("EXHAUST_MANY", CardSelectTask::EXHAUST_MANY)
        .value("EXHUME", CardSelectTask::EXHUME)
        .value("FORETHOUGHT", CardSelectTask::FORETHOUGHT)
        .value("GAMBLE", CardSelectTask::GAMBLE)
        .value("HEADBUTT", CardSelectTask::HEADBUTT)
        .value("HOLOGRAM", CardSelectTask::HOLOGRAM)
        .value("LIQUID_MEMORIES_POTION", CardSelectTask::LIQUID_MEMORIES_POTION)
        .value("MEDITATE", CardSelectTask::MEDITATE)
        .value("NIGHTMARE", CardSelectTask::NIGHTMARE)
        .value("RECYCLE", CardSelectTask::RECYCLE)
        .value("SECRET_TECHNIQUE", CardSelectTask::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardSelectTask::SECRET_WEAPON)
        .value("SEEK", CardSelectTask::SEEK)
        .value("SETUP", CardSelectTask::SETUP)
        .value("WARCRY", CardSelectTask::WARCRY);

    pybind11::enum_<GameAction::RewardsActionType> rewardsActionType(m, "RewardsActionType");
    rewardsActionType.value("CARD", GameAction::RewardsActionType::CARD)
        .value("GOLD", GameAction::RewardsActionType::GOLD)
        .value("KEY", GameAction::RewardsActionType::KEY)
        .value("POTION", GameAction::RewardsActionType::POTION)
        .value("RELIC", GameAction::RewardsActionType::RELIC)
        .value("CARD_REMOVE", GameAction::RewardsActionType::CARD_REMOVE)
        .value("SKIP", GameAction::RewardsActionType::SKIP);

    pybind11::enum_<CharacterClass> characterClass(m, "CharacterClass", pybind11::metaclass(enum_metaclass));
    characterClass.value("IRONCLAD", CharacterClass::IRONCLAD)
            .value("SILENT", CharacterClass::SILENT)
            .value("DEFECT", CharacterClass::DEFECT)
            .value("WATCHER", CharacterClass::WATCHER)
            .value("INVALID", CharacterClass::INVALID);

    pybind11::enum_<Room> roomEnum(m, "Room", pybind11::metaclass(enum_metaclass));
    roomEnum.value("SHOP", Room::SHOP)
        .value("REST", Room::REST)
        .value("EVENT", Room::EVENT)
        .value("ELITE", Room::ELITE)
        .value("MONSTER", Room::MONSTER)
        .value("TREASURE", Room::TREASURE)
        .value("BOSS", Room::BOSS)
        .value("BOSS_TREASURE", Room::BOSS_TREASURE)
        .value("NONE", Room::NONE)
        .value("INVALID", Room::INVALID);

    pybind11::enum_<CardRarity>(m, "CardRarity", pybind11::metaclass(enum_metaclass))
        .value("COMMON", CardRarity::COMMON)
        .value("UNCOMMON", CardRarity::UNCOMMON)
        .value("RARE", CardRarity::RARE)
        .value("BASIC", CardRarity::BASIC)
        .value("SPECIAL", CardRarity::SPECIAL)
        .value("CURSE", CardRarity::CURSE)
        .value("INVALID", CardRarity::INVALID);

    pybind11::enum_<CardColor>(m, "CardColor", pybind11::metaclass(enum_metaclass))
        .value("RED", CardColor::RED)
        .value("GREEN", CardColor::GREEN)
        .value("PURPLE", CardColor::PURPLE)
        .value("COLORLESS", CardColor::COLORLESS)
        .value("CURSE", CardColor::CURSE)
        .value("INVALID", CardColor::INVALID);

    pybind11::enum_<CardType>(m, "CardType", pybind11::metaclass(enum_metaclass))
        .value("ATTACK", CardType::ATTACK)
        .value("SKILL", CardType::SKILL)
        .value("POWER", CardType::POWER)
        .value("CURSE", CardType::CURSE)
        .value("STATUS", CardType::STATUS)
        .value("INVALID", CardType::INVALID);

    pybind11::enum_<CardId>(m, "CardId", pybind11::metaclass(enum_metaclass))
        .value("INVALID", CardId::INVALID)
        .value("ACCURACY", CardId::ACCURACY)
        .value("ACROBATICS", CardId::ACROBATICS)
        .value("ADRENALINE", CardId::ADRENALINE)
        .value("AFTER_IMAGE", CardId::AFTER_IMAGE)
        .value("AGGREGATE", CardId::AGGREGATE)
        .value("ALCHEMIZE", CardId::ALCHEMIZE)
        .value("ALL_FOR_ONE", CardId::ALL_FOR_ONE)
        .value("ALL_OUT_ATTACK", CardId::ALL_OUT_ATTACK)
        .value("ALPHA", CardId::ALPHA)
        .value("AMPLIFY", CardId::AMPLIFY)
        .value("ANGER", CardId::ANGER)
        .value("APOTHEOSIS", CardId::APOTHEOSIS)
        .value("APPARITION", CardId::APPARITION)
        .value("ARMAMENTS", CardId::ARMAMENTS)
        .value("ASCENDERS_BANE", CardId::ASCENDERS_BANE)
        .value("AUTO_SHIELDS", CardId::AUTO_SHIELDS)
        .value("A_THOUSAND_CUTS", CardId::A_THOUSAND_CUTS)
        .value("BACKFLIP", CardId::BACKFLIP)
        .value("BACKSTAB", CardId::BACKSTAB)
        .value("BALL_LIGHTNING", CardId::BALL_LIGHTNING)
        .value("BANDAGE_UP", CardId::BANDAGE_UP)
        .value("BANE", CardId::BANE)
        .value("BARRAGE", CardId::BARRAGE)
        .value("BARRICADE", CardId::BARRICADE)
        .value("BASH", CardId::BASH)
        .value("BATTLE_HYMN", CardId::BATTLE_HYMN)
        .value("BATTLE_TRANCE", CardId::BATTLE_TRANCE)
        .value("BEAM_CELL", CardId::BEAM_CELL)
        .value("BECOME_ALMIGHTY", CardId::BECOME_ALMIGHTY)
        .value("BERSERK", CardId::BERSERK)
        .value("BETA", CardId::BETA)
        .value("BIASED_COGNITION", CardId::BIASED_COGNITION)
        .value("BITE", CardId::BITE)
        .value("BLADE_DANCE", CardId::BLADE_DANCE)
        .value("BLASPHEMY", CardId::BLASPHEMY)
        .value("BLIND", CardId::BLIND)
        .value("BLIZZARD", CardId::BLIZZARD)
        .value("BLOODLETTING", CardId::BLOODLETTING)
        .value("BLOOD_FOR_BLOOD", CardId::BLOOD_FOR_BLOOD)
        .value("BLUDGEON", CardId::BLUDGEON)
        .value("BLUR", CardId::BLUR)
        .value("BODY_SLAM", CardId::BODY_SLAM)
        .value("BOOT_SEQUENCE", CardId::BOOT_SEQUENCE)
        .value("BOUNCING_FLASK", CardId::BOUNCING_FLASK)
        .value("BOWLING_BASH", CardId::BOWLING_BASH)
        .value("BRILLIANCE", CardId::BRILLIANCE)
        .value("BRUTALITY", CardId::BRUTALITY)
        .value("BUFFER", CardId::BUFFER)
        .value("BULLET_TIME", CardId::BULLET_TIME)
        .value("BULLSEYE", CardId::BULLSEYE)
        .value("BURN", CardId::BURN)
        .value("BURNING_PACT", CardId::BURNING_PACT)
        .value("BURST", CardId::BURST)
        .value("CALCULATED_GAMBLE", CardId::CALCULATED_GAMBLE)
        .value("CALTROPS", CardId::CALTROPS)
        .value("CAPACITOR", CardId::CAPACITOR)
        .value("CARNAGE", CardId::CARNAGE)
        .value("CARVE_REALITY", CardId::CARVE_REALITY)
        .value("CATALYST", CardId::CATALYST)
        .value("CHAOS", CardId::CHAOS)
        .value("CHARGE_BATTERY", CardId::CHARGE_BATTERY)
        .value("CHILL", CardId::CHILL)
        .value("CHOKE", CardId::CHOKE)
        .value("CHRYSALIS", CardId::CHRYSALIS)
        .value("CLASH", CardId::CLASH)
        .value("CLAW", CardId::CLAW)
        .value("CLEAVE", CardId::CLEAVE)
        .value("CLOAK_AND_DAGGER", CardId::CLOAK_AND_DAGGER)
        .value("CLOTHESLINE", CardId::CLOTHESLINE)
        .value("CLUMSY", CardId::CLUMSY)
        .value("COLD_SNAP", CardId::COLD_SNAP)
        .value("COLLECT", CardId::COLLECT)
        .value("COMBUST", CardId::COMBUST)
        .value("COMPILE_DRIVER", CardId::COMPILE_DRIVER)
        .value("CONCENTRATE", CardId::CONCENTRATE)
        .value("CONCLUDE", CardId::CONCLUDE)
        .value("CONJURE_BLADE", CardId::CONJURE_BLADE)
        .value("CONSECRATE", CardId::CONSECRATE)
        .value("CONSUME", CardId::CONSUME)
        .value("COOLHEADED", CardId::COOLHEADED)
        .value("CORE_SURGE", CardId::CORE_SURGE)
        .value("CORPSE_EXPLOSION", CardId::CORPSE_EXPLOSION)
        .value("CORRUPTION", CardId::CORRUPTION)
        .value("CREATIVE_AI", CardId::CREATIVE_AI)
        .value("CRESCENDO", CardId::CRESCENDO)
        .value("CRIPPLING_CLOUD", CardId::CRIPPLING_CLOUD)
        .value("CRUSH_JOINTS", CardId::CRUSH_JOINTS)
        .value("CURSE_OF_THE_BELL", CardId::CURSE_OF_THE_BELL)
        .value("CUT_THROUGH_FATE", CardId::CUT_THROUGH_FATE)
        .value("DAGGER_SPRAY", CardId::DAGGER_SPRAY)
        .value("DAGGER_THROW", CardId::DAGGER_THROW)
        .value("DARKNESS", CardId::DARKNESS)
        .value("DARK_EMBRACE", CardId::DARK_EMBRACE)
        .value("DARK_SHACKLES", CardId::DARK_SHACKLES)
        .value("DASH", CardId::DASH)
        .value("DAZED", CardId::DAZED)
        .value("DEADLY_POISON", CardId::DEADLY_POISON)
        .value("DECAY", CardId::DECAY)
        .value("DECEIVE_REALITY", CardId::DECEIVE_REALITY)
        .value("DEEP_BREATH", CardId::DEEP_BREATH)
        .value("DEFEND_BLUE", CardId::DEFEND_BLUE)
        .value("DEFEND_GREEN", CardId::DEFEND_GREEN)
        .value("DEFEND_PURPLE", CardId::DEFEND_PURPLE)
        .value("DEFEND_RED", CardId::DEFEND_RED)
        .value("DEFLECT", CardId::DEFLECT)
        .value("DEFRAGMENT", CardId::DEFRAGMENT)
        .value("DEMON_FORM", CardId::DEMON_FORM)
        .value("DEUS_EX_MACHINA", CardId::DEUS_EX_MACHINA)
        .value("DEVA_FORM", CardId::DEVA_FORM)
        .value("DEVOTION", CardId::DEVOTION)
        .value("DIE_DIE_DIE", CardId::DIE_DIE_DIE)
        .value("DISARM", CardId::DISARM)
        .value("DISCOVERY", CardId::DISCOVERY)
        .value("DISTRACTION", CardId::DISTRACTION)
        .value("DODGE_AND_ROLL", CardId::DODGE_AND_ROLL)
        .value("DOOM_AND_GLOOM", CardId::DOOM_AND_GLOOM)
        .value("DOPPELGANGER", CardId::DOPPELGANGER)
        .value("DOUBLE_ENERGY", CardId::DOUBLE_ENERGY)
        .value("DOUBLE_TAP", CardId::DOUBLE_TAP)
        .value("DOUBT", CardId::DOUBT)
        .value("DRAMATIC_ENTRANCE", CardId::DRAMATIC_ENTRANCE)
        .value("DROPKICK", CardId::DROPKICK)
        .value("DUALCAST", CardId::DUALCAST)
        .value("DUAL_WIELD", CardId::DUAL_WIELD)
        .value("ECHO_FORM", CardId::ECHO_FORM)
        .value("ELECTRODYNAMICS", CardId::ELECTRODYNAMICS)
        .value("EMPTY_BODY", CardId::EMPTY_BODY)
        .value("EMPTY_FIST", CardId::EMPTY_FIST)
        .value("EMPTY_MIND", CardId::EMPTY_MIND)
        .value("ENDLESS_AGONY", CardId::ENDLESS_AGONY)
        .value("ENLIGHTENMENT", CardId::ENLIGHTENMENT)
        .value("ENTRENCH", CardId::ENTRENCH)
        .value("ENVENOM", CardId::ENVENOM)
        .value("EQUILIBRIUM", CardId::EQUILIBRIUM)
        .value("ERUPTION", CardId::ERUPTION)
        .value("ESCAPE_PLAN", CardId::ESCAPE_PLAN)
        .value("ESTABLISHMENT", CardId::ESTABLISHMENT)
        .value("EVALUATE", CardId::EVALUATE)
        .value("EVISCERATE", CardId::EVISCERATE)
        .value("EVOLVE", CardId::EVOLVE)
        .value("EXHUME", CardId::EXHUME)
        .value("EXPERTISE", CardId::EXPERTISE)
        .value("EXPUNGER", CardId::EXPUNGER)
        .value("FAME_AND_FORTUNE", CardId::FAME_AND_FORTUNE)
        .value("FASTING", CardId::FASTING)
        .value("FEAR_NO_EVIL", CardId::FEAR_NO_EVIL)
        .value("FEED", CardId::FEED)
        .value("FEEL_NO_PAIN", CardId::FEEL_NO_PAIN)
        .value("FIEND_FIRE", CardId::FIEND_FIRE)
        .value("FINESSE", CardId::FINESSE)
        .value("FINISHER", CardId::FINISHER)
        .value("FIRE_BREATHING", CardId::FIRE_BREATHING)
        .value("FISSION", CardId::FISSION)
        .value("FLAME_BARRIER", CardId::FLAME_BARRIER)
        .value("FLASH_OF_STEEL", CardId::FLASH_OF_STEEL)
        .value("FLECHETTES", CardId::FLECHETTES)
        .value("FLEX", CardId::FLEX)
        .value("FLURRY_OF_BLOWS", CardId::FLURRY_OF_BLOWS)
        .value("FLYING_KNEE", CardId::FLYING_KNEE)
        .value("FLYING_SLEEVES", CardId::FLYING_SLEEVES)
        .value("FOLLOW_UP", CardId::FOLLOW_UP)
        .value("FOOTWORK", CardId::FOOTWORK)
        .value("FORCE_FIELD", CardId::FORCE_FIELD)
        .value("FOREIGN_INFLUENCE", CardId::FOREIGN_INFLUENCE)
        .value("FORESIGHT", CardId::FORESIGHT)
        .value("FORETHOUGHT", CardId::FORETHOUGHT)
        .value("FTL", CardId::FTL)
        .value("FUSION", CardId::FUSION)
        .value("GENETIC_ALGORITHM", CardId::GENETIC_ALGORITHM)
        .value("GHOSTLY_ARMOR", CardId::GHOSTLY_ARMOR)
        .value("GLACIER", CardId::GLACIER)
        .value("GLASS_KNIFE", CardId::GLASS_KNIFE)
        .value("GOOD_INSTINCTS", CardId::GOOD_INSTINCTS)
        .value("GO_FOR_THE_EYES", CardId::GO_FOR_THE_EYES)
        .value("GRAND_FINALE", CardId::GRAND_FINALE)
        .value("HALT", CardId::HALT)
        .value("HAND_OF_GREED", CardId::HAND_OF_GREED)
        .value("HAVOC", CardId::HAVOC)
        .value("HEADBUTT", CardId::HEADBUTT)
        .value("HEATSINKS", CardId::HEATSINKS)
        .value("HEAVY_BLADE", CardId::HEAVY_BLADE)
        .value("HEEL_HOOK", CardId::HEEL_HOOK)
        .value("HELLO_WORLD", CardId::HELLO_WORLD)
        .value("HEMOKINESIS", CardId::HEMOKINESIS)
        .value("HOLOGRAM", CardId::HOLOGRAM)
        .value("HYPERBEAM", CardId::HYPERBEAM)
        .value("IMMOLATE", CardId::IMMOLATE)
        .value("IMPATIENCE", CardId::IMPATIENCE)
        .value("IMPERVIOUS", CardId::IMPERVIOUS)
        .value("INDIGNATION", CardId::INDIGNATION)
        .value("INFERNAL_BLADE", CardId::INFERNAL_BLADE)
        .value("INFINITE_BLADES", CardId::INFINITE_BLADES)
        .value("INFLAME", CardId::INFLAME)
        .value("INJURY", CardId::INJURY)
        .value("INNER_PEACE", CardId::INNER_PEACE)
        .value("INSIGHT", CardId::INSIGHT)
        .value("INTIMIDATE", CardId::INTIMIDATE)
        .value("IRON_WAVE", CardId::IRON_WAVE)
        .value("JAX", CardId::JAX)
        .value("JACK_OF_ALL_TRADES", CardId::JACK_OF_ALL_TRADES)
        .value("JUDGMENT", CardId::JUDGMENT)
        .value("JUGGERNAUT", CardId::JUGGERNAUT)
        .value("JUST_LUCKY", CardId::JUST_LUCKY)
        .value("LEAP", CardId::LEAP)
        .value("LEG_SWEEP", CardId::LEG_SWEEP)
        .value("LESSON_LEARNED", CardId::LESSON_LEARNED)
        .value("LIKE_WATER", CardId::LIKE_WATER)
        .value("LIMIT_BREAK", CardId::LIMIT_BREAK)
        .value("LIVE_FOREVER", CardId::LIVE_FOREVER)
        .value("LOOP", CardId::LOOP)
        .value("MACHINE_LEARNING", CardId::MACHINE_LEARNING)
        .value("MADNESS", CardId::MADNESS)
        .value("MAGNETISM", CardId::MAGNETISM)
        .value("MALAISE", CardId::MALAISE)
        .value("MASTERFUL_STAB", CardId::MASTERFUL_STAB)
        .value("MASTER_OF_STRATEGY", CardId::MASTER_OF_STRATEGY)
        .value("MASTER_REALITY", CardId::MASTER_REALITY)
        .value("MAYHEM", CardId::MAYHEM)
        .value("MEDITATE", CardId::MEDITATE)
        .value("MELTER", CardId::MELTER)
        .value("MENTAL_FORTRESS", CardId::MENTAL_FORTRESS)
        .value("METALLICIZE", CardId::METALLICIZE)
        .value("METAMORPHOSIS", CardId::METAMORPHOSIS)
        .value("METEOR_STRIKE", CardId::METEOR_STRIKE)
        .value("MIND_BLAST", CardId::MIND_BLAST)
        .value("MIRACLE", CardId::MIRACLE)
        .value("MULTI_CAST", CardId::MULTI_CAST)
        .value("NECRONOMICURSE", CardId::NECRONOMICURSE)
        .value("NEUTRALIZE", CardId::NEUTRALIZE)
        .value("NIGHTMARE", CardId::NIGHTMARE)
        .value("NIRVANA", CardId::NIRVANA)
        .value("NORMALITY", CardId::NORMALITY)
        .value("NOXIOUS_FUMES", CardId::NOXIOUS_FUMES)
        .value("OFFERING", CardId::OFFERING)
        .value("OMEGA", CardId::OMEGA)
        .value("OMNISCIENCE", CardId::OMNISCIENCE)
        .value("OUTMANEUVER", CardId::OUTMANEUVER)
        .value("OVERCLOCK", CardId::OVERCLOCK)
        .value("PAIN", CardId::PAIN)
        .value("PANACEA", CardId::PANACEA)
        .value("PANACHE", CardId::PANACHE)
        .value("PANIC_BUTTON", CardId::PANIC_BUTTON)
        .value("PARASITE", CardId::PARASITE)
        .value("PERFECTED_STRIKE", CardId::PERFECTED_STRIKE)
        .value("PERSEVERANCE", CardId::PERSEVERANCE)
        .value("PHANTASMAL_KILLER", CardId::PHANTASMAL_KILLER)
        .value("PIERCING_WAIL", CardId::PIERCING_WAIL)
        .value("POISONED_STAB", CardId::POISONED_STAB)
        .value("POMMEL_STRIKE", CardId::POMMEL_STRIKE)
        .value("POWER_THROUGH", CardId::POWER_THROUGH)
        .value("PRAY", CardId::PRAY)
        .value("PREDATOR", CardId::PREDATOR)
        .value("PREPARED", CardId::PREPARED)
        .value("PRESSURE_POINTS", CardId::PRESSURE_POINTS)
        .value("PRIDE", CardId::PRIDE)
        .value("PROSTRATE", CardId::PROSTRATE)
        .value("PROTECT", CardId::PROTECT)
        .value("PUMMEL", CardId::PUMMEL)
        .value("PURITY", CardId::PURITY)
        .value("QUICK_SLASH", CardId::QUICK_SLASH)
        .value("RAGE", CardId::RAGE)
        .value("RAGNAROK", CardId::RAGNAROK)
        .value("RAINBOW", CardId::RAINBOW)
        .value("RAMPAGE", CardId::RAMPAGE)
        .value("REACH_HEAVEN", CardId::REACH_HEAVEN)
        .value("REAPER", CardId::REAPER)
        .value("REBOOT", CardId::REBOOT)
        .value("REBOUND", CardId::REBOUND)
        .value("RECKLESS_CHARGE", CardId::RECKLESS_CHARGE)
        .value("RECURSION", CardId::RECURSION)
        .value("RECYCLE", CardId::RECYCLE)
        .value("REFLEX", CardId::REFLEX)
        .value("REGRET", CardId::REGRET)
        .value("REINFORCED_BODY", CardId::REINFORCED_BODY)
        .value("REPROGRAM", CardId::REPROGRAM)
        .value("RIDDLE_WITH_HOLES", CardId::RIDDLE_WITH_HOLES)
        .value("RIP_AND_TEAR", CardId::RIP_AND_TEAR)
        .value("RITUAL_DAGGER", CardId::RITUAL_DAGGER)
        .value("RUPTURE", CardId::RUPTURE)
        .value("RUSHDOWN", CardId::RUSHDOWN)
        .value("SADISTIC_NATURE", CardId::SADISTIC_NATURE)
        .value("SAFETY", CardId::SAFETY)
        .value("SANCTITY", CardId::SANCTITY)
        .value("SANDS_OF_TIME", CardId::SANDS_OF_TIME)
        .value("SASH_WHIP", CardId::SASH_WHIP)
        .value("SCRAPE", CardId::SCRAPE)
        .value("SCRAWL", CardId::SCRAWL)
        .value("SEARING_BLOW", CardId::SEARING_BLOW)
        .value("SECOND_WIND", CardId::SECOND_WIND)
        .value("SECRET_TECHNIQUE", CardId::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardId::SECRET_WEAPON)
        .value("SEEING_RED", CardId::SEEING_RED)
        .value("SEEK", CardId::SEEK)
        .value("SELF_REPAIR", CardId::SELF_REPAIR)
        .value("SENTINEL", CardId::SENTINEL)
        .value("SETUP", CardId::SETUP)
        .value("SEVER_SOUL", CardId::SEVER_SOUL)
        .value("SHAME", CardId::SHAME)
        .value("SHIV", CardId::SHIV)
        .value("SHOCKWAVE", CardId::SHOCKWAVE)
        .value("SHRUG_IT_OFF", CardId::SHRUG_IT_OFF)
        .value("SIGNATURE_MOVE", CardId::SIGNATURE_MOVE)
        .value("SIMMERING_FURY", CardId::SIMMERING_FURY)
        .value("SKEWER", CardId::SKEWER)
        .value("SKIM", CardId::SKIM)
        .value("SLICE", CardId::SLICE)
        .value("SLIMED", CardId::SLIMED)
        .value("SMITE", CardId::SMITE)
        .value("SNEAKY_STRIKE", CardId::SNEAKY_STRIKE)
        .value("SPIRIT_SHIELD", CardId::SPIRIT_SHIELD)
        .value("SPOT_WEAKNESS", CardId::SPOT_WEAKNESS)
        .value("STACK", CardId::STACK)
        .value("STATIC_DISCHARGE", CardId::STATIC_DISCHARGE)
        .value("STEAM_BARRIER", CardId::STEAM_BARRIER)
        .value("STORM", CardId::STORM)
        .value("STORM_OF_STEEL", CardId::STORM_OF_STEEL)
        .value("STREAMLINE", CardId::STREAMLINE)
        .value("STRIKE_BLUE", CardId::STRIKE_BLUE)
        .value("STRIKE_GREEN", CardId::STRIKE_GREEN)
        .value("STRIKE_PURPLE", CardId::STRIKE_PURPLE)
        .value("STRIKE_RED", CardId::STRIKE_RED)
        .value("STUDY", CardId::STUDY)
        .value("SUCKER_PUNCH", CardId::SUCKER_PUNCH)
        .value("SUNDER", CardId::SUNDER)
        .value("SURVIVOR", CardId::SURVIVOR)
        .value("SWEEPING_BEAM", CardId::SWEEPING_BEAM)
        .value("SWIFT_STRIKE", CardId::SWIFT_STRIKE)
        .value("SWIVEL", CardId::SWIVEL)
        .value("SWORD_BOOMERANG", CardId::SWORD_BOOMERANG)
        .value("TACTICIAN", CardId::TACTICIAN)
        .value("TALK_TO_THE_HAND", CardId::TALK_TO_THE_HAND)
        .value("TANTRUM", CardId::TANTRUM)
        .value("TEMPEST", CardId::TEMPEST)
        .value("TERROR", CardId::TERROR)
        .value("THE_BOMB", CardId::THE_BOMB)
        .value("THINKING_AHEAD", CardId::THINKING_AHEAD)
        .value("THIRD_EYE", CardId::THIRD_EYE)
        .value("THROUGH_VIOLENCE", CardId::THROUGH_VIOLENCE)
        .value("THUNDERCLAP", CardId::THUNDERCLAP)
        .value("THUNDER_STRIKE", CardId::THUNDER_STRIKE)
        .value("TOOLS_OF_THE_TRADE", CardId::TOOLS_OF_THE_TRADE)
        .value("TRANQUILITY", CardId::TRANQUILITY)
        .value("TRANSMUTATION", CardId::TRANSMUTATION)
        .value("TRIP", CardId::TRIP)
        .value("TRUE_GRIT", CardId::TRUE_GRIT)
        .value("TURBO", CardId::TURBO)
        .value("TWIN_STRIKE", CardId::TWIN_STRIKE)
        .value("UNLOAD", CardId::UNLOAD)
        .value("UPPERCUT", CardId::UPPERCUT)
        .value("VAULT", CardId::VAULT)
        .value("VIGILANCE", CardId::VIGILANCE)
        .value("VIOLENCE", CardId::VIOLENCE)
        .value("VOID", CardId::VOID)
        .value("WALLOP", CardId::WALLOP)
        .value("WARCRY", CardId::WARCRY)
        .value("WAVE_OF_THE_HAND", CardId::WAVE_OF_THE_HAND)
        .value("WEAVE", CardId::WEAVE)
        .value("WELL_LAID_PLANS", CardId::WELL_LAID_PLANS)
        .value("WHEEL_KICK", CardId::WHEEL_KICK)
        .value("WHIRLWIND", CardId::WHIRLWIND)
        .value("WHITE_NOISE", CardId::WHITE_NOISE)
        .value("WILD_STRIKE", CardId::WILD_STRIKE)
        .value("WINDMILL_STRIKE", CardId::WINDMILL_STRIKE)
        .value("WISH", CardId::WISH)
        .value("WORSHIP", CardId::WORSHIP)
        .value("WOUND", CardId::WOUND)
        .value("WRAITH_FORM", CardId::WRAITH_FORM)
        .value("WREATH_OF_FLAME", CardId::WREATH_OF_FLAME)
        .value("WRITHE", CardId::WRITHE)
        .value("ZAP", CardId::ZAP);

    pybind11::enum_<MonsterEncounter> meEnum(m, "MonsterEncounter", pybind11::metaclass(enum_metaclass));
    meEnum.value("INVALID", ME::INVALID)
        .value("CULTIST", ME::CULTIST)
        .value("JAW_WORM", ME::JAW_WORM)
        .value("TWO_LOUSE", ME::TWO_LOUSE)
        .value("SMALL_SLIMES", ME::SMALL_SLIMES)
        .value("BLUE_SLAVER", ME::BLUE_SLAVER)
        .value("GREMLIN_GANG", ME::GREMLIN_GANG)
        .value("LOOTER", ME::LOOTER)
        .value("LARGE_SLIME", ME::LARGE_SLIME)
        .value("LOTS_OF_SLIMES", ME::LOTS_OF_SLIMES)
        .value("EXORDIUM_THUGS", ME::EXORDIUM_THUGS)
        .value("EXORDIUM_WILDLIFE", ME::EXORDIUM_WILDLIFE)
        .value("RED_SLAVER", ME::RED_SLAVER)
        .value("THREE_LOUSE", ME::THREE_LOUSE)
        .value("TWO_FUNGI_BEASTS", ME::TWO_FUNGI_BEASTS)
        .value("GREMLIN_NOB", ME::GREMLIN_NOB)
        .value("LAGAVULIN", ME::LAGAVULIN)
        .value("THREE_SENTRIES", ME::THREE_SENTRIES)
        .value("SLIME_BOSS", ME::SLIME_BOSS)
        .value("THE_GUARDIAN", ME::THE_GUARDIAN)
        .value("HEXAGHOST", ME::HEXAGHOST)
        .value("SPHERIC_GUARDIAN", ME::SPHERIC_GUARDIAN)
        .value("CHOSEN", ME::CHOSEN)
        .value("SHELL_PARASITE", ME::SHELL_PARASITE)
        .value("THREE_BYRDS", ME::THREE_BYRDS)
        .value("TWO_THIEVES", ME::TWO_THIEVES)
        .value("CHOSEN_AND_BYRDS", ME::CHOSEN_AND_BYRDS)
        .value("SENTRY_AND_SPHERE", ME::SENTRY_AND_SPHERE)
        .value("SNAKE_PLANT", ME::SNAKE_PLANT)
        .value("SNECKO", ME::SNECKO)
        .value("CENTURION_AND_HEALER", ME::CENTURION_AND_HEALER)
        .value("CULTIST_AND_CHOSEN", ME::CULTIST_AND_CHOSEN)
        .value("THREE_CULTIST", ME::THREE_CULTIST)
        .value("SHELLED_PARASITE_AND_FUNGI", ME::SHELLED_PARASITE_AND_FUNGI)
        .value("GREMLIN_LEADER", ME::GREMLIN_LEADER)
        .value("SLAVERS", ME::SLAVERS)
        .value("BOOK_OF_STABBING", ME::BOOK_OF_STABBING)
        .value("AUTOMATON", ME::AUTOMATON)
        .value("COLLECTOR", ME::COLLECTOR)
        .value("CHAMP", ME::CHAMP)
        .value("THREE_DARKLINGS", ME::THREE_DARKLINGS)
        .value("ORB_WALKER", ME::ORB_WALKER)
        .value("THREE_SHAPES", ME::THREE_SHAPES)
        .value("SPIRE_GROWTH", ME::SPIRE_GROWTH)
        .value("TRANSIENT", ME::TRANSIENT)
        .value("FOUR_SHAPES", ME::FOUR_SHAPES)
        .value("MAW", ME::MAW)
        .value("SPHERE_AND_TWO_SHAPES", ME::SPHERE_AND_TWO_SHAPES)
        .value("JAW_WORM_HORDE", ME::JAW_WORM_HORDE)
        .value("WRITHING_MASS", ME::WRITHING_MASS)
        .value("GIANT_HEAD", ME::GIANT_HEAD)
        .value("NEMESIS", ME::NEMESIS)
        .value("REPTOMANCER", ME::REPTOMANCER)
        .value("AWAKENED_ONE", ME::AWAKENED_ONE)
        .value("TIME_EATER", ME::TIME_EATER)
        .value("DONU_AND_DECA", ME::DONU_AND_DECA)
        .value("SHIELD_AND_SPEAR", ME::SHIELD_AND_SPEAR)
        .value("THE_HEART", ME::THE_HEART)
        .value("LAGAVULIN_EVENT", ME::LAGAVULIN_EVENT)
        .value("COLOSSEUM_EVENT_SLAVERS", ME::COLOSSEUM_EVENT_SLAVERS)
        .value("COLOSSEUM_EVENT_NOBS", ME::COLOSSEUM_EVENT_NOBS)
        .value("MASKED_BANDITS_EVENT", ME::MASKED_BANDITS_EVENT)
        .value("MUSHROOMS_EVENT", ME::MUSHROOMS_EVENT)
        .value("MYSTERIOUS_SPHERE_EVENT", ME::MYSTERIOUS_SPHERE_EVENT);

    // MonsterId enum binding
    pybind11::enum_<MonsterId> monsterIdEnum(m, "MonsterId", pybind11::metaclass(enum_metaclass));
    monsterIdEnum.value("INVALID", MonsterId::INVALID)
        .value("ACID_SLIME_L", MonsterId::ACID_SLIME_L)
        .value("ACID_SLIME_M", MonsterId::ACID_SLIME_M)
        .value("ACID_SLIME_S", MonsterId::ACID_SLIME_S)
        .value("AWAKENED_ONE", MonsterId::AWAKENED_ONE)
        .value("BEAR", MonsterId::BEAR)
        .value("BLUE_SLAVER", MonsterId::BLUE_SLAVER)
        .value("BOOK_OF_STABBING", MonsterId::BOOK_OF_STABBING)
        .value("BRONZE_AUTOMATON", MonsterId::BRONZE_AUTOMATON)
        .value("BRONZE_ORB", MonsterId::BRONZE_ORB)
        .value("BYRD", MonsterId::BYRD)
        .value("CENTURION", MonsterId::CENTURION)
        .value("CHOSEN", MonsterId::CHOSEN)
        .value("CORRUPT_HEART", MonsterId::CORRUPT_HEART)
        .value("CULTIST", MonsterId::CULTIST)
        .value("DAGGER", MonsterId::DAGGER)
        .value("DARKLING", MonsterId::DARKLING)
        .value("DECA", MonsterId::DECA)
        .value("DONU", MonsterId::DONU)
        .value("EXPLODER", MonsterId::EXPLODER)
        .value("FAT_GREMLIN", MonsterId::FAT_GREMLIN)
        .value("FUNGI_BEAST", MonsterId::FUNGI_BEAST)
        .value("GIANT_HEAD", MonsterId::GIANT_HEAD)
        .value("GREEN_LOUSE", MonsterId::GREEN_LOUSE)
        .value("GREMLIN_LEADER", MonsterId::GREMLIN_LEADER)
        .value("GREMLIN_NOB", MonsterId::GREMLIN_NOB)
        .value("GREMLIN_WIZARD", MonsterId::GREMLIN_WIZARD)
        .value("HEXAGHOST", MonsterId::HEXAGHOST)
        .value("JAW_WORM", MonsterId::JAW_WORM)
        .value("LAGAVULIN", MonsterId::LAGAVULIN)
        .value("LOOTER", MonsterId::LOOTER)
        .value("MAD_GREMLIN", MonsterId::MAD_GREMLIN)
        .value("MUGGER", MonsterId::MUGGER)
        .value("MYSTIC", MonsterId::MYSTIC)
        .value("NEMESIS", MonsterId::NEMESIS)
        .value("ORB_WALKER", MonsterId::ORB_WALKER)
        .value("POINTY", MonsterId::POINTY)
        .value("RED_LOUSE", MonsterId::RED_LOUSE)
        .value("RED_SLAVER", MonsterId::RED_SLAVER)
        .value("REPTOMANCER", MonsterId::REPTOMANCER)
        .value("REPULSOR", MonsterId::REPULSOR)
        .value("ROMEO", MonsterId::ROMEO)
        .value("SENTRY", MonsterId::SENTRY)
        .value("SHELLED_PARASITE", MonsterId::SHELLED_PARASITE)
        .value("SHIELD_GREMLIN", MonsterId::SHIELD_GREMLIN)
        .value("SLIME_BOSS", MonsterId::SLIME_BOSS)
        .value("SNAKE_PLANT", MonsterId::SNAKE_PLANT)
        .value("SNEAKY_GREMLIN", MonsterId::SNEAKY_GREMLIN)
        .value("SNECKO", MonsterId::SNECKO)
        .value("SPHERIC_GUARDIAN", MonsterId::SPHERIC_GUARDIAN)
        .value("SPIKER", MonsterId::SPIKER)
        .value("SPIKE_SLIME_L", MonsterId::SPIKE_SLIME_L)
        .value("SPIKE_SLIME_M", MonsterId::SPIKE_SLIME_M)
        .value("SPIKE_SLIME_S", MonsterId::SPIKE_SLIME_S)
        .value("SPIRE_GROWTH", MonsterId::SPIRE_GROWTH)
        .value("SPIRE_SHIELD", MonsterId::SPIRE_SHIELD)
        .value("SPIRE_SPEAR", MonsterId::SPIRE_SPEAR)
        .value("TASKMASTER", MonsterId::TASKMASTER)
        .value("THE_CHAMP", MonsterId::THE_CHAMP)
        .value("THE_COLLECTOR", MonsterId::THE_COLLECTOR)
        .value("THE_GUARDIAN", MonsterId::THE_GUARDIAN)
        .value("THE_MAW", MonsterId::THE_MAW)
        .value("TIME_EATER", MonsterId::TIME_EATER)
        .value("TORCH_HEAD", MonsterId::TORCH_HEAD)
        .value("TRANSIENT", MonsterId::TRANSIENT)
        .value("WRITHING_MASS", MonsterId::WRITHING_MASS);

    pybind11::enum_<RelicId> relicEnum(m, "RelicId", pybind11::metaclass(enum_metaclass));
    relicEnum.value("AKABEKO", RelicId::AKABEKO)
        .value("ART_OF_WAR", RelicId::ART_OF_WAR)
        .value("BIRD_FACED_URN", RelicId::BIRD_FACED_URN)
        .value("BLOODY_IDOL", RelicId::BLOODY_IDOL)
        .value("BLUE_CANDLE", RelicId::BLUE_CANDLE)
        .value("BRIMSTONE", RelicId::BRIMSTONE)
        .value("CALIPERS", RelicId::CALIPERS)
        .value("CAPTAINS_WHEEL", RelicId::CAPTAINS_WHEEL)
        .value("CENTENNIAL_PUZZLE", RelicId::CENTENNIAL_PUZZLE)
        .value("CERAMIC_FISH", RelicId::CERAMIC_FISH)
        .value("CHAMPION_BELT", RelicId::CHAMPION_BELT)
        .value("CHARONS_ASHES", RelicId::CHARONS_ASHES)
        .value("CHEMICAL_X", RelicId::CHEMICAL_X)
        .value("CLOAK_CLASP", RelicId::CLOAK_CLASP)
        .value("DARKSTONE_PERIAPT", RelicId::DARKSTONE_PERIAPT)
        .value("DEAD_BRANCH", RelicId::DEAD_BRANCH)
        .value("DUALITY", RelicId::DUALITY)
        .value("ECTOPLASM", RelicId::ECTOPLASM)
        .value("EMOTION_CHIP", RelicId::EMOTION_CHIP)
        .value("FROZEN_CORE", RelicId::FROZEN_CORE)
        .value("FROZEN_EYE", RelicId::FROZEN_EYE)
        .value("GAMBLING_CHIP", RelicId::GAMBLING_CHIP)
        .value("GINGER", RelicId::GINGER)
        .value("GOLDEN_EYE", RelicId::GOLDEN_EYE)
        .value("GREMLIN_HORN", RelicId::GREMLIN_HORN)
        .value("HAND_DRILL", RelicId::HAND_DRILL)
        .value("HAPPY_FLOWER", RelicId::HAPPY_FLOWER)
        .value("HORN_CLEAT", RelicId::HORN_CLEAT)
        .value("HOVERING_KITE", RelicId::HOVERING_KITE)
        .value("ICE_CREAM", RelicId::ICE_CREAM)
        .value("INCENSE_BURNER", RelicId::INCENSE_BURNER)
        .value("INK_BOTTLE", RelicId::INK_BOTTLE)
        .value("INSERTER", RelicId::INSERTER)
        .value("KUNAI", RelicId::KUNAI)
        .value("LETTER_OPENER", RelicId::LETTER_OPENER)
        .value("LIZARD_TAIL", RelicId::LIZARD_TAIL)
        .value("MAGIC_FLOWER", RelicId::MAGIC_FLOWER)
        .value("MARK_OF_THE_BLOOM", RelicId::MARK_OF_THE_BLOOM)
        .value("MEDICAL_KIT", RelicId::MEDICAL_KIT)
        .value("MELANGE", RelicId::MELANGE)
        .value("MERCURY_HOURGLASS", RelicId::MERCURY_HOURGLASS)
        .value("MUMMIFIED_HAND", RelicId::MUMMIFIED_HAND)
        .value("NECRONOMICON", RelicId::NECRONOMICON)
        .value("NILRYS_CODEX", RelicId::NILRYS_CODEX)
        .value("NUNCHAKU", RelicId::NUNCHAKU)
        .value("ODD_MUSHROOM", RelicId::ODD_MUSHROOM)
        .value("OMAMORI", RelicId::OMAMORI)
        .value("ORANGE_PELLETS", RelicId::ORANGE_PELLETS)
        .value("ORICHALCUM", RelicId::ORICHALCUM)
        .value("ORNAMENTAL_FAN", RelicId::ORNAMENTAL_FAN)
        .value("PAPER_KRANE", RelicId::PAPER_KRANE)
        .value("PAPER_PHROG", RelicId::PAPER_PHROG)
        .value("PEN_NIB", RelicId::PEN_NIB)
        .value("PHILOSOPHERS_STONE", RelicId::PHILOSOPHERS_STONE)
        .value("POCKETWATCH", RelicId::POCKETWATCH)
        .value("RED_SKULL", RelicId::RED_SKULL)
        .value("RUNIC_CUBE", RelicId::RUNIC_CUBE)
        .value("RUNIC_DOME", RelicId::RUNIC_DOME)
        .value("RUNIC_PYRAMID", RelicId::RUNIC_PYRAMID)
        .value("SACRED_BARK", RelicId::SACRED_BARK)
        .value("SELF_FORMING_CLAY", RelicId::SELF_FORMING_CLAY)
        .value("SHURIKEN", RelicId::SHURIKEN)
        .value("SNECKO_EYE", RelicId::SNECKO_EYE)
        .value("SNECKO_SKULL", RelicId::SNECKO_SKULL)
        .value("SOZU", RelicId::SOZU)
        .value("STONE_CALENDAR", RelicId::STONE_CALENDAR)
        .value("STRANGE_SPOON", RelicId::STRANGE_SPOON)
        .value("STRIKE_DUMMY", RelicId::STRIKE_DUMMY)
        .value("SUNDIAL", RelicId::SUNDIAL)
        .value("THE_ABACUS", RelicId::THE_ABACUS)
        .value("THE_BOOT", RelicId::THE_BOOT)
        .value("THE_SPECIMEN", RelicId::THE_SPECIMEN)
        .value("TINGSHA", RelicId::TINGSHA)
        .value("TOOLBOX", RelicId::TOOLBOX)
        .value("TORII", RelicId::TORII)
        .value("TOUGH_BANDAGES", RelicId::TOUGH_BANDAGES)
        .value("TOY_ORNITHOPTER", RelicId::TOY_ORNITHOPTER)
        .value("TUNGSTEN_ROD", RelicId::TUNGSTEN_ROD)
        .value("TURNIP", RelicId::TURNIP)
        .value("TWISTED_FUNNEL", RelicId::TWISTED_FUNNEL)
        .value("UNCEASING_TOP", RelicId::UNCEASING_TOP)
        .value("VELVET_CHOKER", RelicId::VELVET_CHOKER)
        .value("VIOLET_LOTUS", RelicId::VIOLET_LOTUS)
        .value("WARPED_TONGS", RelicId::WARPED_TONGS)
        .value("WRIST_BLADE", RelicId::WRIST_BLADE)
        .value("BLACK_BLOOD", RelicId::BLACK_BLOOD)
        .value("BURNING_BLOOD", RelicId::BURNING_BLOOD)
        .value("MEAT_ON_THE_BONE", RelicId::MEAT_ON_THE_BONE)
        .value("FACE_OF_CLERIC", RelicId::FACE_OF_CLERIC)
        .value("ANCHOR", RelicId::ANCHOR)
        .value("ANCIENT_TEA_SET", RelicId::ANCIENT_TEA_SET)
        .value("BAG_OF_MARBLES", RelicId::BAG_OF_MARBLES)
        .value("BAG_OF_PREPARATION", RelicId::BAG_OF_PREPARATION)
        .value("BLOOD_VIAL", RelicId::BLOOD_VIAL)
        .value("BOTTLED_FLAME", RelicId::BOTTLED_FLAME)
        .value("BOTTLED_LIGHTNING", RelicId::BOTTLED_LIGHTNING)
        .value("BOTTLED_TORNADO", RelicId::BOTTLED_TORNADO)
        .value("BRONZE_SCALES", RelicId::BRONZE_SCALES)
        .value("BUSTED_CROWN", RelicId::BUSTED_CROWN)
        .value("CLOCKWORK_SOUVENIR", RelicId::CLOCKWORK_SOUVENIR)
        .value("COFFEE_DRIPPER", RelicId::COFFEE_DRIPPER)
        .value("CRACKED_CORE", RelicId::CRACKED_CORE)
        .value("CURSED_KEY", RelicId::CURSED_KEY)
        .value("DAMARU", RelicId::DAMARU)
        .value("DATA_DISK", RelicId::DATA_DISK)
        .value("DU_VU_DOLL", RelicId::DU_VU_DOLL)
        .value("ENCHIRIDION", RelicId::ENCHIRIDION)
        .value("FOSSILIZED_HELIX", RelicId::FOSSILIZED_HELIX)
        .value("FUSION_HAMMER", RelicId::FUSION_HAMMER)
        .value("GIRYA", RelicId::GIRYA)
        .value("GOLD_PLATED_CABLES", RelicId::GOLD_PLATED_CABLES)
        .value("GREMLIN_VISAGE", RelicId::GREMLIN_VISAGE)
        .value("HOLY_WATER", RelicId::HOLY_WATER)
        .value("LANTERN", RelicId::LANTERN)
        .value("MARK_OF_PAIN", RelicId::MARK_OF_PAIN)
        .value("MUTAGENIC_STRENGTH", RelicId::MUTAGENIC_STRENGTH)
        .value("NEOWS_LAMENT", RelicId::NEOWS_LAMENT)
        .value("NINJA_SCROLL", RelicId::NINJA_SCROLL)
        .value("NUCLEAR_BATTERY", RelicId::NUCLEAR_BATTERY)
        .value("ODDLY_SMOOTH_STONE", RelicId::ODDLY_SMOOTH_STONE)
        .value("PANTOGRAPH", RelicId::PANTOGRAPH)
        .value("PRESERVED_INSECT", RelicId::PRESERVED_INSECT)
        .value("PURE_WATER", RelicId::PURE_WATER)
        .value("RED_MASK", RelicId::RED_MASK)
        .value("RING_OF_THE_SERPENT", RelicId::RING_OF_THE_SERPENT)
        .value("RING_OF_THE_SNAKE", RelicId::RING_OF_THE_SNAKE)
        .value("RUNIC_CAPACITOR", RelicId::RUNIC_CAPACITOR)
        .value("SLAVERS_COLLAR", RelicId::SLAVERS_COLLAR)
        .value("SLING_OF_COURAGE", RelicId::SLING_OF_COURAGE)
        .value("SYMBIOTIC_VIRUS", RelicId::SYMBIOTIC_VIRUS)
        .value("TEARDROP_LOCKET", RelicId::TEARDROP_LOCKET)
        .value("THREAD_AND_NEEDLE", RelicId::THREAD_AND_NEEDLE)
        .value("VAJRA", RelicId::VAJRA)
        .value("ASTROLABE", RelicId::ASTROLABE)
        .value("BLACK_STAR", RelicId::BLACK_STAR)
        .value("CALLING_BELL", RelicId::CALLING_BELL)
        .value("CAULDRON", RelicId::CAULDRON)
        .value("CULTIST_HEADPIECE", RelicId::CULTIST_HEADPIECE)
        .value("DOLLYS_MIRROR", RelicId::DOLLYS_MIRROR)
        .value("DREAM_CATCHER", RelicId::DREAM_CATCHER)
        .value("EMPTY_CAGE", RelicId::EMPTY_CAGE)
        .value("ETERNAL_FEATHER", RelicId::ETERNAL_FEATHER)
        .value("FROZEN_EGG", RelicId::FROZEN_EGG)
        .value("GOLDEN_IDOL", RelicId::GOLDEN_IDOL)
        .value("JUZU_BRACELET", RelicId::JUZU_BRACELET)
        .value("LEES_WAFFLE", RelicId::LEES_WAFFLE)
        .value("MANGO", RelicId::MANGO)
        .value("MATRYOSHKA", RelicId::MATRYOSHKA)
        .value("MAW_BANK", RelicId::MAW_BANK)
        .value("MEAL_TICKET", RelicId::MEAL_TICKET)
        .value("MEMBERSHIP_CARD", RelicId::MEMBERSHIP_CARD)
        .value("MOLTEN_EGG", RelicId::MOLTEN_EGG)
        .value("NLOTHS_GIFT", RelicId::NLOTHS_GIFT)
        .value("NLOTHS_HUNGRY_FACE", RelicId::NLOTHS_HUNGRY_FACE)
        .value("OLD_COIN", RelicId::OLD_COIN)
        .value("ORRERY", RelicId::ORRERY)
        .value("PANDORAS_BOX", RelicId::PANDORAS_BOX)
        .value("PEACE_PIPE", RelicId::PEACE_PIPE)
        .value("PEAR", RelicId::PEAR)
        .value("POTION_BELT", RelicId::POTION_BELT)
        .value("PRAYER_WHEEL", RelicId::PRAYER_WHEEL)
        .value("PRISMATIC_SHARD", RelicId::PRISMATIC_SHARD)
        .value("QUESTION_CARD", RelicId::QUESTION_CARD)
        .value("REGAL_PILLOW", RelicId::REGAL_PILLOW)
        .value("SSSERPENT_HEAD", RelicId::SSSERPENT_HEAD)
        .value("SHOVEL", RelicId::SHOVEL)
        .value("SINGING_BOWL", RelicId::SINGING_BOWL)
        .value("SMILING_MASK", RelicId::SMILING_MASK)
        .value("SPIRIT_POOP", RelicId::SPIRIT_POOP)
        .value("STRAWBERRY", RelicId::STRAWBERRY)
        .value("THE_COURIER", RelicId::THE_COURIER)
        .value("TINY_CHEST", RelicId::TINY_CHEST)
        .value("TINY_HOUSE", RelicId::TINY_HOUSE)
        .value("TOXIC_EGG", RelicId::TOXIC_EGG)
        .value("WAR_PAINT", RelicId::WAR_PAINT)
        .value("WHETSTONE", RelicId::WHETSTONE)
        .value("WHITE_BEAST_STATUE", RelicId::WHITE_BEAST_STATUE)
        .value("WING_BOOTS", RelicId::WING_BOOTS)
        .value("CIRCLET", RelicId::CIRCLET)
        .value("RED_CIRCLET", RelicId::RED_CIRCLET)
        .value("INVALID", RelicId::INVALID);

    pybind11::enum_<Potion> potionEnum(m, "Potion", pybind11::metaclass(enum_metaclass));
    potionEnum.value("INVALID", Potion::INVALID)
        .value("EMPTY_POTION_SLOT", Potion::EMPTY_POTION_SLOT)
        .value("AMBROSIA", Potion::AMBROSIA)
        .value("ANCIENT_POTION", Potion::ANCIENT_POTION)
        .value("ATTACK_POTION", Potion::ATTACK_POTION)
        .value("BLESSING_OF_THE_FORGE", Potion::BLESSING_OF_THE_FORGE)
        .value("BLOCK_POTION", Potion::BLOCK_POTION)
        .value("BLOOD_POTION", Potion::BLOOD_POTION)
        .value("BOTTLED_MIRACLE", Potion::BOTTLED_MIRACLE)
        .value("COLORLESS_POTION", Potion::COLORLESS_POTION)
        .value("CULTIST_POTION", Potion::CULTIST_POTION)
        .value("CUNNING_POTION", Potion::CUNNING_POTION)
        .value("DEXTERITY_POTION", Potion::DEXTERITY_POTION)
        .value("DISTILLED_CHAOS", Potion::DISTILLED_CHAOS)
        .value("DUPLICATION_POTION", Potion::DUPLICATION_POTION)
        .value("ELIXIR_POTION", Potion::ELIXIR_POTION)
        .value("ENERGY_POTION", Potion::ENERGY_POTION)
        .value("ENTROPIC_BREW", Potion::ENTROPIC_BREW)
        .value("ESSENCE_OF_DARKNESS", Potion::ESSENCE_OF_DARKNESS)
        .value("ESSENCE_OF_STEEL", Potion::ESSENCE_OF_STEEL)
        .value("EXPLOSIVE_POTION", Potion::EXPLOSIVE_POTION)
        .value("FAIRY_POTION", Potion::FAIRY_POTION)
        .value("FEAR_POTION", Potion::FEAR_POTION)
        .value("FIRE_POTION", Potion::FIRE_POTION)
        .value("FLEX_POTION", Potion::FLEX_POTION)
        .value("FOCUS_POTION", Potion::FOCUS_POTION)
        .value("FRUIT_JUICE", Potion::FRUIT_JUICE)
        .value("GAMBLERS_BREW", Potion::GAMBLERS_BREW)
        .value("GHOST_IN_A_JAR", Potion::GHOST_IN_A_JAR)
        .value("HEART_OF_IRON", Potion::HEART_OF_IRON)
        .value("LIQUID_BRONZE", Potion::LIQUID_BRONZE)
        .value("LIQUID_MEMORIES", Potion::LIQUID_MEMORIES)
        .value("POISON_POTION", Potion::POISON_POTION)
        .value("POTION_OF_CAPACITY", Potion::POTION_OF_CAPACITY)
        .value("POWER_POTION", Potion::POWER_POTION)
        .value("REGEN_POTION", Potion::REGEN_POTION)
        .value("SKILL_POTION", Potion::SKILL_POTION)
        .value("SMOKE_BOMB", Potion::SMOKE_BOMB)
        .value("SNECKO_OIL", Potion::SNECKO_OIL)
        .value("SPEED_POTION", Potion::SPEED_POTION)
        .value("STANCE_POTION", Potion::STANCE_POTION)
        .value("STRENGTH_POTION", Potion::STRENGTH_POTION)
        .value("SWIFT_POTION", Potion::SWIFT_POTION)
        .value("WEAK_POTION", Potion::WEAK_POTION);

    // PlayerStatus enum bindings
    pybind11::enum_<PlayerStatus> playerStatus(m, "PlayerStatus", pybind11::metaclass(enum_metaclass));
    playerStatus.value("INVALID", PlayerStatus::INVALID)
        .value("DOUBLE_DAMAGE", PlayerStatus::DOUBLE_DAMAGE)
        .value("DRAW_REDUCTION", PlayerStatus::DRAW_REDUCTION)
        .value("FRAIL", PlayerStatus::FRAIL)
        .value("INTANGIBLE", PlayerStatus::INTANGIBLE)
        .value("VULNERABLE", PlayerStatus::VULNERABLE)
        .value("WEAK", PlayerStatus::WEAK)
        .value("BIAS", PlayerStatus::BIAS)
        .value("CONFUSED", PlayerStatus::CONFUSED)
        .value("CONSTRICTED", PlayerStatus::CONSTRICTED)
        .value("ENTANGLED", PlayerStatus::ENTANGLED)
        .value("FASTING", PlayerStatus::FASTING)
        .value("HEX", PlayerStatus::HEX)
        .value("LOSE_DEXTERITY", PlayerStatus::LOSE_DEXTERITY)
        .value("LOSE_STRENGTH", PlayerStatus::LOSE_STRENGTH)
        .value("NO_BLOCK", PlayerStatus::NO_BLOCK)
        .value("NO_DRAW", PlayerStatus::NO_DRAW)
        .value("WRAITH_FORM", PlayerStatus::WRAITH_FORM)
        .value("BARRICADE", PlayerStatus::BARRICADE)
        .value("BLASPHEMER", PlayerStatus::BLASPHEMER)
        .value("CORRUPTION", PlayerStatus::CORRUPTION)
        .value("ELECTRO", PlayerStatus::ELECTRO)
        .value("SURROUNDED", PlayerStatus::SURROUNDED)
        .value("MASTER_REALITY", PlayerStatus::MASTER_REALITY)
        .value("PEN_NIB", PlayerStatus::PEN_NIB)
        .value("WRATH_NEXT_TURN", PlayerStatus::WRATH_NEXT_TURN)
        .value("AMPLIFY", PlayerStatus::AMPLIFY)
        .value("BLUR", PlayerStatus::BLUR)
        .value("BUFFER", PlayerStatus::BUFFER)
        .value("COLLECT", PlayerStatus::COLLECT)
        .value("DOUBLE_TAP", PlayerStatus::DOUBLE_TAP)
        .value("DUPLICATION", PlayerStatus::DUPLICATION)
        .value("ECHO_FORM", PlayerStatus::ECHO_FORM)
        .value("FREE_ATTACK_POWER", PlayerStatus::FREE_ATTACK_POWER)
        .value("REBOUND", PlayerStatus::REBOUND)
        .value("MANTRA", PlayerStatus::MANTRA)
        .value("ACCURACY", PlayerStatus::ACCURACY)
        .value("AFTER_IMAGE", PlayerStatus::AFTER_IMAGE)
        .value("BATTLE_HYMN", PlayerStatus::BATTLE_HYMN)
        .value("BRUTALITY", PlayerStatus::BRUTALITY)
        .value("BURST", PlayerStatus::BURST)
        .value("COMBUST", PlayerStatus::COMBUST)
        .value("CREATIVE_AI", PlayerStatus::CREATIVE_AI)
        .value("DARK_EMBRACE", PlayerStatus::DARK_EMBRACE)
        .value("DEMON_FORM", PlayerStatus::DEMON_FORM)
        .value("DEVA", PlayerStatus::DEVA)
        .value("DEVOTION", PlayerStatus::DEVOTION)
        .value("DRAW_CARD_NEXT_TURN", PlayerStatus::DRAW_CARD_NEXT_TURN)
        .value("ENERGIZED", PlayerStatus::ENERGIZED)
        .value("ENVENOM", PlayerStatus::ENVENOM)
        .value("ESTABLISHMENT", PlayerStatus::ESTABLISHMENT)
        .value("EVOLVE", PlayerStatus::EVOLVE)
        .value("FEEL_NO_PAIN", PlayerStatus::FEEL_NO_PAIN)
        .value("FIRE_BREATHING", PlayerStatus::FIRE_BREATHING)
        .value("FLAME_BARRIER", PlayerStatus::FLAME_BARRIER)
        .value("FOCUS", PlayerStatus::FOCUS)
        .value("FORESIGHT", PlayerStatus::FORESIGHT)
        .value("HELLO_WORLD", PlayerStatus::HELLO_WORLD)
        .value("INFINITE_BLADES", PlayerStatus::INFINITE_BLADES)
        .value("JUGGERNAUT", PlayerStatus::JUGGERNAUT)
        .value("LIKE_WATER", PlayerStatus::LIKE_WATER)
        .value("LOOP", PlayerStatus::LOOP)
        .value("MAGNETISM", PlayerStatus::MAGNETISM)
        .value("MAYHEM", PlayerStatus::MAYHEM)
        .value("METALLICIZE", PlayerStatus::METALLICIZE)
        .value("NEXT_TURN_BLOCK", PlayerStatus::NEXT_TURN_BLOCK)
        .value("NOXIOUS_FUMES", PlayerStatus::NOXIOUS_FUMES)
        .value("OMEGA", PlayerStatus::OMEGA)
        .value("PANACHE", PlayerStatus::PANACHE)
        .value("PHANTASMAL", PlayerStatus::PHANTASMAL)
        .value("PLATED_ARMOR", PlayerStatus::PLATED_ARMOR)
        .value("RAGE", PlayerStatus::RAGE)
        .value("REGEN", PlayerStatus::REGEN)
        .value("RITUAL", PlayerStatus::RITUAL)
        .value("RUPTURE", PlayerStatus::RUPTURE)
        .value("SADISTIC", PlayerStatus::SADISTIC)
        .value("STATIC_DISCHARGE", PlayerStatus::STATIC_DISCHARGE)
        .value("THORNS", PlayerStatus::THORNS)
        .value("THOUSAND_CUTS", PlayerStatus::THOUSAND_CUTS)
        .value("TOOLS_OF_THE_TRADE", PlayerStatus::TOOLS_OF_THE_TRADE)
        .value("VIGOR", PlayerStatus::VIGOR)
        .value("WAVE_OF_THE_HAND", PlayerStatus::WAVE_OF_THE_HAND)
        .value("EQUILIBRIUM", PlayerStatus::EQUILIBRIUM)
        .value("ARTIFACT", PlayerStatus::ARTIFACT)
        .value("DEXTERITY", PlayerStatus::DEXTERITY)
        .value("STRENGTH", PlayerStatus::STRENGTH)
        .value("THE_BOMB", PlayerStatus::THE_BOMB);

    // MonsterStatus enum bindings
    pybind11::enum_<MonsterStatus> monsterStatus(m, "MonsterStatus", pybind11::metaclass(enum_metaclass));
    monsterStatus.value("ARTIFACT", MonsterStatus::ARTIFACT)
        .value("BLOCK_RETURN", MonsterStatus::BLOCK_RETURN)
        .value("CHOKED", MonsterStatus::CHOKED)
        .value("CORPSE_EXPLOSION", MonsterStatus::CORPSE_EXPLOSION)
        .value("LOCK_ON", MonsterStatus::LOCK_ON)
        .value("MARK", MonsterStatus::MARK)
        .value("METALLICIZE", MonsterStatus::METALLICIZE)
        .value("MINION", MonsterStatus::MINION)
        .value("MINION_LEADER", MonsterStatus::MINION_LEADER)
        .value("PAINFUL_STABS", MonsterStatus::PAINFUL_STABS)
        .value("PLATED_ARMOR", MonsterStatus::PLATED_ARMOR)
        .value("POISON", MonsterStatus::POISON)
        .value("REGEN", MonsterStatus::REGEN)
        .value("REGROW", MonsterStatus::REGROW)
        .value("SHIFTING", MonsterStatus::SHIFTING)
        .value("STASIS", MonsterStatus::STASIS)
        .value("ASLEEP", MonsterStatus::ASLEEP)
        .value("BARRICADE", MonsterStatus::BARRICADE)
        .value("SHACKLED", MonsterStatus::SHACKLED)
        .value("STRENGTH", MonsterStatus::STRENGTH)
        .value("VULNERABLE", MonsterStatus::VULNERABLE)
        .value("WEAK", MonsterStatus::WEAK)
        .value("ANGRY", MonsterStatus::ANGRY)
        .value("BEAT_OF_DEATH", MonsterStatus::BEAT_OF_DEATH)
        .value("CURIOSITY", MonsterStatus::CURIOSITY)
        .value("CURL_UP", MonsterStatus::CURL_UP)
        .value("ENRAGE", MonsterStatus::ENRAGE)
        .value("FADING", MonsterStatus::FADING)
        .value("FLIGHT", MonsterStatus::FLIGHT)
        .value("GENERIC_STRENGTH_UP", MonsterStatus::GENERIC_STRENGTH_UP)
        .value("INTANGIBLE", MonsterStatus::INTANGIBLE)
        .value("MALLEABLE", MonsterStatus::MALLEABLE)
        .value("MODE_SHIFT", MonsterStatus::MODE_SHIFT)
        .value("RITUAL", MonsterStatus::RITUAL)
        .value("SLOW", MonsterStatus::SLOW)
        .value("SPORE_CLOUD", MonsterStatus::SPORE_CLOUD)
        .value("THIEVERY", MonsterStatus::THIEVERY)
        .value("THORNS", MonsterStatus::THORNS)
        .value("TIME_WARP", MonsterStatus::TIME_WARP)
        .value("INVINCIBLE", MonsterStatus::INVINCIBLE)
        .value("REACTIVE", MonsterStatus::REACTIVE)
        .value("SHARP_HIDE", MonsterStatus::SHARP_HIDE);

    // Stance enum binding
    pybind11::enum_<Stance> stance(m, "Stance", pybind11::metaclass(enum_metaclass));
    stance.value("NEUTRAL", Stance::NEUTRAL)
        .value("CALM", Stance::CALM)
        .value("WRATH", Stance::WRATH)
        .value("DIVINITY", Stance::DIVINITY);

    // Lookup functions for dynamic enum conversion
    m.def("getCardName", &sts::getCardName, "Get card name by CardId");
    m.def("getCardStringId", &sts::getCardStringId, "Get card string ID by CardId");
    m.def("getRelicName", &sts::getRelicName, "Get relic name by RelicId");
    m.def("getRelicId", &sts::getRelicId, "Get relic string ID by RelicId");
    m.def("getMonsterName", &sts::getMonsterName, "Get monster name by MonsterId");
    m.def("getMonsterIdString", &sts::getMonsterIdString, "Get monster string ID by MonsterId");
    m.def("getPlayerStatusForString", &SimHelpers::getPlayerStatusForString, "Get PlayerStatus enum from string name");
    
    // Array access functions for efficient reverse lookup
    m.def("getAllCardStringIds", []() {
        std::vector<std::pair<int, std::string>> result;
        constexpr int card_count = sizeof(sts::cardStringIds) / sizeof(sts::cardStringIds[0]);
        for (int i = 0; i < card_count; i++) {
            result.emplace_back(i, sts::getCardStringId(static_cast<sts::CardId>(i)));
        }
        return result;
    }, "Get all card string IDs with their enum indices");
    
    m.def("getAllRelicNames", []() {
        std::vector<std::pair<int, std::string>> result;
        constexpr int relic_count = sizeof(sts::relicNames) / sizeof(sts::relicNames[0]);
        for (int i = 0; i < relic_count; i++) {
            result.emplace_back(i, sts::getRelicName(static_cast<sts::RelicId>(i)));
        }
        return result;
    }, "Get all relic names with their enum indices");
    
    m.def("getAllMonsterStringIds", []() {
        std::vector<std::pair<int, std::string>> result;
        constexpr int monster_count = sizeof(sts::monsterIdStrings) / sizeof(sts::monsterIdStrings[0]);
        for (int i = 0; i < monster_count; i++) {
            result.emplace_back(i, sts::getMonsterIdString(static_cast<sts::MonsterId>(i)));
        }
        return result;
    }, "Get all monster string IDs with their enum indices");

    m.attr("MAX_POTION_CAPACITY") = MAX_POTION_CAPACITY;

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

// os.add_dll_directory("C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin")
