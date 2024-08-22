//
// Created by keega on 9/24/2021.
//
#include <sstream>
#include <algorithm>

#include "constants/Rooms.h"
#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "game/Map.h"

#include "slaythespire.h"

namespace sts::py {

    pybind11::array_t<int> getFixedObservation(const GameContext &gc) {
        std::vector<int> ret(fixed_observation_space_size);

        int offset = 0;

        ret[offset++] = std::min(gc.curHp, playerHpMax);
        ret[offset++] = std::min(gc.maxHp, playerHpMax);
        ret[offset++] = std::min(gc.gold, playerGoldMax);
        ret[offset++] = gc.floorNum;
        ret[offset++] = getBossEncoding(gc.boss);

        return to_numpy(ret);
    }

    pybind11::array_t<int> getFixedObservationMaximums() {
        std::vector<int> ret(fixed_observation_space_size);
        int spaceOffset = 0;

        ret[0] = playerHpMax;
        ret[1] = playerHpMax;
        ret[2] = playerGoldMax;
        ret[3] = 60;
        ret[4] = numBosses;

        return to_numpy(ret);
    }

    NNCardsRepresentation getCardRepresentation(const Deck &deck) {
        std::vector<CardId> cards;
        std::vector<int> upgrades;
        for (int i = 0; i < deck.size(); ++i) {
            cards.push_back(deck.cards[i].id);
            upgrades.push_back(deck.cards[i].getUpgraded());
        }
        return NNCardsRepresentation {
            .cards = to_numpy(cards),
            .upgrades = to_numpy(upgrades)
        };
    }

    NNRelicsRepresentation getRelicRepresentation(const RelicContainer &relics) {
        std::vector<RelicId> relicIds;
        std::vector<int> relicCounters;
        for (int i = 0; i < relics.size(); ++i) {
            relicIds.push_back(relics.relics[i].id);
            relicCounters.push_back(relics.relics[i].data);
        }
        return NNRelicsRepresentation {
            .relics = to_numpy(relicIds),
            .relicCounters = to_numpy(relicCounters)
        };
    }


    NNRepresentation getNNRepresentation(const GameContext &gc) {
        NNRepresentation rep;
        rep.fixedObservation = getFixedObservation(gc);
        rep.deck = getCardRepresentation(gc.deck);
        rep.relics = getRelicRepresentation(gc.relics);
        rep.map = getNNMapRepresentation(*gc.map);
        rep.mapX = gc.curMapNodeX;
        rep.mapY = gc.curMapNodeY;
        return rep;
    }

    int getBossEncoding(MonsterEncounter boss) {
        switch (boss) {
            case ME::SLIME_BOSS:
                return 0;
            case ME::HEXAGHOST:
                return 1;
            case ME::THE_GUARDIAN:
                return 2;
            case ME::CHAMP:
                return 3;
            case ME::AUTOMATON:
                return 4;
            case ME::COLLECTOR:
                return 5;
            case ME::TIME_EATER:
                return 6;
            case ME::DONU_AND_DECA:
                return 7;
            case ME::AWAKENED_ONE:
                return 8;
            case ME::THE_HEART:
                return 9;
            default:
                assert(false);
        }
    }

    void play() {
        sts::SimulatorContext ctx;
        sts::ConsoleSimulator sim;
        sim.play(std::cin, std::cout, ctx);
    }

    search::ScumSearchAgent2* getAgent() {
        static search::ScumSearchAgent2 *agent = nullptr;
        if (agent == nullptr) {
            agent = new search::ScumSearchAgent2();
            agent->pauseOnCardReward = true;
        }
        return agent;
    }

    void playout(GameContext &gc) {
        auto agent = getAgent();
        agent->playout(gc);
    }

    std::vector<Card> getCardReward(GameContext &gc) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;

        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return {};
        }

        const auto &r = gc.info.rewardsContainer;
        const auto &cardList = r.cardRewards[r.cardRewardCount-1];
        return std::vector<Card>(cardList.begin(), cardList.end());
    }

    // BEGIN MAP THINGS ****************************

    NNMapRepresentation getNNMapRepresentation(const Map &map) {
        std::array<std::array<int, 7>, 16> ids;
        int id = 0;
        bool haveLastRow = false;
        std::vector<int> xs, ys, edgeStarts, edgeEnds;
        std::vector<Room> roomTypes;

        // First pass: collect data
        for (int y = 0; y < 15; ++y) {
            for (int x = 0; x < 7; ++x) {
                const MapNode& node = map.getNode(x,y);
                if (node.room != Room::NONE) {
                    ids[y][x] = id++;
                    roomTypes.push_back(node.room);
                    xs.push_back(x);
                    ys.push_back(y);
                    if (y == 14) {
                        haveLastRow = true;
                    }
                }
            }
        }
        if (haveLastRow) {
            ids[15][3] = id++; // boss
            roomTypes.push_back(Room::BOSS);
            xs.push_back(3);
            ys.push_back(15);
        }

        for (int y = 0; y < 15; ++y) {
            for (int x = 0; x < 7; ++x) {
                const MapNode& node = map.getNode(x,y);
                if (node.room != Room::NONE) {
                    for (int k = 0; k < node.edgeCount; ++k) {
                        edgeStarts.push_back(ids[y][x]);
                        edgeEnds.push_back(ids[y+1][node.edges[k]]);
                    }
                }
            }
        }

        // Create numpy arrays from collected data
        return NNMapRepresentation {
            .xs = to_numpy(xs),
            .ys = to_numpy(ys),
            .roomTypes = to_numpy(roomTypes),
            .edgeStarts = to_numpy(edgeStarts),
            .edgeEnds = to_numpy(edgeEnds)
        };
    }

    Room getRoomType(const Map &map, int x, int y) {
        if (x < 0 || x > 6 || y < 0 || y > 14) {
            return Room::INVALID;
        }

        return map.getNode(x,y).room;
    }

    bool hasEdge(const Map &map, int x, int y, int x2) {
        if (x == -1) {
            return map.getNode(x2,0).edgeCount > 0;
        }

        if (x < 0 || x > 6 || y < 0 || y > 14) {
            return false;
        }


        auto node = map.getNode(x,y);
        for (int i = 0; i < node.edgeCount; ++i) {
            if (node.edges[i] == x2) {
                return true;
            }
        }
        return false;
    }

}
