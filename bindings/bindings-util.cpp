//
// Created by keega on 9/24/2021.
//
#include <sstream>
#include <algorithm>

#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "game/Map.h"

#include "slaythespire.h"

namespace sts::py {

    std::array<int,fixed_observation_space_size> getFixedObservation(const GameContext &gc) {
        std::array<int,fixed_observation_space_size> ret {};

        int offset = 0;

        ret[offset++] = std::min(gc.curHp, playerHpMax);
        ret[offset++] = std::min(gc.maxHp, playerHpMax);
        ret[offset++] = std::min(gc.gold, playerGoldMax);
        ret[offset++] = gc.floorNum;
        ret[offset++] = getBossEncoding(gc.boss);

        return ret;
    }

    std::array<int,fixed_observation_space_size> getFixedObservationMaximums() {
        std::array<int,fixed_observation_space_size> ret {};
        int spaceOffset = 0;

        ret[0] = playerHpMax;
        ret[1] = playerHpMax;
        ret[2] = playerGoldMax;
        ret[3] = 60;
        ret[4] = numBosses;

        return ret;
    }
    
    NNCardsRepresentation getCardRepresentation(const Deck &deck) {
        NNCardsRepresentation rep;
        for (int i = 0; i < deck.size(); ++i) {
            rep.cards.push_back(deck.cards[i].id);
            rep.upgrades.push_back(deck.cards[i].getUpgraded());
        }
        return rep;
    }
    
    NNRelicsRepresentation getRelicRepresentation(const RelicContainer &relics) {
        NNRelicsRepresentation rep;
        for (int i = 0; i < relics.size(); ++i) {
            rep.relics.push_back(relics.relics[i].id);
            rep.relicCounters.push_back(relics.relics[i].data);
        }
        return rep;
    }
    
    
    NNRepresentation getNNRepresentation(const GameContext &gc) {
        NNRepresentation rep;
        rep.fixedObservation = getFixedObservation(gc);
        rep.deck = getCardRepresentation(gc.deck);
        rep.relics = getRelicRepresentation(gc.relics);
        rep.map = getNNMapRepresentation(*gc.map);
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
        NNMapRepresentation nnMap;
        int id = 0;
        bool haveLastRow = false;
        for (int y = 0; y < 15; ++y) {
            for (int x = 0; x < 7; ++x) {
                const MapNode& node = map.getNode(x,y);
                if (node.room != Room::NONE) {
                    ids[y][x] = id++;
                    nnMap.roomTypes.push_back(node.room);
                    nnMap.xs.push_back(x);
                    nnMap.ys.push_back(y);
                    if (y == 14) {
                        haveLastRow = true;
                    }
                }
            }
        }
        if (haveLastRow) {
            ids[15][3] = id++; // boss
            nnMap.roomTypes.push_back(Room::BOSS);
        }
        for (int y = 0; y < 15; ++y) {
            for (int x = 0; x < 7; ++x) {
                const MapNode& node = map.getNode(x,y);
                if (node.room!= Room::NONE) {
                    for (int k = 0; k < node.edgeCount; ++k) {
                        nnMap.edgeStarts.push_back(ids[y][x]);
                        nnMap.edgeEnds.push_back(ids[y+1][node.edges[k]]);
                    }
                }
            }
        }
        return nnMap;
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