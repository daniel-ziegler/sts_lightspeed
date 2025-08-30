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
#include "game/Deck.h"

#include "slaythespire.h"

namespace sts::py {

    constexpr int getMaxScreenState() {
        return static_cast<int>(ScreenState::BATTLE);
    }

    constexpr int getMaxCardSelectScreenType() {
        return static_cast<int>(CardSelectScreenType::BONFIRE_SPIRITS);
    }

    constexpr int getMaxFloor() {
        // Act 4 Heart fight is floor 56, which is the maximum possible floor
        return 56;
    }

    pybind11::array_t<int> getFixedObservation(const GameContext &gc) {
        std::vector<int> ret(fixed_observation_space_size);

        int offset = 0;

        ret[offset++] = std::min(gc.curHp, playerHpMax);
        ret[offset++] = std::min(gc.maxHp, playerHpMax);
        ret[offset++] = std::min(gc.gold, playerGoldMax);
        ret[offset++] = gc.floorNum;
        ret[offset++] = getBossEncoding(gc.boss);
        ret[offset++] = gc.info.toSelectCount;

        return to_numpy(ret);
    }

    pybind11::array_t<int> getFixedObservationMaximums() {
        std::vector<int> ret(fixed_observation_space_size);
        int offset = 0;

        ret[offset++] = playerHpMax;
        ret[offset++] = playerHpMax;
        ret[offset++] = playerGoldMax;
        ret[offset++] = getMaxFloor();
        ret[offset++] = numBosses;
        ret[offset++] = Deck::MAX_SIZE; // max cards to select

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
        
        // Get potion slots up to capacity (including empty ones) to preserve indices
        std::vector<Potion> potions;
        for (int i = 0; i < gc.potionCapacity; ++i) {
            potions.push_back(gc.potions[i]);
        }
        rep.potions = to_numpy(potions);
        
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
        std::vector<int> xs, ys;
        std::vector<Room> roomTypes;
        std::vector<std::vector<int>> pathXs;

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

        // Second pass: create path_xs for each room
        for (int y = 0; y < 15; ++y) {
            for (int x = 0; x < 7; ++x) {
                const MapNode& node = map.getNode(x,y);
                if (node.room != Room::NONE) {
                    std::vector<int> roomPaths(3, -1);  // Initialize with -1 (no edge)
                    
                    // For each room, check the three possible directions: left (x-1), straight (x), right (x+1)
                    for (int k = 0; k < node.edgeCount; ++k) {
                        int edgeX = node.edges[k];
                        
                        // For the last row (y=14), edges can go to the boss at x=3
                        if (y == 14) {
                            assert(edgeX == 3);  // Boss is always at x=3
                            // Count the boss edge as "straight" with its actual x value
                            roomPaths[1] = edgeX;
                        } else {
                            // Assert that edgeX is within expected range for normal rows
                            assert(edgeX >= 0 && edgeX < 7);
                            
                            if (edgeX == x - 1) {
                                roomPaths[0] = edgeX;  // left
                            } else if (edgeX == x) {
                                roomPaths[1] = edgeX;  // straight
                            } else if (edgeX == x + 1) {
                                roomPaths[2] = edgeX;  // right
                            } else {
                                // This should never happen with the current map generation
                                assert(false && "Unexpected edge direction");
                            }
                        }
                    }
                    pathXs.push_back(roomPaths);
                }
            }
        }
        
        // Handle boss room (always at x=3, y=15)
        if (haveLastRow) {
            // Boss has no outgoing edges - it's the destination
            pathXs.push_back({-1, -1, -1});
        }

        // Create 2D numpy array for pathXs
        auto pathXsArray = pybind11::array_t<int>(
            {static_cast<pybind11::ssize_t>(pathXs.size()), static_cast<pybind11::ssize_t>(3)}, 
            {}
        );
        auto pathXsAccessor = pathXsArray.mutable_unchecked<2>();
        for (pybind11::ssize_t i = 0; i < pathXs.size(); ++i) {
            for (pybind11::ssize_t j = 0; j < 3; ++j) {
                pathXsAccessor(i, j) = pathXs[i][j];
            }
        }

        // Create numpy arrays from collected data
        return NNMapRepresentation {
            .xs = to_numpy(xs),
            .ys = to_numpy(ys),
            .roomTypes = to_numpy(roomTypes),
            .pathXs = pathXsArray
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
