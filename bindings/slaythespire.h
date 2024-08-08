//
// Created by keega on 9/24/2021.
//

#ifndef STS_LIGHTSPEED_SLAYTHESPIRE_H
#define STS_LIGHTSPEED_SLAYTHESPIRE_H

#include <vector>
#include <unordered_map>
#include <array>

#include "constants/Rooms.h"

namespace sts {
    
    namespace search {
        class ScumSearchAgent2;
    }


    class GameContext;
    class Map;

    namespace py {
        static constexpr int fixed_observation_space_size = 5;
        static constexpr int playerHpMax = 200;
        static constexpr int playerGoldMax = 1800;
        static constexpr int cardCountMax = 7;
        static constexpr int numBosses = 10;

        struct NNCardsRepresentation {
            std::vector<CardId> cards;
            std::vector<int> upgrades;
        };
        
        struct NNRelicsRepresentation {
            std::vector<RelicId> relics;
            std::vector<int> relicCounters;
        };
        
        struct NNMapRepresentation {
            std::vector<int> xs;
            std::vector<int> ys;
            std::vector<Room> roomTypes;
            std::vector<int> edgeStarts;
            std::vector<int> edgeEnds;
            // todo current pos, burning elite pos
        };
        
        
        struct NNRepresentation {
            std::array<int, fixed_observation_space_size> fixedObservation;
            NNCardsRepresentation deck;
            NNRelicsRepresentation relics;
            NNMapRepresentation map;
            // todo history
        };

        void play();

        search::ScumSearchAgent2* getAgent();
        void setGc(const GameContext &gc);
        GameContext* getGc();

        void playout();
        std::vector<Card> getCardReward(GameContext &gc);
        void pickRewardCard(GameContext &gc, Card card);
        void skipRewardCards(GameContext &gc);

        NNMapRepresentation getNNMapRepresentation(const Map &map);
        Room getRoomType(const Map &map, int x, int y);
        bool hasEdge(const Map &map, int x, int y, int x2);
        
        int getBossEncoding(MonsterEncounter boss);

        std::array<int,py::fixed_observation_space_size> getFixedObservationMaximums();
        std::array<int,py::fixed_observation_space_size> getFixedObservation(const GameContext &gc);
        py::NNCardsRepresentation getCardRepresentation(const Deck &deck);
        py::NNRelicsRepresentation getRelicRepresentation(const RelicContainer &relics);

        
        py::NNRepresentation getNNRepresentation(const GameContext &gc);


    };


}


#endif //STS_LIGHTSPEED_SLAYTHESPIRE_H
