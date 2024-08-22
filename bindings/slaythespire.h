//
// Created by keega on 9/24/2021.
//

#ifndef STS_LIGHTSPEED_SLAYTHESPIRE_H
#define STS_LIGHTSPEED_SLAYTHESPIRE_H

#include <vector>
#include <unordered_map>
#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "constants/Cards.h"
#include "constants/MonsterEncounters.h"
#include "constants/Relics.h"
#include "constants/Rooms.h"
#include "game/Card.h"
#include "game/Deck.h"
#include "game/RelicContainer.h"

namespace sts {

    namespace search {
        class ScumSearchAgent2;
    }


    class GameContext;
    class Map;

    namespace py {
        template<typename T> pybind11::array_t<T> to_numpy(const std::vector<T>& vec) {
            auto result = pybind11::array_t<T>(vec.size());
            auto r = result.template mutable_unchecked<1>();
            for (pybind11::ssize_t i = 0; i < vec.size(); ++i) {
                r(i) = vec[i];
            }
            return result;
        }

        static constexpr int fixed_observation_space_size = 5;
        static constexpr int playerHpMax = 200;
        static constexpr int playerGoldMax = 1800;
        static constexpr int cardCountMax = 7;
        static constexpr int numBosses = 10;

        struct NNCardsRepresentation {
            pybind11::array_t<CardId> cards;
            pybind11::array_t<int> upgrades;

            pybind11::dict as_dict() const;
        };

        struct NNRelicsRepresentation {
            pybind11::array_t<RelicId> relics;
            pybind11::array_t<int> relicCounters;

            pybind11::dict as_dict() const;
        };

        struct NNMapRepresentation {
            pybind11::array_t<int> xs;
            pybind11::array_t<int> ys;
            pybind11::array_t<Room> roomTypes;
            pybind11::array_t<int> edgeStarts;
            pybind11::array_t<int> edgeEnds;
            // todo burning elite pos

            pybind11::dict as_dict() const;
        };


        struct NNRepresentation {
            pybind11::array_t<int> fixedObservation;
            NNCardsRepresentation deck;
            NNRelicsRepresentation relics;
            NNMapRepresentation map;
            int mapX, mapY;
            // todo history

            pybind11::dict as_dict() const;
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

        pybind11::array_t<int> getFixedObservationMaximums();
        pybind11::array_t<int> getFixedObservation(const GameContext &gc);
        py::NNCardsRepresentation getCardRepresentation(const Deck &deck);
        py::NNRelicsRepresentation getRelicRepresentation(const RelicContainer &relics);


        py::NNRepresentation getNNRepresentation(const GameContext &gc);


    };


}


#endif //STS_LIGHTSPEED_SLAYTHESPIRE_H
