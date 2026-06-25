//
// Created by gamerpuppy on 7/24/2021.
//

#ifndef STS_LIGHTSPEED_CARDMANAGER_H
#define STS_LIGHTSPEED_CARDMANAGER_H

#include <vector>
#include <array>

#include "sts_common.h"

#include "combat/CardInstance.h"
#include "combat/CardPile.h"
#include "game/Random.h"
#include "game/Deck.h"

namespace sts {

    class GameContext;
    class Card;

    struct CardManager {

        static constexpr int MAX_HAND_SIZE = 10;
        static constexpr int MAX_GROUP_SIZE = 64;

        int nextUniqueCardId = 0; // unique card ids that are less than the masterDeckSize are non-temporary

        int cardsInHand = 0;
        std::array<CardInstance, MAX_HAND_SIZE> hand;
        std::array<CardInstance, MAX_HAND_SIZE> limbo; // used only for end of turn during discard, for retained cards
        std::array<CardInstance,2> stasisCards { CardId::INVALID, CardId::INVALID }; // for bronze automaton fight

#ifdef sts_card_manager_use_fixed_list
        fixed_list<CardInstance, MAX_GROUP_SIZE> drawPile;
        fixed_list<CardInstance, MAX_GROUP_SIZE> discardPile;
        fixed_list<CardInstance, MAX_GROUP_SIZE> exhaustPile;
#else
        CardPile drawPile;
        CardPile discardPile;
        CardPile exhaustPile;
#endif
        int handNormalityCount = 0;
        int handPainCount = 0;
        int strikeCount = 0;
        int handBloodCardCount = 0;
        int drawPileBloodCardCount = 0;
        int discardPileBloodCardCount = 0;

        void init(const GameContext &gc, BattleContext &bc); // returns count of innate cards

        CardInstance createDeckCardInstance(const Card &card, int deckIdx);
        void createTempCardInDrawPile(Random &rng, CardInstance c); // shuffled in
        void createTempCardInDiscard(CardInstance c);
        void createTempCardInHand(CardInstance c);

        void removeFromDrawPileAtIdx(int idx);
        // Remove the first draw-pile card matching `match` by id + upgrade count, returning it (or an
        // INVALID CardInstance if none matched). Encapsulates the pile index -- callers reconstructing
        // an observed outcome (e.g. forcing Havoc's drawn card) shouldn't depend on the unknown order.
        CardInstance removeFromDrawPile(const CardInstance &match);
        CardInstance popFromDrawPile(Random &rng);

        void removeFromHandAtIdx(int idx); // this method is dangerous if used in the wrong place.
        void removeFromHandById(std::uint16_t uniqueId); // can do more than one card if they have the same uniqueId, does this happen?
        void removeFromDiscard(int idx);
        void removeFromExhaustPile(int idx);

        void moveToHand(const CardInstance &c);
        void moveToExhaustPile(const CardInstance &c);

        void moveToDrawPileTop(const CardInstance &c);
        void moveToDrawPileBottom(const CardInstance &c);
        // Add a card to the draw pile's UNKNOWN region (no known position, no rng). For
        // reconstructing a draw pile whose order the player can't actually see: the searcher then
        // draws it stochastically (chance nodes) instead of treating the reconstructed order as
        // known future draws. Use this, not moveToDrawPileTop, when rebuilding a converted state.
        void moveToDrawPileUnknown(const CardInstance &c);
        void shuffleIntoDrawPile(Random &rng, const CardInstance &c);

        void moveToDiscardPile(const CardInstance &c);
        void moveDiscardPileIntoToDrawPile(Random &rng);

        // **************
        void notifyAddCardToCombat(const CardInstance &c);
        void notifyRemoveFromCombat(const CardInstance &c);

        void notifyAddToHand(const CardInstance &c);
        void notifyRemoveFromHand(const CardInstance &c);

        void notifyAddToDrawPile(const CardInstance &c);
        void notifyRemoveFromDrawPile(const CardInstance &c);

        void notifyAddToDiscardPile(const CardInstance &c);
        void notifyRemoveFromDiscardPile(const CardInstance &c);
        // **************

        void eraseAtIdxInHand(int idx); // does not call notifyRemoveFromHand
        int getRandomCardIdxInHand(Random &rng);
        void resetAttributesAtEndOfTurn();

        // special helpers
        void draw(BattleContext &bc, int amount);
        void onTookDamage(); // update blood for blood, masterful stab
        void findAndUpgradeSpecialData(std::int16_t uniqueId, int amount);
        void onBuffCorruption();
        void clear(); // clear all card piles and reset counters
        
        bool operator==(const CardManager &rhs) const;

        // operator== with the hand compared as a multiset: hand order is not gameplay-
        // meaningful, so search-state dedup must unify permutations (the searcher continues
        // from the surviving node's concrete order — a legal equivalent continuation).
        [[nodiscard]] bool equalForSearch(const CardManager &rhs) const;
    };

    std::ostream &operator <<(std::ostream &os, const CardManager &c);

}

#endif //STS_LIGHTSPEED_CARDMANAGER_H
