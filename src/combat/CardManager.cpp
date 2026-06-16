//
// Created by gamerpuppy on 7/24/2021.
//

#include "combat/CardManager.h"

#include <set>
#include <cstdlib>
#include <execinfo.h>

#include <algorithm>

#include "combat/BattleContext.h"
#include "game/GameContext.h"
#include "game/Card.h"

#include "sim/search/BattleSearcher.h"

using namespace sts;

void CardManager::init(const sts::GameContext &gc, BattleContext &bc) {

//    masterDeckSize = gc.deck.size();
    nextUniqueCardId = gc.deck.size();
    handPainCount = false;
    handNormalityCount = false;
    strikeCount = 0;

    drawPile.clear();
    discardPile.clear();
    exhaustPile.clear();
    drawPile.setOrderObserved(bc.player.hasRelic<RelicId::FROZEN_EYE>());

    int innateCount = 0;
    bool isInnateMemo[Deck::MAX_SIZE];
    for (int deckIdx = 0; deckIdx < gc.deck.size(); ++deckIdx) {
        bool isBottled = std::find(gc.deck.bottleIdxs.begin(), gc.deck.bottleIdxs.end(), deckIdx) != gc.deck.bottleIdxs.end();
        isInnateMemo[deckIdx] = gc.deck.cards[deckIdx].isInnate() || isBottled;
        if (isInnateMemo[deckIdx]) {
            ++innateCount;
        }
    }

    if (bc.player.hasRelic<RelicId::FROZEN_EYE>()) {
        // Full order is genuinely observed: materialize a concrete shuffle (innates on top),
        // exactly as a real shuffle would resolve it.
        fixed_list<int, Deck::MAX_SIZE> idxs(gc.deck.size());
        for (int i = 0; i < idxs.size(); ++i) {
            idxs[i] = i;
        }
        java::Collections::shuffle(idxs.begin(), idxs.end(), java::Random(bc.rng.randomLong()));

        // normals first (bottom), then innates on top, each set in shuffled order
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < gc.deck.size(); ++i) {
                const int deckIdx = idxs[i];
                if (isInnateMemo[deckIdx] == (pass == 1)) {
                    drawPile.addToTop(createDeckCardInstance(gc.deck.cards[deckIdx], deckIdx));
                }
            }
        }
    } else {
        // Unknown order: no rng — the shuffle's randomness binds at draw time. Innates form
        // the known top in deck order (their internal order is unobservable in practice;
        // SEARCH_MODEL_INACCURACIES.md #3).
        for (int deckIdx = 0; deckIdx < gc.deck.size(); ++deckIdx) {
            const auto c = createDeckCardInstance(gc.deck.cards[deckIdx], deckIdx);
            if (isInnateMemo[deckIdx]) {
                drawPile.addToTop(c);
            } else {
                drawPile.add(c);
            }
        }
    }

    if (innateCount > bc.player.cardDrawPerTurn) {
        bc.addToBot( Actions::DrawCards(innateCount-bc.player.cardDrawPerTurn) );
    }
}

CardInstance CardManager::createDeckCardInstance(const Card &card, int deckIdx) {
    CardInstance c(card);
    c.setUniqueId(deckIdx);
#ifdef sts_asserts
    if (card.getId() == CardId::INVALID) {
        std::cerr << *g_debug_bc << '\n';
        search::g_debug_scum_search->printSearchStack(std::cerr, true);
        std::cerr << "attempted to create invalid deck instance in draw pile" << std::endl;
        assert(false);
    }
#endif
    notifyAddCardToCombat(c);
    notifyAddToDrawPile(c);
    return c;
}

void CardManager::createTempCardInDrawPile(Random &rng, CardInstance c) {
#ifdef sts_asserts
    if (c.getId() == CardId::INVALID) {
        std::cerr << *g_debug_bc << '\n';
        search::g_debug_scum_search->printSearchStack(std::cerr, true);
        std::cerr << "attempted to create invalid card in draw pile" << std::endl;
        assert(false);
    }
#endif

    c.uniqueId = static_cast<std::int16_t>(nextUniqueCardId++);
    notifyAddCardToCombat(c);
    notifyAddToDrawPile(c);
    drawPile.shuffleIn(rng, c);
}

void CardManager::createTempCardInDiscard(CardInstance c) {
    c.uniqueId = static_cast<std::int16_t>(nextUniqueCardId++);
#ifdef sts_asserts
    if (c.getId() == CardId::INVALID) {
        std::cerr << *g_debug_bc << '\n';
        search::g_debug_scum_search->printSearchStack(std::cerr, true);
        std::cerr << "attempted to create invalid card in discard" << std::endl;
        assert(false);
    }
#endif
    notifyAddCardToCombat(c);
    notifyAddToDiscardPile(c);
    discardPile.add(c);
}

void CardManager::createTempCardInHand(CardInstance c) {
    c.uniqueId = static_cast<std::int16_t>(nextUniqueCardId++);
#ifdef sts_asserts
    if (c.getId() == CardId::INVALID) {
        std::cerr << *g_debug_bc << '\n';
        search::g_debug_scum_search->printSearchStack(std::cerr, true);
        std::cerr << "attempted to create invalid card in hand" << std::endl;
        assert(false);
    }
#endif
    notifyAddCardToCombat(c);
    notifyAddToHand(c);
    hand[cardsInHand++] = c;
}

// **************** START Remove Methods ****************

void CardManager::removeFromDrawPileAtIdx(int idx) {
    notifyRemoveFromDrawPile(drawPile[idx]);
    drawPile.removeAt(idx);
}

CardInstance CardManager::popFromDrawPile(Random &rng) {
    auto c = drawPile.drawTop(rng);
    notifyRemoveFromDrawPile(c);
    return c;
}

void CardManager::removeFromHandAtIdx(int idx) {
    notifyRemoveFromHand(hand[idx]);
    eraseAtIdxInHand(idx);
}

void CardManager::removeFromHandById(std::uint16_t uniqueId) {
    for (int i = 0; i < cardsInHand; ++i) {
        if (hand[i].getUniqueId() == uniqueId) {
            notifyRemoveFromHand(hand[i]);
            eraseAtIdxInHand(i);
        }
    }
}

void CardManager::removeFromDiscard(int idx) {
    notifyRemoveFromDiscardPile(discardPile[idx]);
    discardPile.removeAt(idx);
}

void CardManager::removeFromExhaustPile(int idx) {
    exhaustPile.removeAt(idx);
}

// **************** END Remove Methods ****************

// **************** START Move Methods ****************

void CardManager::moveToHand(const CardInstance &c) {
#ifdef sts_asserts
    assert(cardsInHand < 10);
#endif
    notifyAddToHand(c);
    hand[cardsInHand++] = c;
}

void CardManager::moveToExhaustPile(const CardInstance &c) {
    notifyRemoveFromCombat(c);
    exhaustPile.add(c);
}


void CardManager::moveToDrawPileBottom(const CardInstance &c) {
    // An INVALID instance reaching a pile move is a rare engine bug (first observed with
    // chest relics in play, ~1/1000 games, timing-dependent). Dropping the card keeps the
    // battle playable; asserting would abort the whole hosting process. The warning line
    // is the detection signal -- grep logs for "WARNING: dropped INVALID".
    if (c.getId() == CardId::INVALID) {
        std::cerr << "WARNING: dropped INVALID card on moveToDrawPileBottom" << std::endl;
        return;
    }
    notifyAddToDrawPile(c);
    drawPile.addToBottom(c);
}

void CardManager::moveToDrawPileTop(const CardInstance &c) {
    if (c.getId() == CardId::INVALID) {
        std::cerr << "WARNING: dropped INVALID card on moveToDrawPileTop" << std::endl;
        return;
    }
    notifyAddToDrawPile(c);
    drawPile.addToTop(c);
}

void CardManager::moveToDrawPileUnknown(const CardInstance &c) {
    if (c.getId() == CardId::INVALID) {
        std::cerr << "WARNING: dropped INVALID card on moveToDrawPileUnknown" << std::endl;
        return;
    }
    notifyAddToDrawPile(c);
    drawPile.add(c);  // joins the unknown region; the searcher draws it stochastically
}

void CardManager::shuffleIntoDrawPile(Random &rng, const CardInstance &c) {
    notifyAddToDrawPile(c);
    drawPile.shuffleIn(rng, c);
}

void CardManager::moveToDiscardPile(const CardInstance &c) {
    // todo check flurries, weave
    if (c.getId() == CardId::INVALID) {
        // One-line battle fingerprint for root-causing (g_debug_bc is thread_local, set by
        // the executing battle): seed/turn/monster identify the fight class.
        // STS_INVALID_VERBOSE=1 additionally dumps the full battle state + search action
        // stack on the first occurrence (debug sessions only -- the dump is large).
        std::cerr << "WARNING: dropped INVALID card on moveToDiscardPile";
        if (g_debug_bc != nullptr) {
            std::cerr << " [seed " << g_debug_bc->seed << " turn " << g_debug_bc->turn
                      << " m0 " << static_cast<int>(g_debug_bc->monsters.arr[0].id)
                      << " mAlive " << g_debug_bc->monsters.monstersAlive
                      << " cardQ " << g_debug_bc->cardQueue.size
                      << " hand " << g_debug_bc->cards.cardsInHand << "]";
        }
        std::cerr << std::endl;
        static bool dumped = false;
        if (!dumped && std::getenv("STS_INVALID_VERBOSE") != nullptr && g_debug_bc != nullptr) {
            dumped = true;
            std::cerr << *g_debug_bc << '\n';
            if (search::g_debug_scum_search != nullptr) {
                search::g_debug_scum_search->printSearchStack(std::cerr, true);
            }
            std::cerr << "curCardQueueItem: card " << g_debug_bc->curCardQueueItem.card
                      << " isEndTurn " << g_debug_bc->curCardQueueItem.isEndTurn
                      << " triggerOnUse " << g_debug_bc->curCardQueueItem.triggerOnUse
                      << " purgeOnUse " << g_debug_bc->curCardQueueItem.purgeOnUse << '\n';
            std::cerr << "actionQueue (" << g_debug_bc->actionQueue.size << "):";
            {
                int ci = g_debug_bc->actionQueue.front;
                for (int i = 0; i < g_debug_bc->actionQueue.size; ++i) {
                    if (ci >= g_debug_bc->actionQueue.getCapacity()) ci = 0;
                    std::cerr << ' ' << g_debug_bc->actionQueue.arr[ci];
                    ++ci;
                }
            }
            std::cerr << '\n';
            void *frames[32];
            const int n = backtrace(frames, 32);
            backtrace_symbols_fd(frames, n, 2);
            std::cerr.flush();
        }
        return;
    }
    notifyAddToDiscardPile(c);
    discardPile.add(c);
}

void CardManager::moveDiscardPileIntoToDrawPile(Random &rng) {
    drawPileBloodCardCount += discardPileBloodCardCount;
    discardPileBloodCardCount = 0;
    drawPile.absorb(rng, discardPile);
}

// **************** END Move Methods ****************


// **************** BEGIN NOTIFY METHODS ****************

void CardManager::notifyAddCardToCombat(const CardInstance &c) {
    if (c.isStrikeCard()) {
        ++strikeCount;
    }
}

void CardManager::notifyRemoveFromCombat(const CardInstance &c) {
    if (c.isStrikeCard()) {
        --strikeCount;
    }
}

void CardManager::notifyAddToHand(const CardInstance &c) {
#ifdef sts_asserts
    if (c.getId() == CardId::INVALID) {
        std::cerr << *g_debug_bc << '\n';
        search::g_debug_scum_search->printSearchStack(std::cerr, true);
        std::cerr << "attempted to notify of invalid card in hand" << std::endl;
        assert(false);
    }
#endif

    if (c.isBloodCard()) {
        ++handBloodCardCount;
    }

    switch (c.id) {
        case CardId::NORMALITY:
            ++handNormalityCount;
            break;

        case CardId::PAIN:
            ++handPainCount;
            break;

        default:
            break;
    }
}

void CardManager::notifyRemoveFromHand(const CardInstance &c) {
    if (c.isBloodCard()) {
        --handBloodCardCount;
    }

    switch (c.id) {
        case CardId::NORMALITY:
            --handNormalityCount;
            break;

        case CardId::PAIN:
            --handPainCount;
            break;

        default:
            break;
    }
}

void CardManager::notifyAddToDrawPile(const CardInstance &c) {

#ifdef sts_asserts
    if (c.getId() == CardId::INVALID) {
        std::cerr << *g_debug_bc << '\n';
        search::g_debug_scum_search->printSearchStack(std::cerr, true);
        std::cerr << "attempted to notify of invalid card in draw pile" << std::endl;
        assert(false);
    }
#endif

    if (c.isBloodCard()) {
        ++drawPileBloodCardCount;
    }
}

void CardManager::notifyRemoveFromDrawPile(const CardInstance &c) {
    if (c.isBloodCard()) {
        --drawPileBloodCardCount;
    }
}

void CardManager::notifyAddToDiscardPile(const CardInstance &c) {
#ifdef sts_asserts
    if (c.getId() == CardId::INVALID) {
        std::cerr << *g_debug_bc << '\n';
        search::g_debug_scum_search->printSearchStack(std::cerr, true);
        std::cerr << "attempted to notify of invalid card in discard pile" << std::endl;
        assert(false);
    }
#endif

    if (c.isBloodCard()) {
        ++discardPileBloodCardCount;
    }
}


void CardManager::notifyRemoveFromDiscardPile(const CardInstance &c) {
    if (c.isBloodCard()) {
        --discardPileBloodCardCount;
    }
}

// **************** END NOTIFY METHODS ****************

void CardManager::eraseAtIdxInHand(int idx) {
#ifdef sts_asserts
    if (idx >= cardsInHand) {
        assert(false);
    }
#endif

    for (int x = idx; x < cardsInHand-1; ++x) {
        hand[x] = hand[x+1];
    }
    --cardsInHand;
}

int CardManager::getRandomCardIdxInHand(Random &rng) {
    return rng.random(cardsInHand-1);
}

void CardManager::resetAttributesAtEndOfTurn() {
    for (int i = 0; i < cardsInHand; ++i) {
        hand[i].setCostForTurn(hand[i].cost);
    }

    discardPile.mutateAll([](CardInstance &c) { c.setCostForTurn(c.cost); });
    drawPile.mutateAll([](CardInstance &c) { c.setCostForTurn(c.cost); });
}

// **************** BEGIN SPECIAL HELPERS ****************

void CardManager::draw(BattleContext &bc, int amount) {
    int evolve = bc.player.getStatus<PS::EVOLVE>();
    int fireBreathing = bc.player.getStatus<PS::FIRE_BREATHING>();

    for (int i = 0; i < amount; i++) {
        auto c = popFromDrawPile(bc.rng);

        if (bc.player.hasStatus<PS::CONFUSED>()) {
            if (c.cost >= 0) {  // todo status and curses affected by this?
                const auto newCost = static_cast<std::int8_t>(bc.rng.random(3));
                if (c.cost != newCost) {
                    c.costForTurn = newCost;
                    c.cost = newCost;
                }
                c.freeToPlayOnce = false;
            }
        }

        if (c.getType() == CardType::SKILL) {
            if (bc.player.hasStatus<PS::CORRUPTION>()) {
                c.setCostForTurn(-9);
            }

        } else if (c.getType() == CardType::STATUS) {
            if (evolve) {
                bc.addToBot( Actions::DrawCards(evolve) );
            }
            if (fireBreathing) {
                bc.addToBot( Actions::DamageAllEnemy(fireBreathing) );
            }
            if (c.getId() == CardId::VOID) {
                // game adds action to bottom of the queue but I think it is ok to do directly
                bc.player.energy = std::max(0, bc.player.energy-1);
            }

        } else if (c.getType() == CardType::CURSE) {
            if (fireBreathing) {
                bc.addToBot( Actions::DamageAllEnemy(fireBreathing) );
            }

        }

        // do we need to check this?
        if (cardsInHand < 10) {
            moveToHand(c);
        } else {
            moveToDiscardPile(c);
        }
    }

}

void CardManager::onTookDamage() {
    // this method will fail catastrophically if the bloodCardCounts are not correct
    const bool hasAnyBloodCards = handBloodCardCount | drawPileBloodCardCount | discardPileBloodCardCount;
    if (!hasAnyBloodCards) {
        return;
    }

    int i = 0;
    int foundBloodCards = 0;
    while (foundBloodCards < handBloodCardCount) {
        if (hand[i].isBloodCard()) {
            hand[i].tookDamage();
            ++foundBloodCards;
        }
        ++i;
    }

    if (drawPileBloodCardCount > 0) {
        drawPile.mutateAll([](CardInstance &c) {
            if (c.isBloodCard()) {
                c.tookDamage();
            }
        });
    }

    if (discardPileBloodCardCount > 0) {
        discardPile.mutateAll([](CardInstance &c) {
            if (c.isBloodCard()) {
                c.tookDamage();
            }
        });
    }
}

void upgrade(CardInstance &c, int upgradeAmount) {
    c.specialData += upgradeAmount;
}

// for ritual dagger, rampage
void CardManager::findAndUpgradeSpecialData(const std::int16_t uniqueId, const int upgradeAmount) {

    // special checks for most common scenarios
    for (int i = discardPile.size()-1; i >= 0; --i) {
        if (discardPile[i].uniqueId == uniqueId) {
            discardPile.mutateAt(i, [=](CardInstance &c) { upgrade(c, upgradeAmount); });
            return;
        }
    }

    for (int i = exhaustPile.size()-1; i >= 0; --i) {
        if (exhaustPile[i].uniqueId == uniqueId) {
            exhaustPile.mutateAt(i, [=](CardInstance &c) { upgrade(c, upgradeAmount); });
            return;
        }
    }

    for (int i = drawPile.size()-1; i >= 0; --i) {
        if (drawPile[i].uniqueId == uniqueId) {
            drawPile.mutateAt(i, [=](CardInstance &c) { upgrade(c, upgradeAmount); });
            return;
        }
    }

    for (int i = 0; i < cardsInHand; ++i) {
        auto &c = hand[i];
        if (c.uniqueId == uniqueId) {
            upgrade(c, upgradeAmount);
            return;
        }
    }

}

void CardManager::onBuffCorruption() {
    // game does modifyCostForCombat here but I don't think its necessary as skills cant cost more than 4?
    for (int i = 0; i < cardsInHand; ++i) {
        auto &c = hand[i];
        if (c.getType() == CardType::SKILL && c.cost > 0) {
            c.cost = 0;
            c.costForTurn = 0;
        }
    }

    // probably only need to do hand?

    drawPile.mutateAll([](CardInstance &c) {
        if (c.getType() == CardType::SKILL && c.cost > 0) {
            c.cost = 0;
            c.costForTurn = 0;
        }
    });

    discardPile.mutateAll([](CardInstance &c) {
        if (c.getType() == CardType::SKILL && c.cost > 0) {
            c.cost = 0;
            c.costForTurn = 0;
        }
    });

    exhaustPile.mutateAll([](CardInstance &c) {
        if (c.getType() == CardType::SKILL && c.cost > 0) {
            c.cost = 0;
            c.costForTurn = 0;
        }
    });


}

// **************** END SPECIAL HELPERS ****************

std::set<CardInstance> unordered(const std::vector<CardInstance> &v) {
    return std::set<CardInstance>(v.begin(), v.end());
}

bool CardManager::operator==(const CardManager &rhs) const {
    return nextUniqueCardId == rhs.nextUniqueCardId &&
           cardsInHand == rhs.cardsInHand &&
           std::equal(hand.begin(), hand.begin() + cardsInHand, rhs.hand.begin()) &&  // slots past cardsInHand are stale
           limbo == rhs.limbo &&
           stasisCards == rhs.stasisCards &&
           drawPile == rhs.drawPile &&
           discardPile == rhs.discardPile &&
           exhaustPile == rhs.exhaustPile &&
           handNormalityCount == rhs.handNormalityCount &&
           handPainCount == rhs.handPainCount &&
           strikeCount == rhs.strikeCount &&
           handBloodCardCount == rhs.handBloodCardCount &&
           drawPileBloodCardCount == rhs.drawPileBloodCardCount &&
           discardPileBloodCardCount == rhs.discardPileBloodCardCount;
}

bool CardManager::equalForSearch(const CardManager &rhs) const {
    // cheap scalar fields and exact pile compares first; the hand-multiset check (two scratch
    // sorts) runs only when everything else already matched
    if (!(cardsInHand == rhs.cardsInHand &&
          nextUniqueCardId == rhs.nextUniqueCardId &&
          handNormalityCount == rhs.handNormalityCount &&
          handPainCount == rhs.handPainCount &&
          strikeCount == rhs.strikeCount &&
          handBloodCardCount == rhs.handBloodCardCount &&
          drawPileBloodCardCount == rhs.drawPileBloodCardCount &&
          discardPileBloodCardCount == rhs.discardPileBloodCardCount &&
          limbo == rhs.limbo &&
          stasisCards == rhs.stasisCards &&
          drawPile == rhs.drawPile &&
          discardPile == rhs.discardPile &&
          exhaustPile == rhs.exhaustPile)) {
        return false;
    }
    // hand as a multiset: sort scratch copies into canonical order, then compare exactly
    std::array<CardInstance, MAX_HAND_SIZE> lhsHand, rhsHand;
    std::copy(hand.begin(), hand.begin() + cardsInHand, lhsHand.begin());
    std::copy(rhs.hand.begin(), rhs.hand.begin() + cardsInHand, rhsHand.begin());
    std::sort(lhsHand.begin(), lhsHand.begin() + cardsInHand);
    std::sort(rhsHand.begin(), rhsHand.begin() + cardsInHand);
    return std::equal(lhsHand.begin(), lhsHand.begin() + cardsInHand, rhsHand.begin());
}

namespace sts {

//    std::ostream &operator<<(std::ostream &os, const CardInstance &c) {
//        return os << "("
//            << c.getName()
//            << ", uid:" << std::to_string(c.uniqueId)
//            << ", u:" << std::to_string(c.upgraded)
//            << ", c:" << c.cost
//            << ", ct:" << c.costForTurn
//            << ")";
//    }

    template<typename Forward_Iterator>
    void printArray(std::ostream &os, Forward_Iterator begin, Forward_Iterator end) {
        os << "{ ";
        while (begin != end && begin+1 != end) {
            os << *begin << ", ";
            ++begin;
        }
        if (begin != end) {
            os << *begin;
        }
        os << " }";
    }

    std::ostream &operator<<(std::ostream &os, const CardManager &c) {
        os << "CardManager: {";

        os << "\n\tdrawPile: " << c.drawPile.size() << " ";
        printArray(os, c.drawPile.begin(), c.drawPile.end());

        os << ",\n\tdiscardPile: " << c.discardPile.size() << " ";
        printArray(os, c.discardPile.begin(), c.discardPile.end());

        os << ",\n\texhaustPile: " << c.exhaustPile.size() << " ";
        printArray(os, c.exhaustPile.begin(), c.exhaustPile.end());

        os << ",\n\thand: " << c.cardsInHand << " ";
        printArray(os, c.hand.begin(), c.hand.begin()+c.cardsInHand);

        os << "\n\tstasisCards{" << c.stasisCards[0] << "," << c.stasisCards[1] << "}";

        os << "\n\t" << "handNormalityCount: " << c.handNormalityCount;
        const auto s = ", ";
        os << s << "handPainCount: " << c.handPainCount;
        os << s << "strikeCount: " << c.strikeCount;

        os << s << "handBloodCardCount: " << c.handBloodCardCount;
        os << s << "drawPileBloodCardCount: " << c.drawPileBloodCardCount;
        os << s << "discardPileBloodCardCount: " << c.discardPileBloodCardCount;

        os << "\n}\n";

        return os;
    }

    void CardManager::clear() {
        // Clear all card piles
        cardsInHand = 0;
        drawPile.clear();
        discardPile.clear();
        exhaustPile.clear();
        
        // Reset all counters
        handNormalityCount = 0;
        handPainCount = 0;
        strikeCount = 0;
        handBloodCardCount = 0;
        drawPileBloodCardCount = 0;
        discardPileBloodCardCount = 0;
        
        // Reset other fields
        nextUniqueCardId = 0;
        stasisCards[0] = CardInstance(CardId::INVALID);
        stasisCards[1] = CardInstance(CardId::INVALID);
        
        // Clear hand and limbo arrays (not strictly necessary but good for consistency)
        std::fill(hand.begin(), hand.end(), CardInstance(CardId::INVALID));
        std::fill(limbo.begin(), limbo.end(), CardInstance(CardId::INVALID));
    }


}
