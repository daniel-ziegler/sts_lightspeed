#ifndef STS_LIGHTSPEED_CARDPILE_H
#define STS_LIGHTSPEED_CARDPILE_H

#include <algorithm>
#include <vector>

#include "combat/CardInstance.h"

namespace sts {

    // A card pile represented as the player's information set rather than a concrete order:
    // [0, size-K) is the *unknown region*, kept canonically sorted so equal information sets
    // compare equal in the search's transposition table; [size-K, size) is the *known top* in
    // true order (top = back), populated by effects that place cards at definite positions
    // (Headbutt, Warcry, innates). Discard/exhaust piles are the degenerate K = 0 case.
    //
    // All mutation goes through this interface so the invariant cannot be broken at call sites:
    // insertions keep the unknown region sorted, removals fix the known-top bookkeeping, and
    // value mutations (upgrades, blood-card triggers — card values participate in the ordering)
    // go through mutateAt/mutateAll, which restore the invariant afterwards. Read access is
    // const-only by construction. Indices in this API are absolute positions in the underlying
    // vector (stable between a const enumeration and the removal that resolves it).
    class CardPile {
        std::vector<CardInstance> cards;
        int knownTopCount = 0;

        [[nodiscard]] auto unknownEnd() { return cards.end() - knownTopCount; }
        [[nodiscard]] bool inKnownTop(int idx) const { return idx >= size() - knownTopCount; }

    public:
        // Position-less insert: joins the unknown region (sorted); never consumes rng.
        void add(const CardInstance &c) {
            cards.insert(std::upper_bound(cards.begin(), unknownEnd(), c), c);
        }

        // Definite placement on top of the pile (Headbutt, Warcry, innates); never consumes rng.
        void addToTop(const CardInstance &c) {
            cards.push_back(c);
            ++knownTopCount;
        }

        void removeAt(int idx) {
            if (inKnownTop(idx)) {
                --knownTopCount;
            }
            cards.erase(cards.begin() + idx);
        }

        // Read + remove in one step (card selects, stasis).
        CardInstance takeAt(int idx) {
            CardInstance c = cards[idx];
            removeAt(idx);
            return c;
        }

        void clear() {
            cards.clear();
            knownTopCount = 0;
        }

        [[nodiscard]] const CardInstance &operator[](int idx) const { return cards[idx]; }
        [[nodiscard]] const CardInstance &back() const { return cards.back(); }
        [[nodiscard]] int size() const { return static_cast<int>(cards.size()); }
        [[nodiscard]] bool empty() const { return cards.empty(); }
        [[nodiscard]] auto begin() const { return cards.cbegin(); }
        [[nodiscard]] auto end() const { return cards.cend(); }

        // Number of cards whose position is known (the top of the pile). Exposed for state
        // hashing/equality and debug printing only — a known top [A,B] and an unknown {A,B} are
        // different information sets. Dynamics code never needs it.
        [[nodiscard]] int knownCount() const { return knownTopCount; }

        // Apply f to one card / every card, then restore the sort invariant. Only the unknown
        // region is re-sorted: known-top positions are meaningful and survive value mutation.
        template <typename F> void mutateAt(int idx, F &&f) {
            f(cards[idx]);
            if (!inKnownTop(idx)) {
                std::sort(cards.begin(), unknownEnd());
            }
        }
        template <typename F> void mutateAll(F &&f) {
            for (auto &c : cards) {
                f(c);
            }
            std::sort(cards.begin(), unknownEnd());
        }

        // Whole-pile transfer (deck reshuffle); positional knowledge does not survive the
        // transfer, so contents leave as a plain vector (unknown region sorted, then known top).
        [[nodiscard]] std::vector<CardInstance> takeAll() {
            std::vector<CardInstance> out = std::move(cards);
            cards.clear();
            knownTopCount = 0;
            return out;
        }

        bool operator==(const CardPile &rhs) const = default;
    };

}

#endif //STS_LIGHTSPEED_CARDPILE_H
