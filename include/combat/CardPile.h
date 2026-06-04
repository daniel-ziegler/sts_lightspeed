#ifndef STS_LIGHTSPEED_CARDPILE_H
#define STS_LIGHTSPEED_CARDPILE_H

#include <algorithm>
#include <cassert>
#include <vector>

#include "combat/CardInstance.h"
#include "game/Random.h"

namespace sts {

    // A card pile represented as the player's information set rather than a concrete order:
    // [0, B) is the *known bottom* in true order (Forethought placements; index 0 is drawn
    // dead last), [B, size-K) is the *unknown region*, kept canonically sorted so equal
    // information sets compare equal in the search's transposition table, and [size-K, size)
    // is the *known top* in true order (top = back), populated by effects that place cards at
    // definite positions (Headbutt, Warcry, innates). Discard/exhaust piles are the degenerate
    // B = K = 0 case. Draw order: known top, then unknown (sampled), then known bottom.
    //
    // Callers never branch on the known/unknown split: mutating operations take a Random& and
    // consume rng exactly when the operation is genuinely stochastic for the player (a draw from
    // the unknown region, an insertion relative to a known top), which is precisely when the
    // searcher's rng-counter detection should see a chance event. Randomness that real StS
    // resolves at shuffle time is deferred to draw time — the induced distribution over draw
    // sequences is identical (exchangeability; see SEARCH_MODEL_INACCURACIES.md #4).
    //
    // orderObserved (Frozen Eye) means the player legitimately sees the full order: every
    // shuffle materializes a concrete order via rng and the entire pile is "known top" (K ==
    // size always), reproducing legacy concrete-pile dynamics exactly.
    //
    // All mutation goes through this interface so the invariants cannot be broken at call
    // sites: insertions keep the unknown region sorted, removals fix the known-top bookkeeping,
    // and value mutations (upgrades, blood-card triggers — card values participate in the
    // ordering) go through mutateAt/mutateAll, which restore the invariant afterwards. Read
    // access is const-only by construction. Indices in this API are absolute positions in the
    // underlying vector (stable between a const enumeration and the removal that resolves it).
    class CardPile {
        std::vector<CardInstance> cards;
        int knownTopCount = 0;     // K; orderObserved keeps K == size, B == 0
        int knownBottomCount = 0;  // B
        bool orderObserved = false;

        [[nodiscard]] auto unknownBegin() { return cards.begin() + knownBottomCount; }
        [[nodiscard]] auto unknownEnd() { return cards.end() - knownTopCount; }
        [[nodiscard]] int unknownCount() const { return size() - knownTopCount - knownBottomCount; }
        [[nodiscard]] bool inKnownTop(int idx) const { return idx >= size() - knownTopCount; }
        [[nodiscard]] bool inKnownBottom(int idx) const { return idx < knownBottomCount; }

        // A single-card pile is fully known either way; canonicalize to K = B = 0 so it hashes
        // and compares identically however it was produced (known remnant vs fresh insert).
        void normalizeSingleton() {
            if (!orderObserved && cards.size() == 1) {
                knownTopCount = 0;
                knownBottomCount = 0;
            }
        }

    public:
        // Must be set while the pile is empty (battle init); makes every subsequent shuffle
        // concrete (legacy dynamics) as the price of genuine full-order knowledge.
        void setOrderObserved(bool observed) {
            assert(cards.empty());
            orderObserved = observed;
        }

        // Position-less insert: joins the unknown region (sorted); never consumes rng.
        void add(const CardInstance &c) {
            assert(!orderObserved);
            cards.insert(std::upper_bound(unknownBegin(), unknownEnd(), c), c);
        }

        // Definite placement on top of the pile (Headbutt, Warcry, innates); never consumes rng.
        void addToTop(const CardInstance &c) {
            cards.push_back(c);
            ++knownTopCount;
            normalizeSingleton();
        }

        // Definite placement on the bottom (Forethought); never consumes rng. The card is drawn
        // after the entire unknown region (dead last until a reshuffle).
        void addToBottom(const CardInstance &c) {
            cards.insert(cards.begin(), c);
            if (orderObserved) {
                ++knownTopCount;
            } else {
                ++knownBottomCount;
                normalizeSingleton();
            }
        }

        // Shuffle a card into the pile at a position the player does not choose. The legacy gap
        // choice never lands the card on the very top, so the pre-insert top remains the next
        // draw: with no known top we first sample one unknown card and promote it to known top
        // (the player effectively learns their next draw one observation early — a mild,
        // player-favorable timing shift, but the joint distribution over draw sequences is
        // exact; the chance node simply moves here from the following draw, which becomes a
        // deterministic pop). The inserted card then joins the pile below that top: gaps whose
        // location the player could track afterwards are determinized as chance outcomes (the
        // K-1 gaps strictly within the known top, the B gaps within/below the known bottom —
        // SEARCH_MODEL_INACCURACIES.md #1), while the boundary gaps adjacent to the known
        // stacks are NOT player-distinguishable from the unknown interior and fold into "joins
        // the unknown region", whose exchangeability reproduces the legacy positional
        // distribution exactly.
        void shuffleIn(Random &rng, const CardInstance &c) {
            if (orderObserved) {
                // legacy concrete insertion (gap [0, N-1]; never the very top)
                const int idx = cards.empty() ? 0 : rng.random(size() - 1);
                cards.insert(cards.begin() + idx, c);
                ++knownTopCount;
                return;
            }
            if (knownTopCount == 0 && unknownCount() > 0) {
                // promote the pre-insert top: uniform sample from the unknown region
                const int idx = knownBottomCount
                        + (unknownCount() == 1 ? 0 : rng.random(unknownCount() - 1));
                const CardInstance top = cards[idx];
                cards.erase(cards.begin() + idx);
                cards.push_back(top);
                knownTopCount = 1;
            }
            if (knownBottomCount == 0 && knownTopCount < 2) {
                add(c);
                normalizeSingleton();
                return;
            }
            const int g = rng.random(size() - 1);
            if (g < knownBottomCount) {
                // within/below the known bottom stack — track it from here on
                cards.insert(cards.begin() + g, c);
                ++knownBottomCount;
            } else if (g > size() - knownTopCount) {
                // strictly between two known-top cards — track it from here on
                cards.insert(cards.begin() + g, c);
                ++knownTopCount;
            } else {
                // boundary gaps or unknown interior: one information-set outcome
                cards.insert(std::upper_bound(unknownBegin(), unknownEnd(), c), c);
            }
        }

        // Draw the top card. Deterministic pop while a known top exists; otherwise the deferred
        // shuffle randomness binds here: one rng draw samples uniformly from the unknown region
        // (none needed when it holds a single card). With the unknown region empty, the known
        // bottom pops deterministically in order.
        CardInstance drawTop(Random &rng) {
            CardInstance c;
            if (knownTopCount > 0) {
                c = cards.back();
                cards.pop_back();
                --knownTopCount;
            } else if (unknownCount() > 0) {
                const int idx = knownBottomCount
                        + (unknownCount() == 1 ? 0 : rng.random(unknownCount() - 1));
                c = cards[idx];
                cards.erase(cards.begin() + idx);
            } else {
                c = cards[knownBottomCount - 1];
                cards.erase(cards.begin() + knownBottomCount - 1);
                --knownBottomCount;
            }
            normalizeSingleton();
            return c;
        }

        // Take all of `from` (a reshuffle source, e.g. the discard pile) into this pile and
        // erase all order knowledge. No rng unless the order is genuinely observed, in which
        // case a concrete shuffle materializes (legacy behavior).
        void absorb(Random &rng, CardPile &from) {
            cards.insert(cards.end(), from.cards.begin(), from.cards.end());
            from.cards.clear();
            from.knownTopCount = 0;
            from.knownBottomCount = 0;
            if (orderObserved) {
                java::Collections::shuffle(cards.begin(), cards.end(), java::Random(rng.randomLong()));
                knownTopCount = size();
            } else {
                knownTopCount = 0;
                knownBottomCount = 0;
                std::sort(cards.begin(), cards.end());
            }
        }

        // In-place reshuffle (Deep Breath): all positional knowledge dissolves; no rng unless
        // the order is observed.
        void reshuffleSelf(Random &rng) {
            if (orderObserved) {
                java::Collections::shuffle(cards.begin(), cards.end(), java::Random(rng.randomLong()));
                knownTopCount = size();
            } else {
                knownTopCount = 0;
                knownBottomCount = 0;
                std::sort(cards.begin(), cards.end());
            }
        }

        void removeAt(int idx) {
            if (inKnownTop(idx)) {
                --knownTopCount;
            } else if (inKnownBottom(idx)) {
                --knownBottomCount;
            }
            cards.erase(cards.begin() + idx);
            normalizeSingleton();
        }

        // Read + remove in one step (card selects, Violence, stasis). The removed card is
        // observed by the effect, so per-outcome removal by index is exact.
        CardInstance takeAt(int idx) {
            CardInstance c = cards[idx];
            removeAt(idx);
            return c;
        }

        void clear() {
            cards.clear();
            knownTopCount = 0;
            knownBottomCount = 0;
        }

        [[nodiscard]] const CardInstance &operator[](int idx) const { return cards[idx]; }
        [[nodiscard]] const CardInstance &back() const { return cards.back(); }
        [[nodiscard]] int size() const { return static_cast<int>(cards.size()); }
        [[nodiscard]] bool empty() const { return cards.empty(); }
        [[nodiscard]] auto begin() const { return cards.cbegin(); }
        [[nodiscard]] auto end() const { return cards.cend(); }

        // Number of cards whose position is known (top/bottom of the pile). Exposed for state
        // hashing/equality and debug printing only — a known top [A,B] and an unknown {A,B} are
        // different information sets. Dynamics code never needs them.
        [[nodiscard]] int knownCount() const { return knownTopCount; }
        [[nodiscard]] int knownBottom() const { return knownBottomCount; }
        [[nodiscard]] bool orderIsObserved() const { return orderObserved; }

        // Apply f to one card / every card, then restore the sort invariant. Only the unknown
        // region is re-sorted: known top/bottom positions are meaningful and survive value
        // mutation. Mutations usually leave the order intact (per-turn cost resets are no-ops
        // most of the time), so check sortedness first — linear vs n log n on the common path.
        template <typename F> void mutateAt(int idx, F &&f) {
            f(cards[idx]);
            if (!inKnownTop(idx) && !inKnownBottom(idx)
                && !std::is_sorted(unknownBegin(), unknownEnd())) {
                std::sort(unknownBegin(), unknownEnd());
            }
        }
        template <typename F> void mutateAll(F &&f) {
            for (auto &c : cards) {
                f(c);
            }
            if (!std::is_sorted(unknownBegin(), unknownEnd())) {
                std::sort(unknownBegin(), unknownEnd());
            }
        }

        bool operator==(const CardPile &rhs) const = default;
    };

}

#endif //STS_LIGHTSPEED_CARDPILE_H
