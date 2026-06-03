# Search-model inaccuracies (battle engine vs. real Slay the Spire)

Deliberate, documented deviations between this engine's battle model and real StS semantics.
Each entry: what differs, why we accept it, and the bound on the error. Add to this file whenever
a new modeling shortcut is taken.

The framing throughout: search states represent the **player's information set**, not the engine's
ground truth (this is the `rerandomize` philosophy — hidden information is resampled per chance
outcome rather than peeked). Deviations below are places where the exact information set is too
rich to represent and we approximate it.

## 1. Shuffle-into-pile with known stacks: bounded determinization

**Context.** The draw pile is represented as a sorted unknown multiset plus a known top stack of
K cards (Headbutt/Warcry placements, innates at battle start) and a known bottom stack of B
cards (Forethought). "Shuffle a card into the draw pile" (Wild Strike, Power Through, Reckless
Charge, parasite implants, ...) inserts at a uniform gap — in real StS the player does not
observe where it lands, so their true belief after inserting past a known stack is a *mixture*
("my Headbutted card is still on top with probability (N-1)/N").

**Model.** Gaps whose location the player could track afterwards are determinized as chance
outcomes: the K-1 gaps strictly within the known top, and the B gaps within/below the known
bottom (each lands the card at a definite tracked slot). The boundary gaps directly adjacent to
the known stacks are *not* player-distinguishable from the unknown interior, so they fold into a
single "joins the unknown region" outcome — exchangeability of the unknown multiset then
reproduces the legacy positional distribution exactly. All outcome marginals match the legacy
engine's uniform gap choice (1/N per within-stack gap, (U+1)/N for the unknown region), and the
induced draw-sequence distribution is exact (validated by the `verify_draw_dist` χ² harness).
With B = 0 and K < 2 there are no trackable gaps: the insert is deterministic for the player —
no rng, no chance node.

**The inaccuracy.** Each within-stack chance outcome is a *definite* state, so a search line
conditions on where the inserted card landed — information the real player lacks until they
draw. The search is slightly clairvoyant about one card's position, only in lines where a
shuffle-into lands strictly inside a known stack (requires K ≥ 2 or B ≥ 1 at insertion time).

**Why accepted.** The exact belief state is unrepresentable without belief-MDP machinery, and
the marginals are exact — the error is bounded to the value difference between *anticipating*
and *being surprised by* one card's position, typically a status/curse worth little either way.

## 2. Frozen Eye: exact, at the cost of legacy branching

Frozen Eye legitimately reveals the full draw-pile order. Modeled exactly by materializing the
order: while the player has Frozen Eye, every shuffle (battle init, reshuffle-on-empty, Deep
Breath) performs a concrete rng shuffle and sets K = pile size (everything is "known top").
Draws pop deterministically; shuffle-into uses the K>0 path above with K = N (which reproduces
legacy uniform-index insertion exactly). No information inaccuracy — but battles with Frozen Eye
forgo the canonicalization benefits (shuffles are wide chance nodes again, fewer transpositions),
which is the correct price of genuine order knowledge.

## 3. Innate/bottled cards at battle start: internal order fixed

Real StS shuffles innate+bottled cards among themselves on top of the pile; we place them as a
known top in deterministic (sorted) order and consume no rng for the initial shuffle. With ≤5
innate cards they are all drawn turn 1 regardless, so the internal order is unobservable in
practice. (If innate count ever exceeded initial hand size the model would fix an order the real
game randomizes — negligible for Ironclad pools.)

## 4. Unknown-pile draws defer randomness (not an inaccuracy — equivalence note)

Drawing from the sorted unknown multiset samples uniformly without replacement at draw time;
real StS fixes a concrete order at shuffle time and draws deterministically. The induced
distribution over draw sequences is identical (exchangeability), and the player's information
set is identical at every decision point. This is a representation change, not a semantics
change — recorded here to head off confusion.

## 5. Shuffle-in top-promotion: exact distribution, early knowledge timing

The legacy engine (mirroring the game) inserts a shuffled-in card at a uniform gap that
*excludes the very top of the pile* — the inserted card is never the next card drawn. To
represent this exactly, a shuffle-in with no known top first samples one unknown card and
promotes it to the known top (it was the pre-insert top: uniform over the unknown region from
the player's view), then the inserted card joins the unknown region below it. The joint
distribution over draw sequences is exact, and the chance node merely *moves* — insert-time
promotion (U outcomes) replaces the following draw's sampling (which becomes a deterministic
pop).

**The inaccuracy** is knowledge timing only: the player learns their next draw at insert time
rather than at the draw itself, so decisions between the shuffle-in and the next draw are
slightly clairvoyant about one card — mild and bounded (the same card would be revealed one
observation later anyway). When a known top already exists (K ≥ 1) the exclusion is automatic
and no promotion happens.
