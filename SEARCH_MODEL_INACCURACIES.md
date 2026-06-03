# Search-model inaccuracies (battle engine vs. real Slay the Spire)

Deliberate, documented deviations between this engine's battle model and real StS semantics.
Each entry: what differs, why we accept it, and the bound on the error. Add to this file whenever
a new modeling shortcut is taken.

The framing throughout: search states represent the **player's information set**, not the engine's
ground truth (this is the `rerandomize` philosophy — hidden information is resampled per chance
outcome rather than peeked). Deviations below are places where the exact information set is too
rich to represent and we approximate it.

## 1. Shuffle-into-pile with a known top: bounded determinization

**Context.** The draw pile is represented as a sorted unknown multiset plus a known top stack of
K cards (Headbutt/Warcry placements, innates at battle start). "Shuffle a card into the draw
pile" (Wild Strike, Power Through, Reckless Charge, parasite implants, ...) inserts at a uniform
position among all N+1 gaps — in real StS the player does not observe where it lands, so their
true belief after inserting past a known top is a *mixture* ("my Headbutted card is still on top
with probability N/(N+1)").

**Model.** With K = 0 the insertion is deterministic (joins the unknown multiset; no rng, no
chance node). With K > 0 the insertion is a chance event over K+2 outcomes: one for each slot
relative to the known stack (the card joins the known top at that definite slot), plus one for
"joins the unknown region". Probabilities are exact (1/(N+1) per known slot, U/(N+1) for
unknown).

**The inaccuracy.** Each chance outcome is a *definite* state, so a search line conditions on
where the inserted card landed — information the real player lacks until they draw. The search is
slightly clairvoyant about one card's position, only in lines where a shuffle-into occurs while a
known top exists.

**Why accepted.** The exact belief state is unrepresentable without belief-MDP machinery. The two
cheap alternatives are both worse: always-below-known-top makes Headbutt setups unbreakable
(player-favorable dynamics error); dissolving the known top makes the next draw uniform
(destroys nearly all of Headbutt's real value, a large dynamics error in a common Ironclad
interaction). The accepted error is bounded to the value difference between *anticipating* and
*being surprised by* one card's position — typically a status/curse worth little either way.

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
