# Plan: canonical deck representation for the battle search

Goal: unify more transposition states and shrink chance-node branching by representing piles as
the player's information set — sorted discard/exhaust, draw pile = sorted unknown multiset +
known top stack — and deferring shuffle randomness to draw time. Companion doc:
`SEARCH_MODEL_INACCURACIES.md` (accepted deviations live there).

## Representation (CardManager)

- `discardPile`, `exhaustPile`: kept **physically sorted** (CardInstance's defaulted `<=>`).
- `drawPile`: top = back (unchanged). Invariant: `[0, size-K)` is the **unknown region, kept
  sorted**; `[size-K, size)` is the **known top in true order**. New field `int drawKnownTop`.
- Physical canonicalization means `equalForSearch`/hash work unchanged and stay cheap.

## Dynamics changes

| event | today | new |
|---|---|---|
| battle init | concrete shuffle (1 rng) | unknown region sorted; innate+bottled = known top; **no rng** (inaccuracy #3) |
| reshuffle-on-empty (`_EmptyDeckShuffle`) | shuffle discard (1 rng) | sorted discard merges into unknown region; K=0; **no rng**; onShuffle relic triggers unchanged |
| `_ShuffleDrawPile` (Deep Breath) | shuffle in place (1 rng) | K=0, re-sort; **no rng** |
| shuffle-into (`shuffleIntoDrawPile`, `_ShuffleTempCardIntoDrawPile`) | uniform concrete index (rng) | **K=0: deterministic insort, no rng.** K>0: rng gap g∈[0,N]; g≤K joins known top at slot g, else insort unknown — K+2-outcome chance event, exact probabilities (inaccuracy #1) |
| draw / play-top (Havoc) | pop back (deterministic; order was fixed earlier) | K>0: pop known top, deterministic. K=0: **rng-sample uniform from unknown region** — randomness binds at observation |
| `moveToDrawPileTop` (Headbutt/Warcry) | push back | push back, ++K |
| random-card-from-draw effects | rng index | rng-sample unknown region (identical distribution); known-top cards excluded only if the real effect excludes them (audit) |

**Frozen Eye (explicit):** checked once at battle init → `bool drawOrderKnown`. While set, every
shuffle event performs a legacy concrete rng shuffle and sets **K = size** (the entire pile is
"known top"); draws pop deterministically; shuffle-into runs the K>0 path with K=N, which exactly
reproduces legacy uniform insertion. Exact semantics, no inaccuracy; such battles simply forgo
unification benefits (inaccuracy doc #2). No other Ironclad-pool source of full-order knowledge
is known — Phase A audit confirms.

**Searcher: zero changes.** Stochastic-action detection (rng counter), `Random(base+N)` outcome
resampling, and post-state dedup already do the right thing; the representation makes equal
information-sets actually compare equal. A reshuffle+draw-5 chance node goes from
always-distinct concrete orders to 5-card multisets that collide heavily.

## Phases

**A. Instrumentation + audit (first, independent).**
- BattleSearcher counters: nodes created; chance-**sibling reuse** (resampled outcome == existing
  sibling, incl. the dedup-hit-already-a-sibling case); **true transpositions** (dedup hit not a
  sibling / any dedup hit on deterministic-edge expansion); chance outcomes sampled; nodes per
  simulation. Aggregate across threads; `STATS` line in eval_states output next to SCORE.
- Capture **before** numbers: states_fs boss sets + a uniform set, budgets 1000 and 5000.
- Code audit deliverable: complete inventory of (a) pile-order readers, (b) pile mutators,
  (c) rng-consuming pile events, (d) anything indexing piles across an action boundary,
  (e) order-knowledge sources (expected: Frozen Eye only).

**B. Discard + exhaust sorting (small, independent).** Physically sorted at mutation. Dynamics-
neutral (full shuffle erases order; the uniqueId fast-path at CardManager.cpp:498 falls back to
scan). Not byte-identical (same rng over a different input order) — validate distributionally +
SCORE-neutral + unification stats ↑.

**C. Draw pile knownTop + unknown multiset (core).** Representation, all table-row changes above,
Frozen Eye mode, `drawKnownTop` in equalForSearch/hash. Unit-level: distribution-equivalence
harness (many-seed draw-frequency χ² old vs new from identical states; Headbutt/Wild Strike
interaction cases explicitly).

**D. Validation + retune.**
- Unification telemetry before/after on identical sets/budgets (sibling-reuse %, transposition %,
  nodes/sim).
- ⚠ Replay-compat break: all existing recorded state sets are invalid under new dynamics →
  regenerate via collect_boss_states2 (~1h/set); keep the old-engine commit for reference.
- Quality: per-battle SCORE at fixed sims and fixed time on regenerated sets; **unified
  full-game A/B** (eval_hero, 1000 paired seeds, old vs new engine) — cross-engine win rates are
  comparable even though individual games diverge.
- Re-tune chance widening afterward (cheaper, unifying chance nodes plausibly shift the optimum;
  the in-flight round-3 study is the old-representation baseline).

## Risks / open items
- Audit completeness is the gate for C (order-readers indexed across action boundaries).
- uniqueId participates in CardInstance equality → cross-instance unification (Strike#1↔Strike#2)
  remains unaddressed (separate canonicalization project; known ceiling on wins here).
- Rare recording divergence (~0.1% of collection games, unexplained) — scanner stays in the
  collection flow; unrelated but shares the replay machinery.
