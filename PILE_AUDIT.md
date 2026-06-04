# Pile-usage audit (canonical-deck Phase A deliverable)

Every site touching `drawPile` / `discardPile` / `exhaustPile` outside the searcher, classified.
Gate for Phases B/C of `CANONICAL_DECK_PLAN.md`.

## Classification legend
- **SIZE/CONTENT** — reads size or membership only; order-independent; no change needed.
- **MUTATOR** — Phase B/C work list (must maintain sort / known-top invariants).
- **RNG-EVENT** — consumes rng over pile contents; distribution-invariant under sorted
  representation (different concrete rng stream, same outcome distribution).
- **INDEX-CHOICE** — exposes pile indices across an action boundary (card-select screens);
  indices remain *internally consistent* under sorted piles (enumeration and resolution see the
  same ordering) but their meaning differs from legacy ⇒ breaks replay of old recordings.

## drawPile

| site | what | class |
|---|---|---|
| CardManager: draw pop (`drawPile.back()`), moveToDrawPileTop, insertToDrawPile, shuffleIntoDrawPile, createTempCardInDrawPile, removeFromDrawPileAtIdx, moveDiscardPileIntoToDrawPile, notifyAdd/Remove (blood counts) | core mutators | MUTATOR (Phase C table) |
| Actions.cpp:215 `_ShuffleDrawPile` (Deep Breath) | in-place shuffle | MUTATOR → re-sort, K=0 |
| Actions.cpp:226/249/509 `_ShuffleTempCardIntoDrawPile` / `_PutRandomCardsInDrawPile` | rng-index insert | MUTATOR + RNG-EVENT → K-aware insert (chance event only when K>0) |
| Actions.cpp:556 `_ViolenceAction` | random ATTACKs pulled from pile by index | RNG-EVENT; selection = uniform subset of attacks (order-invariant). ⚠ Phase C: removal may hit known-top cards — resolve known-top membership exactly (observable, deterministic per outcome) |
| Actions.cpp:763 `_DrawToHandAction` + BattleContext:3082 `chooseDrawToHandCards` | enumerate matching cards → card-select → remove by idx | INDEX-CHOICE (stable within action; reveals content only, fine) |
| Actions.cpp:911 (Apotheosis-class: upgrade all in pile) | mutates card values in place | MUTATOR ⚠ re-sort after value mutation (sort key includes upgrade/cost) |
| BattleContext:1098 Mind Blast (`size()` as damage), 2461-2535 draw/reshuffle plumbing, 629 updateCardsOnExit, 782 emptiness | size/content | SIZE/CONTENT |
| BattleContext:3079 `putOnTopOfDrawPile` (Warcry) | hand → top | MUTATOR → ++K |
| CardInstance.cpp:305/314 (`std::find` membership for cost triggers) | membership | SIZE/CONTENT |
| MonsterSpecific:3472 Bronze Automaton stasis | rarity-tiered rng pick, card removed & observed | RNG-EVENT (stasisHelper is content-based + rng tiebreak ⇒ distribution-invariant); chance event resolves which card — exact |
| Player.cpp:668 emptiness | size | SIZE/CONTENT |

## discardPile

| site | what | class |
|---|---|---|
| moveToDiscardPile (every card play), removeByUniqueId (back() fast path falls back to scan — stays correct), moveDiscardPileIntoToDrawPile | core mutators | MUTATOR (insort on add) |
| BattleContext:3008/3061 `chooseHeadbuttCard(discardIdx)` + the discard card-select enumeration | player picks a discard card by index | INDEX-CHOICE |
| CardManager:482 blood-card iteration, content scans | content | SIZE/CONTENT |
| `_EmptyDeckShuffle` reads discard as shuffle input | consumed by full shuffle | dynamics-neutral under sorting |
| stasis fallback (discard variant) | as above | RNG-EVENT |

## exhaustPile

| site | what | class |
|---|---|---|
| add-on-exhaust, removeByUniqueId | mutators | MUTATOR (insort) |
| BattleContext:3043 `chooseExhumeCard(exhaustIdx)` + exhaust card-select | index choice | INDEX-CHOICE |
| content scans (CardManager:516 etc.) | content | SIZE/CONTENT |

## Order-knowledge sources (survey result)
**Frozen Eye only** (handled exactly per the plan: materialize order, K=size). No Ironclad-pool
card grants draw-order knowledge; innate/bottled initial placement covered by inaccuracies doc #3.

## Key correction surfaced by this audit
**Phase B (discard/exhaust sorting) already breaks replay of existing recordings**: recorded
card-select actions (Headbutt/Exhume/draw-to-hand) store pile indices whose meaning depends on
pile ordering. Although discard sorting is dynamics-neutral for play, replaying an old recording
on the sorted engine resolves those indices to different cards ⇒ divergence. State-set
regeneration is therefore required from Phase B onward, not just Phase C. (Searches and fresh
recordings are self-consistent — enumeration and resolution share one ordering.)
