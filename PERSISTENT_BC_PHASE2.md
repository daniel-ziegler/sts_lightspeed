# Persistent BattleContext bridge — Phase 2 plan (flip search onto the persistent bc)

Updates the Phase-2 sketch in `PERSISTENT_BC_PLAN.md` with what we now know after Phase 1 is fully
built and the divergence data is in. Read that doc first for the why; this is the how.

## Where we are

- **Phase 0 / 1 done.** `Monster::setMove` bound; the shadow (`_shadow_card_play_check`) carries the
  prior decision's reconstructed bc + chosen action, advances it with `Action.execute`, and diffs the
  deterministic result against this decision's fresh reconstruction — for CARD and END_TURN. Logging
  only; live play still runs on per-decision reconstruction (`convert_combat_state`, "packing").
- **Phase 1 has served its purpose.** Across archived runs: 30k `[shadow ok]`, ~1.7k `[shadow DIVERGE]`.
  Today's breakdown of the divergences reframes Phase 2:

| Divergence | Count | What it actually is |
|---|---|---|
| monster hp/block | 1238 | **mostly a measurement artifact** — the shadow diffs *two independently packed* reconstructions; when their slot order differs, the same hit looks like it hit different monsters (the SlaverBlue/SlaverRed swap). Real play is self-consistent (one reconstruction drives both search and the live command). |
| hand size | 416 | ET `+3` draw-count signal + per-card draw mismatches. Partly real (draw reconciliation), partly RNG. |
| Havoc | ~73 | shadow can't reproduce the random top-of-draw Havoc plays. Not an engine bug. |
| player hp/block/energy | 140 | the genuinely real, high-value ones (e.g. the Bronze Automaton Stasis hand-count, Feed/minion) |

**The reframe:** the dominant "monster hp" bucket is largely the *packing-twice* artifact, not a live
bug. That is itself the strongest argument for Phase 2: with **one** persistent bc, there is no second
reconstruction to disagree with, so that whole bucket dissolves *and* the hidden-state losses (the
ones that actually cost winrate — `miscInfo`/`specialData`/scaling/`moveHistory`) are maintained for
free. Phase 2 is both the fix and a cleaner measurement.

## Architecture

One `self._pbc` (persistent BattleContext) per combat, gated behind `STS_PERSISTENT_BC` (default off).

```
combat start (first handle_combat of a fight, or _pbc is None):
    self._pbc = convert_combat_state(...)        # full reconstruction ONCE, at a clean point
    # at fight start there is no hidden history: firstTurn() true, miscInfo fresh, slots == native
    # layout (packing-in-order matches the native encounter; summoners use fixed_slot_layouts)

every decision:
    reconcile(self._pbc, snapshot)               # overlay observable truth, KEEP engine hidden state
    searcher = BattleSearcher(self._pbc); searcher.search(sims)
    action = searcher.get_best_action()
    send map_search_action_to_spirecomm(action, self._pbc, ...) to live
    action.execute(self._pbc)                    # carry the bc forward through the action queue

monster turn (the action was END_TURN):
    execute(END_TURN) runs the monster turn inside the engine;
    next decision's reconcile setMove()s each monster's observed intent + overlays results.

combat end / new combat / mid-fight entry:
    self._pbc = None  -> next combat re-inits; mid-fight entry falls back to per-decision reconstruction.
```

The key difference from Phase 1: today we `execute` the prior bc only to *compare* it, then throw it
away and search on a fresh reconstruction. Phase 2 **keeps** the executed bc and searches on it.

## Reconcile spec (the real work)

`reconcile(pbc, snapshot)` — applied before each search. Field-by-field source of truth:

| Field | Source | How |
|---|---|---|
| Player hp / block / energy | **snapshot** | overlay; if engine ≠ snapshot beyond tol, log fidelity bug, then hard-set |
| Monster hp / block / statuses | **snapshot** | overlay onto the matching slot (slots are stable now — see below) |
| Monster move for the turn | **snapshot intent** | `setMove(observedMove)` per monster (maintains `moveHistory`) |
| Hand / draw / discard / exhaust **contents** | **snapshot** | replace from snapshot (reuse `convert_combat_state` pile logic) — draw order is RNG, snapshot is ground truth |
| `miscInfo` / `uniquePower0/1` / `moveHistory` / phase counters | **engine (pbc)** | keep — the whole point |
| Card `specialData` / `strikeCount` / accumulated scaling / `uniqueId` | **engine (pbc)** | keep |

**Slots are stable by construction now.** Because `_pbc` is built once and only advanced by `execute`,
the engine's own split/summon/death logic owns slot identity. Reconcile overlays hp/block onto the slot
the engine already has — it never re-packs. The slot↔live-monster map is established at init and
maintained by `slot_to_spire` advanced alongside (dead monsters: the engine marks the slot gone; we
keep the map, don't compact).

**How much reconcile is really needed** scales inversely with Phase-1 fidelity. At the limit (engine
faithful) reconcile is just `setMove` + a player-hp drift correction. We start conservative (overlay
everything observable) and remove overlays as the shadow proves each axis clean.

## Forcing details

1. **Player actions** — already covered. `Action.execute(pbc)` runs the card/potion/end-turn/select
   through the engine to the next input point. No new work; this alone preserves specialData/scaling/
   uniqueId, today's most impactful loss.
2. **Monster moves** — `setMove(observedMove)` covers **61/65** monsters (selection reads only
   `moveHistory`). The roll-sweep trick does *not* work (`getMoveForRoll` draws internal rerolls).
3. **The 4 `monsterData` monsters** — `THE_CHAMP`, `DARKLING`, `BOOK_OF_STABBING`, `GREMLIN_WIZARD`
   read/write a selection counter in `miscInfo`. `setMove` won't advance it. Options: (a) a small C++
   `advanceSelectionMiscInfo(observedMove)` hook, or (b) infer the counter from observed move history in
   Python. Only Champ is a boss; bounded surface. **Decide (a) vs (b) before coding the monster path.**
4. **Draw / shuffle** — force from the resulting snapshot hand (simpler than setting pre-shuffle pile
   order, and it's exactly what the snapshot gives). Havoc/Mayhem top-of-draw stays irreducibly
   unforceable mid-turn → overlay from the next snapshot.
5. **Runic Dome** — intent hidden, can't force pre-resolution. Let the engine roll, reconcile observable
   after. The one irreducible case; already handled by the deferred-roll path.

## Risk gating & rollout

- **`STS_PERSISTENT_BC` env, default OFF.** Cannot regress live until proven. `run_live.sh`/`run_batch.sh`
  pass it like the other knobs.
- **Fallback** to per-decision reconstruction whenever `self._pbc is None` (mid-combat entry, new
  combat, or any reconcile/execute exception — never crash live).
- **Shadow stays on** as a continuous assert even after the flip: now it compares pbc-advanced vs fresh
  reconstruction, so any new divergence is still surfaced.
- **Reconcile-time assert**: if engine observable ≠ snapshot beyond tolerance, log `[pbc DESYNC]` with
  the field + delta, then hard-overlay reality and continue. This is the Phase-2 form of the fidelity
  checker.

## Milestones

- **M1 — lifecycle skeleton (no reconcile).** Behind the env flag: init `_pbc` at combat start, search
  on it, `execute` the chosen action, reset to None on combat end / mid-entry. No overlay yet → expect
  drift; goal is just the plumbing + fallback paths. Verify it never crashes live (offline replay +
  one gated live game).
- **M2 — full reconcile overlay.** Implement `reconcile()` overlaying every observable field +
  `setMove`. With full overlay, the pbc should track reality as well as today's reconstruction *plus*
  keep hidden state. Run the shadow alongside; `[pbc DESYNC]` count should be ≤ the Phase-1 real
  divergences (artifact bucket gone).
- **M3 — the 4 monsterData monsters.** Add the chosen counter-advance mechanism; verify Champ/Darkling/
  Book of Stabbing/Gremlin Wizard fights show no selection drift over a full fight.
- **M4 — thin the overlay.** Per axis the shadow shows clean, stop overlaying it (trust the engine).
  Each removal is a measurable fidelity claim.
- **M5 — measure.** Gated 10–20 game live set at fixed seed/ascension, compare to the ~75% offline
  target. Promote clean asserts to hard.

## What Phase 2 deletes (the payoff, concretely)

- **Slot/target "monster hp" divergences** — gone; one bc, native slots, never re-packed.
- **Giant Head It-Is-Time count, the `_MISCINFO_DAMAGE_MOVE_INTS` restore table, Book-of-Stabbing stab
  count, stasis return** — gone as a *class*; these counters advance with the bc instead of being
  reconstructed (the restore table stays only as the mid-entry fallback).
- **Card scaling** (Rampage / Ritual Dagger / Genetic Algorithm / Glass Knife / Perfect Strike) — kept
  by the engine instead of needing per-card `specialData` reconstruction.

## Open decisions (resolve before coding the relevant milestone)

1. **monsterData counter (M3):** C++ `advanceSelectionMiscInfo` hook vs Python inference. C++ is more
   robust (matches engine semantics exactly); Python avoids a rebuild. Lean C++ — it's ~4 monsters and
   we control the engine.
2. **Reconcile granularity (M2):** start by reusing `convert_combat_state`'s pile/overlay code wholesale
   against `_pbc` (overwrite observable, keep hidden), or write a leaner `reconcile`? Start by reusing —
   it's proven — then thin in M4.
3. **slot_to_spire maintenance:** advance it alongside `_pbc` (track engine slot → live index across
   summons/deaths), or recompute by identity each decision? Advancing is correct-by-construction;
   recomputing risks reintroducing the packing artifact. Lean advance.
4. **Mid-turn extra inputs** (card-select screens spawned by a card, e.g. Warcry/Headbutt): confirm
   `execute` runs the queue to *each* input point and we re-enter `handle_combat` cleanly for the sub-
   decision while keeping the same `_pbc`.

## Success criteria

A gated live set where (a) `[pbc DESYNC]` is dominated by the known-irreducible cases (Runic Dome,
Havoc top-of-draw), (b) no live crashes, and (c) winrate moves materially toward the ~75% offline
number — the direct measure that the hidden-state gap, not search strength, was the bottleneck.
