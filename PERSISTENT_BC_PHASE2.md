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
combat start (no _pbc AND live screen is a genuine combat-start):
    self._pbc = convert_combat_state(...)        # full reconstruction ONCE, at a clean point
    # at fight start there is no hidden history: firstTurn() true, miscInfo fresh, slots == native
    # layout (packing-in-order matches the native encounter; summoners use fixed_slot_layouts)

every decision (top-level OR mid-action sub-decision, distinguished by _pbc.inputState):
    reconcile(self._pbc, snapshot)               # overlay observable truth, KEEP engine hidden state;
                                                 # correct moveHistory[0] ONLY where it diverges from intent
    searcher = BattleSearcher(self._pbc); searcher.search(sims)
    action = searcher.get_best_action()
    send map_search_action_to_spirecomm(action, self._pbc, ...) to live
    action.execute(self._pbc)                    # carry the bc forward to the next input point

monster turn (the action was END_TURN):
    execute(END_TURN) runs the monster turn AND rolls each monster's next intent inside the engine;
    next decision's reconcile corrects only the (rare) intents that diverge from what the live game shows.

mid-action input (execute left _pbc.inputState == CARD_SELECT, e.g. Warcry/Headbutt):
    do NOT re-init; the live bot re-enters handle_combat for the GRID/HAND_SELECT screen and we
    keep advancing the SAME _pbc (search offers the card-select actions, execute drains the queue).

combat end / mid-fight entry with no _pbc:
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
| Monster move for the turn | **engine; snapshot corrects** | the engine already rolled the intent at the end of the prior monster turn (`execute(END_TURN)` → `rollMoveFromInputs`). Compare `moveHistory[0]` to the observed intent and **only on mismatch** call `commit_observed_move(observed)` (setMove + drop any deferred roll). Do **not** unconditionally `setMove` every decision — that shifts the already-correct rolled move into `moveHistory[1]`, corrupting `lastMove`/`lastTwoMoves` and the `monsterData` counters. |
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
2. **Monster moves** — the engine rolls the next intent itself at the end of each monster turn, so in
   the persistent bc the move is *already correct* after `execute(END_TURN)` whenever the engine is
   faithful. We never force speculatively; we only **correct on observed divergence** via
   `commit_observed_move(observed)`. The roll-sweep trick does *not* work (`getMoveForRoll` draws
   internal rerolls), but we don't need it — we're not re-deriving the roll, just overwriting its result
   when it disagrees with the live intent.
3. **The 4 `monsterData` monsters** — `THE_CHAMP`, `DARKLING`, `BOOK_OF_STABBING`, `GREMLIN_WIZARD`
   carry a selection/scaling counter that *is* `miscInfo` (`getMoveForRoll`'s `monsterData` param is
   `miscInfoCopy = in.miscInfo`, written back at `Monster.cpp:805`). Because the engine's own roll
   advances `miscInfo` for these exactly like every other monster, **no manual counter advance is needed
   in the faithful path** — this dissolves the original (a)-vs-(b) question. The only residual is a
   *divergent correction*: `commit_observed_move` fixes `moveHistory` but leaves `miscInfo` at whatever
   the (wrong) roll produced. Gremlin Wizard self-heals (`monsterData = 1` unconditionally next roll);
   Book of Stabbing (stab count → hit count) and Champ (phase-2 bit `0x4` + defensive-stance count
   `&0x3`) could carry a stale counter across a correction. This is rare and only on an already-flagged
   `[pbc DESYNC]`. **Resolution: ship M2/M3 with engine-natural advancement + correction-only override;
   add a C++ `correctSelectionMiscInfo(observedMove)` hook *only if* M3 measurement shows residual
   monsterData drift.** C++ over Python because the semantics live in `getMoveForRoll` and we own the
   engine.
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
- **M3 — the 4 monsterData monsters.** With engine-natural advancement (no manual counter code), run
  full Champ/Darkling/Book of Stabbing/Gremlin Wizard fights under the shadow and check for selection
  drift. Add the C++ `correctSelectionMiscInfo(observedMove)` hook *only if* a real divergence-correction
  leaves a stale counter (per Forcing §3 / decision 1) — otherwise this milestone is a verification, not
  new code.
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

## Resolved decisions (2026-06-28, grounded in the engine code)

1. **monsterData counter (M3) — RESOLVED: engine-natural advancement, correction-only override, C++
   hook deferred.** `monsterData` *is* `miscInfo` (`Monster.cpp:801-805`), and the engine rolls each
   monster's next move at the end of its turn, so `execute(END_TURN)` advances the counter for *all*
   monsters by construction — the original C++-vs-Python framing was a false choice born of the
   `setMove`-every-decision model. Correct moves only on divergence via the existing
   `commit_observed_move`. Add a C++ `correctSelectionMiscInfo` hook *only* if M3 shows residual
   monsterData drift after a real correction (Book of Stabbing / Champ); Gremlin Wizard self-heals and
   Darkling's counter-bearing moves are setMove overrides already. See Forcing §3.

2. **Reconcile granularity (M2) — RESOLVED: lean overlay that shares the pile/status helpers but NOT the
   fresh-build seeding.** Do *not* call `convert_combat_state` wholesale on `_pbc`: it re-seeds
   `miscInfo` from the live intent (`_MISCINFO_*` tables), calls `rollMove`, and sets `moveHistory[0]`
   from scratch — all of which would clobber the engine-maintained hidden state that is the entire point
   of Phase 2. Instead factor the *observable overlay* out of `_set_sts_monster_fields` (hp / block /
   powers / pile contents) into a `reconcile()` that writes onto the existing slots and skips the
   move/miscInfo/rollMove seeding. The miscInfo-restore + rollMove path stays where it belongs: the
   fresh-build (`convert_combat_state`) used at init and on the per-decision fallback.

3. **slot_to_spire maintenance — RESOLVED: advance alongside `_pbc`.** Correct-by-construction:
   `_pbc` is built once and only mutated by `execute`, so the engine owns slot identity across
   splits / summons / deaths. Establish the map at init (in-order packing + `fixed_slot_layouts` for
   summoners, exactly as today's `_build_monster_group`) and maintain it incrementally — on death keep
   the slot entry (engine marks the slot gone; never compact), on summon/split map the new live monster
   to the engine's known target slot via the same fixed-layout rule. Recomputing by identity each
   decision is rejected: it reintroduces the packing artifact this whole phase exists to kill.

4. **Mid-turn extra inputs — RESOLVED: key the lifecycle on `inputState`, continue the same `_pbc`.**
   A card needing a selection (Warcry/Headbutt/Armaments/…) drives `bc.inputState` to `CARD_SELECT`
   (`InputState.h`), and the engine search already resolves these internally as `SINGLE/MULTI_CARD_SELECT`
   actions (`_CARD_SELECT_TASK_BY_ACTION`). After `execute(card)`, `_pbc` sits at the `CARD_SELECT`
   input point with its queue partially drained. The live bot re-enters `handle_combat` for the
   GRID/HAND_SELECT screen — so the lifecycle gate must distinguish *new combat* from *mid-action
   sub-decision*: if `_pbc` exists and `_pbc.inputState != PLAYER_NORMAL` (i.e. `EXECUTING_ACTIONS` /
   `CARD_SELECT`), **continue the same `_pbc`** (search offers the card-select actions; `execute` the
   chosen pick drains the rest of the queue). Only init a fresh `_pbc` when there is none and the live
   screen is a genuine combat-start. This makes "first `handle_combat` of a fight" precise instead of
   heuristic.

## Success criteria

A gated live set where (a) `[pbc DESYNC]` is dominated by the known-irreducible cases (Runic Dome,
Havoc top-of-draw), (b) no live crashes, and (c) winrate moves materially toward the ~75% offline
number — the direct measure that the hidden-state gap, not search strength, was the bottleneck.
