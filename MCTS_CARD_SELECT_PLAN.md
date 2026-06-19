# Plan: drive in-combat card-select with the combat MCTS

## Problem

After deleting the out-of-combat heuristics, comm.py fail-louds on in-combat card-select
screens (HAND_SELECT, or GRID while `in_combat`). These appear when the policy plays a card
whose resolution needs a follow-up pick: Armaments (upgrade a hand card), Headbutt (a discard
card to the draw top), Warcry, Dual Wield, Exhume, Forethought, etc. The net picks such cards
often, so this blocks completing many live games. Per the design call, the *select* is the
combat MCTS's job, not the policy's.

## How the engine already models it (key facts)

- `BattleContext` is a state machine on `inputState` (`include/combat/InputState.h`). Playing
  Armaments in-sim sets `inputState = CARD_SELECT` and `cardSelectInfo.cardSelectTask = ARMAMENTS`;
  the search then resolves the pick as ordinary child actions. So a *normal* search already
  handles selects end-to-end — the only problem is the **live callback boundary**: comm.py sends
  `PlayCardAction(Armaments)`, the live game plays it, and asks for the select on a *separate*
  state-change callback where we must answer.
- The select action carries `getSelectIdx()`, an index into a pile that depends on the task
  (`src/sim/search/Action.cpp:isValidSingleCardSelectAction`):
  - hand: `ARMAMENTS` (upgradeable only), `DUAL_WIELD` (attack/power only), `FORETHOUGHT`,
    `WARCRY`, `EXHAUST_ONE`, `SETUP`, `RECYCLE`, `NIGHTMARE`
  - discardPile: `HEADBUTT`, `HOLOGRAM`, `MEDITATE`, `LIQUID_MEMORIES_POTION`
  - exhaustPile: `EXHUME` (not Exhume itself)
  - drawPile: `SEEK`, `SECRET_TECHNIQUE` (skills), `SECRET_WEAPON` (attacks)
  - multi-select: `EXHAUST_MANY`, `GAMBLE` (handled via `getSelectedIdxs()`; rarer)
- `BattleContext::openSimpleCardSelectScreen(task, count)` (BattleContext.cpp:2921) sets exactly
  `inputState`, `cardSelectTask`, `pickCount`, `canPickZero=false`, `canPickAnyNumber=false` — a
  clean entry point to put a reconstructed bc into the select state.
- The live screen names the triggering action in `game.current_action` (e.g. Headbutt shows
  `BetterDiscardPileToHandAction`; captured GRID, `num_cards=1`, pool = discard pile).

## Chosen approach: reconstruct the select state and re-search (approach A)

Approach B (descend handle_combat's existing search tree to the select child) is rejected:
`BattleSearcherNode` exposes only `simulation_count`/`evaluation_sum` (no child edges), and the
select might not have been expanded; it would also need fragile cross-callback state.

Approach A re-searches at the actual select screen, with no carried state:

1. Reconstruct the bc from the live (mid-select) combat_state: `gc = spirecomm_to_gamecontext;
   bc, slot_to_spire = convert_combat_state`. The live piles already reflect the in-progress
   resolution (triggering card removed from hand, energy spent), so the bc is the correct
   pre-select position.
2. Map `game.current_action` (Java action class name) -> `CardSelectTask` via a new table.
3. `bc.open_card_select(task, num_cards)` (new binding).
4. Configure the searcher exactly as `handle_combat` does (`search_agent.configure_searcher`),
   `search()`, `get_best_action()`.
5. `sel_idx = best.get_select_idx()` indexes the task's pile (above). Translate that pile index
   to the matching card on the live screen (match by card id + position; piles are reconstructed
   in live order) and return `CardSelectAction([live_card])`.
6. Unmapped `current_action` -> fail loud (log the name to extend the table), consistent with the
   no-heuristic policy.

## Work items

1. **Binding** (`bindings/slaythespire.cpp`): bind the `CardSelectTask` enum and
   `bc.open_card_select(task, count)` -> `openSimpleCardSelectScreen`. (`input_state` is already
   bound; `get_select_idx`/`get_selected_idxs` already exist.)
2. **Verify the searcher runs from a CARD_SELECT root** (the one real unknown). Offline-replay the
   captured Headbutt GRID state: reconstruct bc, `open_card_select(HEADBUTT, 1)`, configure +
   `search()`, confirm `get_root_edges()` are select actions over the discard pile and
   `get_best_action()` returns a valid `select_idx`. If the searcher assumes a PLAYER_NORMAL root,
   adjust (e.g. a small engine entry point) before proceeding.
3. **comm.py `mcts_card_select_action()`** + route HAND_SELECT / in-combat GRID to it in
   `handle_screen` (replacing today's `NotImplementedError`).
4. **Tables**:
   - `_CARD_SELECT_TASK_BY_ACTION`: Java action class -> `CardSelectTask`. Seed from decompiled
     `com/megacrit/cardcrawl/actions/**` + live captures; start with the Ironclad set
     (Armaments, Headbutt=`BetterDiscardPileToHandAction`, Warcry, Dual Wield, Exhume,
     Forethought) and extend as `current_action` names surface. Audit like the move-byte table.
   - `_CARD_SELECT_POOL_BY_TASK`: task -> which live screen list / pile to translate the index
     against (hand / discard / exhaust / draw).
5. **Multi-select** (`EXHAUST_MANY`/`GAMBLE`, count>1): use `get_selected_idxs()` ->
   `CardSelectAction([...])`. Rare for Ironclad; can land in a follow-up, fail loud meanwhile.
6. **Validation**: offline replay for Headbutt (have a capture) and any others we can capture;
   then live — confirm Armaments/Headbutt/etc. resolve and games complete past them.

## Risks / unknowns

- **Searcher from a CARD_SELECT root** (work item 2) — the gating unknown; verify first.
- **`current_action` -> task table completeness** — fail-loud surfaces gaps; fill incrementally.
- **Pile-index ↔ live-screen-order match** — both come from the live piles; match by id+position
  to be safe against any reordering.
- **Chained selects** (Armaments upgrades, turn continues; a card that triggers two selects) — we
  answer one screen per callback; the live game re-presents the next, each re-searched. Should
  compose, but watch Nightmare/Setup-style multi-step cards.
- **Reconstruction faithfulness of the mid-select bc** — relies on the live combat_state being
  complete at the select screen (it is for the captured Headbutt). Low risk; fail loud catches
  any divergence.
