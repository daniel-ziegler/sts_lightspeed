# Persistent BattleContext bridge — plan

## Problem

The live bridge (`comm.py`) rebuilds a fresh `BattleContext` from the spirecomm snapshot at **every**
combat decision (`convert_combat_state`). The snapshot only carries *observable* state, so every
rebuild **drops hidden state** the engine needs to play correctly:

- monster phase / counter state in `miscInfo`, `uniquePower0/1` (Champ phase + stance count, Time
  Eater turn count, Awakened One revive, Bronze Automaton / Slime Boss / Collector spawn counters),
- `moveHistory` (drives `firstTurn()` and "can't repeat move N times" gating),
- card `specialData` (Rampage / Ritual Dagger / Genetic Algorithm / Glass Knife scaling),
- `strikeCount` / per-turn play counts, accumulated scaling.

Offline `agent.playout_battle(gc)` keeps **one continuous `BattleContext`** that the engine advances
move-by-move, so all of that is maintained for free → ~75% win. Live reconstructs lossily → ~10%.

We confirmed the cheap explanations are **not** the gap: tree-reuse off costs only a couple percent,
and RNG fidelity is no longer maintained so same-seed offline replay diverges (can't diff
trajectories). The remaining gap is hidden-state / fidelity, and there is no single static assert
that covers every axis. We need to stop losing the hidden state in the first place.

## Goal

Maintain **one persistent `BattleContext` per combat**, advanced by the engine as the real fight
progresses, instead of rebuilding it each decision. Feed the engine the **observed** non-deterministic
outcomes (monster moves, card draws) so its hidden counters stay correct, and use a light observable
overlay as a safety net + **continuous fidelity assert**.

## Core design: engine-advanced with forced inputs

A persistent bc is only correct if the engine's play matches reality. RNG fidelity is broken, so we
**force** the engine at every non-determinism point using what the real game reveals:

1. **Init at combat start.** Build the bc once via the existing full `convert_combat_state` — valid
   because at fight start there is no hidden history (`firstTurn()` genuinely true, `miscInfo` fresh).
2. **Player actions: apply exactly.** When our bot plays a card / uses a potion / ends turn, apply the
   *same* action to the persistent bc through the engine, then send it to the real game (current path).
   We know our own actions precisely — zero divergence here.
3. **Monster turn: force observed moves.** After `execute(END_TURN)` advances the bc through the
   monster turn, override each monster's next move with the real game's observed intent via
   `setMove(observedMove)` (maintains `moveHistory`, which drives selection for 61/65 monsters). The 4
   `monsterData`-using monsters (Champ/Darkling/Book of Stabbing/Gremlin Wizard) additionally need their
   selection counter reconciled — see Scoping findings. (The engine's own rolled move during
   `execute` is discarded by the override; we only need the observed one for the search's next decision.)
4. **Draw: force observed cards.** When the engine would draw/shuffle (turn start, Battle Trance,
   Pommel Strike…), override the result with the cards the next snapshot actually shows in hand.
5. **Observable overlay = safety net + assert.** After applying our action and reconciling, overlay
   player HP/block/energy and monster HP/block/statuses from the snapshot. Critically: if the engine —
   *fed the real moves and draws* — still disagrees with reality beyond tolerance, that is an **engine
   fidelity bug**. Log it (it's exactly what we're hunting), then hard-overlay reality to stay aligned.

Insight: this makes the bridge a **continuous fidelity checker**. Every live turn becomes an assertion
that "engine + real inputs == real result." That surfaces the remaining bugs without needing
RNG-matched replay.

## What forcing covers

In-combat RNG that actually moves hidden state: **monster move selection** and **shuffle/draw order**
(plus a handful of card effects — Chaos/Wish/Discovery). StS monster *damage* is deterministic per
move, so forcing move + draw covers ~all divergence that matters. Card-effect RNG is rare and can fall
back to overlay.

## Scoping findings (verified against the code)

**Player-action apply is already done.** `Action::execute(BattleContext&)` is bound
(`bindings/slaythespire.cpp:423`). Calling `action.execute(persistent_bc)` runs a card / potion /
end-turn / card-select through the engine, executing the action queue to the next input point. So
driving the persistent bc forward on *our* moves needs **no new engine work** — and that alone keeps
card `specialData`, `strikeCount`, accumulated scaling, and `uniqueId` correct, which is the most
common (and most impactful) reconstruction loss today.

**Monster-move forcing is mostly `setMove`, with a tiny residual.** `Monster::setMove(moveId)` just
updates `moveHistory[0/1]` (no miscInfo side effects). That is *sufficient* for the **61 of 65**
monsters whose `getMoveForRoll` decides purely from `moveHistory` (`lastMove`/`lastTwoMoves`) + the
roll — `setMove(observedMove)` keeps their future selection correct for free.

The brute-force-the-roll idea does **not** work: `getMoveForRoll` is not pure in `roll` — it draws
extra `bc.rng.randomBoolean()` rerolls internally (ACID_SLIME_M, many slimes), so you can't reproduce
an arbitrary observed move by sweeping `roll` 0..99.

Only **4 monsters** read/write `monsterData` (their selection-time miscInfo) in `getMoveForRoll`:
**`THE_CHAMP`, `DARKLING`, `BOOK_OF_STABBING`, `GREMLIN_WIZARD`** (7 references total; MonsterSpecific.cpp
~1893+). For these, `setMove` alone won't advance the selection counter — they need per-monster
handling (infer the counter from observed move history, or a dedicated "advance miscInfo for the
observed move" hook). Only **Champ** is a boss, so the residual per-monster surface is small and
bounded — a far cry from "reconstruct all hidden state per boss." (Note: miscInfo *also* feeds damage
in `takeTurn` for Louse/Hexaghost/etc., but that axis is already covered by the existing
`_MISCINFO_DAMAGE_MOVE_INTS` reconstruction + intent-damage assert.)

## Bindings needed (pybind)

Much smaller than first thought:

- **`Monster::setMove(moveId)`** — expose (the primary move-forcing primitive; `miscInfo`,
  `uniquePower0/1`, `moveHistory` reads are already bound rw).
- Per-monster miscInfo advancement for the **4** `monsterData` monsters — either a small C++
  `advanceSelectionMiscInfo(observedMove)` hook, or reconstruct the counter in Python from observation.
- `Action::execute` — **already bound** (player actions).
- Draw/hand/pile reconciliation — **reuse the existing per-decision reconstruction** as an overlay; no
  new binding.
- HP/block/statuses read for the reconciliation assert — already exposed.

## Reconciliation rules (which fields win)

| Field class | Source of truth |
|---|---|
| Player HP / block / energy | snapshot (overlay; assert engine ≈ snapshot) |
| Monster HP / block / statuses | snapshot (overlay) |
| Hand / draw / discard / exhaust **contents** | snapshot (RNG draw is ground truth) |
| Monster **move** for the turn | snapshot intent → forced into engine |
| `miscInfo` / `uniquePower` / `moveHistory` / phase counters | **engine** (the whole point) |
| Card `specialData` / `strikeCount` / scaling | **engine** |

## Edge cases

- **Combat entered mid-fight** (bot restart / reconnect / first decision we ever see in a fight):
  no persistent bc → fall back to current per-decision reconstruction. Detect by `persistent_bc is None`.
- **Runic Dome** (intent hidden): can't force pre-resolution; let the engine roll (deferred-roll path),
  then reconcile observable after the monster acts. The one irreducible case.
- **Desync beyond tolerance**: log as fidelity bug, hard-overlay reality, continue (never crash live).
- **Cards that draw/shuffle mid-turn**: force from the resulting snapshot.
- Ironclad-only for now (no stance/orb) simplifies, though the engine handles those anyway.

## Status (2026-06-21)

- **Phase 0 done:** `Monster::setMove(moveId)` exposed (`bindings/slaythespire.cpp`, built OK).
- **Phase 1a done (shadow, logging only):** `comm.py` carries the prior decision's reconstructed bc +
  chosen action; for a CARD play, `handle_combat` advances the prior bc via `Action.execute` and diffs
  the deterministic result (player/monster hp/block/energy, hand size) vs this decision's ground-truth
  reconstruction. Emits `[shadow ok|DIVERGE|ERR] (CARD|ET) after <desc>`. Validated offline.
- **Phase 1b done (END_TURN shadow):** same method now also handles END_TURN. Because the
  reconstruction sets each monster's *current* move from the visible intent, `execute(END_TURN)` runs
  the real monster moves, so any post-monster-turn hp/block divergence is a genuine monster-turn
  fidelity bug. No turn-guard on the ET branch (it legitimately crosses the boundary). Validated
  offline (player takes monster damage / block absorbs / hand redraws). Caveat: END_TURNs issued at the
  `get_next_action_in_game` line ~2419 path (nothing playable) aren't shadowed yet — only
  search-chosen END_TURNs via `handle_combat`. Acceptable; most go through `handle_combat`.
- **v9 run (3 games, Phase 1a only — launched before the 1b edit):** capture
  `runs/comm_capture_v9.jsonl`, errlog completion baseline 27 (watch 30), ckpt iter_2005. Monitor
  collects `[shadow ...]` counts + win/loss.

**v9 shadow results (1 game before a leaked-proc contaminated the completion count — ignore winrate):**
564 `[shadow ok]`, 11 `[shadow DIVERGE]`, 0 ERR. Breakdown:
- 8× **Havoc+** — hidden draw-pile order (Havoc plays top-of-draw; engine's top card differs). EXPECTED,
  not a bug. A Phase-2 reconcile consideration (can't predict top-of-draw).
- **Feed → BronzeOrb** (`php +3`) and **Bash/Feed → BronzeOrb** (`hand -1`): two real Bronze Automaton bugs.

**FIX #1 done (Feed/minion):** the live `'Minion'` monster power was in `_ALL_POWER_IDS` but not the
status map, so `apply_monster_power` *dropped* it — reconstructed Bronze Orbs/Daggers/TorchHeads lacked
`MS::MINION`, so Feed (which checks `!hasStatus<MINION>()`, Actions.cpp:985) wrongly raised max HP on a
minion kill. Bound `MonsterStatus::MINION` (bindings) + mapped `'Minion'` in comm.py. Rebuilt.

**STILL OPEN:**
- **Bronze Automaton hand `-1`** (Bash/Feed → BronzeOrb): engine ends up 1 card short. Hypothesis:
  **Stasis** card-return not modeled (a Bronze Orb dying returns the stolen card to hand), or the
  stasis'd card isn't reconstructed. `apply_monster_power` drops `Stasis`. Investigate engine + recon.
- **Feed → GremlinWizard** (`php +3`): wizard isn't a minion, so kill→+3 is correct *if* it died — so
  this is a kill-detection / monster-HP / vulnerable divergence (engine thought Feed killed, live no).
- **Run hygiene:** a leaked prior-run comm.py inflated the completion count (saw "Game 9,10,1"). Before
  any winrate measurement, hard-verify `ps aux|grep comm.py` == 0 AND `tasklist java` == 0 after kill,
  and baseline by the v9-portion of the errlog, not the whole file.

**Next:** (1) build done → relaunch a clean set (Phase 1b shadow now live) to get `(ET)` monster-turn
divergences + confirm Feed/minion fixed; (2) chase the Stasis hand-count bug; (3) then Phase 2.

### Phase 2 design (flip search onto the persistent bc) — depends on Phase 1 divergence being small

If the shadow shows `execute` is faithful (few DIVERGE), the persistent bc we already advance each
decision *is* accurate, so Phase 2 is light:
- Gate behind `STS_PERSISTENT_BC` env (default off) so it can't regress live until proven.
- Lifecycle: build `self._pbc` once at combat start (existing `convert_combat_state`); each decision,
  **reconcile** it to the snapshot, search on it, then `execute` the chosen action to carry forward.
- **Reconcile overlay** (the real work): overwrite observable fields from the snapshot (player
  hp/block/energy, monster hp/block/statuses), `setMove(visibleIntent)` each monster, replace
  hand/draw/discard/exhaust contents (reuse the pile logic in `convert_combat_state`), but KEEP engine
  hidden state (miscInfo/specialData/moveHistory). The lighter the divergence Phase 1 shows, the less
  reconcile is needed — at the limit, trust the bc and only `setMove` + fix HP drift.
- Fallback to per-decision reconstruction when `self._pbc is None` (mid-combat entry / new combat) and
  as a cross-check. Keep the shadow running as a continuous assert.

## Phased rollout

- **Phase 0 — infra.** Add the bindings above. Unit-test on captured fights: init bc, force the
  recorded moves + draws, step through, confirm it reaches the same observable states.
- **Phase 1 — shadow mode (high value, zero risk).** Maintain the persistent bc *in parallel* with the
  existing per-decision reconstruction, but keep driving the live bot from the **old** path. Each turn,
  run the reconciliation assert (engine-prediction vs reality) and log divergences. This is a
  continuous fidelity checker that finds the remaining bugs **without** putting the persistent bc on the
  critical path. Fix surfaced bugs here.
- **Phase 2 — flip the search to the persistent bc.** Once Phase 1 divergences are small/understood,
  run MCTS on the persistent bc. Keep per-decision reconstruction as the mid-combat-entry fallback and
  as a cross-check.
- **Phase 3 — tighten + measure.** Promote asserts to hard where clean, run a 10–20 game live set,
  compare to the ~75% offline target.

## Why this is the right investment

It fixes *all* hidden-state axes at once (rather than per-boss whack-a-mole on state we can't even fully
observe), and Phase 1 turns the system into a self-diagnosing fidelity harness before we ever risk the
live bot. Main cost: the forcing bindings + reconciliation logic, and engine fidelity bugs it surfaces
(which we'd have to fix regardless to close the gap).

## Open questions

- Does a clean "apply player action + run queue to next input" entry already exist in the search apply
  code we can expose directly, or does it need a wrapper?
- Granularity of `force_draw`: set draw-pile order pre-shuffle, or directly set the resulting hand?
  (Resulting-hand is simpler and matches what the snapshot gives us.)
- How much card-effect RNG (Chaos/Wish/etc.) appears in Ironclad runs — is overlay-only fine for those?
