# Remaining shadow divergences

Living list of known divergences flagged by the persistent-bc shadow check
(`_shadow_card_play_check` in `comm.py`, logged as `[shadow DIVERGE]`). The shadow advances the prior
decision's reconstructed `BattleContext` by the observed action and diffs deterministic scalars
(player hp/block/energy, hand size, per-monster hp/block) against the next decision's freshly
reconstructed state. A divergence means the engine simulated the action differently than the live
game — a fidelity gap contributing to the live↔offline winrate difference — OR a shadow-measurement
artifact. The shadow is **logging-only**; it never affects live play.

Status legend: **DEFERRED** (understood, fix not worth it yet) · **OPEN** (not root-caused).

Already fixed (for context, not divergences anymore): Iron Wave double-`calculateCardBlock` (engine,
0 now), Havoc forcing reset-order bug (141→~0), cardDrawPerTurn/Snecko, Slaver's Collar energy,
Berserk, the act-4 Smoke Bomb/Surrounded model.

**Shadow cross-combat artifact (FIXED).** The ET divergence count was inflated by ~15% (19/126 in the
20-game run) by the shadow comparing a stale `prev_bc` from the PREVIOUS combat against a fresh fight
— e.g. last game's dead player (php 0) diffed vs the new game's full-HP turn 1, surfacing as scary
"player dies turn 1" lines. The only cross-combat guard was a monster-COUNT check, which passes when
two fights share a count. `_shadow_card_play_check` now also gates on floor identity (a combat lives
on one floor) and skips when `self.game.floor != prev_floor`.

**Slaver's Collar in an EVENT boss fight (FIXED).** The single biggest *real* ET cluster was a
Mind Bloom Guardian fight (`room_type == EventRoom`) with Ice Cream: Slaver's Collar's +1 energy was
gated on `room_type in (MonsterRoomElite, MonsterRoomBoss)`, but Mind Bloom spawns the boss in an
EventRoom, so the +1 was dropped → energyPerTurn 3 not 4 → a 1-energy deficit that Ice Cream conserves
and compounds EVERY turn (one root cause, ~14 divergence lines). SlaversCollar.java actually gates on
`getCurrRoom().eliteTrigger || any monster.type == BOSS`, so the bridge now grants it when
`room_type == MonsterRoomElite` OR the encounter is a boss (`_infer_boss_encounter(...) != INVALID`,
whose keys mirror the Java BOSS-type set). Verified offline against the captured fight: reconstructed
energy now matches live exactly across all 14 turns. (Minor open edge: Colosseum elites in an EventRoom
rely on `eliteTrigger`, which the bridge can't see — Slaver's Collar there is still ungranted.)

---

## 1. Havoc played on an EMPTY draw pile — DEFERRED
Tagged `force=no-top` and now logged as `[shadow unverifiable]` rather than `[shadow DIVERGE]`.

Havoc plays the top card of the draw pile. When the draw pile is empty, `playTopCardInDrawPile`
queues an `EmptyDeckShuffle` (uses `bc.rng`) then replays the top — so the played card comes from a
reshuffle whose order the shadow can't reproduce from the pre-play state. The non-empty case is fully
forced and verified (`force=forced`); only the empty-pile case is affected.

**Not unsolvable** — deferred options:
- Replay the reshuffle deterministically: thread the live shuffle-RNG state into `prev_bc.rng` so its
  `EmptyDeckShuffle` produces the same order. Needs the live RNG counter exposed/tracked.
- Pre-place the post-hoc-identified played card on `prev_bc`'s draw-top so no reshuffle happens. The
  played card is `truth.exhaust[-1]` for a normal card (Havoc exhausts it last) — but a **Power**
  card played off the top never enters exhaust, so that read isn't 100%; would also need to detect
  the new/incremented player power for the power-card case, plus a discard→draw-top move.

Low value (logging-only, rare) so deferred.

---

## 1b. Mayhem at turn start — BEST-EFFORT (like Havoc)
The dominant `pblock pred 0 vs live N` / `hand` ET class was **Mayhem** ("at the start of your turn,
play the top card of your draw pile"). Mayhem fires pre-draw (Java `atStartOfTurn`, engine
`applyStartOfTurnPowers` precedes `DrawCards`), so the card it plays is the draw-pile top observed at
the end-turn decision — the same thing the Havoc path forces. So the shadow now forces that top onto
`prev_bc` before executing END_TURN (gated on `player.getStatus(MAYHEM) > 0`): a stable top is then
verified (`[shadow ok]`). But the top often shifts between end-turn and the next-turn play (the monster
turn shuffles status cards into the draw pile, the pile reshuffles, stacked Mayhem plays >1 card), so a
residual diff can't be attributed to a real mis-sim vs Mayhem's draw uncertainty -> logged
`[shadow unverifiable] (ET) ... Mayhem`. Offline over the v15 capture: of ~65 Mayhem end-turn pairs,
forcing verified 17 outright (up from 14 unforced); the rest are unverifiable.

## 1c. Relic onEquip HP/gold double-count — FIXED (was the dominant ET class)
`spirecomm_to_gamecontext` set `gc.max_hp` from the live snapshot (which already includes Pear's +10
maxHP pickup), then `gc.obtain_relic(Pear)` re-fired the +10 -> 100 vs live 90. The search played the
whole game with ~10 phantom HP (under-valued defense). Surfaced as `php pred consistently +10 vs live`
(55/101 ET divergences in one 20-game run). Fixed by overwriting cur_hp/max_hp/gold with live truth
AFTER the obtain_relic loop. Verified offline: the entire +10 class disappears.

## 1d. Runic Dome hidden monster moves — OBSERVE-THEN-FORCE (like Havoc/Mayhem)
Runic Dome hides the upcoming intent, so the reconstruction defers the move (`rollMove` ->
`pending_move_rolls`) and END_TURN rolls a guess from `bc.rng` != live's actual hidden move. But
CommunicationMod still reports `last_move_id`, so the move a monster MADE this turn is its last move at
the next decision (already in `truth_bc.monsters[slot].moveHistory[1]`).
`_force_observed_monster_moves` commits that onto prev_bc before advancing (new `commit_observed_move`
binding = setMove + cancelPendingMove). A still-divergent forced end-turn is a REAL signal (move was
right); an unobserved move stays unverifiable. Offline on the RD game: 124 end-turns -> ok, 21 real
signal (mostly the 4 miscInfo monsters Champ/Darkling/Book of Stabbing/Gremlin Wizard whose hidden
per-hit damage can't be recovered under RD), 3 unverifiable.

## 2. End-turn (ET) divergences — OPEN (the hard core)
By far the largest class. Executing `END_TURN` runs the real monster moves, so a post-monster-turn
mismatch is a genuine monster-turn fidelity gap (the boss concern). Sub-patterns observed:

- **`php pred 0` / large `php` gaps** (e.g. `php pred 18 vs live 80`): the sim predicts the player
  takes far more damage on the enemy turn than they really did — often predicting a lethal that the
  live player survived. Monster attack-damage misprediction (intent/move-byte, block timing, or a
  damage modifier not modeled). This is the most important class for winrate (drives over-defensive /
  fatal play).
- **`hand pred 0 vs live N`** (N = 5/7/9...): the sim ends with an empty hand while the live player
  drew a fresh hand. Usually paired with a `php` gap — the sim thinks the player died (no next-turn
  draw). When `php` matches but hand is still 0, suspect a draw/reshuffle or draw-count gap.
- **`energy pred N vs live N±1`**: small energy mismatch at turn start (energy-relic / carry-over
  reconstruction, e.g. Ice Cream, or an energy source applied at battle start).
- **`pblock` mismatches**: end-of-turn block retention (Barricade/Calipers/Blur) or block applied by
  a monster-turn effect.

Examples:
```
[shadow DIVERGE] (ET) after { end turn }: php pred 84 vs live 88; pblock pred 3 vs live 0; hand pred 0 vs live 7
[shadow DIVERGE] (ET) after { end turn }: php pred 18 vs live 80; hand pred 7 vs live 5
[shadow DIVERGE] (ET) after { end turn }: energy pred 2 vs live 3
```

---

## 3. CARD divergences (non-Havoc, `force=forced`) — OPEN
The draw was forced correctly, so these are real per-card effect mismatches.

- **X-cost cards energy** — e.g. `Whirlwind+ energy pred 0 vs live 1`: the sim consumes all energy,
  the live game keeps 1. X-cost reconstruction (energy available, or an energy refund) is off.
- **`hand pred > live` by ~3 for assorted cards** (Bloodletting, Juggernaut, Sword Boomerang+, none
  of which draw): the sim's hand is larger than live after the play. Suspect a hand-reconstruction
  gap or an intervening draw/discard between decisions the one-step shadow doesn't attribute. Note
  several show the engine hand-size cap (10).
- **`Battle Trance energy pred 5 vs live 6`**: energy off by 1 after a 0-cost draw card.

Examples:
```
[shadow DIVERGE] (CARD) after { use card (Whirlwind+) }: energy pred 0 vs live 1 [force=forced]
[shadow DIVERGE] (CARD) after { use card (Bloodletting) }: hand pred 10 vs live 7 [dex=0 str=0 frail=2 force=forced]
[shadow DIVERGE] (CARD) after { use card (Battle Trance) }: energy pred 5 vs live 6 [str=4 frail=1 force=forced]
```

---

## How to refresh this list
With a run's errlog scoped (`STS_COMM_CAPTURE` run, truncated per launch):
```
grep -a '\[shadow DIVERGE\] (ET)'  "$ERRLOG" | head
grep -a '\[shadow DIVERGE\] (CARD)' "$ERRLOG" | grep -v Havoc | head
grep -ao 'force=[a-z-]*' "$ERRLOG" | sort | uniq -c
```
The CARD lines carry `[dex= str= frail= preblk= prehp= force=]` context for root-causing.
