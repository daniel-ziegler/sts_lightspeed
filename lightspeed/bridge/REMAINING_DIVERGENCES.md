# Remaining live<->engine divergences

Living list of known fidelity gaps between the engine and the live game, as surfaced by the live
bridge (this package). Two detection channels:

- **Driven path (crashes, not logs).** With `STS_PBC_DRIVE` a fight runs on one engine-advanced
  `BattleContext`; observable divergence fails loud — the per-decision conversion asserts (intent
  damage, card base damage), pbc/live select mismatches, and the decided-outcome check (engine
  thinks the fight ended; live continues). Every grind crash is a fidelity bug to root-cause;
  none are currently known open (drive51's four crashes each produced a fix: Heavy Blade/Mind
  Blast/Rampage/Searing Blow damage display, Mayhem post-draw, Havoc'd Perfected Strike, stale
  select parks).
- **Shadow check (logging only).** `_shadow_card_play_check` (`agent.py`) advances the PRIOR
  decision's fresh reconstruction by the observed action and diffs deterministic scalars (player
  hp/block/energy, hand size, per-monster hp/block) against the next decision's reconstruction,
  logging `[shadow DIVERGE]`. It measures the same engine step the drive executes but from a
  lossier baseline (per-decision reconstruction), so most residual noise lives here. **Slated for
  removal** once the classes below are either fixed or covered by driven-path crash checks.

Status legend: **ARTIFACT** (shadow measurement noise, not a real gap) · **DEFERRED**
(understood; fix not worth it yet) · **OPEN** (not fully root-caused).

## Open / deferred classes

1. **Identical-monster slot churn — ARTIFACT.** After a kill, the fresh reconstruction packs
   survivors at different slots than the engine-advanced bc (which keeps INVALID gaps); twin
   monsters (2x BronzeOrb / Dagger / Spiker / JawWorm) churn assignment between decisions.
   Signature: pred and live are the same multiset, or paired +K/-K on twins. A fix would be
   shadow-side only (multiset compare / identity alignment).
2. **RNG-target rolls — ARTIFACT (unverifiable).** Shield Gremlin protects an engine-rolled
   ally; a Havoc'd attack picks a random target; Juggernaut fires its per-block-gain damage at a
   random enemy (the drive52 g11 SpireShield/Spear lines — every block card "dealt 7" to a
   different monster in engine vs live). The one-step shadow can't know the live roll; these
   could be reclassified `[shadow unverifiable]` the way Runic Dome moves already are.
3. **Havoc on an EMPTY draw pile — DEFERRED.** `playTopCardInDrawPile` queues an
   `EmptyDeckShuffle` (`bc.rng`) whose order the replay can't reproduce, so the played card
   can't be forced. Non-empty Havoc chains are exact: `_pbc_force_live_draw_order` forces the
   full live pile order before every driven advance, so recursive Havoc-into-Havoc replays
   reality. Rare, and the next per-decision reconcile masks it.
4. **Mayhem draw-top shift — DEFERRED (unverifiable residue).** Mayhem fires post-draw (engine
   fixed 2026-07-03) and the top it plays is forced when observable, but the top can shift
   between the end-turn decision and the play (monster-turn status shuffles, a reshuffle,
   stacked Mayhem), leaving some end-turns unverifiable rather than wrong.
5. **Writhing Mass residual hp-only ET diffs — OPEN.** Malleable / Flail block are modeled now
   (Hand of Greed routed through `attacked()`, asc-correct Flail block), but a small hp-only
   end-turn residue in WM fights is still unexplained.
6. **Energy ±1 — RESOLVED as noise/RNG (2026-07-03 audit).** Every named model checked against
   the decompiled Java came back correct: Art of War (attacksPlayedThisTurn gate, verified by
   replaying a captured no-attack end-turn — the engine's +1 matched live), Happy Flower's
   ++counter==3 convention, Gremlin Horn's death proc, recharge's reset (monster-turn energy
   procs wiped identically), Ancient Tea Set (battle-start-only, can't re-fire on a converted
   bc), and the per-combat relic counters restored from live. The surviving lines split into
   (a) Snecko Eye / Mummified Hand cost rolls inside the one-step window — live RNG, now tagged
   `[shadow unverifiable]` when the diff is energy-only under those relics — and (b) stale
   energy emits (both directions in the same game with no counter relic). The driven pbc
   re-syncs costs/energy from live every decision, so neither affects live play.
7. **Runic Dome hidden miscInfo — DEFERRED.** Under Dome, made moves are force-committed onto
   the shadow (`commit_observed_move`), but the miscInfo monsters (Champ / Darkling / Book of
   Stabbing / Gremlin Wizard) keep hidden per-hit state that can't be recovered, so their
   forced end-turns can still genuinely diverge.
8. **Colosseum elites + Slaver's Collar — DEFERRED edge.** The +1 energy gates on the room's
   `eliteTrigger`, which the bridge can't see for an EventRoom elite (Colosseum), so the Collar
   goes ungranted there.

## Fixed (context; no longer divergences)

In fix order: Iron Wave double-`calculateCardBlock`; Havoc draw-order forcing;
cardDrawPerTurn/Snecko; Slaver's Collar energy (incl. event-boss rooms); Berserk; the act-4
Smoke Bomb/Surrounded gate; relic onEquip HP/gold double-count; the shadow cross-combat floor
gate; Panache countdown-vs-damage mapping + `panacheCounter`; Centennial Puzzle used-flag and
Necronomicon `activated` (mod fork exports relic `grayscale`/`activated`; the "Awakened One
halfDead" class was actually Necronomicon double-play); Writhing Mass Flail block; Time Eater
usedHaste (miscInfo) inference; Time Warp forced end-of-turn; Mayhem moved post-draw; Havoc'd
Perfected Strike counting its in-flight copy (`autoplay`); Heavy Blade / Mind Blast / Rampage /
Searing Blow base-damage reconstruction; Awakened One rebirth (engine no longer declares victory
with a half-dead AO pending; conversion keeps half-dead monsters, parked on their revival move,
and restores AO's stage from its stage-2-only moves — the drive52 g3 phantom victory); Smoke Bomb
escape transient (a live escape plays a ~2.5s animation during which the room still reports
COMBAT with the potion consumed; the drive waits it out instead of crashing — the drive52 g7
crash. Java's escape fires in ANY room whose phase is COMBAT, events included; the only blocks
are canUse's boss-type/BackAttack checks, which the engine mirrors).

## How to refresh this list

With a run's errlog scoped (`STS_COMM_CAPTURE` run, truncated per launch):
```
grep -a '\[shadow DIVERGE\] (ET)'  "$ERRLOG" | head
grep -a '\[shadow DIVERGE\] (CARD)' "$ERRLOG" | grep -v Havoc | head
grep -ao 'force=[a-z-]*' "$ERRLOG" | sort | uniq -c
```
The CARD lines carry `[dex= str= frail= preblk= prehp= force=]` context for root-causing.
On the driven path, also check the crash markers: `Game error:`, `predicted combat over`,
`invalid on driven`, `not parked at expected`.
