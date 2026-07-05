# Correctness issues (unresolved)

Issues found during autonomous work that could not be root-caused to full correctness.
Each entry stays until resolved; resolution notes go to the relevant commit / doc.

### 2026-07-05 — Runic Dome + Time Eater + select-opening play: forced turn can materialize guessed moves

Residual sliver of the dome/Time-Warp hazard (main paths fixed in dd0bc8e and 9c69157): when a
monster's Time Warp counter is at 11 and the triggering play OPENS A CARD SELECT
(Armaments/Warcry/...), the pbc must park at the select input to drive the pick, so the card
advance cannot be deferred -- the forced end-of-turn after the select completion still
materializes engine-guessed monster moves under hidden intents. Requires Runic Dome + Time
Eater + the 12th play being a select-opener; logged loudly when it happens
("[pbc] Time Warp primed with hidden intents on a select-opening play"). A phantom decided
outcome there would crash with that line as the marker.

## Resolved

### 2026-07-05 — Necronomicon trigger-gate edges vs live (engine) — RESOLVED 2026-07-05 (9c69157)

Live's gate (Necronomicon.class bytecode): `costForTurn >= 2 && !card.freeToPlayOnce`, OR
`cost == -1 && energyOnUse >= 2` (X-cost, no freeToPlayOnce exclusion). The engine checked the
queue-item's freeToPlay flag instead of the card's freeToPlayOnce, and applied it to both
branches: a Forethought'd >=2-cost attack was phantom-duplicated, and a free-played X-cost
attack (Whirlwind) with 2+ energy banked was not duplicated. Gate now transcribes the bytecode;
normal-play duplication regression-validated on the g20 capture (GL 104->56 unchanged).

### 2026-07-05 — Runic Dome + Time Eater forced end-turn (card advance) — RESOLVED 2026-07-05 (9c69157)

Time Warp's forced end-of-turn inside a CARD advance materialized guessed (dome-hidden) monster
moves. Card plays with a Time Warp counter at 11 and hidden intents now defer like END_TURN
(observed moves injected at the next decision), except the select-opening sliver above.

### 2026-07-04 — g38 phantom PLAYER_LOSS (Time Eater, floor 50) — RESOLVED 2026-07-05 (e9f82d8)

The a20h10k-redo replay of the same seed (2LVX3CDNDVR9Q, temp-0 deterministic) reproduced the
phantom with the c740d3e forensics armed. The dump showed the carried pbc's Time Eater with
uniquePower0=0 and Strength 7 vs live's Time Warp 4 / Strength 5: the reconcile's blanket
uniquePower0/1 transplant clobbered the live-observed Time Warp counter (the engine stores the
status IN uniquePower0) with an engine-evolved value that never got corrected again. The
drifted counter hit its in-engine trigger (11 -> +2 Str, end-turn-early) during a card-play
advance, parking the pbc a full turn ahead of live; the auto END_TURN advance then ran a
second monster turn whose engine-rolled Reverberate (22x3) killed the 45-hp player. The old
replay attempts couldn't see it because they rebuilt hidden state via the same transplant from
FRESH reconstructions (correct counter), not the drifted carried one. Fix: transplant
uniquePower0 only for Hexaghost (the engine's one genuinely hidden uniquePower client);
everything else keeps the reconstruction's live-observed values. Validated on the captured
phantom state. Era impact: F3-era games with Time Eater fights are tainted (TE_FIGHT marker).
