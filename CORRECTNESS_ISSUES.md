# Correctness issues (unresolved)

Issues found during autonomous work that could not be root-caused to full correctness.
Each entry stays until resolved; resolution notes go to the relevant commit / doc.

(none currently)

## Resolved

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
