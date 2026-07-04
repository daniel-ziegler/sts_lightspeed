# Correctness issues (unresolved)

Issues found during autonomous work that could not be root-caused to full correctness.
Each entry stays until resolved; resolution notes go to the relevant commit / doc.

## 2026-07-04 — g38 phantom PLAYER_LOSS (Time Eater, floor 50) — UNRESOLVED

During the a20h10k live grind, the driven persistent bc reached PLAYER_LOSS on a turn-10
advance (Second Wind play or the auto END_TURN that followed) while the live fight continued —
the decided-outcome crash fired correctly at turn 11 (seed 2LVX3CDNDVR9Q, burned).

What was ruled out by faithful offline replay (fresh conversion + hidden-state transplants +
forced live draw order, exact captured states):
- The replayed advances stay UNDECIDED and reproduce live's turn-11 state exactly
  (player 45/110 hp, Time Eater Ripple +20 block, Time Warp 4).
- All damage paths (Ripple deals none; a Head Slam park would explain the kill but the fresh
  park, the carried park, and the DESYNC oracle all agree on RIPPLE).
- Non-damage LOSS sources: the movesThisTurn>200 runaway guard and the empty-piles can't-win
  check don't fire on the replayed state (piles non-empty, player has Thorns).

Why it's stuck: the arm was silent and the pbc is dropped at arm time, so the actually-diverged
state was unrecoverable. Mitigation shipped (c740d3e): every decided driven advance now dumps
pre/post bc + the raw live message to `runs/pbc_decided_dumps.jsonl`. Next occurrence carries
its own evidence — re-open from that dump.
