# RL experiment log

Running log of training/eval experiments (RL session). Newest entries first within each day.
See COORD.md for the MCTS-session side.

## ⚠ Standing caveat: draw-order clairvoyance (discovered 2026-06-03)

The deployed battle search inherits the concrete draw-pile order at every decision root —
**all absolute win rates below are inflated** (root-hiding probe: heroe2-270 @1000 sims drops
69.4% → 35.2% honest; CardPile belief search recovers 56.2%). Note these honest figures are
LOWER BOUNDS on honest-era performance: they evaluate a policy trained on clairvoyant battles
with cheat-tuned search knobs — the ~34pp is the information value to this cheat-adapted
system, not the intrinsic cost of honesty. Relative comparisons within cheat mode (paired
A/Bs, schedule effects) likely keep their direction but are conditioned on the cheat.
Open decisions: move RL collection to honest battles (expect a large apparent drop that is
NOT a regression); retune search knobs under honest dynamics; honest-era retraining will
recover an unknown chunk of the gap. All entries below predate honest mode unless noted.

---

## 2026-07-03 (combat objective: unified gold delta — no-harm gate PASSED)

goldWeight (0.25/gold, replacing goldLossWeight) now scores a root-baselined delta of EFFECTIVE
gold — pocket + stolen gold held by not-escaped Looters/Muggers (exitBattle refunds it) — so Hand
of Greed kills are finally worth ~5-6 HP-equivalent to the search, while escaped-thief losses
price identically to the old penalty. Gate: 600 matched seeds, heart1.pt.iter_2575, A0, 1000
sims, eval_hero control (pre-change engine) vs treatment. Result: control 501/600 (83.5%) vs
treatment 502/600 (83.7%); discordant pairs 11 treatment-only vs 10 control-only, McNemar
z=+0.22 — no significant difference, no-harm bar met. Commit b75776f.

## 2026-06-20 (heart1 card-preference re-analysis @iter 1950/1955 + boss/ascension conditioning)

Re-ran the card-preference diagnostics on the latest heart1 checkpoint (iter 1950, episodes
iter_1906-1955 = 50 training iterations ≈ 24.6k games / 2.15M decisions; 933k card-acquisition
decisions). Headline structure unchanged from the 2026-06-09/19 runs: identity alone predicts
~55% of picks (McFadden R2 +0.28), +context ~67%, duplicate-aversion still ~0 (deck_count
−0.001). **But the taste content improved:** Perfected Strike fell from rank 5/107 (+3.19) to
46/107 (−0.24) — the deck-blind unconditional-PStrike symptom from 2026-06-09 has self-corrected
— and a coherent power/exhaust archetype now leads (Corruption #1, then Feel No Pain / Dark
Embrace / Barricade / Offering). Wild Strike is the most-avoided (−4.57).

**New question: does the policy condition card choice on the upcoming act boss?** Two methods,
both say *weakly*:
1. Correlational logit (`analyze_card_boss.py`): per-card × boss interaction on top of
   identity+act adds only **+0.10 nats/decision** (+3.7pp acc, 0.583→0.620) beyond the act.
2. Counterfactual probe (`analyze_boss_probe.py`): override ONLY fixed_observation[4] (boss
   token; encoding 0-8, see bindings-util.cpp) across the act's 3 bosses on real card-choice
   states. Mean pairwise **TV 0.053** (act1 0.047 / act2 0.048 / act3 0.068) vs a 0.84 cross-
   state scale; top card pick flips in only 8.6% of states. Biggest pair: act3 Donu&Deca vs
   AwakenedOne (0.076).

**Ascension conditioning (`analyze_ascension_probe.py`, re-run):** card-choice TV only ~0.067
across the *entire* A0→A20 range — about as weak as boss. The VALUE head, by contrast, reads
ascension strongly (mean V +0.880→+0.519 A0→A20, corr −0.83, lower at A20 in 100% of states).
So the net ingests the tokens; the policy just barely lets them move card picks.

**Per-card attribution (`analyze_card_token_sensitivity.py`):** maps each logit slot back to its
offered card (slot j == cards_offered[j]). Magnitudes are single-digit pp everywhere. Boss
swings top out ~0.05 (act3: Mayhem/Evolve/Panic Button/Apotheosis) with directions mostly at the
noise floor — no defensible boss-specific plan. Ascension is the more coherent, larger signal,
sharpest in act 3: UP at A20 = Apotheosis (+0.058, the single biggest effect) + immediate
defense (Shrug It Off, Ghostly Armor); DOWN = slow scaling / setup (Demon Form, Rupture, Feed,
Berserk, Entrench) and combo enablers (Dual Wield, Double Tap). The right qualitative adjustment,
applied weakly.

Conclusion: consistent with the deck-blindness story — card taste is a near-fixed identity
ranking with only marginal context conditioning; the upcoming boss is among the *weakest*
signals (strongest in act 3, where boss identity matters most), ascension slightly stronger and
directionally sensible. No training change made; diagnostics only.

## 2026-06-12 (battle-outcome prediction: pretraining/aux task — design + datagen + SL gate launched)

**Hypothesis**: making the trunk predict a SPECIFIC battle's ΔHP from (state, encounter) teaches
combat-strength features the value/policy heads learn only weakly. Gate = held-out value EV on
heart1 episodes (battle_value_sl.py; protocol = value_sl.py's seed-split). Design locked in
`BATTLE_OUTCOME_PLAN.md`: 20-bucket %-of-maxHP output (DEATH / 5% damage bins / EXACT-0 / gains
5%-fine to +20 then coarse — Burning Blood puts modal easy-win at ~+6HP) vs scaled-float
comparison; encounter is HEAD-ONLY (trunk inputs identical to policy net → clean transfer).

**Infra** (commits e0a411f, 108c65c, 727d102): `GameContext.copy()` + `playout_battle(gc,
encounter=)` override bindings; all battle randomness (env + searcher) derives from
`gc.seed+floorNum`, so seed reassignment on a copy = full honest reroll (verified 8/8 distinct).
⚠ static `cardColors` table misaligned (Bullet Time→RED) — add-card mutation pool uses the
engine's real reward pools (`get_card_pool`) instead. `gen_battle_outcomes.py`: heart1-iter-1035
policy plays, every battle entry snapshotted, variants simmed to completion (2 real rerolls +
6 deck mutations + 2 alt encounters from empirical (act,kind) pools); episode-schema parquet.
Datagen on AWS spot c7a.16xlarge (~165 rows/game, 2000 games train + 120-game val with
32 rerolls/(state,enc) for distribution-level eval). NOTE: datagen engine predates the Runic
Dome merge (acabcd4) — Dome battles in the data are intent-clairvoyant (<1 HP/battle, accepted).

**SL experiment queue** (heart1 box GPU; data 338k train / 201k val rows). Value-EV gate
(held-out, seed-split; baseline `value_base` **0.8079**):

| variant | val EV | |
|---|---|---|
| value_base / value_lr1e-4 | 0.8079 / 0.8048 | from-scratch baseline |
| probe_random (frozen random trunk) | 0.6449 | linear-probe floor |
| probe_b20 / probe_b0 (frozen battle-pretrained) | 0.6340 / 0.6430 | **at the random floor** |
| finetune_b20 / finetune_b0 (warm-start, all params) | 0.7832 / 0.7814 | below baseline |
| mt_b20_c1 / mt_b0_c1 / mt_b20_c03 (joint) | 0.7882 / 0.7899 / 0.7904 | below baseline |

**Verdict: NEGATIVE — battle-outcome prediction does not improve the value fit, and mildly
competes with it.** (1) Frozen battle-pretrained trunks probe at the random-init floor → the
representation has ~zero linearly-decodable value signal beyond random projections
(caveat-independent). (2) Warm-start and joint training all land ~2-3pp *below* from-scratch,
across both head types and coef 1.0/0.3. (3) Distribution-level head eval (dedicated val,
6279 groups × 32 rerolls; irreducible CE floor 1.59 raw / ~1.69 Miller-Madow): the battle head
is itself a *reasonable* predictor — bucket CE 1.936 (KL ~0.34 above floor, ~0.24 bias-corrected;
74% of the marginal→floor gap captured), float mean-ΔHP MAE 4.4% of maxHP (~3.5 HP). So the
negative is not unlearnability. (4) The multitask heads are *worse* battle predictors than the
dedicated pretrain (CE 1.98 vs 1.94; MSE 0.0069 vs 0.0044) — the two tasks share little useful
structure and mildly interfere in a shared trunk.

**Caveats.** The value-EV baseline is 0.81 (dense floor-bonus returns nearly saturated by
trivial features — floor/HP/ascension), so the gate has limited headroom for *any* auxiliary
signal and is a weak discriminator vs the old sparse 0.42-ceiling regime. And the gate measures
only value EV — it does NOT test whether the aux loss helps the *policy* representation during
RL. Per BATTLE_OUTCOME_PLAN.md's decision gate (no EV win + flat probe transfer ⇒ stop before RL
spend), **Phase 4 RL integration is NOT justified on this evidence**; held pending user call.
Results: `lambda_results/bvsl_results.jsonl`. Spot box terminated (338k+201k rows, ~$5, 3.2h).

## 2026-06-13/14 (MCTS session: Dome rank corrected + console teleport)

**Dome blindness cost is battle-DEPTH-driven, not ascension** (eval_states `hideIntents=1`, pure
blindness no energy, paired @1000 sims, Δscore / Δbattle-wins): trivial low-floor (h1dev) −0.81 /
−0.18pp; deep acts-1-4 (heart1, median fl 20) −4.73 / −1.56pp; genuine asc 16-20 (heart1, shallow
fl~10 — policy dies early) −1.60 / −0.67pp. ⚠ The "asc 16-20 = −4.73" behind the earlier
ascension-conditional rank (`e289471`) was a MISLABEL — that collection silently defaulted to
asc 0; the −4.73 set is asc-0-deep. `b959617` reverts `getBossRelicOrdering(RelicId)` to a single
**tier 2** for Dome (Choker/Snecko tier), since the decision-relevant cost is the deep battles a
Dome holder carries it through. **New tooling:** console teleport (`7e84551`) — `./main replay
<stateFile> <i>` drops a human into a recorded pre-battle state (faithful RNG) to retry a lost
battle; `collect_states_asc.py --only-losses` sources MCTS losses under a checkpoint (shared
replay in `sim/StateReplay.h`). The teleport's `list` command is what caught the mislabel.

## 2026-06-11 (MCTS session: Runic Dome intent clairvoyance fixed)

**Last known intent cheat closed** (`boss-eval@2563b1d`): with Runic Dome the search planned
against the concrete rolled move; rolls now defer their rng under `bc.intentsHidden` with the
volatile roll inputs snapshotted at the true roll time, materializing inside END_TURN (chance
node) or at Spot Weakness. Distribution exactly vanilla — verify_intent harness 0/22000
mismatches; winrate_mt byte-identical gate-off; gate-on diverges in exactly the 26/200
Dome-picking games. **Open follow-up:** expert boss-relic picker still ranks Dome as the #1
pick (ordering -1, tuned under clairvoyance) — needs re-rank + deployment gate; honest-Dome
games are presumably weaker now until that lands. Details in COORD.md.

## 2026-06-10 (map-SL revival: choice-dependent reachability vs per-choice cone aggregate)

**Motivation.** heart1 routes poorly to burning elites (emerald key, usually 2-4 rows ahead) —
a *multi-hop* lookahead the existing map SL probes never tested (all were one-hop "is the
option's immediate destination an elite"). The production binary `reachable` bit collapses
exactly the "must commit left *now*" distinction. Added to `sl_repr_lab.py`: (a) two multi-hop
routing tasks — `route_deep_elite` (option whose cone holds an elite beyond its immediate
dest) and `route_burning` (option whose cone reaches the burning-elite node, from the new
burningEliteX/Y obs field); (b) new representation arms on the R5b production baseline —
`RV` (binary reachable → per-frontier-column multi-hot `reach_via`, sinusoidal FixedVec),
`RVe` (same set as a SUM of a learned per-column table SHARED with the path token's x →
identity binding, the "sum the x embeddings" idea), `PC` (forward-cone aggregate
minE/maxE/dist_rest + a reaches_burn bit ON the path-choice token), and combos. Ran on the box
A10 (laptop 3060 too small: dim-256 full-attn OOMs >batch128), 10 heart1 parquets (28k path
decisions, ~1k held-out/task), 20 epochs, seed 0.

**Result (best held-out acc).** route_burning (base 0.41): Rbase 0.58, RV 0.59, RVe 0.56,
**PC/RV+PC/RVe+PC 1.00**. route_deep_elite (base 0.47): Rbase 0.61, RV 0.60, RVe 0.57,
**PC/RV+PC/RVe+PC 1.00**. elite one-hop control 1.00 everywhere (saturated, as in prior lab).

**Verdict.** The per-choice forward-cone aggregate (PC) *solves* both multi-hop routing tasks;
`reach_via` (the per-node multi-hop, raw reachability) barely beats baseline. The info is
*present* in reach_via but the net can't learn to *aggregate it across the DAG* at a realistic
budget — precomputing the aggregate onto the choice token makes routing trivial. Confirms the
standing repr-lab lesson ("option grounding = lookup, not a learned multi-hop attention
program") for the multi-hop case. `RV+PC == PC` (reach_via adds nothing on top). Learned shared
embedding `RVe` is *worst* — consistent with "sinusoids load-bearing, learned tables crater."
Honesty caveat: PC's cone features encode nearly exactly the lookahead each task needs, so this
shows representation *sufficiency* (the net exploits precomputed per-choice aggregates, not raw
per-node reachability), not yet a win-rate gain — that's the RL test. Mess notes: orphaned
child PID survived `kill <wrapper>` and held 11.8 GB → OOM'd the relaunch (kill the actual
`python3` pid / use setsid); batch 512 needs ~17 GB (co-resident with heart1's 4.6 GB), batch
256 ~8-12 GB fits.

**Production-port candidate (pending go-ahead):** add the per-choice cone aggregate
(frontier-node minE/maxE/dist_rest, scaled, + reaches_burn) to the `paths` token in
collate_fn/network as ZERO-INIT DictAdd components → warm-starts heart1 bit-identically, then
learns to use them. Skip reach_via (no policy benefit). Non-path states keep the existing
per-node agg + binary reachable (already in production) for the value head.

**Deployed to heart1 @iter 850** (`1cbede1` net + `5db2284` lab; rl_train `<this commit>` adds a
fresh-optimizer fallback when a warm-start changes the param set). Per-choice cone added as
zero-init DictAdd components on the `paths` token (golden check: max |logit|/|value| diff = 0
vs iter_850). Resumed bit-identically, 256 games, entropy floored 0.0167 — **representation
change in isolation for 50 steps** (user call, cleaner attribution), then layer in entropy decay
(0.0167→~0.0083) + num-games 256→512 at iter ~900. Fresh optimizer moments on resume (added
params break the 2-group optimizer's saved state; net weights still load exactly).

**Schedule changes layered in @iter 895→900** (after the 50-step repr-only window): num-games
256→512, entropy decay restart 0.0166667→0.0083333 over 100 iters anchored at decay-start 900
(holds at floor until 900 then halves), lr untouched (still 2e-5). Resume from iter_895 loaded
optimizer state cleanly this time (the cone param set is now stable, so saved moments match) —
momentum preserved. Repr-only window (850-895) stayed healthy: heart kills in the prior
0.05-0.13 band, invdrops frozen at 256 (cone change is obs-only, doesn't touch the engine),
warm-start continuity exact. Watch emerald-key rate + act-4 reach over the next few dozen iters
for the cone feature paying off.

**More-aggressive heart reward @iter 910** (`446ad8c`): CLEAR_BONUS 0.2→0 (clearing act 3 only
earns its floor reward -- no stop bonus), KEY_VALUE 0.1→0.05, HEART_BONUS 0.3→0.6 (heart kill
totals ≥1.0). New terminal progression (floors 51/54/55): act3 0k 0.27 < 1k 0.32 < 2k 0.37 <
3-key act-4 death 0.43 < heart kill 1.04 -- strictly monotone (act-4 death still outscores any
act-3 win on floor+keys alone, so no clear bonus needed). Resumed from iter_910, optimizer
momentum preserved (reward change adds no params). Stacks on the cone feature + 512 games +
entropy decay, so cone attribution is now entangled (user opted to deploy now). Watching for
the heart-directed shift: less act-3 dawdling, higher key-rate / act-4 reach / heart kills.

**Cone-feature isolated effect (clean 45-iter window 850→895, before any schedule/reward
change).** Windowed A/B vs matched pre-cone window 805-849 (identical hparams, 256 games/ent
0.0167/lr 2e-5): avg_floor **+0.54** (sig), avg_keys **+0.026** (sig, marginal), win/act3/
act4reach/heart all flat. Direct routing A/B (`eval_burn_route.py`, 3314 identical
burning-reachable path decisions, iter_850's cone params load zero = pre-cone vs iter_895's 45
trained iters): argmax picks a burning-reaching option **0.569 → 0.621 (+5.2pp)**, prob-mass
0.563 → 0.598 (+3.5pp), vs 0.511 uniform. Verdict: the cone feature **works** -- the policy
demonstrably learned to route to burning elites more (the capability), but the behavioral payoff
was modest because the OLD reward (CLEAR_BONUS 0.2, KEY 0.1) under-incentivized chasing the
emerald key / heart. Capability now meets incentive via the aggressive reward.

**Gradual lr decay to half @iter 910**: bases reset to the current effective lrs (policy
3e-5→2e-5, value 1e-4→6.6667e-5) with lr-final-frac 0.66667→0.5, decay-start 185→910, steps
100 — so lr holds at current (2e-5 / 6.67e-5) at iter 910 then geometrically halves to 1e-5 /
3.33e-5 by ~iter 1010, no discontinuity. The schedule scales policy+value by one factor, so
both halve (value re-fits to the new reward scale in the first few iters at 6.67e-5, well before
it decays). All four changes (cone / 512 games / entropy decay / aggressive reward / lr halve)
now stacked on heart1.

## 2026-06-08 (heart1 schedule + engine update @185)

**heart1 lr decay engaged @iter 185** (pre-agreed condition met: kl 0.0045→0.0077, clipfrac
0.050→0.069 over iters 165-180 as the entropy coef fell): lr → 1/3 over 100 iters, anchored.
Same bounce loaded the **boss-eval@0245769 merge** (`b2c2514`): battle-end detail eval terms
gold/maxHp/parasite as engine defaults (MCTS-session gated 79.2% vs 77.8%) — thief-gold
protection, Feed/maxHp value, Writhing Mass implant avoidance, all heart-run-relevant.

## 2026-06-09 (card-choice + value-head deck-conditioning diagnosis)

**Why heart1 plateaus at mediocre strategies: it's deck-BLIND, and that's a value-target/credit
problem, not exploration temperature or representation.** Three nested analyses on the last
~50 iters (analyze_card_choices.py, analyze_value_probe.py, analyze_rep_probe.py):

1. *Policy card choice* (conditional logit, 409k reward+shop acquisitions): card IDENTITY alone
   predicts 64% of picks (McFadden R2 0.38); full context only adds to 0.71. deck_count coef
   ~-0.04/SD, n_pstrike ~-0.05/SD -> near-zero duplicate aversion / deck-conditioning. Average
   TASTE is fine (top: Panache/Reaper/Apotheosis/Feed/Demon Form; bottom: Flex/True Grit/Rampage)
   -- Perfected Strike is a rank-5/107 UNCONDITIONAL grab (good in the abstract, bad as a fixed
   policy that can't tell a 3-attack deck from a 12-attack one).
2. *Critic deck-conditioning* (counterfactual dV(card|deck) = V(deck+c)-V(deck) on real states):
   dV is near-constant across decks (sd 0.005-0.019 on a ~1.1 scale). Control-deconfounded
   synergy slopes ~0: Perfected Strike vs #strikes -0.006/SD (wrong sign), Body Slam vs #block
   +0.006/SD (negligible), True Grit upgrade valued BACKWARDS (-0.021). The critic carries no
   deck-conditional card value.
3. *Representation capacity* (linear probe from the pooled vector the value head reads): recovers
   deck composition at high R2 -- #strikes 0.80, #pstrikes 0.87, #block 0.79, deck_size 0.97.
   The value head is a linear map on this same vector, so the info is present and linearly
   accessible; the head just never learned to use it.

Conclusion: the trunk encodes the deck; the on-policy returns (under a deck-blind policy) never
made deck-conditioning pay, so neither critic nor policy learned it -- a self-reinforcing
equilibrium. Fixable training-side (no arch change). Per-decision entropy can't break it
(synergy decks need a COORDINATED multi-pick build; per-decision entropy explores picks
independently). Candidate levers: card-acquisition-specific exploration temperature; deck-
archetype diversity/intrinsic reward; population with varied card biases.

## 2026-06-08 (heart reward v4 + lr un-anneal)

**Heart reward v4 (`d3b2924`): monotone in true progress.** v3 had an inversion -- a 2-key
act-3 stop (0.7) outscored a 3-key act-4 death (0.584) because the win bonus was exclusive to
stopping, even though an act-4 death also cleared act 3; this drove the keyless/low-key act-3
equilibrium. v4 restructures as level(floor/190, cap 0.3) + 0.2 act-3-clear (stop OR push) +
0.1/key (earned once act 3 cleared) + 0.3 heart. Act-4 death gets the clear bonus too + always
3 keys, so it strictly beats any stop. Terminal chain strictly increasing: act3 0k 0.47 < 1k
0.57 < 2k 0.67 < act4 death 0.78 < heart 1.09. Partial keys still rewarded; early-death keys
clawed back. Swapped into heart1 live (it had been banked as a general agent; user chose to
redirect it after all).

**lr un-annealed 2x** (1e-5 -> 2e-5): clipfrac had fallen to ~0.025 (over-annealed); raised
lr_final_frac 0.333 -> 0.667 (past the decay window, so this just lifts the floor). Same bounce.


**INVALID-card root cause found + fixed (`55902fa`): action-queue ring desync on victory.**
`clearPostCombatActions` compacted the queue (size decremented) without updating `back`,
breaking `back == front + size`. Any post-victory `addToBot` (Self-Forming Clay / Rupture /
Red Skull / Gremlin Horn reacting to the surviving whitelisted beat-of-death damage) then
pushed past the live region, and later pops walked the stale gap — **resurrecting cleared or
already-executed actions**. A resurrected OnAfterCardUsed disposed the consumed-power husk
(onAfterUseCard deliberately sets a played power's id to INVALID) → the INVALID pile moves /
original abort. The bug predates chests (any victory tail could double-fire resurrected
actions, silently corrupting post-battle HP); chests+powers made it visible. Debug chain:
warn fingerprint (seed/turn/monster) → deterministic replay from heart1's episode parquet
(forcing recorded decisions; 226 drops reproduced exactly, Heart fight floor 55 turn 7) →
in-process backtraces → husk detector → queue-dump showing the stale gap. Verified 226→0 on
the repro; deployed to heart1 (supervisor bounce).

## 2026-06-07 (heart reward v2)

**heart1 reward revised @iter ~70 (`0a5b9a2`)**: avg keys decayed 1.96→1.27 over iters 18-69 —
pure PBRS gives no NET key incentive and the heart bonus is unreachable early, so the pickup
habit was extinguishing. New reward: level term capped 0.3 (floor/190), act-3-only win capped
0.5 total, heart kill +0.5, and **+0.1/key real terminal reward** (in the base potential:
dense credit at pickup, kept at terminal; PBRS shaping_key_coef dropped as redundant).
Outcome ordering now: heart ~1.09 > act3+2keys 0.70 > 3-key act-4 loss 0.58 > keyless act3 win
0.50 — dying in act 4 with keys deliberately beats winning act 3 without them. Restarted from
the latest checkpoint (value head re-anchors to the new terminal scale over a few iters).

## 2026-06-07

**Chest value measured: +6.6pp causal.** honest1-440, 519 paired headline seeds @1k sims:
chests-open 0.834 vs chests-skipped 0.769 (McNemar 83/49, z=2.96, p=0.003); independent
600-game confirmation at seeds 3M+: 0.823. So the honest A0 champion is ~0.83 with chests,
and the skip bug cost ~7pp all along.

**Two more chest-era bugs found + fixed:** (1) MAX_RELICS 25→40 — chest relics push real
games past the old collate cap (python assert killed the NN service game). (2) The
INVALID-card pile-move assert fired inside search rollouts (~1/1000 chest-enabled games,
timing-dependent, NOT reproducible in 600 deterministic-seed attempts) and aborted the whole
process — now warn-and-drop (grep logs for "WARNING: dropped INVALID"); root cause FOUND + FIXED
(`9b95037`): _DiscardNoTriggerCard read the mutable curCardQueueItem at execution time instead
of capturing its card at queue time — stale reads hit the end-turn/battle-end items whose
default CardInstance is INVALID (the one observed dump: dead Heart, empty card queue, one
pending action). warn-and-drop stays as a tripwire.

**honest1asc paused at its anneal peak (~iter 255-260, six straight highs, 0.422 @255;
16-20 band 0.18).** Resumable: absolute-anchored schedules, checkpoints on box.

**heart1 launched** (box, supervisor-wrapped auto-resume): fresh from-scratch, uniform A0-20,
reward-function heart, shaping-key-coef 0.1, 256 games/iter, entropy 0.05 flat, lr 3e-5/1e-4,
battle-timeout 60, 30 workers. New inputs (ascension/keys/burning) now RANDOM-init — the
zero-init era ended with the fresh run; old checkpoints evaluate faithfully only at tag
`honest1-eval-compat`.


**Heart-run support built (`aa8f739`) — not yet deployed; honest1asc continues.** Full act-4
stack: key flags + burning-elite position in the obs (zero-init, golden-checked no-op on old
checkpoints), TAKE_KEY + OPEN_CHEST policy actions, 'heart' reward fn (floor/114 uncapped to 57;
act-3-only win +0.25 vs heart kill +0.5 — only a heart kill totals ~1.0), shaping_key_coef PBRS
term, heart_win/act3_win/act4_reach/avg_keys stats. Validated end-to-end: all 3 keys obtainable
via the policy interface, act-4 transition, Shield & Spear + Corrupt Heart beaten under real
search. Plan: fresh from-scratch run (uniform A0-20) — the current policy likely learned
never-RECALL, so no warm start.

**⚠ BUGFIX with run-wide implications: every prior run skipped every chest.** SearchAgent's
treasure-room fallback had open/skip inverted (`GameAction(takeChest)` → idx1=1 = skip), and
construct_choice didn't surface the screen to the NN — so ALL collection and evals to date
(incl. honest1's 0.794 and the 62.5-62.7% baselines) played without chest relics. Fixed +
chests are now an NN decision. Future evals get a strictly richer game; absolute numbers are
not directly comparable to pre-fix ones (direction: pre-fix figures are handicapped).

## 2026-06-07

**Battle-end detail eval terms GATED IN as engine defaults (MCTS session, boss-eval@0245769):**
goldLossWeight 0.25 (escaped-thief gold; 100g==25HP calibration), maxHpWeight 2.0 (delta vs
search root; Feed/Darkstone), parasitePenalty 12 (Writhing Mass implant, mirrors exitBattle).
Targeted A/Bs (honest1-440 slices @1k sims): thief gold lost 27.2→4.1 (~0.1 HP); implant 68%→24%
(HP up); Feed decks +1.0 maxHp/battle (HP up). Deployment gate 79.2% vs 77.8% (flips +91/−77,
floor +0.29). eval_states gained a DETAIL line (goldLost/parasiteRate/maxHpGain); encounter
state slices in states_h1/.

**DPW widening split by chance-node category: knob landed (6b4934f), defaults unchanged.**
WIDEN telemetry: cap binds 39-45% of END_TURN visits vs 3-10% card; card widen executes are
56-62% wasted sibling re-rolls. 180-trial 4D tune (honest1-440 dev set): best region card
5.0/0.84 + endTurn 5.7/0.49, +0.8 dev / +0.9 holdout — deployment gate TIE (77.7 vs 77.8,
flips +129/−130). Third knob-split evaporation in a row (chance/det exploration, now widening);
only the mechanism-backed objective change survived its gate. Perf note for later: adaptive
stop-widening after K sibling collisions could reclaim most of the wasted card executes
without quality risk.

## 2026-06-06

**honest1asc: 512 games/iter + gentle anneal @~iter 175** (user call, on the first signs of an
A20-era climb: iters 165/170 = 0.316/0.293 vs era mean ~0.21; bands 0.58/0.36/0.14/0.10 and
0.47/0.40/0.10/0.11). Restarted at the next checkpoint with --num-games-per-step 512 (halves
per-level variance: ~24 games/level/iter) and geometric decay of entropy coef 0.05→0.0167 and
lr→1/3 over 100 iters, anchored at the resume iter. A20-era context: flat ~0.21 through iters
106-160 (pooled 10-iter blocks 0.198-0.235), training health clean throughout (EV 0.70→0.75,
clipfrac ~0.05). First A20 (double-boss) wins at iters 120/150.

**Chance-vs-det UCB exploration split: tuned + gated, defaults stay (25,25) (MCTS session).**
New knob `explorationParameterChance` (boss-eval@010ed99, byte-identical at default): separate
UCB constant for stochastic edges at decision nodes. Tuned on fresh honest1-440 production
states (2346 dev / 1156 holdout, acts balanced, run_episode-collected; 138-trial Optuna TPE
@1000 sims, full-set evals; spot box, one reclaim — checkpointed db, deterministic
re-collection). Proxy basin (expl≈18, ec≈12) and holdout (+0.56) did NOT survive the gate:
paired 1000-seed eval_hero (honest1-440 @1k sims, seeds 8.2M; control 25/25 = 77.7%, floor
48.12): **A (18,12) tie** (77.8%, flips +135/−134; floor +0.53 p=0.049), **B (25,12) −5.9pp**
(71.8%, flips +117/−148 p=0.065; floor −0.97 p=0.005). Reading: deployment is sensitive to the
det:chance exploration *balance*, not the absolute level — both-scaled-down ties, chance-only
cut loses; ec≥35 is catastrophic even per-battle (ec=70 → score 47 vs 83). Proxy↔deployment
divergence recurred despite production-strength states; 1000-seed paired gate stays mandatory.
Artifacts: `states_h1/` in the boss-eval worktree (state sets, tune db/csv, gate csvs).

**honest1 concluded at iter ~456.** Anneal floored @355 (entropy coef 0.0025, lr 3e-6/1e-5);
high-water 0.816 @436, then 0.758-0.785 — plateau called. Process stopped (last checkpoint
iter_455); local three-stage eval queued: (1) checkpoint screen 401/431/436/455 × 400 paired
seeds @1k sims (screen seeds 2,000,000+ disjoint from the headline set), (2) headline champion
vs heroe2-270 paired on identical seeds (honest engine, McNemar), (3) routing intervention:
champion ± --randomize-paths (causal price of the routing policy).

**honest1 final evals (local, 1k sims, honest engine).** Stage 1 screen (400 seeds @2,000,000+):
435=440=0.823 > 455=0.805 > 430=0.798 > 400=0.795; 435/440 perfectly discordance-balanced
(41/41) → champion = **iter_440**. Stage 2 headline (1000 held-out seeds @1,000,000+):
**0.794 ± 0.013** — +17pp over the prior honest champion (heroe2-270 = 62.5%), and above the
cheat-era champion's clairvoyant 0.768. Stage 3 routing intervention (same 1000 seeds,
--randomize-paths): 0.696 vs 0.794 → **the learned routing policy is causally worth +9.8pp**
(discordant 213 vs 115, McNemar z=5.41, p=6e-8). The R5b+aux map-representation program is
validated end-to-end: encoding → SL learnability → RL routing weights → causal win-rate value.

**honest1asc dialed up to A15 @iter 40** (user call: "enough wins even on A5"). Resumed from
the iter_40 checkpoint (optimizer included) with --max-ascension 15 → uniform 0-15 (seed % 16,
~16 games/level/iter). 0-5 era summary (iters 1-40): mixture 0.395→0.33 dip→0.418; A0 held
0.42-0.55 (≈ honest1's level at the fork), A4/A5 ~0.2-0.3. New mechanics now in
distribution: A6 90% start HP, A7+ monster HP, A10 Ascender's Bane, A11 two potion slots,
A12 fewer upgraded rewards, A13 less boss gold, A14 lower max HP, A15 hard event pool —
all engine-side, pipeline smoke-tested at A15 (8/8 games clean; A10/A14 start states verified).

**honest1asc launched (box, 30 workers): ascension 0-5 uniform mixture.** Warm start from
honest1.pt.iter_155 — the last pre-anneal (entropy 0.05, lr 3e-5) checkpoint, per plan: keep
exploration high for the new task distribution; fresh AdamW. No decay scheduled yet (anneal
manually later, as with honest1). Level dealt by `seed % 6` (reproducible). Infrastructure
(`7a63fb1`): ascension appended to the fixed observation (6→7, max 20) and treated as a
zero-init categorical embedding — golden-checked bit-identical to the old net at A0, so the
warm start preserves honest1's A0 policy exactly; old parquet (6-element fixed obs) still
collates (→A0). Per-level win rates (win_rate_asc0..5) now in stats jsonl. Engine effects
verified at A1 (act-1 elites 4.44→7.11 avg), A2+ (monster HP gates), A5 (75% post-boss heal),
A6 boundary (90% start HP — outside our mixture). Search knobs remain the A0-tuned set.

## 2026-06-05

**Engine throughput A/B on live honest1 (collect s/iter @ 256 games):** `c6b4d84` ~265s (win
0.50) → `51156d0` ~380-460s (win 0.53-0.68; march=native ruled out by a no-march rebuild arm)
→ **`d1189ce` ~285-320s (win 0.63-0.70) ← deployed.** Verdict: the uid-relabeling dedup
(`38205a3`) regresses wall time ~30-40% on strong-policy deep-game collection despite its
weak-policy telemetry showing +12%; the d1189ce alloc/sort round is genuinely fast. Flagged in
COORD for re-gating on strong states. honest1 quality unaffected throughout (0.703 @ iter 251,
new high; entropy 0.61, coef 0.013, lr 1.1e-5). UPDATE: MCTS session confirmed on iter_320 states
and REVERTED uid-dedup upstream (`1a8d1b1`, merged as `a4caa5b`); box build dir rebuilt —
search-identical to the deployed d1189ce, picked up at the next natural restart.

**honest1: annealing engaged @155 (entropy 0.05→0.0025 + lr→0.1×, 200 iters each); engine
upgraded to `51156d0` @190** (perf round + uid-blind transposition dedup, perm-reroot reverted
after gate failure; quality-neutral, ~12% faster search — MCTS-session gates). Win crossed 0.5
@181 honest. Box build now uses -march=native via the new STS_MARCH_NATIVE opt-in.

## 2026-06-04

**R5b encoding validated in RL — honest1 learns a real routing policy.** Conditional logit on
honest1's iter-90..101 dumps (32.8k informative path decisions): REST +1.51±0.03 (4.5x odds vs
MONSTER, z=56), SHOP +1.34 (z=43), EVENT +0.64 (z=41), ELITE +0.22 (z=8), and hp_frac x REST
-1.72±0.13 (z=-13; strong rest-when-hurt). The cheat-era policy with 3x the training never
exceeded ±0.1 nats (all n.s.). aux_room_loss collapsed to ~0.003 by iter 6 (grounding circuit
instant, as the lab predicted). honest1 win-curve pace ~matches the best cheat-era recipe per
iteration (0.43 @ iter 96 honest vs hient ~0.42 cheating) at ~2x the wall speed.

**honest1 launched** — first honest-era hero run (new Lambda box 192.9.243.58, A10/30 cores,
persistent vol ~/sts-ca). From scratch on rerandomize2 @c6b4d84: honest CardPile engine with
tuned defaults (expl 25, widening 3.70/0.52, EvalWeights unchanged; +ActionQueue/Transmutation/
ethereal fixes), R5b encoding + dest-room aux (coef 0.1), PPO epochs 2, 256 games, batch 192,
lr 3e-5/1e-4, entropy 0.05 flat (decay + lr decay held in reserve for the post-climb phase),
1000 sims, --seed 1, --save-episodes from iter 1, 30 workers (collect ~104s/iter — ~2x faster
than the old box). Reference points: honest heroe2-270 @engine defaults = 62.5%; early win
rates will look far below cheat-era curves by construction.

## 2026-06-03

**R5b encoding + aux loss in production** (`33bd7db`): repr-lab winner ported to
network.py/rl_train.py — dest room on path options, ego-rel/reachable/scaled-aggregate node
features (exact collate-time DP, cross-checked vs lab on 300 rows; reachability frontier
includes Winged-Boots extra options), boss→categorical, dest-room aux head
(aux_dest_room_coef=0.1, logged as aux_room_loss). Breaks checkpoint compat (tag
`cheat-era-final` for old nets). Ready for the honest-era from-scratch run alongside the
entropy + lr decay schedules; awaiting honest-dynamics knob retune (MCTS session).

**Repr lab CONCLUDED (3 rounds, 73 cells, `sl_repr_lab.py` + results CSVs).** Final picture:
- *Grounding (take-elite/rest CE)*: baseline = seed lottery ({0.50..0.999} across replicates);
  every relational arm saturates at 1.000 every seed.
- *Routing (pick option minimizing dist-to-rest / avoiding elites)*: R0 0.918 → R4 0.968 →
  **R5b 0.999**. Dropping path_xs (R6) costs 3–4pp: keep it.
- *Queries*: dist-to-rest MAE 0.17 (R0) → 0.007 (R4/R5b); max-elites 0.169 → 0.005 (R5b);
  depth-3 elite count 0.116 → 0.005.
- *Low-data (⅛ ≈ 900 examples)*: **R0 at chance on everything; R4 still saturates** — the
  RL-relevance clincher (RL's signal is weaker still).
- *Embedding family*: **sinusoids are load-bearing** — learned tables at equal structure (R7)
  crater (elite 0.98→0.57, dist 0.17→0.66); even on R4 features they cost 19pp routing (R4L).
  R5's oracle pathology was unscaled magnitudes in additive token sums (R5b scaling fixes it;
  R5L tables also work but trail R5b).
- *Aux losses* (free self-supervised labels): dest_room aux lifts R0 routing 91.8→95.5 and
  pins elite ≈0.998 without any repr change; queries aux +2pp; combining doesn't stack.

**Production recommendation (honest-era run):** adopt R5b encoding — path option += destination
room type; map nodes += ego-relative (dx,dy) + reachable flag + [0,1]-scaled (minE, maxE,
distRest) aggregates; keep path_xs and sinusoids; fix boss→categorical in fixed_obs; scale
discipline for any added feature. Add dest_room (+ optionally queries) aux heads to RL training.

**Hient vs heroe2, paired held-out evals (cheat-mode).** Plateau called at iter ~440
(band 0.70±0.04, 75 iters past high-water). Checkpoint screen (400 paired seeds @1k sims):
iter_370 (training high-water 0.766) evals 0.688 — winner's curse confirmed; iter_425 wins
(0.725). Headline tier (1000 identical seeds, tuned engine, McNemar):
**hient-425 0.729 vs heroe2-240 0.663 (+6.6pp, p=7e-4)**; vs heroe2-270 0.711 (+1.8pp, p=0.36,
n.s. — heroe2-270 is much stronger under the tuned engine than its old-engine evals suggested).
10k-sim tier (500 paired seeds): **hient-425 0.768** vs heroe2-270 0.744 (+2.4pp, p=0.34) vs
heroe2-240 0.714 (+5.4pp, p=0.034). Verdict: hient-425 nominally best at both sim counts,
decisively > heroe2-240; statistically a peer of heroe2-270. Closes the cheat-mode era;
hient paused at iter_440 (resumable), box idle for honest-era work.

**LR decay schedule added** (`2763271`): geometric to `lr_final_frac` over `lr_decay_steps`
from absolute `lr_decay_start`; actual lrs logged per-iter. For the next hero run.

**Repr lab round 3 (partial): sinusoids exonerated.** Learned-table embeddings at identical
structure (R7) crater everything (elite 0.98→0.57, dist_rest MAE 0.17→0.66); even on R4
features they cost 19pp on routing (R4L). The smooth shared sinusoid code is what enables
coordinate matching in attention. R5's failure was unscaled magnitudes in additive token sums
(fixed by [0,1] scaling in R5b), not the embedding family. Aux-loss scaffolding cells pending.

## 2026-06-02

**Repr lab rounds 1–2** (`sl_repr_lab.py`, fresh RL-sized nets on 43.5k dumped path decisions,
fixed-budget protocol). Round 1: baseline encoding learns map tasks but *unreliably*
(plateau-then-takeoff; rest task across seeds: {0.50, 0.65, 0.99}); R1 (dest room type on path
option) fixes grounding instantly; R3 (ego-relative coords + reachability flag) makes DAG
queries near-exact (dist-to-rest MAE 0.17→0.008); R4=R1+R3 best everywhere; R5 (unscaled
oracle aggregates) *degrades* unrelated tasks — embedding-sum interference. Round 2:
**R5b (R4 + [0,1]-scaled per-node aggregates) solves routing (99.9%)**; R6 shows path_xs still
needed (−3-4pp routing without it); seed replicates: R4 1.000 everywhere, R0 a lottery;
**⅛-data regime: R0 at chance, R4 saturates at ~900 examples** — likely why the RL policy
never learned node preferences. Production candidate: R4 features + scaled aggregates,
keep path_xs, boss-encoding → categorical, scale discipline for added features.

**SL map probe** (`sl_map_probe.py`): is the map encoding learnable at all? Yes — take-elite
reaches 99.4% (80 epochs) and rest/monster ~98% (20), with leftmost control at 100% instantly.
But every map task shows a plateau before takeoff = expensive multi-hop circuit. Map-encoding
code scan: no bugs (mask conventions, pathXs, boss row all correct); weakness is structural
(option = bare x; destination resolution = 3-hop attention program; no is_current at act start).

**Path-choice statistics** (episode dumps iters 306–311): policy is statistically uniform over
immediate node types. Head-to-head 49–51% on all major pairs (EVENT vs MONSTER 50.0%, n=8446);
chosen-option prob ≈ 1/n_options. Conditional logit (17.4k informative decisions): all type
utilities within ±0.1 nats of MONSTER (n.s.); only hp×REST −0.30±0.16 (right sign, tiny) and
act-3 ELITE/EVENT leans (~2σ). Conclusion: path policy carries the residual entropy; no
node-type preference learned.

**Engine swap mid-run (iter 320)**: box rebuilt on rerandomize2+boss-eval merge — tuned
SearchAgent defaults (expl 9.9, widening 4.6/0.37, weights 53/11/37/3.4/1.75/1.5), boss
widening reverted (deployment regression), explicit era knobs dropped from launch. Training
band stepped 0.64–0.70 → 0.70–0.77 (~+3–5pp at collection; tuned-vs-legacy measured +10.5pp
at deployment by MCTS session — since revealed to be within cheat mode).

**Episode dumps + encounter column**: hient restarted @305 with --save-episodes;
`encounter` column added (most recent battle id) for per-encounter battle-outcome stats
(~21.5k decisions/iter, 2.3MB parquet).

## 2026-06-01 → 06-02 (hero run)

**ppo_hient promoted to hero** after winning the entropy bracket (0.05 sweet spot; 0.566@190).
Resumed @195 with 40 workers (collect 480s→205s) + **exponential entropy-coef decay 0.05→0.0025
over 200 iters** (absolute-anchored schedule, effective coef logged). Decay phase: win 0.52→~0.71,
entropy 0.86→0.50, smooth conversion, no sharpening pathology; engine swap at 320 confounds
~+3–5pp of it. Post-decay band 0.70±0.04 at coef floor.

## 2026-06-01

**Entropy-coef phase analysis.** Effective entropy weight in trajectory-return units =
coef·N̄·σ_adv (PPO) / coef·N̄ (GRPO), N̄≈62–70 decisions/traj — the missing N̄ was a ~70×
error in the phase-plot indifference lines. Effective pressure orders outcomes monotonically:
hient 0.51 ret/nat → win 0.52↑; grpo_a 0.64 → peak 0.46; ent10 0.89 → ~0.30; ent25 → stalled.
Also explains GRPO's late decay (its 0.01 coef ≈ 0.64 ret/nat — not negligible).

**GRPO concluded: PPO strictly dominates.** grpo_a (RLOO, critic-free, from scratch): peak 0.46
at ~1.7× PPO wall time, then decays as groups go homogeneous (entropy 0.78→0.91, win →0.28-0.36).
Root cause: trajectory-broadcast credit vs per-step GAE. Code removed (`bd77ef7`), implementation
at tag `grpo-final`. Side findings: low-HP penalty net-negative (−0.125 win for marginal safety);
sampling temp 1.3 stalls learning.

**Entropy bracket** (from-scratch PPO e2, lr 3e-5, 256 games): coef 0.05 ≫ 0.10 (~0.30 plateau)
≫ 0.25 (stalled ~0.02). Steep falloff between 0.05 and 0.10.

## 2026-05-31

**epochs=2 broke the PPO plateau** (fork hero@170, only change): win ~0.59 → 0.65–0.70 by 270
with healthier optimization (grad norm declining, KL ~0.009). heroe2-240 = 0.706±0.020 @10k sims
on 500 held-out seeds vs hero-130 = 0.612. Refuted: lr/2, entropy cut.

## 2026-05-26/27 (pre-log backfill)

**PBRS reward shaping**: HP/#upgrades potential terms; coeffs from floor-controlled OLS on
episode dumps → shaping_upg_coef 0.035 / offset 0.307 used by all later runs.
**Value-function SL**: held-out EV ceiling ~0.42 is irreducible variance, not capacity
(train EV→1.0 overfit check); smaller nets win offline.
**MCTS perf**: profiling-driven ~1.36× battle-search speedup (rerandomize2).
