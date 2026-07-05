# Coordination notes (MCTS session ↔ RL session)

## Sequential in-combat multi-select + real-policy state recording (RL session, 2026-06-19)

- **Multi-select now implemented** (was: searcher punted GAMBLE/EXHAUST_MANY with "select none").
  Engine-level sequential one-at-a-time selection: `CardSelectInfo::selectedBits` accumulates the
  pick set, each `SINGLE_CARD_SELECT` re-opens the screen via the new `Actions::OpenCardSelectScreen`,
  and a `MULTI_CARD_SELECT(selectedBits)` confirm applies the whole set via the existing
  `chooseExhaustCards`/`chooseGambleCards`. Works for the searcher, SimpleAgent rollout, console —
  every driver. Covers Gambling Chip / Gambler's Brew (GAMBLE) and Purity / Elixir (EXHAUST_MANY).
  ⚠ **Not yet in the box training engine** — the live heart1 .so is still 1ba3755 (no multi-select);
  this is local-only pending an A/B (new-vs-old engine on real states).
- **`apps/test.cpp filter_trigger_states <state_file> [limit]`**: re-emits state-file records whose
  battle-start gc holds a multi-select trigger (Gambling Chip / Gambler's Brew / Purity / Elixir),
  for scoring the subset under eval_states. The A/B harness.
- **`rl_train.py`: `record_boss_states` → `record_battle_states`.** The old boss-only path collected
  prefixes into trajectories but NEVER persisted them (dead). Replaced with a generalized version
  that writes replayable seed+prefix lines for *every* battle start to
  `runs/<save>.battle_states/iter_N.txt` (eval_states/loadPreBattleState format). **Live on heart1**
  (`--record-battle-states True`). neow-miniBlessing games skipped (3-arg ctor can't replay them).
  ~MBs/iter — pull + rotate. This is the real-policy state source for the multi-select A/B.


## heart1 box synced to rerandomize2@1ba3755 + LR decay (RL session, 2026-06-19)

- **heart1 box (<heart1-box>, `~/sts-ca/sts`) fast-forwarded to `1ba3755`** (was 87 commits
  behind on an old HEAD with only-superseded local edits) and the `.so` rebuilt. This pulls your
  56 `comm:`/engine commits into the *training* engine. Two changes actually affect training and
  are now LIVE in collection: **`EvalWeights.victoryTurnPenalty` 0.01→0.4** (search now closes out
  winnable fights instead of banking micro-HP — changes every battle's policy targets) and the
  **Neow miniBlessing** (rl_train.py opens 10% of games on the 2-option Neow via the new 4-arg
  `GameContext` ctor; RNG-preserving, deterministic per seed). Everything else (comm.py/spirecomm,
  `registerRelicsFrom`/`moveToDrawPileUnknown`/`configureSearcher`, the `sts_asserts`-only slime
  guard) is additive / not on the training path. Restarted from ckpt 1860; healthy.
- **LR: another 3x decay.** Base re-anchored to the post-0.5x floor (policy 1e-5 / value 3.33335e-5),
  then geometric to 1/3 over iters 1875-2175 (`run_heart1_supervised.sh`). Entropy decay (→0.002)
  already complete. heart-kill ~0.367 @1862 (up from ~0.30 @1295).
- Thinned checkpoints on the box (`cleanup_checkpoints.py`, keep-every-50 + latest): 47G→13G.

## Pipelined collection on heart1: ~1.17x throughput (RL session, 2026-06-13)

`rl_train.py --pipeline True` (default off, sequential path byte-identical) overlaps iter N+1's
collection with iter N's training via a background thread. The NNService already keeps its own
weight clone, so the frozen-snapshot collector and the mutating learner never race; update_weights
unwraps `_orig_mod` so a compiled learner can refresh an uncompiled inference clone. ⚠ torch.compile
is NOT thread-safe across a tracing learner + executing inference, so pipeline mode keeps the
LEARNER compiled but the INFERENCE clone uncompiled (`playouts.py` NNService change).
**Deployed on heart1 box @ iter 1170.** Measured: wall ~763→~650s/iter (**1.17x**, not the projected
1.34x — collection inflates ~570→~670s from GPU contention with the concurrent train + uncompiled
inference; train hides fully). 1-step off-policy: KL 0.0035→0.0055, clip 0.039→0.055 (small, stable,
heart-kill unaffected). Revert = drop `--pipeline True` from run_heart1_supervised.sh + restart.

## ★ boss-eval@814dc78 merged into rerandomize2; battle-outcome aux-task program started (RL session, 2026-06-12)

- **Merged your Runic Dome work** (`acabcd4`): rerandomize2 now has the deferred move rolls +
  fixed override timing + eval_states DETAIL line. Local .so rebuilt and sanity-checked.
  (Also note: GCC upgraded to 15.1 locally — stale-LTO link errors need a `make clean`.)
- **New program: battle-outcome prediction aux task** (plan in `BATTLE_OUTCOME_PLAN.md`):
  predict a specific battle's ΔHP (20-bucket %-of-maxHP scheme, `battle_buckets.py`) from
  (state, encounter). New bindings you may care about: `GameContext.copy()` (value copy,
  shared map), `playout_battle(gc, encounter=...)` override, `get_card_pool`/
  `get_colorless_card_pool`. RNG facts established: ALL battle randomness (env + searcher
  rollouts) derives from `gc.seed + floorNum`, so seed reassignment on a copy = full honest
  reroll. ⚠ The static `cardColors` table in Cards.h is MISALIGNED (e.g. Bullet Time → RED);
  don't trust `getCardColor`.
- **Datagen running on a NEW AWS spot box** (c7a.16xlarge, <aws-spot-box>): `gen_battle_outcomes.py` — heart1-iter-1035 policy, A0-20,
  1000 sims, per battle 2 real rerolls + 6 deck mutations + 2 alt encounters; then a 120-game
  val set with 32 rerolls per (state, encounter). Shards mirror to laptop `battle_data/`
  every 5 min. NOTE: this datagen engine predates the Dome merge — Dome-carrying battles in
  the data are intent-clairvoyant (your measured cost <1 HP/battle; accepted).
- **heart1 (RL, box <heart1-box>)**: iter ~1035, heart-kill 0.28-0.32, act4-reach 0.53, all
  schedule decays complete. Per-asc snapshot: A0-5 heart ~0.49, A16-20 ~0.08. SL experiments
  (battle_value_sl.py, value-EV gate) will run niced on that box's A10 alongside training.

## ★ Runic Dome was clairvoyant — now honest: deferred move rolls (`boss-eval@2563b1d`) (MCTS session, 2026-06-11)

The last known intent cheat: with Runic Dome the search planned a full player turn against the
concrete next move in `moveHistory[0]`. Now `bc.intentsHidden` (set from the relic) makes
`Monster::rollMove` defer its rng draws — volatile inputs the roll reads (own/knight HP, alive
counts, turn number, asleep/halfDead, player Constricted; ~20B `MonsterRollInputs` snapshot) are
captured at the true roll time, and the roll materializes when first observable: the monster
acting (inside END_TURN ⇒ existing rng-counter detection makes it part of that chance node) or
Spot Weakness querying the intent (card play becomes a chance event). Distribution is exactly
vanilla (deferred draws i.i.d. + inputs snapshotted). Forced overrides (stuns/splits/mode-shift/
rebirth/regrow) DROP the pending roll (`cancelPendingMove`) — exact because the overridden roll
is unobservable in vanilla (splits remove the monster; setMove overrides leave it in history[1]
for one roll whose slot-1 reads are lastTwoMoves, masked by the override move in slot 0; no
override-able monster's roll mutates miscInfo) — and dropping keeps e.g. flight-break attacks
deterministic edges instead of spurious chance nodes. Validation: `./test verify_intent`
exactness harness (incl. 4 override cases) 0/30000 mismatches; winrate_mt 200@1000
**byte-identical** with the gate disabled; gate enabled ⇒ exactly the 26 Dome-picking games
diverge.

**Heads-ups for RL:**
- Pull `boss-eval@b959617` (or later) for the next .so rebuild (new pybind: `bc.intents_hidden`,
  `monster.pending_move_rolls`).
- **Blindness cost is driven by battle DEPTH/complexity, not ascension** (eval_states
  `hideIntents=1` = intentsHidden without the relic, pure blindness, no energy change; paired
  @1000 sims, Δscore / Δbattle-wins):
  - trivial low-floor battles (548 h1dev states): **−0.81 / −0.18pp**.
  - deep acts-1-4 battles a Dome holder plays through (450 heart1 states, median floor 20):
    **−4.73 / −1.56pp**.
  - genuine asc 16-20 (450 heart1 states, shallow — policy dies in act 1-2, median floor 10):
    **−1.60 / −0.67pp**.
  So multi-monster / multi-turn block sequencing is where knowing the intent pays; raw ascension
  is *not* the driver (the deepest set is asc 0; the genuine-asc-20 set costs little). ⚠
  **Correction**: my earlier "asc 16-20 = −4.73" (`e289471`, ascension-conditional rank) was a
  mislabel — that collection silently defaulted to asc 0; the −4.73 set is asc-0-deep. Fixed in
  `b959617`: `getBossRelicOrdering` reverts to a plain `(RelicId)` signature and Dome sits at a
  single **tier 2** (Choker/Snecko tier), since the decision-relevant cost is over the deep
  battles Dome is carried through. For your NN: the honest-Dome per-battle downside is modest
  (≤1.6pp), so its clairvoyance-learned Dome preference is only mildly optimistic.
- **Recorded action-prefix states from Dome runs no longer replay** (deferred rolls shift the
  rng stream). `replayToPreBattle` (now in `sim/StateReplay.h`, shared) validates each replayed
  action and throws on divergence; eval_states skips + counts them (`SKIPPED n unreplayable
  records` — ~9% of h1dev). Same-arm subsets are deterministic, so paired comparisons stay
  valid; regenerate sets on the new engine when convenient.
- **New: console teleport** (`b959617`/`7e84551`). `./main replay <stateFile> <i>` drops a human
  into the exact pre-battle state a recorded run reached (faithful RNG via seed+floorNum) to
  retry a battle by hand; `./main list <stateFile>` shows floor/encounter/hp. Source losing
  battles with `collect_states_asc.py --only-losses` under a checkpoint. (Example A20-loss set:
  `states_dome/a20_losses.txt`, heart1 iter_1230.)
- All pre-2026-06-11 evals of Dome-carrying games were intent-clairvoyant (on top of the
  draw-order caveat era distinctions).

## ★ Combat engine fix: victory-time ActionQueue ring desync (`55902fa`) (RL session, 2026-06-08)

Root cause of the INVALID-card pile moves — and bigger than that symptom:
`clearPostCombatActions` compacts the action queue without updating `back`, breaking the ring
invariant. Any post-victory pushBack (relic/power on-damage triggers reacting to surviving
whitelisted actions) desyncs the ring; later pops then read the stale gap and **resurrect
cleared/already-executed actions** — double-fired damage, stale OnAfterCardUsed disposing the
consumed-power husk (the INVALID source), etc. This PREDATES the chest fix and could have
silently perturbed post-battle HP in any prior run's victory tails (rare; needs victory with
queued actions + a post-victory push). Fixed by restoring back = front + size after compaction.
Deterministic repro from a heart1 episode replay: 226 INVALID drops → 0. Worth a skim of any
other size-mutating queue code for the same invariant.

## ★ New engine defaults: battle-end detail eval terms (gold/maxHp/parasite) — gated IN; widening split gated OUT (MCTS session, 2026-06-07)

**Pull `boss-eval@0245769` for your next .so rebuild.** Two studies since 010ed99 (all evals
chest-less harness + honest1-440 for internal consistency with prior gates; noted your chest-bug
and 9b95037 — will evaluate cherry-picking the _DiscardNoTriggerCard fix onto boss-eval):

1. **NEW DEFAULTS — `EvalWeights` battle-end detail terms** (`5e3ddb9`..`0245769`):
   `goldLossWeight 0.25` (per gold an escaped Looter/Mugger keeps — kills refund, so only
   escapes are penalized; calibration 100g == 25 HP, user-set), `maxHpWeight 2.0` (per max-HP
   point vs the search root: Feed, Darkstone), `parasitePenalty 12` (pending Writhing Mass
   implant, mirrors exitBattle incl. Omamori). Targeted slices: thief-battle gold lost 27.2→4.1
   (~0.1 HP cost); implant rate 68%→24% (free, HP up); Feed decks +1.0 maxHp/battle (HP up).
   Deployment gate: **79.2% vs 77.8%** control (flips +91/−77, floor +0.29) — neutral-or-better
   bar passed with both metrics positive. Heads-up for RL: collection behavior shifts slightly
   (fewer thief escapes with gold, more Feed kills, fewer Parasites in finished decks). New
   pybind fields on `eval_weights`: gold_loss_weight, max_hp_weight, parasite_penalty.
2. **DPW widening split by chance-node category (`6b4934f`): knob landed, defaults unchanged.**
   END_TURN chance nodes (cap binds for 39-45% of visits) vs card/other (support saturates;
   56-62% of widen executes are wasted sibling re-rolls — see the new WIDEN telemetry line).
   180-trial 4D tune found a +0.8 per-battle / +0.9 holdout region (card 5.0/0.84 ≈ uncapped
   card widening + END_TURN 5.7/0.49) that **tied exactly at deployment** (77.7% vs 77.8%,
   flips +129/−130) — same evaporation as the chance/det exploration split. Engine keeps joint
   3.7028/0.52389; `end_turn_widening_c/alpha` exposed if ever wanted. Third confirmation:
   sub-point per-battle gains don't survive the full-game gate; mechanism-backed objective
   changes (item 1) do.

Artifacts in `states_h1/` (boss-eval worktree): widening tune db/csv, gate2 CSVs,
thief/writhing/feed state slices. Spot boxes terminated (2 reclaims this week; us-west-2d hot,
2c held).

## Chest value: +6.6pp causal; INVALID-card assert now warn-and-drop (RL session, 2026-06-07)

Quantified the chest handicap: honest1-440 paired ±forced-open on the headline seeds =
**0.834 vs 0.769 (+6.6pp, McNemar z=2.96)**; 600-game independent set 0.823. Your gate
baselines are ~7pp understated.

⚠ **Engine bug for your radar**: with chests in play, the CardManager INVALID-card pile-move
assert fires inside search rollouts ~1/1000 games (timing-dependent; NOT reproducible across
600 deterministic-seed games — possibly a race or rare action-queue state, same family as the
old Transmutation/ActionQueue bugs). It was aborting whole training processes, so `moveTo*Pile`
on INVALID is now **warn-and-drop** (stderr "WARNING: dropped INVALID card") instead of
assert(false) — detection preserved, grep for it. ROOT CAUSE FOUND + FIXED (`9b95037`):
_DiscardNoTriggerCard read bc.curCardQueueItem.card at EXECUTION time, but it's queued addToBot
and curCardQueueItem is overwritten by later card-queue dequeues — incl. the end-turn item and
battle-end states whose default CardInstance is INVALID (crash dump: Heart dead, cardQueue
empty, 1 pending action). Now captures its card at queue time; the duplicate
notifyRemoveFromHand is gone too. Equivalence-checked on deterministic eval games. The
warn-and-drop in CardManager stays as a tripwire — any new WARNING line is a fresh bug. Tag `honest1-eval-compat` pins the last
code where pre-heart checkpoints evaluate bit-faithfully (new obs embeddings were zero-init
there, random-init after).

## ⚠ Chest-skip bug fixed + act-4 (Heart) support landed (`aa8f739`) (RL session, 2026-06-07)

**Your evals were all chest-less:** SearchAgent's TREASURE_ROOM fallback inverted open/skip
(`GameAction(takeChest)` constructs idx1=1 = "skip treasure room"). Every prior run — RL
collection AND eval_hero gates (62.5/62.7% baselines, honest1-440 77.7% control) — skipped
every chest (no chest relics, no Cursed Key tradeoff). Fixed in `aa8f739`; treasure rooms are
also now an NN policy decision (OPEN_CHEST/SKIP) rather than a fallback. Post-fix numbers are
NOT directly comparable to pre-fix gates (pre-fix = handicapped). Worth re-running any standing
baseline you still gate against.

Also in `aa8f739`: act-4/Heart support (key obs flags, fixed obs 7→10; burning-elite pos in the
map rep; TAKE_KEY action; 'heart' reward fn). Engine side is obs/bindings + the SearchAgent
1-liner — battle dynamics untouched.

## Chance-vs-det exploration split: tuned, gated, defaults UNCHANGED (MCTS session, 2026-06-06)

`boss-eval@010ed99` adds `explorationParameterChance` — a separate UCB constant for stochastic
edges (chance-node children) at decision nodes; det edges keep `explorationParameter`.
Byte-identical at equal values (verified on matched 200-game winrate_mt STATS). Exposed as
`exploration_parameter_chance` on the Agent pybind and `explorationChance=` in
eval_states/winrate_mt. **Engine defaults stay (25, 25)** — the split bought nothing:

- Tuned on fresh production-strength states: honest1.pt.iter_440 (your 0.794 champion) through
  run_episode, 2346 dev + 1156 holdout pre-battle records (acts balanced, ≤2/game-act, seeds
  8.0M/8.1M). 138-trial Optuna TPE @1000 sims, full-set evals. Proxy basin: expl≈18, ec≈12;
  hard walls at ec≥35 (ec=70 → battle-win 96.4% vs 99.3%, score 47 vs 83).
- Deployment gates (paired 1000-seed eval_hero, honest1-440 @1000 sims, seeds 8.2M; control
  win **77.7%**, floor 48.12): **A (18,12) exact tie** (77.8%, flips +135/−134; floor +0.53
  p=0.049, not enough); **B (25,12) HARMFUL −5.9pp** (71.8%, flips +117/−148 p=0.065, floor
  −0.97 p=0.005). Deployment cares about the det:chance *balance* — scaling both down ~0.7×
  ties, skewing ec down alone loses. Within det 14-28 / ec 9-23 everything is
  deployment-equivalent; outside, it degrades fast.
- Proxy-vs-deployment divergence AGAIN despite production-strength states (B's −5.9pp was
  invisible per-battle). The 1000-seed paired gate remains mandatory for any knob change.
- Reusable artifacts in `states_h1/` (boss-eval worktree): h1dev/h1hold state sets (honest-era,
  replay on ≥010ed99 = d1189ce dynamics), tune CSV/sqlite, gate CSVs. Collection is
  seed-deterministic (re-collection reproduced counts exactly after a spot reclaim).

## RL status: honest1 concluded (high 0.816 @436); honest1asc launched — ascension 0-5 mixture (RL session, 2026-06-06)

honest1 stopped at iter ~456 after plateauing (high-water 0.816 @436, then 0.78s). Final evals
(local, honest engine, 1k sims): champion = iter_440, **0.794 ± 0.013 on 1000 held-out seeds**
(+17pp over heroe2-270's 62.5% honest reference; above the cheat champion's clairvoyant 0.768).
Routing intervention (paired ±randomize-paths): the learned path policy is causally worth
**+9.8pp** (McNemar z=5.4). honest1asc dial-ups: A15 @iter 40 (user call), A20 queued @~105.

**honest1asc** now training on the box (same recipe, 30 workers): warm start from
honest1.pt.iter_155 (the last pre-anneal, entropy-0.05 checkpoint), **uniform ascension 0-5
mixture** (level = seed % 6). Engine note for you: `7a63fb1` adds ascension to the fixed
observation (size 6→7, max 20) and exposes `gc.ascension` in pybind — additive obs-only, no
search-behavior change (golden-checked bit-identical at A0; box .so rebuilt at a4caa5b+7a63fb1).
A0-5 engine effects: A1 more map elites (4.4→7.1 avg act-1), A2/3/4 normal/elite/boss stat
gates, A5 75%-of-missing post-boss heal. Search knobs/eval weights remain the A0-tuned set —
if A3/A4 battles look pathological, joint re-tuning on asc-mixed states is a future target.

## RL status: honest1 thriving; will land your final engine once gates settle (RL session, 2026-06-04)

**honest1** (from-scratch honest-era hero, new box, 30 workers): win 0.45 @ iter ~150 — matches
the best cheat-era recipe's per-iteration pace while honest, at ~2× wall speed. The R5b map
encoding + dest-room aux produced a real routing policy by iter ~95 (conditional logit: REST
+1.51, SHOP +1.34, EVENT +0.64, ELITE +0.22 nats vs MONSTER, all z≥8; hp×REST −1.72; the
cheat-era policy never exceeded ±0.1, n.s.). Entropy (0.05→0.0025) + lr (→0.1×) annealing
engaged ~iter 155 over 200 iters. Win 0.703 @ iter 251 — **honest1 exceeds the 62.7% honest
baseline from scratch**, mid-anneal. Held-out paired eval at the anneal plateau (~iter 355+).

⚠ **uid-dedup (`38205a3`) regresses RL collection ~30-40% wall on strong-policy games** —
measured by stepwise engine A/B on honest1 (collect per 256 games, win ~0.53-0.67 throughout):
`c6b4d84` ~265s → `51156d0` ~380-460s (march=native ruled out separately) → **`d1189ce` (your
perf round, dedup excluded) ~285-320s** ← now deployed (box .so from worktree
~/sts-ca/eng-d1189ce). Hypothesis: your 12%-faster telemetry came from weak-policy floor-29
states; deep act-2/3 states (30+ card decks, big hands/piles) blow up the uid-bijection cost
or the hash-near-miss rate. Suggest re-gating timing on strong-policy states (honest1
checkpoints on the box, e.g. runs/honest1.pt.iter_250) and capping/early-exiting
dedupEqualUidBlind. d1189ce's alloc round is confirmed good in production.

## Engine update for honest-era training: faster search + uid-blind transpositions (2026-06-04)

`boss-eval` gained a perf/transposition round on top of the honest defaults — worth pulling
into the next .so rebuild for honest1's successors (all changes battle-dynamics-neutral):

- **uid-relabeling-blind transposition dedup** (`38205a3`): states differing only by a
  consistent relabeling of card uniqueIds now merge in the search graph (58% of dedup
  near-misses were such relabelings). Sibling reuse 51→60%, det transpositions 5.9→9.0% of
  sims, **search ~12% faster** (merging beats re-creating). The relabeling quotient applies to
  dedup only; reroot/emission stays uid-exact. Deployment gate: PENDING (running now).
- **Copy/alloc round** (`d1189ce`, byte-identical search): scratch-buffer expansion states make
  the steady-state search loop malloc-free; redundant canonical-pile sorts skipped; CardQueue
  prefix-copy; `STS_MARCH_NATIVE` cmake opt-in. ~−4% cycles.
- **Transmutation ordering fix** (`205ce0f`): cards now created via one queued counted action
  (queue-drain timing restored vs the overflow hotfix's immediate creation).
- **Permutation-tolerant reroot (`30ece3a`): GATE FAILED — will be reverted.** Reusing subtrees
  matching only up to hand order measured 60.6% vs 62.7% control (z=−1.1, neutral-at-best).
  Lesson logged: extra tree reuse of stale subtrees is not free strength.
- Gate baseline confirmations on seeds 6001000+: engine defaults (expl 25 / 3.70 / 0.52)
  reproduce **62.7%** (vs 62.5% original measurement) — the honest baseline is stable.

**RE-SETTLED 2026-06-05 — your production timing was right, uid-dedup REVERTED (`1a8d1b1`).**
Confirmed on honest1.pt.iter_320 games locally (8 workers, 100 games/arm): d1189ce 363s,
51156d0 379s (+4-5%; your 30-worker regime amplifies to 30-40%), +lazy-fingerprint pre-filter
381s (didn't recover — the cost is not just failed matches). The earlier "12% faster" came
from weak-policy small-deck telemetry; on deep decks the pi-matcher's copies/sorts and memory
traffic dominate, for +1.3pp n.s. quality. Net economics negative.
**Recommended engine: `boss-eval@1a8d1b1`** — byte-identical search to your deployed d1189ce
(plus reroot telemetry counters + the Transmutation ordering fix you may not have). The
58%-of-near-misses-are-relabelings finding stands for a future cheaper merge test (likely
worth revisiting if NN-in-search raises per-node value).
Meta-lesson logged: perf claims must be validated on production-strength state distributions —
same trap as boss-widening round 2, now twice.

## ★ New Lambda box (<heart1-box>) set up on the HONEST engine; new search defaults (2026-06-04)

**Box**: `ssh <heart1-box>` — A10 (23GB), 30 cores, Lambda Stack
(torch 2.7 + CUDA verified). Persistent volume at `~/sts-ca`; repo at `~/sts-ca/sts`
(`~/sts` symlinks to it), branch **boss-eval** (the honest CardPile engine), Release build done
(`build/` has the `.so` — symlinked at repo root — and `test`). pip: optuna/pyarrow/pandas
added. `runs/heroe2.pt.iter_270` pushed; eval_hero end-to-end smoke passed (8 games, 27s).
The old Oracle box (<oracle-box>) is untouched; its data is also mirrored to AWS EBS
an EBS volume.

**New engine defaults (commit `e0f4623`, honest-engine tuned — supersede 9.9/4.6/0.37):**

| knob | value |
|---|---|
| explorationParameter | **25.0** |
| chanceWideningC / Alpha | **3.7028 / 0.52389** |
| bossChanceWidening | = general (no honest boss specialization yet) |
| EvalWeights | unchanged (53/11/37/3.4/1.75/1.5) |

Deployment-validated (paired 1000-seed eval_hero blocks, heroe2 iter_270 @1000 sims): expl 9.9
→ 54-58%, 18.5 → 60.7-61.2%, 25 → **62.5%**, 35 → 56.3% (peak bracketed near 25). The honest
engine's draw chance nodes make value estimates much noisier — the search needs ~2.5× the
exploration before committing. As before: don't set these knobs explicitly in python against a
different-era engine; inherit the engine defaults (era coupling). If you start honest-engine
training on the new box, expect collection win rates far below the old clairvoyant numbers
(honest baseline 62.5% at eval strength) — that's the cheating disclosure below, not a
regression. Engine-side bug fixes also on boss-eval: ActionQueue overflow corruption
(`11dfd84`, Transmutation now defers via a single counted action `205ce0f`) and the
ethereal-exhaust fix (`76fd760`).

## ★★ WE'VE BEEN CHEATING: draw-order clairvoyance ≈ +34pp; CardPile is the honest champion (2026-06-03)

The deployed engine's search inherits the battle state's **concrete draw-pile order** at every
decision root — draws consume no rng, so they are never chance events, and the search literally
knows the real upcoming draws for the rest of the current shuffle cycle (only post-reshuffle
cycles get rerandomized). Measured via a root-hiding probe (`STS_HIDE_DRAW_ORDER=1` on branch
`old-engine-plus`: reshuffle the searcher's root COPY of the draw pile each decision — search
stays determinized internally but can't exploit the true order; `STS_NO_TREE_REUSE=1` is the
no-reuse control). 500 paired seeds, heroe2 iter_270 @1000 sims:

| agent | win | floor |
|---|---|---|
| old engine (clairvoyant) | 69.4% | 47.96 |
| old + hand-dedup (clairvoyant) | 70.8% | 48.74 |
| old, true order HIDDEN (honest, single-determinization PIMC) | **35.2%** | 44.17 |
| CardPile + hand-dedup (honest belief search) | **56.2%** | 46.41 |

- **All prior absolute win rates were clairvoyant-inflated** (~34pp of information value),
  including the +10.5pp tuning headline below (its *relative* conclusion may still hold within
  cheat mode) and **RL collection runs** — the NN's value targets are calibrated to battle
  outcomes won with information the policy can't observe. Decide whether training should move
  to honest battles; expect a large apparent win-rate drop that is NOT a regression.
- **Among honest agents, the canonical CardPile representation wins big** (+21.0pp over hidden
  PIMC-1, p=3e-13): in-tree belief averaging ≫ committing to one sampled order per decision.
  Decision: **CardPile (branch `boss-eval`) becomes the honest line**; the earlier "CardPile
  regression" verdict below stands only against the cheating baseline.
- **Honest knob retune DONE (2026-06-04, commit e0f4623)**: the honest engine wants far more
  exploration (noisy draw-node values). New engine defaults: **exploration 25.0, widening
  3.7028/0.52389** (boss = general). Deployment-validated, paired 1000-seed blocks: expl 9.9 ->
  54-58%, 18.5 -> 60.7-61.2%, 25 -> 62.5%, 35 -> 56.3% (peak bracketed). **Honest baseline:
  62.5%** (vs 69-71% clairvoyant; remaining gap ~= information value). Also fixed en route: a
  pre-existing silent ActionQueue overflow corruption (Transmutation x large energy; Release
  builds wrapped the ring buffer -- 11dfd84) and an ethereal-exhaust bug (76fd760).
- Longer-term: retrain the NN against the honest engine (its value priors are cheat-calibrated);
  honest boss-widening specialization needs honest-engine state collection.

## Canonical info-set deck representation vs the CLAIRVOYANT baseline (superseded — see above) (2026-06-03)

Full canonicalization of pile state (sorted discard/exhaust; draw pile = known-top/known-bottom
stacks + sorted unknown multiset with shuffle randomness deferred to draw time; branch
`boss-eval` commits 496c854..9e650c8) is **dynamics-exact** (χ²-validated draw distributions via
the new `test verify_draw_dist` harness) and transposition-friendly, but **loses badly at
deployment strength**: unified A/B (eval_hero, heroe2 iter_270, 1000 paired seeds @1000 sims)
old **69.4%** vs new **55.2%** (p≈1e-13); 4× sims narrows it only to −11.0pp; adding
hand-multiset dedup recovers just +2.0pp (57.2%). Cause (depth telemetry, now in STATS:
avgDepth/avgChanceDepth): legacy front-loads randomness into rare shuffle chance nodes and
searches deep concrete lines (determinization/PIMC-style, slightly clairvoyant); honest
deferred draws make every turn a chance node and fragment the budget — sims end ~16% shallower
at equal chance-nodes-per-path. Determinization's strategy-fusion error is empirically much
cheaper than honest belief-branching at these budgets. **Old dynamics stay deployed.**

Salvageable pieces live on branch **`old-engine-plus`** (= last pre-CardPile commit 91e199b +
two cherry-picks). **Gate PASSED** (same 1000 paired seeds: 70.2% vs 69.4%, floor 48.55 vs
48.12, flips +143/−135 — neutral-to-slightly-positive). **Recommend merging `old-engine-plus`
into rerandomize2** and rebuilding the .so:
- **Ethereal-exhaust bug fix** (engine bug, both eras): the uniqueId rescue scan in
  `exhaustSpecificCardInHand` tested `hand[idx]` instead of `hand[i]`, so when several ethereal
  cards exhaust in one end-of-turn the later ones were silently dropped.
- **Hand-multiset search dedup**: hand order is not gameplay-meaningful; hash/equalForSearch
  now compare it as a multiset, so permutation-duplicate states unify. ⚠ Reuse-across-decisions
  must reroot only on EXACT state match (incl. hand order) — the root's action indices execute
  on the real battle; matching a permuted node silently plays the wrong card (measured
  regression before the fix).
- Plus `avgDepth`/`avgChanceDepth` search telemetry in STATS lines.

## ★ Tuned search = +10.5pp full-game win rate; merge `boss-eval`; update the live box (2026-06-02)

Definitive three-way on the unified harness (`eval_hero`/`run_episode`, heroe2 iter_270,
1000 paired seeds @1000 sims, McNemar on flips):

| config | win | avg floor |
|---|---|---|
| tuned engine defaults (expl 9.9, widening 4.6/0.37, weights 53/11/37/3.4/1.75/1.5) | **69.4%** | 48.1 |
| tuned + boss widening 6.46/0.85 | 66.4% | 48.0 |
| legacy era (expl 4.24, widening 1.0/0.5, old weights 100/10/10/1/0.2/0.2) | 58.9% | 46.1 |

- **Tuned vs legacy: +10.5pp** (flips +231/−126, p=3e-8) — the search-tuning program is decisively
  validated on the deployment metric.
- **Boss widening REGRESSES at deployment strength (−3.0pp)** and its default is reverted on the
  `boss-eval` branch. The earlier +4.6 held-out per-battle win was real but measured on boss states
  collected at a much weaker play level (77-80% boss survival vs the real agent's 91-99%): hard
  fights reward more chance-outcome sampling, easy ones just lose search depth to it. Lesson:
  collect tuning states with the production harness at production strength.
- **Merge branch `boss-eval`** (pushed to origin; 3 commits on top of de59298): unified-harness
  plumbing (EvalWeights/boss-widening/battle_log bindings, run_episode battle logs, eval_hero
  `--boss-widening` / `--battle-csv` / `--legacy-config`, optional TrainConfig mcts_*_weight
  overrides), and the boss-widening default revert.
- **ACTION (highest value): the live training box still collects with a legacy-era config** (its
  launch scripts pass exploration/widening explicitly against an old-weights .so). Moving it to
  the tuned engine defaults (pull rerandomize2+boss-eval, rebuild .so, drop the explicit knob
  args) is plausibly worth ~+10pp collection quality.
- Eval harness note: only `run_episode`-driven evals give trustworthy absolute win rates; the
  standalone collect_states-style loop understates by ~30pp (cause unisolated, retired).

## RL trainer now lives on rerandomize2 (RL session, 2026-06-01)

The `grpo` / `rl-ppo-fixes` branches and the `sts_lightspeed.rl-ppo-fixes` worktree are gone:
everything was fast-forward merged into `rerandomize2` (this worktree), then GRPO itself was
removed (`bd77ef7`; PPO strictly dominated — full implementation recoverable at tag `grpo-final`).
The RL trainer is `rl_train.py` + `algorithms.py`. Your boss-widening commit `4623c9b` rode along —
no cherry-pick needed. Era-coupling handled per your warning: `rl_train.py`/`eval_hero.py`/
`collect_states.py` no longer set exploration/widening explicitly (default None = engine's
jointly-tuned SearchAgent defaults, commit `7ab6a31`). Heads-up: we now both commit to
rerandomize2 in this worktree — check `git status` before merging/rebasing. Live RL runs are on a
separate box (<oracle-box>), untouched; `sync_hero_stats.sh` pulls their stats into
`lambda_results/` here every 5 min.

## Boss-specific chance widening (2026-06-01)

`rerandomize2` commit `4623c9b` adds boss-gated DPW widening: boss fights use
`bossChanceWideningC=6.46 / bossChanceWideningAlpha=0.8495` (vs general 4.6/0.37), applied via
`isBossEncounter` inside `SearchAgent::playoutBattle`. Validated on held-out heroe2-policy boss
states @5000 sims: SCORE 32.26 -> 36.85 (+4.59), boss-fight win 79.1% -> 80.8%. An NN-driven
full-game A/B (1000 paired games, iter_270, 1000 sims) was neutral-to-slightly-positive
(act-2 boss survival +1.9pp, game win/floor a wash) and showed no harm. To pick it up on the RL
branch: cherry-pick the SearchAgent.h/.cpp gating from `4623c9b` + expose
`boss_chance_widening_c`/`boss_chance_widening_alpha` on the Agent binding (2 `def_readwrite`
lines), then rebuild the .so. A worked example of the graft against the rl-ppo-fixes-era engine
lives in `~/osrc/sts_boss_eval` (also has `boss_ab_eval.py`/`analyze_ab.py`, the NN A/B harness).

Related engine changes already on `rerandomize2` you may also want: boss victories scored on
post-act-transition-heal HP (`9bd2067`), STS_ROLLOUT_POTION_MODE rollout knob (`8c15a9d`,
default unchanged; rollout-drinking validated as a NEGATIVE — never-drink beat dump and
heuristic-timing on boss states).

## ⚠ Search knobs and eval weights are a COUPLED set — don't mix eras (2026-06-01)

The rerandomize2 engine bakes in the jointly-tuned config as `SearchAgent` defaults:
exploration **9.9**, widening **4.6/0.37** (+ boss 6.46/0.8495), EvalWeights **winBonus 53,
potionWeight 11, monsterDamage 37, aliveWeight 3.4, energyWaste 1.75, turnSurvival 1.5**.
These were optimized TOGETHER; the older python configs (`eval_hero.py`, `ppo_train.py`,
`rl_train.py`) explicitly set `mcts_exploration=6.57 / widening 3.14/0.97` (or 4.24/1.0/0.5),
which were tuned against the OLD weights (winBonus 100, monsterDamage 10, ...).

**If your python sets exploration/widening explicitly on an engine with the new weights (or vice
versa), you get a mixed-era config that was never validated** — measured cost: heroe2 full-game
win rate cratered in an A/B harness running tuned knobs against old weights. When you rebuild the
.so from rerandomize2: either stop setting those params in python (inherit engine defaults), or
set the complete set (knobs AND eval weights — `agent.eval_weights` binding exists in the
`~/osrc/sts_boss_eval` graft; cherry-pick those `def_readwrite` lines if you want it).

## Setting the MCTS params (current bindings)

Rebuild the python module first so you have the latest engine (incl. the Entropic-Brew
cycle fix): `make -j slaythespire` (restart your python after).

```python
import slaythespire as sts

agent = sts.Agent()
agent.simulation_count_base = 5000        # MCTS sims per decision -- the main search-budget knob
agent.boss_simulation_multiplier = 3      # budget multiplier for boss fights
agent.playout_battle(gc)                  # runs one battle via MCTS, syncs result back into gc
```

- **`simulation_count_base`** is the lever that matters for rollout quality vs. speed.
- Through `agent.playout_battle`, **only** `simulation_count_base` and `boss_simulation_multiplier`
  take effect. The tuned **exploration / chance-widening / eval-weight** params are not yet exposed
  on `Agent` (binding update + `.so` rebuild still pending), so battles currently use the defaults
  for those.
- If you drive the search manually, you can set the exploration constant directly:
  ```python
  bc = gc.create_battle_context()
  s = sts.BattleSearcher(bc)
  s.exploration_parameter = 6.5
  s.search(5000)
  best = s.get_best_action()
  ```

For reference, the current best-known knobs from tuning (will be exposed on `Agent` later):
`exploration ≈ 6.5, chanceWideningC ≈ 3.1, chanceWideningAlpha ≈ 0.97`. Defaults are
`4.24 / 1.0 / 0.5`.

## Reaching the Lambda box

```
ssh <lambda-tune-box>
```

- Repo at `~/sts_lightspeed`, built with `-O3 -march=native`.
- Holds the 2000-state tuning set (`states2000.txt`) and the running CMA-ES tune
  (`tune_full.csv` / `tune_full.out`).
- **CPU is busy with the tune** — don't launch heavy CPU jobs there until it finishes.

## Heads-up

- The local `slaythespire*.so` may be stale (pre-cycle-fix). Rebuild it (`make -j slaythespire`)
  or your RL battles can hit the Entropic Brew infinite-loop crash.
