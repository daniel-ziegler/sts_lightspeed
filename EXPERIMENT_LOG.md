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
