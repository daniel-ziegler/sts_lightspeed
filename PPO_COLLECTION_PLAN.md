# PPO collection throughput — plan (DEFERRED, 2026-05-26)

Status: **not implemented** — deferred for simplicity. The runs work fine; this is an
efficiency win, not a learning fix.

## Problem
`collect_experience` currently launches `num_games` full episodes per iteration via
`ThreadPoolExecutor` + `as_completed` and **waits for all of them** before training. Deep
(act-3) games take minutes while most finish in well under a minute, so each iteration is
gated by the straggler tail. Measured (2026-05-26):

| box | games/iter, sim | iter time | exp/hr | CPU util | GPU util |
|-----|-----------------|-----------|--------|----------|----------|
| local 3060, 16c | 64, 1000 | ~193 s | ~84 k | ~58% (load 9.4/16) | ~1% |
| lambda A100, 30c | 128, 1500 | ~191 s | ~172 k | ~40–60% (load ~11–18/30) | ~61%* |

The workload is **CPU-MCTS-bound**; the GPU is nearly idle (tiny NN). The ~40–60% CPU
utilization is the episode-length tail (most workers idle while the slowest games finish).
Bumping `num_games` only *amortizes* the tail (smaller fraction with more waves); it doesn't
remove it and it couples batch size to throughput.

## Options considered
1. **Bump num_games** — band-aid; tail persists, batch size coupled to throughput. Rejected.
2. **Async buffer** — persistent workers, harvest completed trajectories into a queue, train
   without pausing in-flight games. ~95% util, but mildly **off-policy** (in-flight games span
   weight updates). PPO tolerates it because `old_log_prob` is recorded per-decision so the
   importance ratio stays correct; staleness ~1 iteration. Viable but off-policy.
3. **CHOSEN (deferred): fixed-rollout synchronous PPO** — persistent parallel workers; to train,
   **pause all workers at a rollout boundary, train, update weights, resume.** Each rollout
   segment is collected under a single policy (on-policy) and there is no episode-completion
   tail (cut at a step budget). This is the standard PPO design; the current full-episode design
   is the unusual one.

## Plan (option 3)
Persistent worker pool plays games continuously, flushing experiences to a shared buffer. When
the buffer reaches the batch target, signal a pause; workers finish their current battle/decision
and hold; main thread trains, updates weights, resumes.

StS-specific details:
- **Atomic battles.** `agent.playout_battle` is one blocking C++ call, so you can't pause
  mid-battle. Pause = "finish current battle/decision, then hold." Residual drain is bounded by
  the longest in-flight battle (≤ the 30 s battle timeout, usually a few s) — vs the current
  full-*episode* tail of minutes.
- **Truncated rollouts → games persist across training steps.** Today a trajectory is a whole
  episode with a 0 terminal bootstrap. With pausing a game is usually mid-episode at the cut, so:
  (a) bootstrap GAE with `V(s_pause)` from the value head instead of 0; (b) carry the game's
  `GameContext` + running reward baseline (`reward_fn_vals` tail) + last value forward so it
  *continues* in the next segment. `compute_advantages` currently assumes complete episodes and
  must be reworked for truncated segments.
- The per-decision `old_log_prob`/value recording already in `run_ppo_episode` stays correct
  (PPO ratio valid even across a weight update, since logprob is captured at decision time).

Expected utilization: removes the episode tail → ~`collect/(collect+train)` ≈ **80–85%** (up from
~50%). The residual idle is the train step itself — games are frozen then to stay on-policy.
(Async/option 2 would fill that too, ~95%, at the off-policy cost we're avoiding.)

## Implementation steps (when picked up)
1. Persistent worker pool + experience buffer + pause/resume barrier (threading.Event), replacing
   the per-iteration `as_completed` collect.
2. Make the episode a *resumable stepper* (keep `run_ppo_episode`'s decision logic; let it yield
   at decision boundaries and hold on the pause event after finishing any in-flight battle).
3. Truncated-rollout GAE: per-game value bootstrap at the pause; carry per-game state across
   segments; a game's experiences are emitted in chunks across multiple training steps.
4. Validation: on a short run, confirm returns/advantages/ratios and win/floor/KL/EV match the
   current synchronous full-episode path; CPU util up and exp/hr up ~1.5×.
5. Smoke-test on a branch, then point the lambda run at it.

## Caveat
This speeds up experiments; it does **not** address the learning ceiling. The SL experiment +
per-group grad norms show the bottleneck is value/advantage-signal quality (irreducible return
variance, EV ~0.38), not data volume.
