# Battle-Outcome Prediction — pretraining / auxiliary task

**Goal.** Teach the trunk to predict *the HP outcome of a specific battle* from `(state × encounter)`.
Hypothesis: this forces deck/HP/relic→combat-strength features that the value and policy heads
currently learn only weakly. First proof point: it should raise the held-out value-SL fit (EV).
Then layer it into RL training as an aux loss and/or a pretraining warm-start.

## Locked design decisions
- **Output:** try **bucketed (primary)** and **float (comparison)**. Prior: bucketed wins.
- **Bucket denominator:** % of **max HP**.
- **Buckets (20 classes):** `DEATH` (battle lost) · `<=-50%` · ten 5% damage bins
  `(-50,-45] … (-5,0)` · `EXACT 0` (unchanged) · gains `(0,+5] (+5,+10] (+10,+15] (+15,+20]
  (+20,+35] (+35,+50] >=+50`. Fine resolution near 0 on the gain side because Burning Blood
  (Ironclad starter, +6 HP end-of-combat heal applied in `exitBattle`) makes the modal easy-win
  outcome ~+6 ≈ 6–12% of max HP, not 0. `EXACT 0` mostly captures flawless wins at full HP
  (heal wasted) or exactly-6 damage.
- **Float variant:** regress `hp_frac_delta = ΔHP / maxHP` (clamped to bucket range); death encoded
  as `-hp_before/max_hp`. MSE, mirroring the value head's raw regression.
- **Encounter conditioning:** **head-only input** — embed the encounter enum and concat to the
  pooled trunk embedding. The trunk's inputs stay byte-identical to the value/policy net, so the
  learned representation transfers cleanly and the trunk is forced to encode deck/HP strength that
  only becomes an outcome when crossed with the encounter.
- **Data:** real states with the **real encounter** they were about to face, generated FRESH from
  the **latest heart1 checkpoint**. Episode parquets are not reusable: in-battle MCTS action bits
  aren't saved, so no deterministic replay to a battle entry; and synthesizing a GameContext from
  obs alone has fidelity gaps (bottled-card bindings, event flags, anything battle-relevant the
  obs doesn't carry). Missing RNG state is NOT a blocker — every sim rerolls RNG anyway.
  Plus mutated decks (below), rerolled RNG per sim, and alternative encounters for the same state
  (sampled from the empirical (act, normal/elite) encounter distribution in the dataset).
- **Sim count for battle sims:** match the training collection MCTS sim count (outcomes match the
  deployment policy). Cheaper setting exposed as a knob.
- **Compute:** a **new spot box** for the Phase-1 generation sweep (isolated from the heart1 box
  192.9.243.58; checkpoint parquet shards off it).

## Verified APIs (the plan rests on these)
- `playout_battle(gc)` (`bindings/slaythespire.cpp:141`): `bc.init(gc)` → MCTS `playoutBattle` at
  `simulation_count_base` → `bc.exitBattle(gc)`. **Consumes `gc` in place**, inits from
  `gc.info.encounter`. Read `cur_hp`/`maxHp`/`outcome` before & after for the delta.
- **GameContext is copyable** — `GameContext() = default`, no deleted copy ctor; only member needing
  thought is `shared_ptr<Map>` (shallow-shared on copy; fine — a battle never mutates the map).
- **Encounter is NOT in the network input today.** `fixed_observation` = 10 scalars
  `[curHp, maxHp, gold, floor, boss, toSelectCount, ascension, redKey, greenKey, blueKey]`
  (`bindings/bindings-util.cpp:33`). Encounter exposed as `gc.encounter` (read-only). New input.
- **`aux_room` head is the wiring template**: head in `network.py`, target in `collate_fn`,
  loss in `algorithms.py` gated by `aux_dest_room_coef`, unpack in `rl_train.py`.
- **`value_sl.py` is the fit harness**: episode parquets, seed-level 85/15 split, trains trunk+value
  head, reports held-out **explained variance**. Re-baseline on heart1 episodes (the ~0.42 ceiling
  is stale ppo data).
- Deck mutation: `gc.obtain_card(card)` / `gc.remove_card(idx)` (`bindings/slaythespire.cpp:424`).

## Phase 0 — bindings (minimal C++) — DONE
1. `GameContext.copy()` → value copy (shallow Map share; battles never mutate the map).
2. `playout_battle(gc, encounter=None)` — optional encounter override, plumbed to the existing
   `BattleContext::init(gc, encounterToInit)` overload. Enables encounter variation on a fixed
   state (encounter is head-only, so the obs row is unchanged — near-free datapoints).
3. RNG reroll needs **no binding**: ALL battle randomness (env + MCTS rollouts) derives from
   `gc.seed + floorNum` (`BattleContext::init_empty`, `BattleSearcher` ctor), and `gc.seed` is
   already writable. Reroll = assign a fresh seed on the copy. Verified: 8/8 distinct ΔHP on a
   forced Gremlin Nob from one snapshot; same-seed runs bit-identical.

(Local build note: GCC 15 upgrade required a full `make clean` rebuild — stale LTO bytecode.)

## Phase 1 — data generation (`gen_battle_outcomes.py`)
Reuse `collect_states.collect_game`'s play loop (NN out-of-combat + MCTS in-battle), with
out-of-combat decisions driven by the **latest heart1 checkpoint** and ascension sampled 0–20
uniform (matching training). At each battle entry (`screen_state == BATTLE`):
- `snap = gc.copy()`; record base `getNNRepresentation(snap)`, `snap.encounter`, `curHp`, `maxHp`,
  `floor`, `act`.
- **Real datapoint(s):** `g = snap.copy(); g.reroll_battle_rng(s); agent.playout_battle(g)` →
  `Δ = g.cur_hp - curHp`, `outcome = g.outcome`. RNG rerolled per sim, so multiple samples per
  state are genuine resamples (train: 1–2 per (state, variant)).
- **Mutations** (each off a fresh `snap.copy()`, then re-derive `getNNRepresentation` of the mutated
  deck, then sim):
  - remove 1–3 random cards
  - remove all copies of 1–2 distinct cards
  - add 1–2 random (class/colorless) cards
  - duplicate 1–2 cards
- **Encounter variation:** for the unmutated state, also sim 1–2 alternative encounters via the
  `playout_battle(g, encounter)` override, sampled from the empirical distribution of encounters
  at the same (act, normal-vs-elite) elsewhere in the dataset. Same obs row, different label —
  teaches the head the encounter axis directly.
- Continue the *actual* game with one canonical `playout_battle(gc)` so later battles are reached.

**Output parquet** = same `obs.*` columns as episodes (so `collate_fn` ingests it unchanged) **plus**
`encounter, hp_before, max_hp, hp_delta, hp_frac_delta, bucket, outcome, mutation_type, rng_seed,
floor, act`. ~10 rows per battle (1–2 real + ~6 mutated + 2 alt-encounter), ~10–15 battles/game →
~100+ rows/game; ~2–3k games → ~200–300k datapoints (value_sl scale).

**Validation set (distribution-level):** for a held-out subset of (state, encounter) pairs (held
out by game seed), reroll **32–64 sims each** to get the empirical bucket distribution. Eval =
cross-entropy of the predicted distribution against the empirical histogram (+ calibration). Much
lower-variance than single-sample accuracy, and it directly tests the bucketed head's calibration.

Notes:
- Parallelize one seed per worker; `nice`'d, `nohup`, incremental parquet shards; **not the laptop**.

## Phase 2 — output heads (both, behind a flag)
- **Bucketed (primary):** `nn.Linear(dim, 20)` on `concat(pooled_trunk, encounter_embed)`, cross-entropy.
- **Float (comparison):** `nn.Linear(dim, 1)` on the same, MSE on `hp_frac_delta`.
- Zero-init the head (bit-identical warm-start trick) so deploying onto heart1 is a no-op until trained.

## Phase 3 — first test: does it improve the value-SL fit?
Extend `value_sl.py` → `battle_value_sl.py`. Re-baseline on **heart1 episodes** first. Three runs,
same seed-level split:
1. **Value-only** → baseline EV.
2. **Multitask:** value MSE (episode states) + `coef ×` battle-outcome loss (Phase-1 data), two data
   streams into one trunk → held-out value EV. Does it beat (1)?
3. **Pretrain → probe:** pretrain trunk on battle-outcome only, then **freeze trunk + linear-probe the
   value head** (and a finetune variant) → EV. Clean representation-transfer read.

Report **bucket vs float** independently: (a) head quality — bucket accuracy/calibration, float
MAE-in-HP — to confirm learnability and pick the winner; (b) downstream value-EV lift each produces.
**Decision gate:** if neither beats baseline EV and the probe transfer is flat, stop before RL spend.

## Phase 4 — fold into RL training (only if Phase 3 positive)
Mirror `aux_room` wiring: `use_battle_outcome_head` + `num_battle_outcome_classes` on `ModelHP`;
encounter head-input; `aux_battle_outcome_coef` loss + logging; unpack in `rl_train.py`. Feed a
**static precomputed battle dataset mixed into training batches** (no per-iter sim cost) and/or use
Phase-3 pretraining as the warm-start init.

## Commit sequence (one step per commit, per repo convention)
1. Phase 0 binding (`GameContext.copy()`) + rebuild.
2. `gen_battle_outcomes.py` + a tiny smoke run (few games) verifying schema + label sanity.
3. Bucketing/label module (shared by gen + training).
4. `battle_value_sl.py` harness + value-only re-baseline on heart1.
5. Battle-outcome head + multitask + pretrain/probe runs; record results in EXPERIMENT_LOG.md.
6. (gated) Phase-4 RL integration.
