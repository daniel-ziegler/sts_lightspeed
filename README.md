# Silver Automaton (silverbot)

A fork of [gamerpuppy/sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) — a
high-performance C++ simulator of the roguelike deckbuilder *Slay the Spire* — that adds
neural-network training for out-of-combat decisions, makes the battle MCTS use random
choice nodes instead of cheating (the upstream search replayed the game's actual RNG stream,
so it "knew" every future shuffle and enemy roll), and fixes a large number of small fidelity
and play-quality issues.

The combined system is Silver Automaton, a bot that plays full Ironclad runs: a
transformer policy network makes every out-of-combat decision (map routing, card rewards, events, shops, rest sites, Neow), and an
expectimax MCTS plays out each battle on the C++ engine. A live-game bridge drives the actual
game with the same agent through a forked CommunicationMod (included as a submodule), which
doubles as an end-to-end fidelity test of the engine.

Retains everything from upstream except for RNG cheat search: the standalone C++ engine with all enemies, all
relics, all Ironclad and colorless cards, the full overworld across all four acts, console play,
save-file loading, and ~1M random playouts in 5 seconds on 16 threads.

## Neural network for out-of-combat decisions

The network makes every non-combat choice: map pathing, card rewards (including skipping and
Singing Bowl), relic and potion pickups, event options, Neow bonuses, shop purchases and
removals, and rest-site actions — a typed action space of
`CARD` / `PATH` / `RELIC` / `POTION` / `EVENT_OPTION` / `FIXED` actions.

**Architecture** (`silverbot/network.py`, `silverbot/inputs.py`). Everything the player can
see is embedded as a set of tokens in a shared space: each deck card (identity + upgrade
count), each relic and potion, each map node (with graph features: room type,
is-current/is-reachable), and a fixed-observation vector (HP, gold, floor, act, upcoming boss,
keys, ...). The *candidate actions* are embedded as tokens in the same space — the offered
cards, the reachable path columns, the event options. A small pre-norm transformer trunk
(4 layers, dim 256, 8 heads, RMSNorm) runs full attention over observation and choice tokens
jointly, and a per-token linear head turns each choice token into a logit, giving a
distribution over exactly the legal actions — new action types need embedding definitions, not
architecture changes. Two extra heads hang off the trunk: a value head over the pooled trunk
embedding (for PPO), and an auxiliary per-path-option destination-room classifier
(a self-supervised grounding loss whose labels come free from the map).

**Training.** Two pipelines share the model and action space:

- *Supervised* (`silverbot/playouts.py` + `silverbot/train.py`): thousands of self-play
  games with Boltzmann-sampled out-of-combat choices and MCTS-played battles, written as
  parquet; the network learns to predict the final win/loss outcome from each decision point.
  This bootstrapped the early agents.
- *RL* (`silverbot/rl_train.py`): PPO with GAE(λ) and entropy regularization on annealed
  schedules. Battles are played by the MCTS, so the network learns drafting, routing, and
  resource decisions *on top of* a strong fixed combat player; credit flows through a reward
  of normalized floor progress plus key and heart-kill bonuses, with potential-based shaping
  on deck quality. Collection runs 512 games per iteration across 30 worker processes,
  pipelined with optimization.

The released `heart1.pt` checkpoint was trained this way, uniformly across ascensions 0-20 on
full 57-floor heart runs. At A0 it kills the heart in ~83% of eval games (600 matched seeds,
1000 search iterations per combat decision). At A20 — the game's hardest setting — it kills the heart
in 18.6% ± 2.4% of simulated games (n=1000, 10000 iterations per combat decision). Real games
are driven via the bridge below and have a somewhat lower winrate due to a few lingering issues.

## Battle MCTS without cheating

Upstream's tree search replayed the game's concrete RNG stream: it knew the shuffle order,
enemy move rolls, card-generation rolls. This fork
replaces it with an expectimax MCTS (`src/sim/search/BattleSearcher.cpp`) adapted to the
engine design, which doesn't provide an easy hook point for random decisions. Hundreds of card and
monster effects consume randomness at arbitrary points, so the search has to discover
stochasticity *after the fact*. Each MCTS iteration does the following:

1. **Closed-loop graph search.** Every decision node stores its full `BattleContext`; the
   search walks pointers instead of replaying actions from the root. (Upstream queued a battle
   state's pending combat effects as `std::function` lambdas, which can't be compared, hashed,
   or cheaply copied; the fork rewrites them as one big tagged union of plain-data action
   structs (`include/combat/Actions.h`), making `BattleContext` — pending effects included —
   trivially copyable and hashable.) Nodes are shared through
   a transposition table whose hash and equality deliberately ignore the RNG state — two
   states differing only in their RNG have the same distribution of futures, so they are the
   same search state. The graph stays acyclic because the key includes the turn and
   cards-played counters, which never decrease along a path.

2. **Selection.** Standard UCT descent, with one twist: the exploration term uses the *edge's*
   visit count, not the child node's total (transposition inflates the latter, which would
   starve exploration of shared states).

3. **Detecting randomness after the fact.** Expanding an edge executes the action on a copy of
   the node's state — after saving the pre-action RNG and noting its counter. If the counter
   didn't move, the action was deterministic: the result is a normal decision node,
   deduplicated through the transposition table. If it *did* move, the action consumed
   randomness, which means the state the engine just produced is only one sample from a
   distribution the search never gets to see explicitly. So the realized result is thrown
   away as "the" outcome, and the edge instead gets a **chance node** recording three things:
   the action, a pointer to the parent decision node (whose stored state is exactly the
   pre-action state), and a `randomnessBase` drawn from the saved pre-action RNG.

4. **Sampling outcomes canonically.** Outcome *N* of a chance node is generated by copying the
   parent's pre-action state, reseeding its RNG to `Random(base + N)`, and re-executing the
   action. `Random` runs its seed through murmur3 twice, so consecutive *N* give decorrelated
   streams — i.i.d. samples from the true outcome distribution. Every outcome, including the
   originally-realized one, is produced this way; there is no special case. Identical outcomes
   merge through the same transposition table, so a chance node's children form an empirical
   distribution over *distinct* results.

5. **Double progressive widening.** A chance node visited *n* times holds at most
   `ceil((n+1)^0.5)` distinct outcomes: below the cap it samples a fresh one, at the cap it
   re-descends into an existing outcome chosen proportionally to its visit count. Low-entropy
   events (a coin flip, a two-move enemy) collapse to a couple of children and converge fast;
   high-entropy events (a full reshuffle) keep deepening the subtrees they already have
   instead of scattering one visit across thousands of orderings. Because descent frequencies
   track the true probabilities, the chance node's running mean converges to the expectation.

6. **Common random numbers.** Sibling stochastic actions at the same decision node draw the
   same `randomnessBase`, so UCB compares them under matched luck rather than one action
   getting a lucky roll and the other an unlucky one.

7. **Rollout and backup.** A brand-new node is evaluated by a fast randomized heuristic-agent
   playout to the end of combat. The terminal score — win bonus, remaining HP, potions kept,
   an effective-gold delta (thief gold, Hand of Greed), small turn penalties, and partial
   credit on losses (enemy HP fraction, energy wasted) — is backed up the descent path, and
   after all iterations the root action with the most visits is played.

The draw pile plugs into this machinery rather than needing its own: `include/combat/CardPile.h`
models the pile as the player's *information set* — an unknown region kept canonically sorted
(so equal information sets transpose), plus known top/bottom stacks for Headbutt/Warcry
put-backs, Forethought, and innates. Shuffle randomness is deferred to draw time
(exchangeability keeps the induced distribution over draw sequences identical), so drawing from
the unknown region consumes RNG — and step 3 automatically turns "which card do I draw" into a
chance node, with no draw-specific logic in the searcher. Frozen Eye legitimately switches the
pile to concrete observed order. Hidden enemy intents under Runic Dome get the same treatment:
the move roll is deferred and resolves as a chance outcome.

To simplify checking whether RNG was consumed, combat runs on a single unified RNG stream (a deliberate
departure from the base game's per-purpose streams). Residual modeling approximations are documented in
`SEARCH_MODEL_INACCURACIES.md`.

## Fidelity and play-quality fixes

- **Era 1 — manual playtesting** (pre-ML; an initial batch was upstreamed, later ones are
  fork-only): dozens of mechanics fixes found by playing the simulator against the real game —
  e.g. Nunchaku, block potion amount, energy cost updates on upgrade, Panache counter not
  resetting on second play, Entrench cost + block overflow, Fire Potion targeting, Mummified
  Hand, Matryoshka, Buffer vs. `loseHp`, Golden Idol, Hypnotizing Colored Mushrooms, heart
  statuses, "transform two" always giving a curse — plus missing implementations (the Empty
  Cage, Meal Ticket, Meat on the Bone, and Orrery relics; the Smoke Bomb potion).
- **Era 2 — source audit + live bridge**: the engine validated against decompiled game source
  and against the live game itself — every decision of a live run is recreated in-engine, and
  any divergence in predicted damage, HP, block, intent, or combat outcome is treated as a
  crash and root-caused. Some representative fixes:
  - Ascension tier gates audited game-wide (`getTriIdx`): Champ Gloat strength, Collector
    block tiers, Gremlin Leader Encourage block.
  - Awakened One rebirth (no phantom victory over a half-dead stage 1); the Darkling
    two-phase revive.
  - Damage pipeline: a shared player damage-modifier pipeline, plus per-card fixes for Heavy
    Blade, Mind Blast, Rampage, Searing Blow, and Perfected Strike (whose Strike count must
    track cards entering and leaving combat, with different rules for duplicated and
    Havoc-played copies).
  - Necronomicon's exact duplication gate (attacks costing ≥2 after cost-for-turn modifiers,
    or X-cost attacks played with ≥2 energy; once per turn) and Centennial Puzzle's
    once-per-combat latch.
  - Runic Dome: enemy move rolls are deferred while intents are hidden, then resolved from
    the moves actually observed.
  - Ritual applying one turn late when gained mid-fight; Mayhem playing its card after the
    turn's draw; Time Eater / Time Warp end-of-turn sequencing.
  - Smoke Bomb's real rules (banned in boss and Surrounded fights, but legal in event combats
    like the Colosseum), stolen-gold accounting, and the act-transition heal gate (a boss
    fight entered through an event room, e.g. Mind Bloom, gives no heal).

  The remaining known engine/live gaps are cataloged in
  `silverbot/bridge/REMAINING_DIVERGENCES.md`.
- **Play-quality fixes** (search plays better, engine unchanged): effective-gold objective and
  other eval-shaping changes, each gated by a matched-seed paired winrate test (`run_paired.sh`,
  McNemar); experiments logged in `EXPERIMENT_LOG.md`.

## Playing with the trained policy

Silver Automaton's current best checkpoint (`heart1.pt`, ~23 MB — trained across all
ascensions, playing through the heart) is published on the
[releases page](https://github.com/daniel-ziegler/sts_lightspeed/releases). Download it to
`runs/` (the default path the tools look for):

```bash
mkdir -p runs
curl -L -o runs/heart1.pt \
  https://github.com/daniel-ziegler/sts_lightspeed/releases/latest/download/heart1.pt
```

Three ways to play it (after building — see below):

```bash
# Watch it play a full run in the console, one decision at a time
python3 -m silverbot.watch_game --model-path runs/heart1.pt \
    --seed 42 --ascension 0 --temperature 0

# Batch evaluation: win rate over N seeded games
python3 -m silverbot.eval_hero --ckpt runs/heart1.pt \
    --n-games 100 --mcts-sims 1000 --temperature 0

# Drive the real game (see the live-game bridge section below)
python3 comm.py --games 10
```

## Live-game bridge

- `silverbot/bridge/` + repo-root `comm.py`: plays the real game through a
  [forked CommunicationMod](https://github.com/daniel-ziegler/CommunicationMod) (the
  `CommunicationMod/` submodule). The fork adds state the bridge needs — per-turn play counts,
  per-card computed base damage, relic `grayscale`/`activated`, draw-pile order — plus
  robustness fixes for a zero-delay controller and a "watch mode" that previews the bot's
  pending choice in-game (see its `CHANGES.md`).
- Persistent `BattleContext`: one engine-advanced battle state carried across a whole combat,
  reconciled against the live game every decision; hidden state (RNG rolls, observed enemy
  moves) transplanted from live observations. Divergence taxonomy in
  `silverbot/bridge/REMAINING_DIVERGENCES.md`.

### Building the mod

Requires JDK 8 and Maven (`apt install openjdk-8-jdk maven`), plus three jars from a Slay the
Spire install placed in `../lib/` relative to the submodule: `desktop-1.0.jar` (game install
dir), `ModTheSpire.jar`, and `BaseMod.jar` (Steam workshop content).

```bash
git submodule update --init CommunicationMod
cd CommunicationMod && mvn -B package   # -> target/CommunicationMod.jar
```

Install by replacing the `CommunicationMod.jar` that ModTheSpire loads (e.g. the Steam workshop
copy). Never swap the jar while the game is running — ModTheSpire lazy-loads patch classes from
disk and will crash on the next class load.

### Running the agent against the real game

Point the mod's `config.properties` at the agent:

```
command=python3 /path/to/sts_lightspeed/comm.py --games 10
```

then launch the modded game; the agent takes over from the main menu. `run_live.sh` automates
the full cycle (kills stale processes, rewrites the mod config with capture name / seed /
ascension / sims as env vars, launches the game). See `COMM_README.md` for details.

## Building and running

C++20 with CMake; nlohmann/json and pybind11 are vendored as git submodules. The Python side
needs 3.10 with `torch`, `pyarrow`, and `tqdm`.

```bash
git submodule update --init json pybind11
cmake . && make -j8
```

This builds the `slaythespire` Python extension module, the `main` interactive console
simulator, and the `test` benchmark/tool suite. All Python runs from the repo root as
`python3 -m silverbot.<module>`, which keeps `slaythespire` and the vendored `spirecomm/`
importable.

```bash
# Console play: enter "<seed> <character> <ascension>", e.g. "12345 I 0"
./main

# Random-playout benchmark
./test simple_agent_mt <threads> <seed> <playouts> <print>

# Battle MCTS from a real save file
./test mcts_save <save_file> <iterations>
```

To train a network from scratch: generate supervised self-play data with
`python3 -m silverbot.playouts` and train on it with `python3 -m silverbot.train`, then use
that checkpoint to initialize PPO in `python3 -m silverbot.rl_train` (see
`run_heart1_supervised.sh` for the full deployed hyperparameters). To play the released
checkpoint or drive the real game, see the sections above.
