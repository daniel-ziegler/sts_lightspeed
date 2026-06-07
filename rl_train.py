#!/usr/bin/env python3

from __future__ import annotations

import random
import argparse
import logging
from dataclasses import dataclass, fields, replace
from typing import List, NamedTuple, Optional, Tuple, get_type_hints, Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import threading
from collections import Counter
import json
import os
import gc

# Run TorchInductor compilation in-process (no async compile-worker subprocess pool). That pool's
# pipe-reader thread uses pickle and races with torch.save() at checkpoint time, intermittently
# crashing with "cannot pickle 'torch._C.PyTorchFileWriter'". Must be set before importing torch.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from network import NN, ModelHP, move_to_device, process_batch, choice_space, collate_fn, load_network_backward_compatible, SeparateValuePolicy, EventFixedInfo, CHOICE_PATHS_OFFSET
from playouts import run_game, NNService, Choice, Decision, ActionType, ChoiceStats, path_to_action_and_desc, construct_choice, flatten_dict
from algorithms import (
    policy_log_probs, importance_ratio, approx_kl, clip_fraction, clipped_surrogate, masked_entropy,
    PPOAlgorithm,
)
import slaythespire as sts

# Set up logging
log = logging.getLogger(__name__)

# Map a choice_space path category to its ActionType (for offline SL labels).
_PATH_TO_ACTIONTYPE = {
    'cards': ActionType.CARD,
    'relics': ActionType.RELIC,
    'potions': ActionType.POTION,
    'paths': ActionType.PATH,
    'fixed': ActionType.FIXED,
}


class Stats:
    """Helper class for accumulating statistics across batches."""
    
    def __init__(self):
        self.sum = 0.0
        self.sum_squared = 0.0
        self.count = 0
    
    def add_samples(self, values: torch.Tensor):
        """Add a batch of values to the running statistics."""
        self.sum += torch.sum(values).item()
        self.sum_squared += torch.sum(values ** 2).item()
        self.count += values.numel()
    
    def mean(self) -> float:
        """Calculate the mean of all accumulated samples."""
        return self.sum / self.count if self.count > 0 else 0.0
    
    def var(self) -> float:
        """Calculate the variance of all accumulated samples."""
        if self.count <= 1:
            return 0.0
        mean_val = self.mean()
        return (self.sum_squared / self.count) - (mean_val ** 2)


class RunningMoments:
    """EWMA estimate of the mean and std of a stream of values.

    Advantage normalization divides by the std; with a small or heavy-tailed batch that
    per-batch std is noisy and the normalization scale jumps between iterations. Tracking
    the mean and E[x^2] with an exponential moving average keeps the scale stable. The
    decay is applied per item (a batch of N items retains (1-decay)^N of the prior
    estimate), so the amount of smoothing is invariant to batch size. decay=1.0 recovers
    pure per-batch normalization.
    """

    def __init__(self, decay: float, eps: float = 1e-8):
        assert 0.0 < decay <= 1.0, f"adv_norm_decay must be in (0, 1], got {decay}"
        self.decay = decay
        self.eps = eps
        self.mean = 0.0
        self.second_moment = 0.0  # EWMA of E[x^2]
        self.initialized = False

    def update(self, values: np.ndarray):
        batch_mean = float(values.mean())
        batch_second = float(np.mean(values ** 2))
        if not self.initialized:
            self.mean = batch_mean
            self.second_moment = batch_second
            self.initialized = True
            return
        # Per-item decay, so smoothing is invariant to batch size: a batch of N items
        # retains (1-decay)^N of the previous estimate. Large batches mostly trust
        # themselves; small/noisy batches lean on accumulated history.
        retain = (1.0 - self.decay) ** values.size
        self.mean = retain * self.mean + (1.0 - retain) * batch_mean
        self.second_moment = retain * self.second_moment + (1.0 - retain) * batch_second

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.second_moment - self.mean ** 2, self.eps)))


@dataclass
class TrainConfig:
    """Training hyperparameters (shared across algorithms)."""
    # Algorithm selection
    algo: str = "ppo"               # {ppo}
    sampling_temperature: float = 1.0  # action-sampling softmax temperature during collection

    # Environment settings
    # Games are dealt ascensions 0..max_ascension uniformly (derived from the game seed, so
    # the same seed always plays the same level and resumes/evals are reproducible).
    # fixed_ascension pins every game to one level instead (per-level evals).
    max_ascension: int = 0
    fixed_ascension: Optional[int] = None
    num_games_per_step: int = 256
    num_epochs: int = 4
    num_workers: int = 40
    inf_batch_size: int = 32
    inf_batch_size_factor: int = 16
    batch_size: int = 128
    
    # PPO hyperparameters
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    # Exponential entropy-coef decay: starting at iteration entropy_coef_decay_start, the
    # effective coefficient decays geometrically from entropy_coef to entropy_coef_final over
    # entropy_coef_decay_steps iterations, then holds at entropy_coef_final. decay_steps=0
    # disables the schedule. decay_start is an ABSOLUTE iteration so a later
    # --resume-from-step continues the same schedule rather than restarting it.
    entropy_coef_final: float = 0.0
    entropy_coef_decay_steps: int = 0
    entropy_coef_decay_start: int = 0
    # Exponential learning-rate decay (same shape as the entropy schedule): from iteration
    # lr_decay_start, both policy and value lr decay geometrically to lr_final_frac of their
    # base values over lr_decay_steps iterations, then hold. decay_steps=0 disables.
    # Anchored at ABSOLUTE iterations so resumes continue the schedule.
    lr_final_frac: float = 0.1
    lr_decay_steps: int = 0
    lr_decay_start: int = 0
    # Weight of the self-supervised destination-room auxiliary loss (per-path-option room
    # classification; labels free from the map obs). Scaffolds the option-grounding circuit
    # at RL signal strength (see EXPERIMENT_LOG.md repr lab). 0 disables.
    aux_dest_room_coef: float = 0.1
    value_coef: float = 0.5
    max_grad_norm: float = 5.0
    
    # Learning rates
    policy_lr: float = 5e-5
    value_lr: float = 1e-4
    weight_decay: float = 1e-4  # L2 regularization strength
    
    # GAE parameters
    gamma: float = 1.00
    gae_lambda: float = 0.97
    adv_norm_decay: float = 5e-4  # per-item EWMA decay for advantage mean/std (batch retains (1-decay)^N; 1.0 = current batch only)

    # MCTS battle search (per-episode agent knobs). exploration/widening default to None =
    # inherit the engine's SearchAgent defaults, which are a JOINTLY-tuned set (exploration,
    # general + boss-gated widening, eval weights). Overriding part of that set yields a
    # mixed-era config that was never validated and measurably hurts win rate -- only set
    # these explicitly together with matching eval weights.
    mcts_simulations: int = 1000
    mcts_exploration: Optional[float] = None
    mcts_widening_c: Optional[float] = None
    mcts_widening_alpha: Optional[float] = None
    mcts_boss_widening_c: Optional[float] = None      # None = engine's boss-gated default
    mcts_boss_widening_alpha: Optional[float] = None
    log_battle_outcomes: bool = False                 # attach per-battle snapshots to trajectories
    randomize_path_choices: bool = False              # intervention eval: uniform-random path picks
    record_boss_states: bool = False                  # attach replayable pre-boss-battle action prefixes
    # Battle-search eval weights (None = engine's jointly-tuned defaults). Like the search knobs,
    # these are a coupled set -- override all of them together or none.
    mcts_win_bonus: Optional[float] = None
    mcts_potion_weight: Optional[float] = None
    mcts_victory_turn_penalty: Optional[float] = None
    mcts_monster_damage_weight: Optional[float] = None
    mcts_alive_weight: Optional[float] = None
    mcts_energy_waste_weight: Optional[float] = None
    mcts_draw_weight: Optional[float] = None
    mcts_turn_survival_weight: Optional[float] = None

    # Reward shaping (potential-based). We extend the existing telescoping potential
    # Phi(s) with shape(s) = shaping_hp_coef*(cur_hp/max_hp) + shaping_upg_coef*num_upgraded.
    # shape is added to NON-terminal potentials only and un-credited (set to 0) at the terminal,
    # so with gamma=1 the undiscounted return telescopes to base_return - (shape(s0) - offset),
    # a per-game constant (s0 is always full-HP/0-upgrades) -> the optimal policy is unchanged.
    # shaping_offset subtracts a constant from the non-terminal potential; it cancels in every
    # per-step delta (policy-neutral) and only shrinks the single terminal clawback (= offset -
    # shape(s_last)). Set it to ~E[shape at last decision] to center that clawback near zero.
    shaping_hp_coef: float = 0.0
    shaping_upg_coef: float = 0.0
    shaping_relic_coef: float = 0.0   # per relic held
    shaping_maxhp_coef: float = 0.0   # per max-HP point
    shaping_key_coef: float = 0.0     # per act-4 key held (heart runs)
    shaping_offset: float = 0.0

    # Per-battle MCTS wall-clock budget (seconds). Sized for the default 1000 sims; raise it
    # when running with many more simulations so long boss fights aren't cut off mid-search.
    battle_timeout: float = 30.0

    # Training settings
    num_iterations: int = 1000
    separate_networks: bool = False  # Use separate policy and value networks
    resume_from_step: int = 0  # Step to resume from (0 = start from beginning)
    seed: Optional[int] = None  # Global RNG seed (torch/numpy/random). None = unseeded.
    
    # Logging
    log_every: int = 10
    save_every: int = 20
    
    # Memory profiling
    memory_profile_iterations: int = 0  # Number of iterations to profile (0 = disabled)



@dataclass
class GameMetrics:
    """Metrics extracted from game state for reward computation."""
    floor_num: int
    cur_hp: int
    max_hp: int
    perfected_strike_count: int
    num_upgraded: int  # count of upgraded cards in deck (for reward shaping)
    num_relics: int    # count of relics held (for reward shaping)
    outcome: sts.GameOutcome
    act: int = 1       # current act; distinguishes a heart win (act 4) from an act-3-only win
    num_keys: int = 0  # act-4 keys held (ruby + emerald + sapphire), for reward shaping
    # MonsterEncounter id of the most recent battle (INVALID=0 before the first one). In episode
    # dumps this enables per-encounter battle-outcome stats: the HP change across the rows where
    # the id flips measures what each fight cost.
    encounter: int = 0

class Experience(NamedTuple):
    """Single step of experience from a game."""
    choice: Choice
    action_idx: int  # Needed for logprobs calculation in PPO training
    log_prob: float
    metrics: GameMetrics
    action_str: str  # Store clean action description for debugging
    choice_type: int  # ActionType value of the chosen action (offline SL label)


class Trajectory(NamedTuple):
    """Complete game trajectory."""
    seed: int
    experiences: List[Experience]
    rewards: List[float]  # Reward for each step
    values: List[float]   # Value prediction for each step
    final_reward: float
    final_metrics: GameMetrics  # Complete final game state metrics
    final_deck: List[sts.Card]  # Final deck state
    final_relics: List[sts.RelicId]  # Final relics
    battle_log: list = []  # per-battle BattleSnapshots (when config.log_battle_outcomes)
    boss_state_records: list = []  # (floor, prefix_bits) per boss battle (when config.record_boss_states)
    ascension: int = 0


def compute_progress_reward(metrics: GameMetrics) -> float:
    """Compute reward based on floor progress and game outcome."""
    floor_reward = min(0.5, metrics.floor_num / 100.0)
    victory_bonus = 0.5 if metrics.outcome == sts.GameOutcome.PLAYER_VICTORY else 0.0
    return floor_reward + victory_bonus


def compute_perfected_strike_reward(metrics: GameMetrics) -> float:
    """Compute reward based on number of Perfected Strikes in deck."""
    return float(metrics.perfected_strike_count)


def compute_victory_reward(metrics: GameMetrics) -> float:
    """Compute sparse victory-only reward (1.0 for victory, 0.0 otherwise)."""
    return 1.0 if metrics.outcome == sts.GameOutcome.PLAYER_VICTORY else 0.0


def compute_heart_reward(metrics: GameMetrics) -> float:
    """Heart-run reward: floor progress (uncapped to floor 57 -> 0.5) plus a split victory
    bonus -- +0.25 for an act-3-only win (no keys), +0.5 for killing the Heart (act 4).
    Only a heart kill reaches a total of ~1.0."""
    floor_reward = metrics.floor_num / 114.0
    if metrics.outcome == sts.GameOutcome.PLAYER_VICTORY:
        victory_bonus = 0.5 if metrics.act >= 4 else 0.25
    else:
        victory_bonus = 0.0
    return floor_reward + victory_bonus


def compute_no_pstrikes_reward(metrics: GameMetrics) -> float:
    """Compute reward that penalizes Perfected Strikes (negative of count)."""
    return -float(metrics.perfected_strike_count)


def compute_shaped_rewards(
    step_metrics: List[GameMetrics],
    final_metrics: GameMetrics,
    reward_fn,
    hp_coef: float,
    upg_coef: float,
    offset: float,
    relic_coef: float = 0.0,
    maxhp_coef: float = 0.0,
    key_coef: float = 0.0,
) -> Tuple[List[float], float]:
    """Per-step rewards as deltas of a potential Phi(s) = base(s) + shape(s).

    base(s)  = reward_fn(s) (floor progress + victory), defined for every state incl. terminal.
    shape(s) = hp_coef*(cur_hp/max_hp) + upg_coef*num_upgraded + relic_coef*num_relics
               + maxhp_coef*max_hp - offset, added to NON-terminal states only; the terminal
               shaping is un-credited (set to 0).

    With gamma=1 the per-step deltas telescope to
        sum(rewards) = base(terminal) - base(s0) - (shape_raw(s0) - offset),
    i.e. shaping changes the return only by a per-game constant (s0 is fixed: full-HP, 0 upgrades,
    1 relic, starting max_hp), so the optimal policy is unchanged. The offset cancels in every
    interior delta and only shrinks the single terminal clawback (= base_delta_T - shape_raw(s_last) + offset).

    Returns (rewards, terminal_base_value).
    """
    base_vals = [reward_fn(m) for m in step_metrics] + [reward_fn(final_metrics)]

    def _shape(m: GameMetrics) -> float:
        hp_frac = (m.cur_hp / m.max_hp) if m.max_hp > 0 else 0.0
        return (hp_coef * hp_frac + upg_coef * m.num_upgraded
                + relic_coef * m.num_relics + maxhp_coef * m.max_hp
                + key_coef * m.num_keys - offset)

    shape_vals = [_shape(m) for m in step_metrics] + [0.0]  # terminal shaping un-credited
    total_vals = [b + s for b, s in zip(base_vals, shape_vals)]
    rewards = [total_vals[i + 1] - total_vals[i] for i in range(len(step_metrics))]
    return rewards, base_vals[-1]


def run_episode(seed: int, service: NNService, reward_fn, battle_executor, config: TrainConfig,
                sample_seed: Optional[int] = None) -> Trajectory:
    """Run a complete game episode and collect experience.

    `seed` is the game/map seed (-> C++ GameContext, fixes the map/card RNG). `sample_seed`
    seeds the action-sampling RNG and defaults to `seed`; decoupling them lets replicates
    draw divergent trajectories from the SAME map."""
    if sample_seed is None:
        sample_seed = seed
    # PPO_LOG_SEEDS=1 prints per-episode seed bookends to isolate which seed crashes
    # the C++ engine (stack-smashing aborts the process; the last >>> line names the culprit).
    _log_seeds = os.environ.get('PPO_LOG_SEEDS') == '1'
    # PPO_LOG_STEPS=1 prints per-game-step info before each C++ call so the last log line
    # before a fatal crash names the screen/action that triggered it.
    _log_steps = os.environ.get('PPO_LOG_STEPS') == '1'
    if _log_seeds:
        print(f"  >>> start seed {seed}", flush=True)
    ascension = (config.fixed_ascension if config.fixed_ascension is not None
                 else seed % (config.max_ascension + 1))
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
    rng = random.Random(sample_seed)
    
    agent = sts.Agent()
    agent.simulation_count_base = config.mcts_simulations
    agent.verbosity_level = 0  # silence per-action battle prints (keep rl_train's own stdout)
    # None = leave the engine's jointly-tuned search defaults in place (see TrainConfig).
    if config.mcts_exploration is not None:
        agent.exploration_parameter = config.mcts_exploration
    if config.mcts_widening_c is not None:
        agent.chance_widening_c = config.mcts_widening_c
    if config.mcts_widening_alpha is not None:
        agent.chance_widening_alpha = config.mcts_widening_alpha
    if config.mcts_boss_widening_c is not None:
        agent.boss_chance_widening_c = config.mcts_boss_widening_c
    if config.mcts_boss_widening_alpha is not None:
        agent.boss_chance_widening_alpha = config.mcts_boss_widening_alpha
    if config.log_battle_outcomes:
        agent.log_battle_outcomes = True
    # Pre-boss-state recording: the mixed action stream (out-of-combat GameAction bits +
    # in-battle search::Action bits) up to each boss battle, replayable by eval_states'
    # loadPreBattleState. _rec is None when recording is off.
    _rec = [] if config.record_boss_states else None
    _boss_records = []
    if config.record_boss_states:
        agent.record_actions = True
    _ew_overrides = [
        ('win_bonus', config.mcts_win_bonus),
        ('potion_weight', config.mcts_potion_weight),
        ('victory_turn_penalty', config.mcts_victory_turn_penalty),
        ('monster_damage_weight', config.mcts_monster_damage_weight),
        ('alive_weight', config.mcts_alive_weight),
        ('energy_waste_weight', config.mcts_energy_waste_weight),
        ('draw_weight', config.mcts_draw_weight),
        ('turn_survival_weight', config.mcts_turn_survival_weight),
    ]
    if any(v is not None for _, v in _ew_overrides):
        ew = agent.eval_weights
        for name, v in _ew_overrides:
            if v is not None:
                setattr(ew, name, v)
        agent.eval_weights = ew
    experiences = []
    values = []  # Collect values separately

    while gc.outcome == sts.GameOutcome.UNDECIDED:
        try:
            if _log_steps:
                print(f"  [seed {seed}] floor={gc.floor_num} hp={gc.cur_hp}/{gc.max_hp} screen={gc.screen_state}", flush=True)
            if gc.screen_state == sts.ScreenState.BATTLE:
                if _log_steps:
                    print(f"  [seed {seed}] BATTLE playout start floor={gc.floor_num}", flush=True)
                if _rec is not None:
                    _pre_bits = len(agent.game_action_history)
                    if gc.cur_room == sts.Room.BOSS:
                        _boss_records.append((gc.floor_num, list(_rec)))
                # Use MCTS agent for battles in background thread
                future = battle_executor.submit(agent.playout_battle, gc)
                
                try:
                    # Wait for completion with timeout. future.result raises
                    # concurrent.futures.TimeoutError, which is a distinct class from the
                    # builtin TimeoutError before Python 3.11 -- catch the right one.
                    future.result(timeout=config.battle_timeout)
                    if _rec is not None:
                        _rec.extend(int(x) for x in agent.game_action_history[_pre_bits:])
                except FuturesTimeoutError:
                    log.warning(f"Battle simulation timed out for seed {seed}. Background thread will continue running.")
                    # End the episode. The outcome will be UNDECIDED.
                    break
                    
            else:
                # Use neural network for non-battle decisions
                obs = sts.getNNRepresentation(gc)
                actions = sts.GameAction.getAllActionsInState(gc)
                
                choice = construct_choice(gc, obs, actions)
                
                if choice is not None:
                    
                    # Count total number of choices available
                    total_choices = (len(choice.cards_offered) + len(choice.relics_offered) + 
                                   len(choice.potions_offered) + len(choice.fixed_actions) + len(choice.paths_offered))
                    
                    if total_choices > 1:
                        # Get network predictions
                        batch_tensors, output = service.get_logits(choice)
                        
                        # Handle value head output
                        if isinstance(output, tuple):
                            logits, value_output = output
                            value = float(value_output) if np.isscalar(value_output) else float(value_output[0])
                        else:
                            logits = output
                            # No separate value service needed with combined wrapper
                            value = 0.0
                        
                        # Convert to probabilities and sample action. sampling_temperature>1
                        # widens the distribution (more diverse trajectories); the stored
                        # log_prob is of the tempered behavior policy actually sampled from.
                        # Default 1.0 is a no-op.
                        logits_tensor = torch.tensor(logits)
                        if config.sampling_temperature != 1.0:
                            logits_tensor = logits_tensor / config.sampling_temperature
                        log_probs = F.log_softmax(logits_tensor, dim=0).numpy()
                        probs = np.exp(log_probs)

                        # Intervention eval: replace the policy with uniform-random on pure
                        # path-choice screens (everything else stays policy-driven), to price
                        # the routing policy's win-rate contribution.
                        if (config.randomize_path_choices and choice.paths_offered
                                and not (choice.cards_offered or choice.relics_offered
                                         or choice.potions_offered or choice.fixed_actions)):
                            chosen_idx = CHOICE_PATHS_OFFSET + rng.randrange(len(choice.paths_offered))
                        else:
                            chosen_idx = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])
                        log_prob = log_probs[chosen_idx]
                        
                        # No perfected strike tracking needed
                        
                        # Convert back to game action
                        path = choice_space.ix_to_path(batch_tensors['choices'], chosen_idx)
                        choice_type = _PATH_TO_ACTIONTYPE[path[0]]

                        # Generate clean action description based on path
                        action, action_desc = path_to_action_and_desc(choice, path, gc)
                        
                        # Extract metrics from game state BEFORE action execution
                        perfected_strike_count = sum(1 for card in gc.deck if card.id == sts.CardId.PERFECTED_STRIKE)
                        metrics = GameMetrics(
                            floor_num=gc.floor_num,
                            cur_hp=gc.cur_hp,
                            max_hp=gc.max_hp,
                            perfected_strike_count=perfected_strike_count,
                            num_upgraded=sum(1 for card in gc.deck if card.upgraded),
                            num_relics=len(gc.relics),
                            outcome=gc.outcome,
                            act=gc.act,
                            num_keys=int(gc.red_key) + int(gc.green_key) + int(gc.blue_key),
                            encounter=int(gc.encounter),
                        )

                        # Store experience data before action execution
                        exp_data = {
                            'choice': choice,
                            'action_idx': chosen_idx,
                            'log_prob': log_prob,
                            'value': value,
                            'action_str': action_desc,
                            'choice_type': int(choice_type),
                        }

                        assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                        if _log_steps:
                            print(f"  [seed {seed}] EXEC (model) floor={gc.floor_num} action={action_desc}", flush=True)
                        action.execute(gc)
                        if _rec is not None:
                            _rec.append(int(action.bits))
                        
                        exp = Experience(
                            choice=exp_data['choice'],
                            action_idx=exp_data['action_idx'],
                            log_prob=exp_data['log_prob'],
                            metrics=metrics,
                            action_str=exp_data['action_str'],
                            choice_type=exp_data['choice_type'],
                        )
                        experiences.append(exp)
                        values.append(exp_data['value'])  # Store value separately
                    else:
                        if _log_steps:
                            print(f"  [seed {seed}] PICK (single-choice) floor={gc.floor_num}", flush=True)
                        action = agent.pick_gameaction(gc)
                        assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                        if _log_steps:
                            print(f"  [seed {seed}] EXEC (single) floor={gc.floor_num} action={action.getDesc(gc)}", flush=True)
                        action.execute(gc)
                        if _rec is not None:
                            _rec.append(int(action.bits))
                else:
                    if _log_steps:
                        print(f"  [seed {seed}] PICK (no-choice) floor={gc.floor_num}", flush=True)
                    action = agent.pick_gameaction(gc)
                    assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                    if _log_steps:
                        print(f"  [seed {seed}] EXEC (no-choice) floor={gc.floor_num} action={action.getDesc(gc)}", flush=True)
                    action.execute(gc)
                    if _rec is not None:
                        _rec.append(int(action.bits))
                
        except Exception as e:
            log.error(f"Error in episode {seed}: {e}")
            raise
    
    # Create final metrics for reward computation
    final_metrics = GameMetrics(
        floor_num=gc.floor_num,
        cur_hp=gc.cur_hp,
        max_hp=gc.max_hp,
        perfected_strike_count=sum(1 for card in gc.deck if card.id == sts.CardId.PERFECTED_STRIKE),
        num_upgraded=sum(1 for card in gc.deck if card.upgraded),
        num_relics=len(gc.relics),
        outcome=gc.outcome,
        act=gc.act,
        num_keys=int(gc.red_key) + int(gc.green_key) + int(gc.blue_key),
        encounter=int(gc.encounter),
    )
    
    # Per-step rewards = deltas of the (base + potential-based shaping) potential.
    rewards, final_base_reward = compute_shaped_rewards(
        [e.metrics for e in experiences], final_metrics, reward_fn,
        config.shaping_hp_coef, config.shaping_upg_coef, config.shaping_offset,
        relic_coef=config.shaping_relic_coef, maxhp_coef=config.shaping_maxhp_coef,
        key_coef=config.shaping_key_coef,
    )
    
    # Add terminal state value (0.0) for GAE bootstrap
    values.append(0.0)
    # Values were collected during the episode

    if _log_seeds:
        print(f"  <<< done  seed {seed} floor={gc.floor_num} outcome={gc.outcome}", flush=True)
    return Trajectory(
        seed=seed,
        experiences=experiences,
        rewards=rewards,
        values=values,
        final_reward=final_base_reward,
        final_metrics=final_metrics,
        final_deck=list(gc.deck),
        final_relics=list(gc.relics),
        battle_log=list(agent.battle_log) if config.log_battle_outcomes else [],
        boss_state_records=_boss_records,
        ascension=ascension,
    )


def collect_experience(config: TrainConfig, service: NNService, reward_fn, jobs) -> List[Trajectory]:
    """Collect experience for a list of CollectionJob (game_seed, sample_seed)."""
    trajectories = []

    def _run(job):
        return run_episode(job.game_seed, service, reward_fn, battle_executor, config,
                           sample_seed=job.sample_seed)

    if config.num_workers == 1:
        # Single-threaded execution for easier debugging (deterministic completion order)
        with ThreadPoolExecutor(max_workers=1) as battle_executor:
            for job in tqdm(jobs, desc="Collecting experience"):
                trajectories.append(_run(job))
    else:
        with ThreadPoolExecutor(max_workers=config.num_workers) as battle_executor:
            with ThreadPoolExecutor(max_workers=config.num_workers) as main_executor:
                futures = [main_executor.submit(_run, job) for job in jobs]
                for future in tqdm(as_completed(futures), total=len(jobs), desc="Collecting experience"):
                    trajectories.append(future.result())

    return trajectories


def print_trajectory(traj: Trajectory, advantages, values=None, returns=None, title: str = ""):
    """Print a human-readable playthrough of one trajectory: a per-step state/choice/action table
    followed by the final outcome, deck, and relics. The Pred Value / Return columns are included
    only when `values`/`returns` are given (PPO); critic-free algos pass just the advantages."""
    print(title or f"=== Trajectory (seed {traj.seed}, ascension {traj.ascension}) ===")
    print(f"Trajectory length: {len(traj.experiences)} steps (ascension {traj.ascension})")
    has_v = values is not None and returns is not None
    header = f"Step | {'State':12s} | {'Choice':20s} | {'Action':20s} | {'Prob':6s} | {'Reward':6s}"
    if has_v:
        header += f" | {'Pred Value':10s} | {'GAE Return':10s}"
    header += f" | {'Advantage':13s}"
    print(header)
    print("-" * 140)

    for t in range(len(traj.experiences)):
        exp = traj.experiences[t]
        # Choice summary - what was offered
        offered_items = []
        if exp.choice.cards_offered:
            offered_items.append(f"{len(exp.choice.cards_offered)}card")
        if exp.choice.relics_offered:
            offered_items.append(f"{len(exp.choice.relics_offered)}rel")
        if exp.choice.potions_offered:
            offered_items.append(f"{len(exp.choice.potions_offered)}pot")
        if exp.choice.fixed_actions:
            offered_items.append(f"{len(exp.choice.fixed_actions)}fix")
        if exp.choice.paths_offered:
            offered_items.append(f"{len(exp.choice.paths_offered)}path")
        choice_desc = '+'.join(offered_items) if offered_items else "none"
        action_desc = exp.action_str[:20] if exp.action_str else "Unknown"
        state_str = f"{exp.metrics.floor_num:>2}: {exp.metrics.cur_hp}/{exp.metrics.max_hp}hp"

        row = (f"{t:4d} | {state_str:12s} | {choice_desc[:20]:20s} | {action_desc[:20]:20s} | "
               f"{np.exp(exp.log_prob):6.3f} | {traj.rewards[t]:6.3f}")
        if has_v:
            row += f" | {values[t]:10.3f} | {returns[t]:10.3f}"
        row += f" | {advantages[t]:13.3f}"
        print(row)

    print("-" * 140)
    fm = traj.final_metrics
    print(f"Final game outcome: {fm.outcome}")
    print(f"Final reward: {traj.final_reward:.3f}, Final state: {fm.floor_num}: {fm.cur_hp}/{fm.max_hp}hp")
    print(f"Final deck ({len(traj.final_deck)} cards):")
    deck_summary = Counter(str(card) for card in traj.final_deck)
    for card_str, count in deck_summary.most_common():
        if count > 1:
            print(f"  {count}x {card_str}")
        else:
            print(f"  {card_str}")
    print(f"Final relics ({len(traj.final_relics)}):")
    for relic in traj.final_relics:
        print(f"  {relic.id}")
    print("=" * 80)


def compute_advantages(trajectories: List[Trajectory], config: TrainConfig, adv_norm: RunningMoments, debug_traj: bool = False) -> tuple[List[Experience], List[float], List[float], List[dict]]:
    """Compute advantages using GAE and prepare training data."""
    all_experiences = []
    all_advantages = []
    all_returns = []
    all_meta = []  # per-experience {seed, outcome, final_floor, reward, value} for episode dumps
    
    # debug_traj_idx = max(range(len(trajectories)), key=lambda i: trajectories[i].final_reward) if debug_traj and trajectories else None
    debug_traj_idx = random.randint(0, len(trajectories) - 1) if debug_traj and trajectories else None
    
    for traj_idx, traj in enumerate(trajectories):
        if not traj.experiences:
            continue
            
        # Use the rewards computed at each step (dense or sparse depending on reward function)
        # len(rewards) = len(experiences), len(values) = len(experiences) + 1 (includes terminal bootstrap)
        values = np.array(traj.values)  # Includes terminal bootstrap value (0.0)
        rewards = np.array(traj.rewards)  # Terminal reward added to last step
        
        # Compute returns and advantages using GAE
        num_steps = len(traj.experiences)
        advantages = np.zeros(num_steps)
        returns = np.zeros(num_steps)
        
        gae = 0
        for t in reversed(range(num_steps)):
            # rewards[t] is reward for step t, values[t+1] is value for next state
            delta = rewards[t] + config.gamma * values[t + 1] - values[t]
            gae = delta + config.gamma * config.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Debug output for random trajectory
        if debug_traj and traj_idx == debug_traj_idx:
            print(f"Rewards array length: {len(traj.rewards)}, first 5 rewards: {traj.rewards[:5]}")
            print(f"Values array length: {len(traj.values)}, first 5 values: {traj.values[:5]}")
            print_trajectory(traj, advantages, values=values, returns=returns,
                             title=f"=== PPO Advantage Calculation Debug (seed {traj.seed}, "
                                   f"ascension {traj.ascension}) ===")
        
        # Store experiences and raw (un-normalized) GAE advantages/returns.
        all_experiences.extend(traj.experiences)
        all_advantages.extend(advantages.tolist())
        all_returns.extend(returns.tolist())
        outcome_label = 1 if traj.final_metrics.outcome == sts.GameOutcome.PLAYER_VICTORY else 0
        all_meta.extend([
            {'seed': traj.seed, 'outcome': outcome_label,
             'final_floor': traj.final_metrics.floor_num,
             'reward': float(rewards[t]), 'value': float(values[t])}
            for t in range(num_steps)
        ])

    # Normalize advantages across the whole batch (never per-trajectory -- that would erase
    # the relative scale between good and bad games, which is most of the signal under sparse
    # win/floor rewards). The mean/std come from an EWMA across iterations so the scale stays
    # stable when a batch is small or heavy-tailed. Returns are left raw -- they are the
    # value-function targets.
    advantages_arr = np.asarray(all_advantages, dtype=np.float64)
    if advantages_arr.size > 0:
        adv_norm.update(advantages_arr)
        advantages_arr = (advantages_arr - adv_norm.mean) / adv_norm.std
    all_advantages = advantages_arr.tolist()

    return all_experiences, all_advantages, all_returns, all_meta


def _serialize_choice(choice: Choice) -> dict:
    """Flatten a Choice to parquet-safe columns matching collate_fn's expectations:
    numpy arrays (incl. 2-D obs.map.pathXs) become (nested) lists, and fixed_actions
    becomes a list of uniform int-valued structs so pyarrow can store it."""
    flat = {}
    for k, v in flatten_dict(choice.as_dict()).items():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            # 2-D (obs.map.pathXs): pyarrow rejects 2-D ndarray cells, store nested lists.
            flat[k] = v.tolist()
        elif k == 'fixed_actions':
            flat[k] = [
                {'action': int(d['action']),
                 'gold': int(d.get('gold', 0)),
                 'card': int(d.get('card', sts.CardId.INVALID)),
                 'relic': int(d.get('relic', sts.RelicId.INVALID)),
                 'info': int(d.get('info', EventFixedInfo.NONE))}
                for d in v
            ]
        else:
            flat[k] = v
    return flat


def save_episodes(experiences: List[Experience], advantages: List[float], returns: List[float], meta: List[dict], path: str):
    """Dump collected decisions to parquet in the SL schema train.py consumes
    (flattened choice + choice_type + chosen_idx + outcome/seed/final_floor/pstrike_count),
    plus PPO extras (reward, value, advantage, return, old_log_prob)."""
    rows = []
    for exp, adv, ret, m in zip(experiences, advantages, returns, meta):
        rows.append({
            **_serialize_choice(exp.choice),
            'choice_type': int(exp.choice_type),
            'chosen_idx': exp.action_idx,
            'outcome': m['outcome'],
            'seed': m['seed'],
            'final_floor': m['final_floor'],
            'encounter': exp.metrics.encounter,  # most recent battle at this decision (0 = none yet)
            'pstrike_count': sum(1 for cid in exp.choice.obs.deck.cards if cid == int(sts.CardId.PERFECTED_STRIKE)),
            'reward': m['reward'],
            'value': m['value'],
            'advantage': float(adv),
            'return': float(ret),
            'old_log_prob': float(exp.log_prob),
        })
    pd.DataFrame(rows).to_parquet(path, engine='pyarrow')


def experiences_to_batches(experiences: List[Experience], advantages: List[float], returns: List[float]) -> List[dict]:
    """Convert PPO experiences to training batches."""
    batch_data = []
    
    for i, exp in enumerate(experiences):
        # Convert Choice to the same format as used in collate_fn
        choice_dict = exp.choice.as_dict()
        flat_dict = {}
        
        # Flatten the nested choice dictionary
        for key, value in choice_dict.items():
            if key == 'obs':
                for obs_key, obs_value in value.items():
                    if isinstance(obs_value, dict):
                        for sub_key, sub_value in obs_value.items():
                            flat_dict[f'obs.{obs_key}.{sub_key}'] = sub_value
                    else:
                        flat_dict[f'obs.{obs_key}'] = obs_value
            elif key == 'cards_offered':
                for cards_key, cards_value in value.items():
                    flat_dict[f'cards_offered.{cards_key}'] = cards_value
            else:
                flat_dict[key] = value
        
        # Add training fields
        flat_dict['chosen_idx'] = exp.action_idx
        flat_dict['old_log_prob'] = exp.log_prob
        flat_dict['advantage'] = advantages[i]
        flat_dict['return'] = returns[i] if returns is not None else 0.0  # None for critic-free algos
        flat_dict['outcome'] = 1.0  # Dummy, not used in training
        
        batch_data.append(flat_dict)
    
    return batch_data


def lr_decay_factor(config: TrainConfig, iteration: int) -> float:
    """Multiplier on the base learning rates at `iteration` (see TrainConfig.lr_decay_*)."""
    if config.lr_decay_steps <= 0:
        return 1.0
    assert config.lr_final_frac > 0, "lr_final_frac must be > 0 when lr decay is enabled"
    frac = (iteration - config.lr_decay_start) / config.lr_decay_steps
    frac = min(max(frac, 0.0), 1.0)
    return config.lr_final_frac ** frac


def effective_entropy_coef(config: TrainConfig, iteration: int) -> float:
    """Entropy coefficient at `iteration` under the exponential decay schedule (see TrainConfig).
    Geometric interpolation entropy_coef -> entropy_coef_final over decay_steps iterations from
    decay_start, clamped to hold at entropy_coef_final afterwards."""
    if config.entropy_coef_decay_steps <= 0:
        return config.entropy_coef
    assert config.entropy_coef_final > 0, "entropy_coef_final must be > 0 when decay is enabled"
    frac = (iteration - config.entropy_coef_decay_start) / config.entropy_coef_decay_steps
    frac = min(max(frac, 0.0), 1.0)
    return config.entropy_coef * (config.entropy_coef_final / config.entropy_coef) ** frac


def train_step(nets, optimizer, experiences: List[Experience], advantages: List[float], returns: List[float], config: TrainConfig, algo=None, iteration: int = -1):
    """Perform one training step. `algo` (an Algorithm) supplies the per-minibatch loss; defaults
    to PPO for backward compatibility."""
    if not experiences:
        return {}
    if algo is None:
        from algorithms import PPOAlgorithm
        algo = PPOAlgorithm()

    if config.separate_networks:
        policy_net, value_net = nets
        device = policy_net.device
        # Set networks to training mode
        policy_net.train()
        value_net.train()
    else:
        net = nets
        device = net.device
        # Set network to training mode
        net.train()
    
    # Convert experiences to batches
    batch_data = experiences_to_batches(experiences, advantages, returns)
    
    # Create custom collate function that adds the per-step training fields
    want_return = algo.needs_return_in_batch()
    def ppo_collate_fn(batch):
        old_log_probs = [x['old_log_prob'] for x in batch]
        advantage_vals = [x['advantage'] for x in batch]

        collated = collate_fn(batch)

        collated['old_log_prob'] = torch.tensor(old_log_probs, dtype=torch.float32)
        collated['advantage'] = torch.tensor(advantage_vals, dtype=torch.float32)
        if want_return:  # value targets only for value-based algos (PPO)
            collated['return'] = torch.tensor([x['return'] for x in batch], dtype=torch.float32)

        return collated
    
    # Create data loader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        batch_data, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=ppo_collate_fn,
        num_workers=2,
    )
    
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_kl_div = 0
    total_grad_norm = 0
    total_policy_grad_norm = 0
    total_value_grad_norm = 0
    total_clipfrac = 0
    total_aux_room = 0.0
    # Stats for explained variance calculation
    target_stats = Stats()
    residual_stats = Stats()
    num_batches = 0
    
    for epoch in range(config.num_epochs):
        for collated_batch in dataloader:
            # Move to device
            collated_batch = move_to_device(collated_batch, device)
            batch_size = len(collated_batch['chosen_idx'])

            # Forward pass
            aux_room_logits = None
            if config.separate_networks:
                # Get policy logits from policy network
                new_logits = policy_net(collated_batch)
                
                # Get values from value network  
                value_output = value_net(collated_batch)
                if isinstance(value_output, tuple):
                    _, new_values = value_output
                else:
                    new_values = torch.zeros(batch_size, device=device)
            else:
                # Single network with value head; aux head active when the aux loss is enabled
                want_aux = config.aux_dest_room_coef > 0
                output = net(collated_batch, return_aux=True) if want_aux else net(collated_batch)
                if want_aux:
                    if len(output) == 3:
                        new_logits, new_values, aux_room_logits = output
                    else:
                        new_logits, aux_room_logits = output
                        new_values = torch.zeros(batch_size, device=device)
                elif isinstance(output, tuple):
                    new_logits, new_values = output
                else:
                    new_logits = output
                    new_values = torch.zeros(batch_size, device=device)

            # Per-minibatch loss from the algorithm strategy.
            total_loss, aux = algo.compute_loss(
                collated_batch, (new_logits, new_values, aux_room_logits), config)

            # Explained-variance accumulation (value-based algos provide ev_target/ev_pred).
            if 'ev_target' in aux:
                target_stats.add_samples(aux['ev_target'])
                residual_stats.add_samples(aux['ev_target'] - aux['ev_pred'])

            # Check for NaN before stepping.
            _pl, _en, _vl = aux['policy_loss'], aux['entropy'], aux.get('value_loss')
            if torch.isnan(_pl) or torch.isnan(_en) or (_vl is not None and torch.isnan(_vl)):
                print(f"NaN detected - policy: {_pl.item()}, "
                      f"value: {_vl.item() if _vl is not None else 'n/a'}, entropy: {_en.item()}; skipping batch")
                continue

            # Backward + grad clip + step (shared across algorithms).
            total_loss.backward()
            # Per-group grad norms (pre-clip): param_groups[0]=policy/trunk, [1]=value head
            # (single-net). Critic-free algos have one param group, so guard the second.
            pg0 = [p.grad.norm() for p in optimizer.param_groups[0]['params'] if p.grad is not None]
            pg1 = ([p.grad.norm() for p in optimizer.param_groups[1]['params'] if p.grad is not None]
                   if len(optimizer.param_groups) > 1 else [])
            policy_grad_norm = torch.norm(torch.stack(pg0)).item() if pg0 else 0.0
            value_grad_norm = torch.norm(torch.stack(pg1)).item() if pg1 else 0.0
            all_params = [p for group in optimizer.param_groups for p in group['params']]
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Accumulate losses and metrics.
            total_policy_loss += _pl.item()
            total_value_loss += _vl.item() if _vl is not None else 0.0
            total_entropy += _en.item()
            total_kl_div += aux['kl_div'].item()
            total_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            total_policy_grad_norm += policy_grad_norm
            total_value_grad_norm += value_grad_norm
            total_clipfrac += aux['clipfrac'].item()
            total_aux_room += aux['aux_room_loss'].item() if 'aux_room_loss' in aux else 0.0
            num_batches += 1
    
    # Calculate explained variance using Stats helper classes
    target_variance = target_stats.var()
    residual_variance = residual_stats.var()
    
    if target_variance > 1e-8:
        explained_variance = 1.0 - (residual_variance / target_variance)
    else:
        explained_variance = 0.0
    
    return {
        'policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0,
        'value_loss': total_value_loss / num_batches if num_batches > 0 else 0,
        'entropy': total_entropy / num_batches if num_batches > 0 else 0,
        'kl_div': total_kl_div / num_batches if num_batches > 0 else 0,
        'grad_norm': total_grad_norm / num_batches if num_batches > 0 else 0,
        'policy_grad_norm': total_policy_grad_norm / num_batches if num_batches > 0 else 0,
        'value_grad_norm': total_value_grad_norm / num_batches if num_batches > 0 else 0,
        'clipfrac': total_clipfrac / num_batches if num_batches > 0 else 0,
        'aux_room_loss': total_aux_room / num_batches if num_batches > 0 else 0,
        'explained_variance': explained_variance,
    }


def robust_save(state_dict_fn, path, tries=4):
    """torch.save(state_dict_fn(), path), retrying on transient teardown-time data races
    (state_dict() / pickling racing with a lingering torch worker thread). state_dict_fn is a
    callable (e.g. net.state_dict) so each attempt re-snapshots. Non-fatal: logs and returns
    False if every attempt fails, since periodic .iter_N checkpoints already cover the run."""
    for attempt in range(tries):
        try:
            torch.save(state_dict_fn(), path)
            return True
        except Exception as e:
            log.warning(f"save to {path} failed (attempt {attempt + 1}/{tries}): "
                        f"{type(e).__name__}: {e}")
            gc.collect()
            time.sleep(0.2)
    log.error(f"giving up on {path} after {tries} attempts; periodic checkpoints remain")
    return False


def main():
    parser = argparse.ArgumentParser(description='PPO training for Slay the Spire')
    parser.add_argument('--init-path', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--save-path', type=str, default='ppo_model.pt',
                        help='Path to save trained model')
    parser.add_argument('--reward-function', type=str, default='smooth',
                        choices=['smooth', 'perfected_strike', 'victory', 'no_pstrikes', 'heart'],
                        help='Reward function to use: smooth (sparse win/loss+floor), perfected_strike (dense card count), victory (sparse 0/1 win/loss), no_pstrikes (dense negative card count), heart (floor/114 + 0.25 act-3 win / 0.5 heart win) (default: smooth)')
    parser.add_argument('--torch-compile', type=str, default='default',
                        help='Torch compile mode: "default", "max-autotune", "reduce-overhead", or "no" to disable')
    parser.add_argument('--save-episodes', action='store_true',
                        help='Dump each iteration of collected decisions to {save_path}.episodes/iter_N.parquet (SL schema + PPO extras) for offline experiments')
    
    # Automatically add all TrainConfig fields as command line arguments
    config_defaults = TrainConfig()
    type_hints = get_type_hints(TrainConfig)
    
    for field in fields(TrainConfig):
        field_name = field.name.replace('_', '-')
        default_value = getattr(config_defaults, field.name)
        field_type = type_hints[field.name]
        
        # Handle Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            # For Optional[T] (which is Union[T, None]), get the non-None type
            non_none_types = [t for t in field_type.__args__ if t is not type(None)]
            field_type = non_none_types[0] if non_none_types else str
        
        # Map to argparse-compatible types
        if field_type == int:
            arg_type = int
        elif field_type == float:
            arg_type = float
        elif field_type == str:
            arg_type = str
        elif field_type == bool:
            arg_type = bool
        else:
            # Fallback to the type of the default value
            arg_type = type(default_value)
            
        parser.add_argument(
            f'--{field_name}',
            type=arg_type,
            default=default_value,
            help=f'PPO config: {field.name} (default: {default_value})'
        )
    
    # Add ModelHP fields as command line arguments with --model.* prefix
    model_hp_defaults = ModelHP()
    model_hp_type_hints = get_type_hints(ModelHP)
    
    for field in fields(ModelHP):
        field_name = f'model.{field.name.replace("_", "-")}'
        default_value = getattr(model_hp_defaults, field.name)
        field_type = model_hp_type_hints[field.name]
        
        # Handle Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            # For Optional[T] (which is Union[T, None]), get the non-None type
            non_none_types = [t for t in field_type.__args__ if t is not type(None)]
            field_type = non_none_types[0] if non_none_types else str
        
        # Map to argparse-compatible types
        if field_type == int:
            arg_type = int
        elif field_type == float:
            arg_type = float
        elif field_type == str:
            arg_type = str
        elif field_type == bool:
            arg_type = bool
        else:
            # Fallback to the type of the default value
            arg_type = type(default_value)
        
        parser.add_argument(
            f'--{field_name}',
            type=arg_type,
            default=default_value,
            help=f'Model hyperparameter: {field.name} (default: {default_value})'
        )
    
    args = parser.parse_args()
    
    # Configure logging - default level is WARNING, set to INFO to see Perfected Strike logs
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.cache_size_limit = 24
    
    # Create TrainConfig from parsed arguments
    config_kwargs = {}
    for field in fields(TrainConfig):
        field_name = field.name.replace('_', '-')
        config_kwargs[field.name] = getattr(args, field_name.replace('-', '_'))
    config = TrainConfig(**config_kwargs)

    # Seed all RNGs for reproducibility when requested (network init, DataLoader shuffle,
    # numpy). Per-episode action sampling uses its own random.Random(seed) and is unaffected.
    # Left unseeded by default to preserve prior behavior. Pair with --num-workers 1 for a
    # fully deterministic run (parallel collection completion order is otherwise nondeterministic).
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        print(f"Seeded RNGs with {config.seed}")

    # Create ModelHP from parsed arguments
    model_hp_kwargs = {}
    for field in fields(ModelHP):
        field_name = f'model.{field.name.replace("_", "-")}'
        arg_name = field_name.replace('-', '_')  # argparse converts dashes to underscores but keeps dots
        model_hp_kwargs[field.name] = getattr(args, arg_name)
    
    # Select reward function
    if args.reward_function == 'smooth':
        reward_fn = compute_progress_reward
    elif args.reward_function == 'perfected_strike':
        reward_fn = compute_perfected_strike_reward
    elif args.reward_function == 'victory':
        reward_fn = compute_victory_reward
    elif args.reward_function == 'no_pstrikes':
        reward_fn = compute_no_pstrikes_reward
    elif args.reward_function == 'heart':
        reward_fn = compute_heart_reward
    else:
        raise ValueError(f"Unknown reward function: {args.reward_function}")
    
    print(f"Using reward function: {args.reward_function}")

    # Algorithm strategy (collection plan + advantage estimation + per-batch loss).
    _ALGOS = {"ppo": PPOAlgorithm}
    if config.algo not in _ALGOS:
        raise ValueError(f"Unknown --algo {config.algo!r}; choose from {sorted(_ALGOS)}")
    algo = _ALGOS[config.algo]()
    print(f"Using algorithm: {algo.name}")
    
    # Create networks based on configuration
    if config.separate_networks:
        # Create separate policy and value networks
        policy_hp_kwargs = {k: v for k, v in model_hp_kwargs.items()}
        policy_hp_kwargs['use_value_head'] = False
        policy_hp = ModelHP(**policy_hp_kwargs)
        
        value_hp_kwargs = {k: v for k, v in model_hp_kwargs.items()}
        value_hp_kwargs['use_value_head'] = True
        value_hp = ModelHP(**value_hp_kwargs)
        
        policy_net = NN(policy_hp).to(device)
        value_net = NN(value_hp).to(device)
        
        
        if config.resume_from_step > 0:
            # Load from specific iteration checkpoints
            policy_path = f"{args.save_path}.policy.iter_{config.resume_from_step}"
            value_path = f"{args.save_path}.value.iter_{config.resume_from_step}"
            optimizer_path = f"{args.save_path}.optimizer.iter_{config.resume_from_step}"
            policy_state = torch.load(policy_path, map_location=device, weights_only=True)
            value_state = torch.load(value_path, map_location=device, weights_only=True)
            policy_net = load_network_backward_compatible(policy_net, policy_state)
            value_net = load_network_backward_compatible(value_net, value_state)
            print(f"Resumed from iteration {config.resume_from_step}: loaded {policy_path} and {value_path}")
        elif args.init_path:
            # Load from init path
            state = torch.load(args.init_path, map_location=device, weights_only=True)
            policy_net = load_network_backward_compatible(policy_net, state)
            # Hack: replace 'policy' with 'value' in the init path to find value weights
            value_path = args.init_path.replace('policy', 'value')
            value_state = torch.load(value_path, map_location=device, weights_only=True)
            value_net = load_network_backward_compatible(value_net, value_state)
            print(f"Loaded policy model from {args.init_path}")
            print(f"Loaded value model from {value_path}")
        
        # Create combined network wrapper
        combined_net = SeparateValuePolicy(policy_net, value_net)
        
        # Separate networks: policy params at policy_lr, value params at value_lr.
        optimizer = torch.optim.AdamW([
            {'params': list(policy_net.parameters()), 'lr': config.policy_lr},
            {'params': list(value_net.parameters()), 'lr': config.value_lr},
        ], weight_decay=config.weight_decay)
        
        # Load optimizer state if resuming
        if config.resume_from_step > 0:
            optimizer_path = f"{args.save_path}.optimizer.iter_{config.resume_from_step}"
            try:
                optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=True)
                optimizer.load_state_dict(optimizer_state)
                print(f"Loaded optimizer state from {optimizer_path}")
            except FileNotFoundError:
                print(f"Warning: Optimizer state file {optimizer_path} not found, starting with fresh optimizer state")
        
        service_net = combined_net
        
        nets = (policy_net, value_net)
        print("Using separate policy and value networks with combined wrapper")
    else:
        # Create single network; value head only if the algorithm uses a critic
        single_hp_kwargs = {k: v for k, v in model_hp_kwargs.items()}
        single_hp_kwargs['use_value_head'] = algo.requires_value_head
        model_hp = ModelHP(**single_hp_kwargs)
        net = NN(model_hp).to(device)
        
        
        if config.resume_from_step > 0:
            # Load from specific iteration checkpoint
            checkpoint_path = f"{args.save_path}.iter_{config.resume_from_step}"
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            net = load_network_backward_compatible(net, state)
            print(f"Resumed from iteration {config.resume_from_step}: loaded {checkpoint_path}")
        elif args.init_path:
            # Load from init path
            state = torch.load(args.init_path, map_location=device, weights_only=True)
            net = load_network_backward_compatible(net, state)
            print(f"Loaded model from {args.init_path}")
        
        if algo.requires_value_head:
            # Shared trunk + policy head train at policy_lr; the value head trains at value_lr.
            value_modules = [net.value_head_norm, net.value_head]
            if net.H.num_value_layers > 0:
                value_modules.append(net.value_layers)
            value_params = [p for m in value_modules for p in m.parameters()]
            value_param_ids = {id(p) for p in value_params}
            policy_params = [p for p in net.parameters() if id(p) not in value_param_ids]
            optimizer = torch.optim.AdamW([
                {'params': policy_params, 'lr': config.policy_lr},
                {'params': value_params, 'lr': config.value_lr},
            ], weight_decay=config.weight_decay)
        else:
            # Critic-free algorithm: a single param group at policy_lr (no value head to split out).
            optimizer = torch.optim.AdamW(net.parameters(), lr=config.policy_lr,
                                          weight_decay=config.weight_decay)
        
        # Load optimizer state if resuming
        if config.resume_from_step > 0:
            optimizer_path = f"{args.save_path}.optimizer.iter_{config.resume_from_step}"
            try:
                optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=True)
                optimizer.load_state_dict(optimizer_state)
                print(f"Loaded optimizer state from {optimizer_path}")
            except FileNotFoundError:
                print(f"Warning: Optimizer state file {optimizer_path} not found, starting with fresh optimizer state")
        
        orig_net = nets = service_net = net  # lol TODO
        print(f"Using single network (value_head={algo.requires_value_head})")
    
    service = NNService(service_net, batch_size=config.inf_batch_size, batch_size_factor=config.inf_batch_size_factor, torch_compile_mode=args.torch_compile)

    # Compile networks after service creation to ensure same compilation state
    if args.torch_compile != 'no':
        compile_mode = args.torch_compile
        if config.separate_networks:
            # Compile the whole SeparateValuePolicy
            service_net = torch.compile(service_net, mode=compile_mode)
        else:
            net = torch.compile(net, mode=compile_mode)
            service_net = net
            nets = net

    if config.resume_from_step > 0:
        print(f"Resuming PPO training from iteration {config.resume_from_step} with {config.num_games_per_step} games per batch")
    else:
        print(f"Starting PPO training with {config.num_games_per_step} games per batch")

    # Persistent advantage normalizer (EWMA of mean/std across iterations).
    adv_norm = RunningMoments(config.adv_norm_decay)

    try:
        for iteration in range(config.resume_from_step, config.num_iterations):
            print(f"\nIteration {iteration + 1}/{config.num_iterations}")
            
            # Collect experience
            start_time = time.time()
            service.update_weights(service_net)
            jobs = algo.collection_plan(config, iteration)
            trajectories = collect_experience(config, service, reward_fn, jobs)
            collect_time = time.time() - start_time
            
            if not trajectories:
                print("No trajectories collected, skipping iteration")
                continue
            
            # Compute statistics. A "win" is any PLAYER_VICTORY (for heart runs that includes
            # act-3-only wins; heart_win_rate below isolates full heart kills).
            def _won(t):
                return t.final_metrics.outcome == sts.GameOutcome.PLAYER_VICTORY
            win_rate = sum(1 for t in trajectories if _won(t)) / len(trajectories)
            avg_floor = sum(t.final_metrics.floor_num for t in trajectories) / len(trajectories)
            avg_reward = sum(t.final_reward for t in trajectories) / len(trajectories)

            print(f"Collected {len(trajectories)} trajectories in {collect_time:.1f}s")
            print(f"Win rate: {win_rate:.3f}, Avg floor: {avg_floor:.1f}, Avg reward: {avg_reward:.3f}")
            # Heart-run breakdown: full heart kills vs act-3-only wins, plus key acquisition.
            heart_stats = {}
            if args.reward_function == 'heart':
                heart_stats['heart_win_rate'] = sum(
                    1 for t in trajectories if _won(t) and t.final_metrics.act >= 4) / len(trajectories)
                heart_stats['act3_win_rate'] = sum(
                    1 for t in trajectories if _won(t) and t.final_metrics.act < 4) / len(trajectories)
                heart_stats['avg_keys'] = sum(t.final_metrics.num_keys for t in trajectories) / len(trajectories)
                heart_stats['act4_reach_rate'] = sum(
                    1 for t in trajectories if t.final_metrics.act >= 4) / len(trajectories)
                print(f"Heart: {heart_stats['heart_win_rate']:.3f} heart wins, "
                      f"{heart_stats['act3_win_rate']:.3f} act3-only wins, "
                      f"{heart_stats['act4_reach_rate']:.3f} reached act4, "
                      f"avg keys {heart_stats['avg_keys']:.2f}")
            # Per-ascension-level breakdown when training on a mixture.
            asc_stats = {}
            if config.max_ascension > 0:
                for a in range(config.max_ascension + 1):
                    ts = [t for t in trajectories if t.ascension == a]
                    if ts:
                        asc_stats[f'win_rate_asc{a}'] = sum(1 for t in ts if _won(t)) / len(ts)
                        asc_stats[f'num_games_asc{a}'] = len(ts)
                print("Per-ascension win rates: " + ", ".join(
                    f"A{a}={asc_stats[f'win_rate_asc{a}']:.3f}(n={asc_stats[f'num_games_asc{a}']})"
                    for a in range(config.max_ascension + 1) if f'win_rate_asc{a}' in asc_stats))
            
            # Prepare training data (with debug output for first trajectory)
            experiences, advantages, returns, ep_meta = algo.compute_advantages(trajectories, config, adv_norm, debug_traj=True)

            if not experiences:
                print("No experiences to train on, skipping iteration")
                continue

            print(f"Training on {len(experiences)} experiences")

            if args.save_episodes:
                ep_dir = f"{args.save_path}.episodes"
                os.makedirs(ep_dir, exist_ok=True)
                ep_path = f"{ep_dir}/iter_{iteration + 1}.parquet"
                save_episodes(experiences, advantages, returns, ep_meta, ep_path)
                print(f"Saved {len(experiences)} decisions to {ep_path}")
            
            # Perform PPO training step (entropy coef / lr possibly decayed for this iteration)
            ent_coef = effective_entropy_coef(config, iteration)
            step_config = replace(config, entropy_coef=ent_coef) if ent_coef != config.entropy_coef else config
            if config.entropy_coef_decay_steps > 0:
                print(f"Entropy coef this iteration: {ent_coef:.5f}")
            lrf = lr_decay_factor(config, iteration)
            if config.lr_decay_steps > 0:
                base_lrs = [config.policy_lr, config.value_lr]
                for gi, group in enumerate(optimizer.param_groups):
                    group['lr'] = base_lrs[min(gi, len(base_lrs) - 1)] * lrf
                print(f"LR factor this iteration: {lrf:.4f}")
            train_start = time.time()
            losses = train_step(nets, optimizer, experiences, advantages, returns, step_config, algo=algo)
            train_time = time.time() - train_start
            
            print(f"Training completed in {train_time:.1f}s")
            print(f"Policy loss: {losses.get('policy_loss', 0):.4f}, "
                  f"Value loss: {losses.get('value_loss', 0):.4f}, "
                  f"Entropy: {losses.get('entropy', 0):.4f}")
            print(f"KL div: {losses.get('kl_div', 0):.6f}, "
                  f"Grad norm: {losses.get('grad_norm', 0):.4f} (policy {losses.get('policy_grad_norm', 0):.4f} / value {losses.get('value_grad_norm', 0):.4f}), "
                  f"Clip frac: {losses.get('clipfrac', 0):.3f}")
            print(f"Value explained variance: {losses.get('explained_variance', 0):.3f}")
            
            # Create comprehensive stats dictionary
            stats = {
                'iteration': iteration + 1,
                'num_trajectories': len(trajectories),
                'collect_time': collect_time,
                'win_rate': win_rate,
                'avg_floor': avg_floor,
                'avg_reward': avg_reward,
                'num_experiences': len(experiences),
                'train_time': train_time,
                'policy_loss': losses.get('policy_loss', 0),
                'value_loss': losses.get('value_loss', 0),
                'entropy': losses.get('entropy', 0),
                'kl_div': losses.get('kl_div', 0),
                'grad_norm': losses.get('grad_norm', 0),
                'policy_grad_norm': losses.get('policy_grad_norm', 0),
                'value_grad_norm': losses.get('value_grad_norm', 0),
                'clipfrac': losses.get('clipfrac', 0),
                'aux_room_loss': losses.get('aux_room_loss', 0),
                'explained_variance': losses.get('explained_variance', 0),
                # EWMA std used to normalize PPO advantages. Logged so the effective
                # return-vs-entropy exchange rate (entropy_coef * adv_norm_std) is recoverable,
                # e.g. for objective-indifference lines in plots. Meaningless for critic-free
                # algos that leave advantages raw and never update adv_norm.
                'adv_norm_std': float(adv_norm.std),
                # Effective (possibly decayed) entropy coefficient used this iteration.
                'entropy_coef': ent_coef,
                # Actual learning rates used this iteration (post-decay).
                'policy_lr': config.policy_lr * lrf,
                'value_lr': config.value_lr * lrf,
                **heart_stats,
                **asc_stats,
            }
            
            # Write stats to JSONL file based on save path
            stats_path = f"{args.save_path}.stats.jsonl"
            with open(stats_path, 'a') as f:
                f.write(json.dumps(stats) + '\n')
            
            # Save model periodically
            if (iteration + 1) % config.save_every == 0:
                if config.separate_networks:
                    policy_net, value_net = nets
                    # Save model states
                    torch.save(policy_net.state_dict(), f"{args.save_path}.policy.iter_{iteration + 1}")
                    torch.save(value_net.state_dict(), f"{args.save_path}.value.iter_{iteration + 1}")
                    # Save optimizer state
                    torch.save(optimizer.state_dict(), f"{args.save_path}.optimizer.iter_{iteration + 1}")
                    print(f"Saved separate network checkpoints at iteration {iteration + 1}")
                else:
                    # Save model state
                    torch.save(orig_net.state_dict(), f"{args.save_path}.iter_{iteration + 1}")
                    # Save optimizer state
                    torch.save(optimizer.state_dict(), f"{args.save_path}.optimizer.iter_{iteration + 1}")
                    print(f"Saved model checkpoint at iteration {iteration + 1}")
    
    finally:
        service.stop()
    
    # Save final model. This runs at process teardown, where the just-finished training loop's
    # last DataLoader iterator (and any lingering torch worker threads) may still be tearing down
    # concurrently -- a data race with torch.save/state_dict() that intermittently corrupts the
    # pickle ("cannot pickle PyTorchFileWriter", "'Tensor' object is not iterable"). gc.collect()
    # forces that teardown to complete first; robust_save retries to absorb any residual race.
    # (Periodic .iter_N checkpoints are written during the loop and are unaffected.)
    gc.collect()
    if config.separate_networks:
        policy_net, value_net = nets
        robust_save(policy_net.state_dict, f"{args.save_path}.policy")
        robust_save(value_net.state_dict, f"{args.save_path}.value")
        robust_save(optimizer.state_dict, f"{args.save_path}.optimizer")
        print(f"Saved final separate networks to {args.save_path}.policy and {args.save_path}.value")
    else:
        robust_save(nets.state_dict, args.save_path)
        robust_save(optimizer.state_dict, f"{args.save_path}.optimizer")
        print(f"Saved final model to {args.save_path}")


if __name__ == "__main__":
    main()