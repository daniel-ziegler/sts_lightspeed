#!/usr/bin/env python3

from __future__ import annotations

import random
import argparse
import logging
from dataclasses import dataclass, fields
from typing import List, NamedTuple, Optional, get_type_hints, Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import threading
from collections import Counter
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from network import NN, ModelHP, move_to_device, process_batch, choice_space, collate_fn, load_network_backward_compatible, SeparateValuePolicy, EventFixedInfo
from playouts import run_game, NNService, Choice, Decision, ActionType, ChoiceStats, path_to_action_and_desc, construct_choice, flatten_dict
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
class PPOConfig:
    """PPO training hyperparameters."""
    # Environment settings
    num_games_per_step: int = 256
    num_epochs: int = 4
    num_workers: int = 40
    inf_batch_size: int = 32
    inf_batch_size_factor: int = 16
    batch_size: int = 128
    
    # PPO hyperparameters
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
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

    # MCTS battle search (per-episode agent knobs)
    mcts_simulations: int = 1000
    mcts_exploration: float = 3 * 2 ** 0.5  # ~4.2426 engine default; tuned ~6.5
    mcts_widening_c: float = 1.0            # tuned ~3.1
    mcts_widening_alpha: float = 0.5        # tuned ~0.97

    # Training settings
    num_iterations: int = 1000
    separate_networks: bool = False  # Use separate policy and value networks
    resume_from_step: int = 0  # Step to resume from (0 = start from beginning)
    
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
    outcome: sts.GameOutcome

class PPOExperience(NamedTuple):
    """Single step of experience from a game."""
    choice: Choice
    action_idx: int  # Needed for logprobs calculation in PPO training
    log_prob: float
    metrics: GameMetrics
    action_str: str  # Store clean action description for debugging
    choice_type: int  # ActionType value of the chosen action (offline SL label)


class PPOTrajectory(NamedTuple):
    """Complete game trajectory."""
    seed: int
    experiences: List[PPOExperience]
    rewards: List[float]  # Reward for each step
    values: List[float]   # Value prediction for each step
    final_reward: float
    final_metrics: GameMetrics  # Complete final game state metrics
    final_deck: List[sts.Card]  # Final deck state
    final_relics: List[sts.RelicId]  # Final relics


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


def compute_no_pstrikes_reward(metrics: GameMetrics) -> float:
    """Compute reward that penalizes Perfected Strikes (negative of count)."""
    return -float(metrics.perfected_strike_count)


def run_ppo_episode(seed: int, service: NNService, reward_fn, battle_executor, config: PPOConfig) -> PPOTrajectory:
    """Run a complete game episode and collect experience for PPO training."""
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    rng = random.Random(seed)
    
    agent = sts.Agent()
    agent.simulation_count_base = config.mcts_simulations
    agent.verbosity_level = 0  # silence per-action battle prints (keep ppo_train's own stdout)
    agent.exploration_parameter = config.mcts_exploration
    agent.chance_widening_c = config.mcts_widening_c
    agent.chance_widening_alpha = config.mcts_widening_alpha
    experiences = []
    values = []  # Collect values separately
    reward_fn_vals = []
    
    while gc.outcome == sts.GameOutcome.UNDECIDED:
        try:
            if gc.screen_state == sts.ScreenState.BATTLE:
                # Use MCTS agent for battles in background thread
                future = battle_executor.submit(agent.playout_battle, gc)
                
                try:
                    # Wait for completion with timeout. future.result raises
                    # concurrent.futures.TimeoutError, which is a distinct class from the
                    # builtin TimeoutError before Python 3.11 -- catch the right one.
                    future.result(timeout=30.0)
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
                        
                        # Convert to probabilities and sample action
                        logits_tensor = torch.tensor(logits)
                        log_probs = F.log_softmax(logits_tensor, dim=0).numpy()
                        probs = np.exp(log_probs)
                        
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
                            outcome=gc.outcome,
                        )
                        reward_fn_vals.append(reward_fn(metrics))
                        
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
                        action.execute(gc)
                        
                        exp = PPOExperience(
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
                        action = agent.pick_gameaction(gc)
                        assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                        action.execute(gc)
                else:
                    action = agent.pick_gameaction(gc)
                    assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                    action.execute(gc)
                
        except Exception as e:
            log.error(f"Error in episode {seed}: {e}")
            raise
    
    # Create final metrics for reward computation
    final_metrics = GameMetrics(
        floor_num=gc.floor_num,
        cur_hp=gc.cur_hp,
        max_hp=gc.max_hp,
        perfected_strike_count=sum(1 for card in gc.deck if card.id == sts.CardId.PERFECTED_STRIKE),
        outcome=gc.outcome,
    )
    
    # Compute shaped rewards with centralized delta calculation
    rewards = []
    
    # Compute all reward values once
    all_reward_values = reward_fn_vals + [reward_fn(final_metrics)]
    
    # Reward shaping: each step gets delta from current state to next state
    for i in range(len(experiences)):
        reward_delta = all_reward_values[i+1] - all_reward_values[i]
        rewards.append(reward_delta)
    
    # Add terminal state value (0.0) for GAE bootstrap
    values.append(0.0)
    # Values were collected during the episode
    
    return PPOTrajectory(
        seed=seed,
        experiences=experiences,
        rewards=rewards,
        values=values,
        final_reward=all_reward_values[-1],
        final_metrics=final_metrics,
        final_deck=list(gc.deck),
        final_relics=list(gc.relics)
    )


def collect_experience(config: PPOConfig, service: NNService, reward_fn, start_seed: int = 0) -> List[PPOTrajectory]:
    """Collect experience from multiple game episodes."""
    trajectories = []
    
    if config.num_workers == 1:
        # Single-threaded execution for easier debugging
        # Create a shared executor for battle simulations
        with ThreadPoolExecutor(max_workers=1) as battle_executor:
            for i in tqdm(range(config.num_games_per_step), desc="Collecting experience"):
                trajectory = run_ppo_episode(start_seed + i, service, reward_fn, battle_executor, config)
                trajectories.append(trajectory)
    else:
        # Multi-threaded execution
        # Create a shared executor for battle simulations
        with ThreadPoolExecutor(max_workers=config.num_workers) as battle_executor:
            with ThreadPoolExecutor(max_workers=config.num_workers) as main_executor:
                futures = [
                    main_executor.submit(run_ppo_episode, start_seed + i, service, reward_fn, battle_executor, config)
                    for i in range(config.num_games_per_step)
                ]
                
                for future in tqdm(as_completed(futures), total=config.num_games_per_step, desc="Collecting experience"):
                    trajectory = future.result()
                    trajectories.append(trajectory)
    
    return trajectories


def compute_advantages(trajectories: List[PPOTrajectory], config: PPOConfig, adv_norm: RunningMoments, debug_traj: bool = False) -> tuple[List[PPOExperience], List[float], List[float], List[dict]]:
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
            print(f"=== PPO Advantage Calculation Debug (seed {traj.seed}) ===")
            print(f"Trajectory length: {len(traj.experiences)} steps")
            print(f"Rewards array length: {len(traj.rewards)}, first 5 rewards: {traj.rewards[:5]}")
            print(f"Values array length: {len(traj.values)}, first 5 values: {traj.values[:5]}")
            print(f"Step | {'State':12s} | {'Choice':20s} | {'Action':20s} | {'Prob':6s} | {'Reward':6s} | {'Pred Value':10s} | {'GAE Return':10s} | {'Raw Advantage':13s}")
            print("-" * 140)
            
            for t in range(len(traj.experiences)):
                exp = traj.experiences[t]
                
                # Get choice summary - what was offered
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
                
                choice_desc = f"{'+'.join(offered_items)}" if offered_items else "none"
                
                # Get action description - use the clean description generated during experience collection
                action_desc = exp.action_str[:20] if exp.action_str else "Unknown"
                
                # Create state string: 18: 20/72hp format
                state_str = f"{exp.metrics.floor_num:>2}: {exp.metrics.cur_hp}/{exp.metrics.max_hp}hp"
                
                print(f"{t:4d} | {state_str:12s} | {choice_desc[:20]:20s} | {action_desc[:20]:20s} | {np.exp(exp.log_prob):6.3f} | {rewards[t]:6.3f} | {values[t]:10.3f} | {returns[t]:10.3f} | {advantages[t]:13.3f}")
            
            print("-" * 140)
            print(f"Final game outcome: {traj.experiences[-1].metrics.outcome}")
            final_metrics = traj.final_metrics
            final_state = f"{final_metrics.floor_num}: {final_metrics.cur_hp}/{final_metrics.max_hp}hp"
            print(f"Final reward: {traj.final_reward:.3f}, Final state: {final_state}")
            
            # Show final deck and relics
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


def save_episodes(experiences: List[PPOExperience], advantages: List[float], returns: List[float], meta: List[dict], path: str):
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
            'pstrike_count': sum(1 for cid in exp.choice.obs.deck.cards if cid == int(sts.CardId.PERFECTED_STRIKE)),
            'reward': m['reward'],
            'value': m['value'],
            'advantage': float(adv),
            'return': float(ret),
            'old_log_prob': float(exp.log_prob),
        })
    pd.DataFrame(rows).to_parquet(path, engine='pyarrow')


def experiences_to_batches(experiences: List[PPOExperience], advantages: List[float], returns: List[float]) -> List[dict]:
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
        
        # Add PPO-specific fields
        flat_dict['chosen_idx'] = exp.action_idx
        flat_dict['old_log_prob'] = exp.log_prob
        flat_dict['advantage'] = advantages[i]
        flat_dict['return'] = returns[i]
        flat_dict['outcome'] = 1.0  # Dummy, not used in PPO
        
        batch_data.append(flat_dict)
    
    return batch_data


def ppo_train_step(nets, optimizer, experiences: List[PPOExperience], advantages: List[float], returns: List[float], config: PPOConfig, iteration: int = -1):
    """Perform one PPO training step."""
    if not experiences:
        return {}
    
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
    
    # Create custom collate function that handles PPO fields
    def ppo_collate_fn(batch):
        # Extract PPO-specific fields before calling main collate_fn
        old_log_probs = [x['old_log_prob'] for x in batch]
        advantage_vals = [x['advantage'] for x in batch]
        return_vals = [x['return'] for x in batch]
        
        # Call the main collate function
        collated = collate_fn(batch)
        
        # Add PPO-specific fields
        collated['old_log_prob'] = torch.tensor(old_log_probs, dtype=torch.float32)
        collated['advantage'] = torch.tensor(advantage_vals, dtype=torch.float32)
        collated['return'] = torch.tensor(return_vals, dtype=torch.float32)
        
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
    total_clipfrac = 0
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
                # Single network with value head
                output = net(collated_batch)
                if isinstance(output, tuple):
                    new_logits, new_values = output
                else:
                    new_logits = output
                    new_values = torch.zeros(batch_size, device=device)
            
            # Get old log probs, advantages, target values from collated batch
            old_log_probs = collated_batch['old_log_prob'].to(device)
            batch_advantages = collated_batch['advantage'].to(device)
            target_values = collated_batch['return'].to(device)
            chosen_indices = collated_batch['chosen_idx'].to(device)
            
            # Compute new log probabilities with numerical stability
            action_probs = F.softmax(new_logits, dim=-1)
            action_log_probs = F.log_softmax(new_logits, dim=-1)
            
            # Get log probs for chosen actions
            batch_indices_tensor = torch.arange(batch_size, device=device)
            new_log_probs = action_log_probs[batch_indices_tensor, chosen_indices]
            
            # Clamp log probs for numerical stability
            new_log_probs = torch.clamp(new_log_probs, min=-20, max=20)
            old_log_probs = torch.clamp(old_log_probs, min=-20, max=20)
            
            # Compute probability ratios
            ratio = torch.exp(new_log_probs - old_log_probs)
            ratio = torch.clamp(ratio, min=1e-8, max=1e8)  # Prevent extreme ratios
            
            # Compute approximate KL divergence
            kl_div = (old_log_probs - new_log_probs).mean()
            
            # Compute clipping fraction
            clipfrac = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
            
            # Clipped surrogate objective
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, target_values)
            
            # Accumulate statistics for explained variance calculation
            target_stats.add_samples(target_values)
            residuals = target_values - new_values
            residual_stats.add_samples(residuals)
            
            # Entropy bonus (with numerical stability for masked actions)
            # Only compute entropy for valid actions (non-inf logits)
            valid_mask = ~torch.isinf(new_logits)
            
            # For each sample, compute entropy only over valid actions
            batch_entropies = []
            for i in range(new_logits.shape[0]):
                valid_actions = valid_mask[i]
                if valid_actions.sum() > 1:  # Need at least 2 valid actions for meaningful entropy
                    valid_probs = action_probs[i][valid_actions]
                    valid_log_probs = action_log_probs[i][valid_actions]
                    sample_entropy = -(valid_probs * valid_log_probs).sum()
                    batch_entropies.append(sample_entropy)
                else:
                    # If only 0 or 1 valid actions, entropy is 0
                    batch_entropies.append(torch.tensor(0.0, device=device))
            
            if batch_entropies:
                entropy = torch.stack(batch_entropies).mean()
            else:
                entropy = torch.tensor(0.0, device=device)
            
            # Optional debug (uncomment for debugging)
            # print(f"Valid actions per sample: {valid_mask.sum(dim=1).float().mean().item():.1f}")
            # print(f"Computed entropy: {entropy.item():.6f}")
            
            # Check for NaN before combining losses
            if torch.isnan(policy_loss) or torch.isnan(value_loss) or torch.isnan(entropy):
                print(f"NaN detected - policy: {policy_loss.item()}, value: {value_loss.item()}, entropy: {entropy.item()}")
                print(f"Ratio stats - min: {ratio.min().item()}, max: {ratio.max().item()}, mean: {ratio.mean().item()}")
                print(f"Advantages stats - min: {batch_advantages.min().item()}, max: {batch_advantages.max().item()}, mean: {batch_advantages.mean().item()}")
                continue  # Skip this batch
            
            # Backward pass - same logic for both separate and single networks
            total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            total_loss.backward()
            all_params = [p for group in optimizer.param_groups for p in group['params']]
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Accumulate losses and metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl_div += kl_div.item()
            total_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            total_clipfrac += clipfrac.item()
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
        'clipfrac': total_clipfrac / num_batches if num_batches > 0 else 0,
        'explained_variance': explained_variance,
    }


def main():
    parser = argparse.ArgumentParser(description='PPO training for Slay the Spire')
    parser.add_argument('--init-path', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--save-path', type=str, default='ppo_model.pt',
                        help='Path to save trained model')
    parser.add_argument('--reward-function', type=str, default='smooth',
                        choices=['smooth', 'perfected_strike', 'victory', 'no_pstrikes'],
                        help='Reward function to use: smooth (sparse win/loss+floor), perfected_strike (dense card count), victory (sparse 0/1 win/loss), no_pstrikes (dense negative card count) (default: perfected_strike)')
    parser.add_argument('--torch-compile', type=str, default='default',
                        help='Torch compile mode: "default", "max-autotune", "reduce-overhead", or "no" to disable')
    parser.add_argument('--save-episodes', action='store_true',
                        help='Dump each iteration of collected decisions to {save_path}.episodes/iter_N.parquet (SL schema + PPO extras) for offline experiments')
    
    # Automatically add all PPOConfig fields as command line arguments
    config_defaults = PPOConfig()
    type_hints = get_type_hints(PPOConfig)
    
    for field in fields(PPOConfig):
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
    
    # Create PPOConfig from parsed arguments
    config_kwargs = {}
    for field in fields(PPOConfig):
        field_name = field.name.replace('_', '-')
        config_kwargs[field.name] = getattr(args, field_name.replace('-', '_'))
    config = PPOConfig(**config_kwargs)
    
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
    else:
        raise ValueError(f"Unknown reward function: {args.reward_function}")
    
    print(f"Using reward function: {args.reward_function}")
    
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
        # Create single network with value head
        single_hp_kwargs = {k: v for k, v in model_hp_kwargs.items()}
        single_hp_kwargs['use_value_head'] = True
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
        print("Using single network with value head")
    
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
            trajectories = collect_experience(config, service, reward_fn, start_seed=iteration * 1000)
            collect_time = time.time() - start_time
            
            if not trajectories:
                print("No trajectories collected, skipping iteration")
                continue
            
            # Compute statistics
            win_rate = sum(1 for t in trajectories if t.final_reward >= 1.0) / len(trajectories)
            avg_floor = sum(t.final_metrics.floor_num for t in trajectories) / len(trajectories)
            avg_reward = sum(t.final_reward for t in trajectories) / len(trajectories)
            
            print(f"Collected {len(trajectories)} trajectories in {collect_time:.1f}s")
            print(f"Win rate: {win_rate:.3f}, Avg floor: {avg_floor:.1f}, Avg reward: {avg_reward:.3f}")
            
            # Prepare training data (with debug output for first trajectory)
            experiences, advantages, returns, ep_meta = compute_advantages(trajectories, config, adv_norm, debug_traj=True)

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
            
            # Perform PPO training step
            train_start = time.time()
            losses = ppo_train_step(nets, optimizer, experiences, advantages, returns, config)
            train_time = time.time() - train_start
            
            print(f"Training completed in {train_time:.1f}s")
            print(f"Policy loss: {losses.get('policy_loss', 0):.4f}, "
                  f"Value loss: {losses.get('value_loss', 0):.4f}, "
                  f"Entropy: {losses.get('entropy', 0):.4f}")
            print(f"KL div: {losses.get('kl_div', 0):.6f}, "
                  f"Grad norm: {losses.get('grad_norm', 0):.4f}, "
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
                'clipfrac': losses.get('clipfrac', 0),
                'explained_variance': losses.get('explained_variance', 0)
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
    
    # Save final model
    if config.separate_networks:
        policy_net, value_net = nets
        torch.save(policy_net.state_dict(), f"{args.save_path}.policy")
        torch.save(value_net.state_dict(), f"{args.save_path}.value")
        torch.save(optimizer.state_dict(), f"{args.save_path}.optimizer")
        print(f"Saved final separate networks to {args.save_path}.policy and {args.save_path}.value")
    else:
        torch.save(nets.state_dict(), args.save_path)
        torch.save(optimizer.state_dict(), f"{args.save_path}.optimizer")
        print(f"Saved final model to {args.save_path}")


if __name__ == "__main__":
    main()