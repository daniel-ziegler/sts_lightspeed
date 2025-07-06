#!/usr/bin/env python3

from __future__ import annotations

import random
import argparse
import logging
from dataclasses import dataclass, fields
from typing import List, NamedTuple, Optional, get_type_hints, Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from network import NN, ModelHP, move_to_device, process_batch, choice_space, collate_fn
from playouts import run_game, NNService, Choice, Decision, ActionType, ChoiceStats
import slaythespire as sts

# Set up logging
log = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO training hyperparameters."""
    # Environment settings
    num_games_per_batch: int = 256
    num_epochs: int = 4
    num_workers: int = 30
    batch_size: int = 128
    
    # PPO hyperparameters
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    
    # Learning rates
    policy_lr: float = 5e-5
    value_lr: float = 1e-4
    
    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Training settings
    num_iterations: int = 1000
    separate_networks: bool = False  # Use separate policy and value networks
    resume_from: int = 0  # Step to resume from (0 = start from beginning)
    
    # Logging
    log_every: int = 10
    save_every: int = 20



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
    action_idx: int
    log_prob: float
    metrics: GameMetrics


class PPOTrajectory(NamedTuple):
    """Complete game trajectory."""
    experiences: List[PPOExperience]
    rewards: List[float]  # Reward for each step
    values: List[float]   # Value prediction for each step
    final_reward: float
    final_floor: int
    pstrike_offers: int  # Number of times Perfected Strike was offered
    pstrike_takes: int   # Number of times Perfected Strike was taken
    pstrike_probs: List[float]  # Probabilities assigned to Perfected Strike when offered


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


def run_ppo_episode(seed: int, service: NNService, reward_fn, value_service=None) -> PPOTrajectory:
    """Run a complete game episode and collect experience for PPO training."""
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    rng = random.Random(seed)
    
    agent = sts.Agent()
    agent.simulation_count_base = 1000
    experiences = []
    values = []  # Collect values separately
    
    # Track Perfected Strike stats
    pstrike_offers = 0
    pstrike_takes = 0
    pstrike_probs = []
    
    # Create timeout handling
    timeout_event = threading.Event()
    
    def timeout_handler():
        timeout_event.set()
        log.warning(f"Battle simulation taking too long for seed {seed}")
    
    while gc.outcome == sts.GameOutcome.UNDECIDED:
        try:
            if gc.screen_state == sts.ScreenState.BATTLE:
                # Use MCTS agent for battles
                timer = threading.Timer(30.0, timeout_handler)
                timer.start()
                
                try:
                    agent.playout_battle(gc)
                finally:
                    timer.cancel()
                    
                if timeout_event.is_set():
                    log.warning(f"Seed {seed} did finish")
                    timeout_event.clear()
                    
            else:
                # Use neural network for non-battle decisions
                obs = sts.getNNRepresentation(gc)
                actions = sts.GameAction.getAllActionsInState(gc)
                
                if gc.screen_state in (sts.ScreenState.REWARDS, sts.ScreenState.SHOP_ROOM, sts.ScreenState.BOSS_RELIC_REWARDS):
                    from playouts import construct_choice
                    choice = construct_choice(gc, obs, actions)
                    
                    # Count total number of choices available
                    total_choices = (len(choice.cards_offered) + len(choice.relics_offered) + 
                                   len(choice.potions_offered) + len(choice.fixed_actions))
                    
                    if total_choices > 1:
                        # Get network predictions
                        batch_tensors, output = service.get_logits(choice)
                        
                        # Handle value head output
                        if isinstance(output, tuple):
                            logits, value_output = output
                            value = float(value_output) if np.isscalar(value_output) else float(value_output[0])
                        else:
                            logits = output
                            # Get value from separate value service if available
                            if value_service is not None:
                                _, value_output = value_service.get_logits(choice)
                                if isinstance(value_output, tuple):
                                    _, value = value_output
                                    value = float(value) if np.isscalar(value) else float(value[0])
                                else:
                                    value = 0.0
                            else:
                                value = 0.0  # No value head
                        
                        # Convert to probabilities and sample action
                        logits_tensor = torch.tensor(logits)
                        log_probs = F.log_softmax(logits_tensor, dim=0).numpy()
                        probs = np.exp(log_probs)
                        
                        chosen_idx = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])
                        log_prob = log_probs[chosen_idx]
                        
                        # Check if Perfected Strike is offered and log probability + decision
                        perfected_strike_offered = any(card.id == sts.CardId.PERFECTED_STRIKE for card in choice.cards_offered)
                        perfected_strike_prob = None
                        
                        if perfected_strike_offered:
                            pstrike_offers += 1
                            # Count total number of valid options
                            total_options = (len(choice.cards_offered) + len(choice.relics_offered) + 
                                           len(choice.potions_offered) + len(choice.fixed_actions))
                            
                            # Find the probability assigned to Perfected Strike
                            for i, card in enumerate(choice.cards_offered):
                                if card.id == sts.CardId.PERFECTED_STRIKE:
                                    # Find the choice index for this card
                                    card_path = ('cards', i)
                                    try:
                                        pstrike_choice_idx = choice_space.path_to_ix(batch_tensors['choices'], card_path)
                                        perfected_strike_prob = float(probs[pstrike_choice_idx])
                                        pstrike_probs.append(perfected_strike_prob)
                                        break
                                    except (IndexError, KeyError):
                                        perfected_strike_prob = 0.0
                                        pstrike_probs.append(0.0)
                        
                        # Convert back to game action
                        path = choice_space.ix_to_path(batch_tensors['choices'], chosen_idx)
                        
                        if path[0] == 'cards':
                            action = choice.card_actions[path[1]]
                            chosen_card = choice.cards_offered[path[1]]
                            if perfected_strike_offered:
                                perfected_strike_taken = chosen_card.id == sts.CardId.PERFECTED_STRIKE
                                if perfected_strike_taken:
                                    pstrike_takes += 1
                                log.info(f"Seed {seed}, Floor {gc.floor_num}: Perfected Strike offered ({total_options} options), prob: {perfected_strike_prob:.3f}, taken: {perfected_strike_taken}")
                        elif path[0] == 'relics':
                            action = choice.relic_actions[path[1]]
                            if perfected_strike_offered:
                                log.info(f"Seed {seed}, Floor {gc.floor_num}: Perfected Strike offered ({total_options} options), prob: {perfected_strike_prob:.3f}, but chose relic instead")
                        elif path[0] == 'potions':
                            action = choice.potion_actions[path[1]]
                            if perfected_strike_offered:
                                log.info(f"Seed {seed}, Floor {gc.floor_num}: Perfected Strike offered ({total_options} options), prob: {perfected_strike_prob:.3f}, but chose potion instead")
                        elif path[0] == 'fixed':
                            action = choice.fixed_actions_list[path[1]]
                            if perfected_strike_offered:
                                log.info(f"Seed {seed}, Floor {gc.floor_num}: Perfected Strike offered ({total_options} options), prob: {perfected_strike_prob:.3f}, but chose fixed action instead")
                        else:
                            raise ValueError(f"Unknown path: {path}")
                        
                        # Store experience data before action execution, but capture metrics after
                        exp_data = {
                            'choice': choice,
                            'action_idx': chosen_idx,
                            'log_prob': log_prob,
                            'value': value
                        }
                        
                        assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                        action.execute(gc)
                        
                        # Extract metrics from game state AFTER action execution for correct reward computation
                        perfected_strike_count = sum(1 for card in gc.deck if card.id == sts.CardId.PERFECTED_STRIKE)
                        metrics = GameMetrics(
                            floor_num=gc.floor_num,
                            cur_hp=gc.cur_hp,
                            max_hp=gc.max_hp,
                            perfected_strike_count=perfected_strike_count,
                            outcome=gc.outcome
                        )
                        
                        exp = PPOExperience(
                            choice=exp_data['choice'],
                            action_idx=exp_data['action_idx'],
                            log_prob=exp_data['log_prob'],
                            metrics=metrics
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
        outcome=gc.outcome
    )
    
    # Compute final reward using the provided reward function
    final_reward = reward_fn(final_metrics)
    
    # Compute shaped rewards with centralized delta calculation
    rewards = []
    
    # Collect all states: experiences + final state
    all_metrics = [exp.metrics for exp in experiences] + [final_metrics]
    
    # Compute all reward values once
    all_reward_values = [reward_fn(metrics) for metrics in all_metrics]
    
    # Proper reward shaping: first step gets 0, subsequent steps get deltas
    if experiences:
        # First step gets 0 reward (reward shaping baseline)
        rewards.append(0.0)
        
        # All subsequent steps get reward deltas
        for i in range(1, len(experiences)):
            reward_delta = all_reward_values[i] - all_reward_values[i-1]
            rewards.append(reward_delta)
        
        # Terminal step: add final delta to the last experience reward
        if len(all_reward_values) > len(experiences):
            final_delta = all_reward_values[-1] - all_reward_values[-2]
            if len(experiences) > 40:  # Debug for longer episodes
                log.debug(f"Final state - floor: {all_metrics[-1].floor_num}, outcome: {all_metrics[-1].outcome}, reward: {all_reward_values[-1]:.6f}")
                log.debug(f"Last exp - floor: {all_metrics[-2].floor_num}, outcome: {all_metrics[-2].outcome}, reward: {all_reward_values[-2]:.6f}")
                log.debug(f"Final delta = {final_delta:.6f}")
                log.debug(f"Last step reward before adding delta = {rewards[-1]:.6f}")
            rewards[-1] += final_delta
            if len(experiences) > 40:
                log.debug(f"Last step reward after adding delta = {rewards[-1]:.6f}")
    
    # Add terminal state value (0.0) for GAE bootstrap
    values.append(0.0)
    # Values were collected during the episode
    
    # Debug: Print what we're actually storing
    if len(experiences) > 40:  # Only for longer episodes
        log.debug(f"Created {len(rewards)} rewards: {rewards[:3]}...{rewards[-3:]}")
        
    # Debug: Check for suspicious high rewards that might indicate the original bug
    for i, (exp, reward) in enumerate(zip(experiences, rewards)):
        if abs(reward - 0.98) < 0.01 and exp.metrics.floor_num >= 45 and exp.metrics.floor_num <= 50:
            log.warning(f"DEBUG FOUND SUSPICIOUS REWARD: Step {i}, floor {exp.metrics.floor_num}, outcome {exp.metrics.outcome}, reward {reward:.6f}")
            log.warning(f"  Expected reward for floor {exp.metrics.floor_num}: {compute_progress_reward(exp.metrics):.6f}")
            log.warning(f"  This matches the original bug description!")
    
    return PPOTrajectory(
        experiences=experiences,
        rewards=rewards,
        values=values,
        final_reward=final_reward,
        final_floor=gc.floor_num,
        pstrike_offers=pstrike_offers,
        pstrike_takes=pstrike_takes,
        pstrike_probs=pstrike_probs
    )


def collect_experience(config: PPOConfig, service: NNService, reward_fn, start_seed: int = 0, value_service=None) -> List[PPOTrajectory]:
    """Collect experience from multiple game episodes."""
    trajectories = []
    
    if config.num_workers == 1:
        # Single-threaded execution for easier debugging
        for i in tqdm(range(config.num_games_per_batch), desc="Collecting experience"):
            trajectory = run_ppo_episode(start_seed + i, service, reward_fn, value_service)
            trajectories.append(trajectory)
    else:
        # Multi-threaded execution
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = [
                executor.submit(run_ppo_episode, start_seed + i, service, reward_fn, value_service)
                for i in range(config.num_games_per_batch)
            ]
            
            for future in tqdm(as_completed(futures), total=config.num_games_per_batch, desc="Collecting experience"):
                trajectory = future.result()
                trajectories.append(trajectory)
    
    return trajectories


def compute_advantages(trajectories: List[PPOTrajectory], config: PPOConfig, debug_first: bool = False) -> tuple[List[PPOExperience], List[float], List[float]]:
    """Compute advantages using GAE and prepare training data."""
    all_experiences = []
    all_advantages = []
    all_returns = []
    
    # Find trajectory with highest reward for debug output
    debug_traj_idx = None
    if debug_first and trajectories:
        # Find trajectory with highest final reward
        best_reward = max(traj.final_reward for traj in trajectories)
        for i, traj in enumerate(trajectories):
            if traj.final_reward == best_reward:
                debug_traj_idx = i
                break
        # Fallback to first trajectory if none found
        if debug_traj_idx is None:
            debug_traj_idx = 0
    
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
        
        # Debug output for highest-reward trajectory
        if debug_first and traj_idx == debug_traj_idx:
            print(f"=== PPO Advantage Calculation Debug (Trajectory {traj_idx} with highest reward: {traj.final_reward:.3f}) ===")
            print(f"Trajectory length: {len(traj.experiences)} steps")
            print(f"Rewards array length: {len(traj.rewards)}, first 5 rewards: {traj.rewards[:5]}")
            print(f"Values array length: {len(traj.values)}, first 5 values: {traj.values[:5]}")
            print(f"Step | {'Floor':5s} | {'Action':30s} | {'Prob':6s} | {'Reward':6s} | {'Pred Value':10s} | {'GAE Return':10s} | {'Raw Advantage':13s}")
            print("-" * 100)
            
            for t in range(len(traj.experiences)):
                exp = traj.experiences[t]
                # Get action description - what was actually chosen
                action_desc = "Unknown"
                
                # Try to get the actual chosen action description
                if exp.choice.cards_offered and exp.action_idx < len(exp.choice.card_actions):
                    chosen_card = exp.choice.cards_offered[exp.action_idx]
                    action_desc = str(chosen_card)[:30]
                elif exp.choice.relics_offered and exp.action_idx < len(exp.choice.relic_actions):
                    chosen_relic = exp.choice.relics_offered[exp.action_idx]
                    action_desc = sts.RelicId(chosen_relic).name[:30]
                elif exp.choice.potions_offered and exp.action_idx < len(exp.choice.potion_actions):
                    chosen_potion = exp.choice.potions_offered[exp.action_idx]
                    action_desc = sts.Potion(chosen_potion).name[:30]
                elif exp.choice.fixed_actions and exp.action_idx < len(exp.choice.fixed_actions_list):
                    action_desc = str(exp.choice.fixed_actions[exp.action_idx])[:30]
                else:
                    # Fallback to showing what was offered
                    offered_items = []
                    if exp.choice.cards_offered:
                        has_pstrike = any(card.id == sts.CardId.PERFECTED_STRIKE for card in exp.choice.cards_offered)
                        cards_str = f"{len(exp.choice.cards_offered)}card"
                        if has_pstrike:
                            cards_str += "*"
                        offered_items.append(cards_str)
                    if exp.choice.relics_offered:
                        offered_items.append(f"{len(exp.choice.relics_offered)}rel")
                    if exp.choice.potions_offered:
                        offered_items.append(f"{len(exp.choice.potions_offered)}pot")
                    if exp.choice.fixed_actions:
                        offered_items.append(f"{len(exp.choice.fixed_actions)}fix")
                    
                    if offered_items:
                        action_desc = f"{'+'.join(offered_items)} idx:{exp.action_idx}"
                    else:
                        action_desc = f"Action idx: {exp.action_idx}"
                
                print(f"{t:4d} | {exp.metrics.floor_num:5d} | {action_desc[:30]:30s} | {np.exp(exp.log_prob):6.3f} | {rewards[t]:6.3f} | {values[t]:10.3f} | {returns[t]:10.3f} | {advantages[t]:13.3f}")
            
            print("-" * 100)
            print(f"Final game outcome: {traj.experiences[-1].metrics.outcome}")
            print(f"Final reward: {traj.final_reward:.3f}, Final floor: {traj.final_floor}")
            print(f"Last step reward (includes terminal): {rewards[-1]:.3f}")
            print("=" * 80)
        
        # Normalize advantages
        if len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / adv_std
            else:
                advantages = advantages - adv_mean
        
        # Store experiences, advantages, and returns
        all_experiences.extend(traj.experiences)
        all_advantages.extend(advantages.tolist())
        all_returns.extend(returns.tolist())
    
    return all_experiences, all_advantages, all_returns


def experiences_to_batches(experiences: List[PPOExperience]) -> List[dict]:
    """Convert PPO experiences to training batches."""
    batch_data = []
    
    for exp in experiences:
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
        flat_dict['outcome'] = 1.0  # Dummy, not used in PPO
        
        batch_data.append(flat_dict)
    
    return batch_data


def ppo_train_step(nets, optimizers, experiences: List[PPOExperience], advantages: List[float], returns: List[float], config: PPOConfig):
    """Perform one PPO training step."""
    if not experiences:
        return {}
    
    # Determine if we have separate networks
    separate_networks = isinstance(nets, tuple)
    
    if separate_networks:
        policy_net, value_net = nets
        policy_optimizer, value_optimizer = optimizers
        device = policy_net.device
        # Set networks to training mode
        policy_net.train()
        value_net.train()
    else:
        net = nets
        optimizer = optimizers
        device = net.device
        # Set network to training mode
        net.train()
    
    # Convert experiences to batches
    batch_data = experiences_to_batches(experiences)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(torch.arange(len(batch_data)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_kl_div = 0
    total_grad_norm = 0
    total_clipfrac = 0
    num_batches = 0
    
    for epoch in range(config.num_epochs):
        for batch_indices in dataloader:
            batch_indices = batch_indices[0]
            mini_batch = [batch_data[i] for i in batch_indices]
            
            # Collate mini-batch
            collated_batch = collate_fn(mini_batch)
            
            # Move to device
            collated_batch = move_to_device(collated_batch, device)
            
            # Forward pass
            if separate_networks:
                # Get policy logits from policy network
                new_logits = policy_net(collated_batch)
                
                # Get values from value network  
                value_output = value_net(collated_batch)
                if isinstance(value_output, tuple):
                    _, new_values = value_output  # Extract values from tuple
                else:
                    new_values = torch.zeros(len(mini_batch), device=device)
            else:
                # Single network with value head
                output = net(collated_batch)
                if isinstance(output, tuple):
                    new_logits, new_values = output
                else:
                    new_logits = output
                    new_values = torch.zeros(len(mini_batch), device=device)
            
            # Get old log probs, advantages, target values
            old_log_probs = torch.tensor([exp.log_prob for exp in [experiences[i] for i in batch_indices]], 
                                       device=device, dtype=torch.float32)
            batch_advantages = torch.tensor([advantages[i] for i in batch_indices], 
                                    device=device, dtype=torch.float32)
            target_values = torch.tensor([returns[i] for i in batch_indices], 
                                       device=device, dtype=torch.float32)
            chosen_indices = torch.tensor([exp.action_idx for exp in [experiences[i] for i in batch_indices]], 
                                        device=device, dtype=torch.long)
            
            # Compute new log probabilities with numerical stability
            action_probs = F.softmax(new_logits, dim=-1)
            action_log_probs = F.log_softmax(new_logits, dim=-1)
            
            # Get log probs for chosen actions
            batch_size = len(mini_batch)
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
            
            # Backward pass
            if separate_networks:
                # Update policy network
                policy_loss_total = policy_loss - config.entropy_coef * entropy
                policy_optimizer.zero_grad()
                policy_loss_total.backward()
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), config.max_grad_norm)
                policy_optimizer.step()
                
                # Update value network
                value_optimizer.zero_grad()
                value_loss.backward()
                value_grad_norm = torch.nn.utils.clip_grad_norm_(value_net.parameters(), config.max_grad_norm)
                value_optimizer.step()
                
                # Average gradient norms for reporting
                grad_norm = (policy_grad_norm + value_grad_norm) / 2
            else:
                # Single network with combined loss
                total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
                optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
                optimizer.step()
            
            # Accumulate losses and metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl_div += kl_div.item()
            total_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            total_clipfrac += clipfrac.item()
            num_batches += 1
    
    return {
        'policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0,
        'value_loss': total_value_loss / num_batches if num_batches > 0 else 0,
        'entropy': total_entropy / num_batches if num_batches > 0 else 0,
        'kl_div': total_kl_div / num_batches if num_batches > 0 else 0,
        'grad_norm': total_grad_norm / num_batches if num_batches > 0 else 0,
        'clipfrac': total_clipfrac / num_batches if num_batches > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='PPO training for Slay the Spire')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--save-path', type=str, default='ppo_model.pt',
                        help='Path to save trained model')
    parser.add_argument('--reward-function', type=str, default='perfected_strike',
                        choices=['smooth', 'perfected_strike', 'victory', 'no_pstrikes'],
                        help='Reward function to use: smooth (sparse win/loss+floor), perfected_strike (dense card count), victory (sparse 0/1 win/loss), no_pstrikes (dense negative card count) (default: perfected_strike)')
    
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
    
    args = parser.parse_args()
    
    # Configure logging - default level is WARNING, set to INFO to see Perfected Strike logs
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.cache_size_limit = 16
    
    # Create PPOConfig from parsed arguments
    config_kwargs = {}
    for field in fields(PPOConfig):
        field_name = field.name.replace('_', '-')
        config_kwargs[field.name] = getattr(args, field_name.replace('-', '_'))
    config = PPOConfig(**config_kwargs)
    
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
        policy_hp = ModelHP(use_value_head=False)
        value_hp = ModelHP(use_value_head=True)
        
        policy_net = NN(policy_hp).to(device)
        value_net = NN(value_hp).to(device)
        
        policy_net = torch.compile(policy_net, mode="default")
        value_net = torch.compile(value_net, mode="default")
        
        if args.model_path:
            if config.resume_from > 0:
                # Load from specific iteration checkpoints
                policy_path = f"{args.model_path}.policy.iter_{config.resume_from}"
                value_path = f"{args.model_path}.value.iter_{config.resume_from}"
                try:
                    policy_state = torch.load(policy_path, map_location=device, weights_only=True)
                    value_state = torch.load(value_path, map_location=device, weights_only=True)
                    policy_net.load_state_dict(policy_state)
                    value_net.load_state_dict(value_state)
                    print(f"Resumed from iteration {config.resume_from}: loaded {policy_path} and {value_path}")
                except FileNotFoundError as e:
                    print(f"Could not find checkpoint files for iteration {config.resume_from}: {e}")
                    print("Loading from base model path instead")
                    state = torch.load(args.model_path, map_location=device, weights_only=True)
                    policy_net.load_state_dict(state, strict=False)
                    value_state = {k: v for k, v in state.items() if not k.startswith('value_head')}
                    value_net.load_state_dict(value_state, strict=False)
                    print(f"Loaded policy model from {args.model_path}")
            else:
                # Load from base model path
                state = torch.load(args.model_path, map_location=device, weights_only=True)
                policy_net.load_state_dict(state, strict=False)
                # Initialize value network with same weights (excluding value head)
                value_state = {k: v for k, v in state.items() if not k.startswith('value_head')}
                value_net.load_state_dict(value_state, strict=False)
                print(f"Loaded policy model from {args.model_path}")
        
        # Create separate optimizers
        policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.policy_lr)
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=config.value_lr)
        
        # Use policy network for action selection
        service = NNService(policy_net, batch_size=32, batch_size_factor=16)
        
        # Create separate value service
        value_service = NNService(value_net, batch_size=32, batch_size_factor=16)
        
        nets = (policy_net, value_net)
        optimizers = (policy_optimizer, value_optimizer)
        print("Using separate policy and value networks")
    else:
        # Create single network with value head
        model_hp = ModelHP(use_value_head=True)
        net = NN(model_hp).to(device)
        net = torch.compile(net, mode="default")
        
        if args.model_path:
            if config.resume_from > 0:
                # Load from specific iteration checkpoint
                checkpoint_path = f"{args.model_path}.iter_{config.resume_from}"
                try:
                    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    net.load_state_dict(state)
                    print(f"Resumed from iteration {config.resume_from}: loaded {checkpoint_path}")
                except FileNotFoundError as e:
                    print(f"Could not find checkpoint file for iteration {config.resume_from}: {e}")
                    print("Loading from base model path instead")
                    state = torch.load(args.model_path, map_location=device, weights_only=True)
                    net.load_state_dict(state, strict=False)  # Allow missing value head weights
                    print(f"Loaded model from {args.model_path}")
            else:
                # Load from base model path
                state = torch.load(args.model_path, map_location=device, weights_only=True)
                net.load_state_dict(state, strict=False)  # Allow missing value head weights
                print(f"Loaded model from {args.model_path}")
        
        # Create service
        service = NNService(net, batch_size=32, batch_size_factor=16)
        
        # Create optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=config.policy_lr)
        
        nets = net
        optimizers = optimizer
        value_service = None  # No separate value service needed
        print("Using single network with value head")
    
    if config.resume_from > 0:
        print(f"Resuming PPO training from iteration {config.resume_from} with {config.num_games_per_batch} games per batch")
    else:
        print(f"Starting PPO training with {config.num_games_per_batch} games per batch")
    
    try:
        for iteration in range(config.resume_from, config.num_iterations):
            print(f"\nIteration {iteration + 1}/{config.num_iterations}")
            
            # Collect experience
            start_time = time.time()
            # Pass value service if using separate networks
            trajectories = collect_experience(config, service, reward_fn, start_seed=iteration * 1000, value_service=value_service)
            collect_time = time.time() - start_time
            
            if not trajectories:
                print("No trajectories collected, skipping iteration")
                continue
            
            # Compute statistics
            win_rate = sum(1 for t in trajectories if t.final_reward >= 1.0) / len(trajectories)
            avg_floor = sum(t.final_floor for t in trajectories) / len(trajectories)
            avg_reward = sum(t.final_reward for t in trajectories) / len(trajectories)
            
            # Compute Perfected Strike statistics
            total_pstrike_offers = sum(t.pstrike_offers for t in trajectories)
            total_pstrike_takes = sum(t.pstrike_takes for t in trajectories)
            all_pstrike_probs = [prob for t in trajectories for prob in t.pstrike_probs]
            
            pstrike_take_rate = total_pstrike_takes / total_pstrike_offers if total_pstrike_offers > 0 else 0.0
            avg_pstrike_prob = np.mean(all_pstrike_probs) if all_pstrike_probs else 0.0
            
            print(f"Collected {len(trajectories)} trajectories in {collect_time:.1f}s")
            print(f"Win rate: {win_rate:.3f}, Avg floor: {avg_floor:.1f}, Avg reward: {avg_reward:.3f}")
            print(f"PStrike: {total_pstrike_offers} offers, {total_pstrike_takes} takes ({pstrike_take_rate:.3f} rate), {avg_pstrike_prob:.3f} avg prob")
            
            # Prepare training data (with debug output for first trajectory)
            experiences, advantages, returns = compute_advantages(trajectories, config, debug_first=True)
            
            if not experiences:
                print("No experiences to train on, skipping iteration")
                continue
            
            print(f"Training on {len(experiences)} experiences")
            
            # Perform PPO training step
            train_start = time.time()
            losses = ppo_train_step(nets, optimizers, experiences, advantages, returns, config)
            train_time = time.time() - train_start
            
            print(f"Training completed in {train_time:.1f}s")
            print(f"Policy loss: {losses.get('policy_loss', 0):.4f}, "
                  f"Value loss: {losses.get('value_loss', 0):.4f}, "
                  f"Entropy: {losses.get('entropy', 0):.4f}")
            print(f"KL div: {losses.get('kl_div', 0):.6f}, "
                  f"Grad norm: {losses.get('grad_norm', 0):.4f}, "
                  f"Clip frac: {losses.get('clipfrac', 0):.3f}")
            
            # Save model periodically
            if (iteration + 1) % config.save_every == 0:
                if config.separate_networks:
                    policy_net, value_net = nets
                    torch.save(policy_net.state_dict(), f"{args.save_path}.policy.iter_{iteration + 1}")
                    torch.save(value_net.state_dict(), f"{args.save_path}.value.iter_{iteration + 1}")
                    print(f"Saved separate network checkpoints at iteration {iteration + 1}")
                else:
                    torch.save(nets.state_dict(), f"{args.save_path}.iter_{iteration + 1}")
                    print(f"Saved model checkpoint at iteration {iteration + 1}")
    
    finally:
        service.stop()
        if value_service is not None:
            value_service.stop()
    
    # Save final model
    if config.separate_networks:
        policy_net, value_net = nets
        torch.save(policy_net.state_dict(), f"{args.save_path}.policy")
        torch.save(value_net.state_dict(), f"{args.save_path}.value")
        print(f"Saved final separate networks to {args.save_path}.policy and {args.save_path}.value")
    else:
        torch.save(nets.state_dict(), args.save_path)
        print(f"Saved final model to {args.save_path}")


if __name__ == "__main__":
    main()