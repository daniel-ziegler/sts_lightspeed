#!/usr/bin/env python3

from __future__ import annotations

import random
import argparse
from dataclasses import dataclass
from typing import List, NamedTuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from network import NN, ModelHP, move_to_device, process_batch, action_logit_space, collate_fn
from playouts import run_game, NNService, Choice, Decision, ActionType, ChoiceStats
import slaythespire as sts


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
    gamma: float = 1.00
    gae_lambda: float = 0.95
    
    # Training settings
    num_iterations: int = 1000
    temperature: float = 1.0
    
    # Logging
    log_every: int = 10
    save_every: int = 100



class PPOExperience(NamedTuple):
    """Single step of experience from a game."""
    choice: Choice
    action_idx: int
    log_prob: float
    value: float
    reward: float


class PPOTrajectory(NamedTuple):
    """Complete game trajectory."""
    experiences: List[PPOExperience]
    final_reward: float
    final_floor: int


def compute_smooth_reward(outcome: sts.GameOutcome, final_floor: int) -> float:
    """Compute smooth reward based on game outcome and floor reached."""
    if outcome == sts.GameOutcome.PLAYER_VICTORY:
        return 1.0
    else:
        # Losses get partial reward based on floor progress (0.0 to 0.5)
        return min(0.5, final_floor / 100.0)


def run_ppo_episode(seed: int, service: NNService, temperature: float = 1.0) -> PPOTrajectory:
    """Run a complete game episode and collect experience for PPO training."""
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    rng = random.Random(seed)
    
    agent = sts.Agent()
    agent.simulation_count_base = 1000
    experiences = []
    
    # Create timeout handling
    timeout_event = threading.Event()
    
    def timeout_handler():
        timeout_event.set()
        print(f"Warning: Battle simulation taking too long for seed {seed}")
    
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
                    print(f"Seed {seed} did finish")
                    timeout_event.clear()
                    
            else:
                # Use neural network for non-battle decisions
                obs = sts.getNNRepresentation(gc)
                actions = sts.GameAction.getAllActionsInState(gc)
                
                if gc.screen_state in (sts.ScreenState.REWARDS, sts.ScreenState.SHOP_ROOM, sts.ScreenState.BOSS_RELIC_REWARDS):
                    from playouts import construct_choice
                    choice = construct_choice(gc, obs, actions)
                    
                    if choice.cards_offered or choice.relics_offered or choice.potions_offered or choice.fixed_actions:
                        # Get network predictions
                        batch_tensors, output = service.get_logits(choice)
                        
                        # Handle value head output
                        if isinstance(output, tuple):
                            logits, values = output
                            value = float(values) if np.isscalar(values) else float(values[0])
                        else:
                            logits = output
                            value = 0.0  # No value head
                        
                        # Convert to probabilities and sample action
                        probs = 1 / (1 + np.exp(-logits))  # sigmoid
                        probs = probs / np.sum(probs)  # normalize
                        
                        # Boltzmann sampling
                        boltz_logits = np.log(np.maximum(probs, 1e-20)) / temperature
                        boltz_logits = boltz_logits - np.max(boltz_logits)
                        exp_logits = np.exp(boltz_logits)
                        boltz_probs = exp_logits / np.sum(exp_logits)
                        
                        chosen_idx = int(rng.choices(range(len(probs)), weights=boltz_probs, k=1)[0])
                        log_prob = np.log(np.maximum(boltz_probs[chosen_idx], 1e-20))
                        
                        # Convert back to game action
                        path = action_logit_space.ix_to_path(batch_tensors['choices'], chosen_idx)
                        
                        if path[0] == 'cards':
                            action = choice.card_actions[path[1]]
                        elif path[0] == 'relics':
                            action = choice.relic_actions[path[1]]
                        elif path[0] == 'potions':
                            action = choice.potion_actions[path[1]]
                        elif path[0] == 'fixed':
                            action = choice.fixed_actions_list[path[1]]
                        else:
                            raise ValueError(f"Unknown path: {path}")
                        
                        # Store experience (reward will be filled in later)
                        exp = PPOExperience(
                            choice=choice,
                            action_idx=chosen_idx,
                            log_prob=log_prob,
                            value=value,
                            reward=0.0  # Will be filled in with final reward
                        )
                        experiences.append(exp)
                    else:
                        action = agent.pick_gameaction(gc)
                else:
                    action = agent.pick_gameaction(gc)
                
                assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                action.execute(gc)
                
        except Exception as e:
            print(f"Error in episode {seed}: {e}")
            raise
    
    # Compute final reward
    final_reward = compute_smooth_reward(gc.outcome, gc.floor_num)
    
    # Fill in rewards for all experiences
    filled_experiences = []
    for exp in experiences:
        filled_exp = PPOExperience(
            choice=exp.choice,
            action_idx=exp.action_idx,
            log_prob=exp.log_prob,
            value=exp.value,
            reward=final_reward
        )
        filled_experiences.append(filled_exp)
    
    return PPOTrajectory(
        experiences=filled_experiences,
        final_reward=final_reward,
        final_floor=gc.floor_num
    )


def collect_experience(config: PPOConfig, service: NNService, start_seed: int = 0) -> List[PPOTrajectory]:
    """Collect experience from multiple game episodes."""
    trajectories = []
    
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [
            executor.submit(run_ppo_episode, start_seed + i, service, config.temperature)
            for i in range(config.num_games_per_batch)
        ]
        
        for future in tqdm(as_completed(futures), total=config.num_games_per_batch, desc="Collecting experience"):
            try:
                trajectory = future.result()
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Failed to collect trajectory: {e}")
    
    return trajectories


def compute_advantages(trajectories: List[PPOTrajectory], config: PPOConfig) -> List[PPOExperience]:
    """Compute advantages using GAE and prepare training data."""
    all_experiences = []
    
    for traj in trajectories:
        if not traj.experiences:
            continue
            
        # Sparse reward: only final step gets reward, others get 0
        values = np.array([exp.value for exp in traj.experiences] + [0.0])  # Add bootstrap
        rewards = np.zeros(len(traj.experiences))
        if len(rewards) > 0:
            rewards[-1] = traj.final_reward  # Only final step gets reward
        
        # Compute returns and advantages using GAE
        advantages = np.zeros(len(traj.experiences))
        returns = np.zeros(len(traj.experiences))
        
        gae = 0
        for t in reversed(range(len(traj.experiences))):
            delta = rewards[t] + config.gamma * values[t + 1] - values[t]
            gae = delta + config.gamma * config.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        if len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / adv_std
            else:
                advantages = advantages - adv_mean
        
        # Create new experiences with advantages and returns
        for i, exp in enumerate(traj.experiences):
            new_exp = PPOExperience(
                choice=exp.choice,
                action_idx=exp.action_idx,
                log_prob=exp.log_prob,
                value=returns[i],  # Use return as target value
                reward=advantages[i]  # Use advantage as reward
            )
            all_experiences.append(new_exp)
    
    return all_experiences


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
        flat_dict['advantage'] = exp.reward  # We stored advantage in reward field
        flat_dict['target_value'] = exp.value  # We stored return in value field
        flat_dict['outcome'] = 1.0  # Dummy, not used in PPO
        
        batch_data.append(flat_dict)
    
    return batch_data


def ppo_train_step(net: NN, optimizer: torch.optim.Optimizer, experiences: List[PPOExperience], config: PPOConfig):
    """Perform one PPO training step."""
    if not experiences:
        return {}
    
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
            collated_batch = move_to_device(collated_batch, net.device)
            
            # Forward pass
            output = net(collated_batch)
            if isinstance(output, tuple):
                new_logits, new_values = output
            else:
                new_logits = output
                new_values = torch.zeros(len(mini_batch), device=net.device)
            
            # Get old log probs, advantages, target values
            old_log_probs = torch.tensor([exp.log_prob for exp in [experiences[i] for i in batch_indices]], 
                                       device=net.device, dtype=torch.float32)
            advantages = torch.tensor([exp.reward for exp in [experiences[i] for i in batch_indices]], 
                                    device=net.device, dtype=torch.float32)
            target_values = torch.tensor([exp.value for exp in [experiences[i] for i in batch_indices]], 
                                       device=net.device, dtype=torch.float32)
            chosen_indices = torch.tensor([exp.action_idx for exp in [experiences[i] for i in batch_indices]], 
                                        device=net.device, dtype=torch.long)
            
            # Compute new log probabilities with numerical stability
            action_probs = F.softmax(new_logits, dim=-1)
            action_log_probs = F.log_softmax(new_logits, dim=-1)
            
            # Get log probs for chosen actions
            batch_size = len(mini_batch)
            batch_indices_tensor = torch.arange(batch_size, device=net.device)
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
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantages
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
                    batch_entropies.append(torch.tensor(0.0, device=net.device))
            
            if batch_entropies:
                entropy = torch.stack(batch_entropies).mean()
            else:
                entropy = torch.tensor(0.0, device=net.device)
            
            # Optional debug (uncomment for debugging)
            # print(f"Valid actions per sample: {valid_mask.sum(dim=1).float().mean().item():.1f}")
            # print(f"Computed entropy: {entropy.item():.6f}")
            
            # Check for NaN before combining losses
            if torch.isnan(policy_loss) or torch.isnan(value_loss) or torch.isnan(entropy):
                print(f"NaN detected - policy: {policy_loss.item()}, value: {value_loss.item()}, entropy: {entropy.item()}")
                print(f"Ratio stats - min: {ratio.min().item()}, max: {ratio.max().item()}, mean: {ratio.mean().item()}")
                print(f"Advantages stats - min: {advantages.min().item()}, max: {advantages.max().item()}, mean: {advantages.mean().item()}")
                continue  # Skip this batch
            
            # Total loss
            total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            
            # Backward pass
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
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of training iterations')
    parser.add_argument('--games-per-batch', type=int, default=256,
                        help='Number of games per training batch')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    config = PPOConfig(
        num_games_per_batch=args.games_per_batch,
        num_iterations=args.iterations
    )
    
    # Create network with value head
    model_hp = ModelHP(use_value_head=True)
    net = NN(model_hp).to(device)
    net = torch.compile(net, mode="default")
    
    if args.model_path:
        state = torch.load(args.model_path, map_location=device, weights_only=True)
        net.load_state_dict(state, strict=False)  # Allow missing value head weights
        print(f"Loaded model from {args.model_path}")
    
    # Create service
    service = NNService(net, batch_size=32, batch_size_factor=16)
    
    # Create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.policy_lr)
    
    print(f"Starting PPO training with {config.num_games_per_batch} games per batch")
    
    try:
        for iteration in range(config.num_iterations):
            print(f"\nIteration {iteration + 1}/{config.num_iterations}")
            
            # Collect experience
            start_time = time.time()
            trajectories = collect_experience(config, service, start_seed=iteration * 1000)
            collect_time = time.time() - start_time
            
            if not trajectories:
                print("No trajectories collected, skipping iteration")
                continue
            
            # Compute statistics
            win_rate = sum(1 for t in trajectories if t.final_reward >= 1.0) / len(trajectories)
            avg_floor = sum(t.final_floor for t in trajectories) / len(trajectories)
            avg_reward = sum(t.final_reward for t in trajectories) / len(trajectories)
            
            print(f"Collected {len(trajectories)} trajectories in {collect_time:.1f}s")
            print(f"Win rate: {win_rate:.3f}, Avg floor: {avg_floor:.1f}, Avg reward: {avg_reward:.3f}")
            
            # Prepare training data
            experiences = compute_advantages(trajectories, config)
            
            if not experiences:
                print("No experiences to train on, skipping iteration")
                continue
            
            print(f"Training on {len(experiences)} experiences")
            
            # Perform PPO training step
            train_start = time.time()
            losses = ppo_train_step(net, optimizer, experiences, config)
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
                torch.save(net.state_dict(), f"{args.save_path}.iter_{iteration + 1}")
                print(f"Saved model checkpoint at iteration {iteration + 1}")
    
    finally:
        service.stop()
    
    # Save final model
    torch.save(net.state_dict(), args.save_path)
    print(f"Saved final model to {args.save_path}")


if __name__ == "__main__":
    main()