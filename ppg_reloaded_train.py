#!/usr/bin/env python3
"""
PPG Reloaded: An improved implementation of Phasic Policy Gradient
Based on "PPG Reloaded: An Empirical Study on What Matters in Phasic Policy Gradient" (ICML 2023)

Key improvements over standard PPG:
1. Policy regularization and data diversity are the critical factors
2. Can achieve similar performance with lower value sample reuse 
3. More frequent feature distillation is acceptable
4. Reduced computational cost to PPO levels
"""

import argparse
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

import slaythespire as sts
from network import NN, ModelHP, choice_space, collate_fn, process_batch, output_to_cpu, move_to_device
from playouts import Choice, construct_choice, pick_card_with_net, flatten_dict


@dataclass
class PPGReloadedConfig:
    """Configuration for PPG Reloaded training"""
    # Basic training parameters
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPG-specific parameters
    n_policy_iterations: int = 4  # N_π - number of policy updates per auxiliary phase
    n_policy_epochs: int = 1      # E_π - epochs per policy update (low for on-policy)
    n_aux_epochs: int = 2         # E_aux - reduced from standard PPG (was 6)
    
    # PPG Reloaded improvements
    policy_reg_coef: float = 0.5  # β - KL regularization coefficient (key improvement)
    data_diversity_buffer_size: int = 10  # Store multiple rollout batches for diversity
    frequent_distillation: bool = True    # More frequent auxiliary phase
    
    # PPO parameters
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Environment parameters
    num_envs: int = 8
    rollout_length: int = 256
    batch_size: int = 2048  # num_envs * rollout_length
    minibatch_size: int = 512
    
    # Model parameters
    model_dim: int = 256
    model_layers: int = 4
    num_value_layers: int = 2     # Use separate value layers
    value_fork_layer: int = 1     # Fork one layer before the end
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    save_path: str = "ppg_reloaded_checkpoints"


class PPGReloadedTrainer:
    """PPG Reloaded trainer implementing the key insights from the paper"""
    
    def __init__(self, config: PPGReloadedConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        model_hp = ModelHP(
            dim=config.model_dim,
            n_layers=config.model_layers,
            use_value_head=True,
            num_value_layers=config.num_value_layers,
            value_fork_layer=config.value_fork_layer,
        )
        
        # Separate policy and value networks for PPG
        self.policy_net = NN(model_hp).to(self.device)
        
        # Value network with value head
        value_hp = ModelHP(
            dim=config.model_dim,
            n_layers=config.model_layers,
            use_value_head=True,
            num_value_layers=config.num_value_layers,
            value_fork_layer=config.value_fork_layer,
        )
        self.value_net = NN(value_hp).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate, eps=1e-8
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=config.learning_rate, eps=1e-8
        )
        
        # PPG Reloaded: Data diversity buffer
        self.experience_buffer = []
        
        # Logging
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(f"runs/ppg_reloaded_{int(time.time())}")
        else:
            self.writer = None
        self.global_step = 0
        self.update_count = 0
        
        os.makedirs(config.save_path, exist_ok=True)
    
    def run_episode(self, max_floor: int = 3) -> Tuple[List[Dict], float, bool, Dict]:
        """Run a single episode and collect experience"""
        gc = sts.GameContext()
        gc.seed = random.randint(0, 2**31 - 1)
        gc.character = sts.Character.IRONCLAD
        gc.ascension_level = 0
        
        episode_data = []
        episode_reward = 0.0
        
        try:
            while not gc.isGameOver() and gc.floor <= max_floor:
                state_obs = self.get_observation(gc)
                
                if not gc.isInANonTransientState():
                    gc.proceedGameContext()
                    continue
                
                actions = gc.getInformationInterfaceActions()
                if not actions:
                    gc.proceedGameContext()
                    continue
                
                choice = construct_choice(gc, actions)
                if choice is None:
                    action_idx = 0 if actions else None
                else:
                    # Get policy prediction
                    batch = self.prepare_batch_for_inference(state_obs, choice)
                    with torch.no_grad():
                        policy_logits, state_value = process_batch(batch, self.policy_net)
                    
                    # Sample action
                    action_probs = F.softmax(policy_logits[0], dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action_idx = action_dist.sample().item()
                    action_log_prob = action_dist.log_prob(torch.tensor(action_idx)).item()
                
                # Store experience
                experience = {
                    'observation': state_obs,
                    'choice': choice,
                    'action_idx': action_idx,
                    'action_log_prob': action_log_prob if choice else 0.0,
                    'state_value': state_value[0].item() if choice else 0.0,
                    'reward': 0.0,  # Will be filled later
                }
                episode_data.append(experience)
                
                # Execute action
                if action_idx is not None and action_idx < len(actions):
                    gc.doAction(actions[action_idx])
                else:
                    gc.proceedGameContext()
            
            # Calculate final reward
            if gc.isGameOver():
                victory = gc.outcome == sts.GameOutcome.VICTORY
                final_reward = 1.0 if victory else 0.0
                episode_reward = final_reward
                
                # Assign rewards (sparse reward at end)
                for exp in episode_data:
                    exp['reward'] = final_reward
                    
                return episode_data, episode_reward, victory, {'floor': gc.floor}
            else:
                # Episode truncated
                return episode_data, 0.0, False, {'floor': gc.floor}
                
        except Exception as e:
            print(f"Episode error: {e}")
            return [], 0.0, False, {'floor': 0}
    
    def get_observation(self, gc: sts.GameContext) -> Dict:
        """Extract observation from game context"""
        return {
            'obs.deck.cards': [int(card.id) for card in gc.deck],
            'obs.deck.upgrades': [card.upgrade for card in gc.deck],
            'obs.relics.relics': [int(relic.id) for relic in gc.relics],
            'obs.potions': [int(potion) for potion in gc.getPotions()],
            'obs.fixed_observation': list(gc.getFixedObservation()),
            'screen_state': int(gc.screenState),
            'obs.map.xs': gc.map.getXs(),
            'obs.map.ys': gc.map.getYs(),
            'obs.map.roomTypes': [int(room) for room in gc.map.getRoomTypes()],
            'obs.map.pathXs': gc.map.getPathXs(),
            'obs.mapX': gc.mapX,
            'obs.mapY': gc.mapY,
        }
    
    def prepare_batch_for_inference(self, obs: Dict, choice: Choice) -> Dict:
        """Prepare a single observation and choice for network inference"""
        # Convert to the format expected by collate_fn
        sample = {**obs}
        
        # Add choice data
        choice_dict = choice.as_dict()
        flattened_choice = flatten_dict(choice_dict)
        
        # Add required fields for collate_fn
        sample.update(flattened_choice)
        sample['chosen_idx'] = 0  # Dummy value
        sample['outcome'] = 0.0   # Dummy value
        
        # Use collate_fn to create proper batch format
        batch = collate_fn([sample])
        return batch
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        
        gae = 0.0
        next_value = 0.0  # Assume terminal state value is 0
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.config.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]
        
        return advantages, returns
    
    def policy_phase_update(self, batch_data: List[Dict]) -> Dict[str, float]:
        """Perform policy phase update (standard PPO)"""
        # Convert batch data to tensors
        observations = []
        choices = []
        actions = []
        old_log_probs = []
        advantages = []
        returns = []
        
        for episode in batch_data:
            episode_obs = []
            episode_choices = []
            episode_actions = []
            episode_log_probs = []
            episode_values = []
            episode_rewards = []
            episode_dones = []
            
            for exp in episode:
                if exp['choice'] is not None:
                    episode_obs.append(exp['observation'])
                    episode_choices.append(exp['choice'])
                    episode_actions.append(exp['action_idx'])
                    episode_log_probs.append(exp['action_log_prob'])
                    episode_values.append(exp['state_value'])
                    episode_rewards.append(exp['reward'])
                    episode_dones.append(False)  # Only last state is terminal
            
            if episode_dones:
                episode_dones[-1] = True
            
            if len(episode_actions) > 0:
                # Compute advantages for this episode
                ep_advantages, ep_returns = self.compute_gae(
                    episode_rewards, episode_values, episode_dones
                )
                
                observations.extend(episode_obs)
                choices.extend(episode_choices)
                actions.extend(episode_actions)
                old_log_probs.extend(episode_log_probs)
                advantages.extend(ep_advantages)
                returns.extend(ep_returns)
        
        if len(actions) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # PPO epochs
        for epoch in range(self.config.n_policy_epochs):
            # Shuffle data
            indices = torch.randperm(len(actions))
            
            for start_idx in range(0, len(actions), self.config.minibatch_size):
                end_idx = min(start_idx + self.config.minibatch_size, len(actions))
                mb_indices = indices[start_idx:end_idx]
                
                mb_observations = [observations[i] for i in mb_indices]
                mb_choices = [choices[i] for i in mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Prepare batch for network
                batch_samples = []
                for i, obs in enumerate(mb_observations):
                    sample = {**obs}
                    choice_dict = mb_choices[i].as_dict()
                    flattened_choice = flatten_dict(choice_dict)
                    sample.update(flattened_choice)
                    sample['chosen_idx'] = mb_actions[i].item()
                    sample['outcome'] = mb_returns[i].item()
                    batch_samples.append(sample)
                
                batch = collate_fn(batch_samples)
                
                # Forward pass
                policy_logits, values = process_batch(batch, self.policy_net)
                
                # Compute policy loss
                action_probs = F.softmax(policy_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(mb_actions)
                entropy = action_dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values.squeeze(), mb_returns)
                
                # Total loss
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                # Backward pass
                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        num_updates = len(actions) // self.config.minibatch_size * self.config.n_policy_epochs
        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
        }
    
    def auxiliary_phase_update(self) -> Dict[str, float]:
        """Perform auxiliary phase update with PPG Reloaded improvements"""
        if len(self.experience_buffer) == 0:
            return {'aux_value_loss': 0.0, 'kl_loss': 0.0}
        
        # PPG Reloaded: Use diverse data from buffer
        all_experiences = []
        for batch in self.experience_buffer:
            all_experiences.extend(batch)
        
        # Filter valid experiences
        valid_experiences = [exp for exp in all_experiences if exp['choice'] is not None]
        if len(valid_experiences) == 0:
            return {'aux_value_loss': 0.0, 'kl_loss': 0.0}
        
        total_aux_loss = 0.0
        total_kl_loss = 0.0
        
        # PPG Reloaded: Reduced auxiliary epochs but more frequent updates
        for epoch in range(self.config.n_aux_epochs):
            # Shuffle experiences
            random.shuffle(valid_experiences)
            
            for start_idx in range(0, len(valid_experiences), self.config.minibatch_size):
                end_idx = min(start_idx + self.config.minibatch_size, len(valid_experiences))
                mb_experiences = valid_experiences[start_idx:end_idx]
                
                # Prepare batch
                batch_samples = []
                old_policy_logits = []
                target_values = []
                
                for exp in mb_experiences:
                    sample = {**exp['observation']}
                    choice_dict = exp['choice'].as_dict()
                    flattened_choice = flatten_dict(choice_dict)
                    sample.update(flattened_choice)
                    sample['chosen_idx'] = exp['action_idx']
                    sample['outcome'] = exp['reward']
                    batch_samples.append(sample)
                    target_values.append(exp['reward'])  # Use actual reward as target
                
                batch = collate_fn(batch_samples)
                target_values = torch.tensor(target_values, dtype=torch.float32, device=self.device)
                
                # Get old policy logits for KL regularization
                with torch.no_grad():
                    old_logits, _ = process_batch(batch, self.policy_net)
                    old_policy_probs = F.softmax(old_logits, dim=-1)
                
                # Forward pass through value network
                _, aux_values = process_batch(batch, self.value_net)
                
                # Auxiliary value loss
                aux_value_loss = F.mse_loss(aux_values.squeeze(), target_values)
                
                # Update value network
                self.value_optimizer.zero_grad()
                aux_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()
                
                # PPG Reloaded: Policy regularization (key improvement)
                # Update policy network's auxiliary value head with KL regularization
                new_logits, new_aux_values = process_batch(batch, self.policy_net)
                new_policy_probs = F.softmax(new_logits, dim=-1)
                
                # KL divergence regularization
                kl_loss = F.kl_div(
                    F.log_softmax(new_logits, dim=-1),
                    old_policy_probs,
                    reduction='batchmean'
                )
                
                # Auxiliary value loss for policy network
                policy_aux_value_loss = F.mse_loss(new_aux_values.squeeze(), target_values)
                
                # Combined loss with strong policy regularization
                total_aux_policy_loss = (
                    policy_aux_value_loss + 
                    self.config.policy_reg_coef * kl_loss
                )
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                total_aux_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                total_aux_loss += aux_value_loss.item()
                total_kl_loss += kl_loss.item()
        
        num_updates = len(valid_experiences) // self.config.minibatch_size * self.config.n_aux_epochs
        return {
            'aux_value_loss': total_aux_loss / max(num_updates, 1),
            'kl_loss': total_kl_loss / max(num_updates, 1),
        }
    
    def train(self):
        """Main training loop for PPG Reloaded"""
        print(f"Starting PPG Reloaded training on {self.device}")
        print(f"Config: {self.config}")
        
        total_episodes = 0
        total_victories = 0
        
        while self.global_step < self.config.total_timesteps:
            # Collect experience batch
            print(f"Collecting experience batch {self.update_count + 1}")
            
            batch_episodes = []
            batch_rewards = []
            batch_victories = []
            
            # Collect episodes in parallel
            with ThreadPoolExecutor(max_workers=self.config.num_envs) as executor:
                futures = [executor.submit(self.run_episode) for _ in range(self.config.num_envs)]
                
                for future in as_completed(futures):
                    episode_data, episode_reward, victory, info = future.result()
                    
                    if len(episode_data) > 0:
                        batch_episodes.append(episode_data)
                        batch_rewards.append(episode_reward)
                        batch_victories.append(victory)
                        
                        total_episodes += 1
                        if victory:
                            total_victories += 1
                        
                        self.global_step += len(episode_data)
            
            if len(batch_episodes) == 0:
                continue
            
            # PPG Reloaded: Add to data diversity buffer
            self.experience_buffer.append(batch_episodes)
            if len(self.experience_buffer) > self.config.data_diversity_buffer_size:
                self.experience_buffer.pop(0)
            
            # Policy phase update
            policy_metrics = self.policy_phase_update(batch_episodes)
            
            # PPG Reloaded: More frequent auxiliary phase
            if (self.update_count + 1) % self.config.n_policy_iterations == 0 or self.config.frequent_distillation:
                aux_metrics = self.auxiliary_phase_update()
            else:
                aux_metrics = {'aux_value_loss': 0.0, 'kl_loss': 0.0}
            
            self.update_count += 1
            
            # Logging
            if self.update_count % self.config.log_interval == 0:
                avg_reward = np.mean(batch_rewards) if batch_rewards else 0.0
                win_rate = total_victories / max(total_episodes, 1)
                
                print(f"Update {self.update_count}, Step {self.global_step}")
                print(f"  Avg Reward: {avg_reward:.3f}")
                print(f"  Win Rate: {win_rate:.3f} ({total_victories}/{total_episodes})")
                print(f"  Policy Loss: {policy_metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {policy_metrics['value_loss']:.4f}")
                print(f"  Aux Value Loss: {aux_metrics['aux_value_loss']:.4f}")
                print(f"  KL Loss: {aux_metrics['kl_loss']:.4f}")
                
                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar('Environment/AverageReward', avg_reward, self.global_step)
                    self.writer.add_scalar('Environment/WinRate', win_rate, self.global_step)
                    self.writer.add_scalar('Loss/PolicyLoss', policy_metrics['policy_loss'], self.global_step)
                    self.writer.add_scalar('Loss/ValueLoss', policy_metrics['value_loss'], self.global_step)
                    self.writer.add_scalar('Loss/AuxValueLoss', aux_metrics['aux_value_loss'], self.global_step)
                    self.writer.add_scalar('Loss/KLLoss', aux_metrics['kl_loss'], self.global_step)
                    self.writer.add_scalar('Loss/Entropy', policy_metrics['entropy'], self.global_step)
            
            # Save checkpoint
            if self.update_count % self.config.save_interval == 0:
                self.save_checkpoint()
        
        print("Training completed!")
        self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'update_count': self.update_count,
        }
        
        save_path = os.path.join(self.config.save_path, f"checkpoint_{self.update_count}.pt")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PPG Reloaded Training")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--policy-reg-coef", type=float, default=0.5, help="Policy regularization coefficient")
    parser.add_argument("--frequent-distillation", action="store_true", help="Enable frequent auxiliary phase updates")
    parser.add_argument("--save-path", type=str, default="ppg_reloaded_checkpoints", help="Save path for checkpoints")
    
    args = parser.parse_args()
    
    config = PPGReloadedConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_envs=args.num_envs,
        policy_reg_coef=args.policy_reg_coef,
        frequent_distillation=args.frequent_distillation,
        save_path=args.save_path,
    )
    
    trainer = PPGReloadedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()