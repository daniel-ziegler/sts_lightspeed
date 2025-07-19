#!/usr/bin/env python3
"""
Unified PPO and PPG training for Slay the Spire.

This trainer supports both algorithms:
- PPO: Set n_policy_iterations=0, n_aux_epochs=0 (no auxiliary phase)
- PPG: Set n_policy_iterations>0, n_aux_epochs>0 (full PPG with auxiliary phase)
"""

import argparse
import time
import random
from dataclasses import dataclass, fields
from typing import List, Dict, Any, Optional, get_type_hints, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import slaythespire as sts
from network import NN, ModelHP, move_to_device, process_batch, load_network_backward_compatible, SeparateValuePolicy
from playouts import NNService
from rl_common import (
    Experience, Trajectory, GameMetrics, REWARD_FUNCTIONS,
    collect_experience, compute_advantages_for_trajectories, experiences_to_batches,
    save_checkpoint, load_checkpoint, create_ppo_collate_fn, log_training_stats
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


@dataclass
class RLConfig:
    """Unified configuration for PPO and PPG training."""
    
    # Algorithm selection
    n_policy_iterations: int = 0      # 0 = PPO mode, >0 = PPG mode
    n_aux_epochs: int = 0             # 0 = no auxiliary phase, >0 = PPG auxiliary epochs
    
    # Environment settings
    num_games_per_step: int = 256
    num_workers: int = 40
    max_floor: int | None = None
    
    # Training settings
    num_iterations: int = 1000
    batch_size: int = 128
    
    # PPO/PPG shared parameters
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 5.0
    
    # PPO-specific parameters
    num_epochs: int = 4               # Policy phase epochs
    
    # PPG-specific parameters  
    policy_reg_coef: float = 0.5      # KL regularization coefficient for auxiliary phase
    behavioral_cloning_coef: float = 1.0  # Behavioral cloning coefficient for auxiliary phase
    data_diversity_buffer_size: int = 10  # Number of trajectory batches to keep for diversity
    
    # PPG Reloaded enhancements
    adaptive_kl_reg: bool = False     # Enable adaptive KL regularization
    kl_target: float = 0.01           # Target KL divergence for adaptive regularization
    kl_adapt_rate: float = 1.5        # Adaptation rate for KL coefficient
    reduced_aux_frequency: bool = True  # Use reduced auxiliary phase frequency for efficiency
    
    # Network settings
    separate_networks: bool = False    # Use separate policy and value networks
    inf_batch_size: int = 32
    inf_batch_size_factor: int = 16
    
    # Model hyperparameters
    model_dim: int = 256
    model_layers: int = 4
    num_value_layers: int = 0
    value_fork_layer: int = 0
    use_value_head: bool = True
    
    # Training control
    resume_from_step: int = 0
    save_path: str = "model"
    
    # Logging
    log_every: int = 10
    save_every: int = 20
    
    @property
    def is_ppo_mode(self) -> bool:
        """Check if configured for PPO mode (no auxiliary phase)."""
        return self.n_policy_iterations == 0 and self.n_aux_epochs == 0
    
    @property
    def is_ppg_mode(self) -> bool:
        """Check if configured for PPG mode (with auxiliary phase)."""
        return self.n_policy_iterations > 0 and self.n_aux_epochs > 0


class UnifiedTrainer:
    """Unified trainer supporting both PPO and PPG algorithms."""
    
    def __init__(self, config: RLConfig, reward_fn, torch_compile_mode: str = 'default'):
        self.config = config
        self.reward_fn = reward_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self._setup_networks()
        
        # Initialize optimizer
        if self.config.separate_networks:
            self.optimizer = torch.optim.AdamW(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # Initialize service
        self.service = NNService(
            self.net, 
            batch_size=config.inf_batch_size,
            batch_size_factor=config.inf_batch_size_factor,
            torch_compile_mode=torch_compile_mode
        )

        # Compile after setting up the service so the state dicts are consistent
        if torch_compile_mode != 'no':
            self.net = torch.compile(self.net, mode=torch_compile_mode)
        
        # PPG-specific: trajectory buffer for data diversity
        self.trajectory_buffer = []
        
        # Logging
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(f"runs/unified_rl_{int(time.time())}")
        else:
            self.writer = None
        
        # Training state
        self.iteration = config.resume_from_step
        
        # PPG Reloaded: adaptive KL regularization
        self.current_kl_coef = config.policy_reg_coef
        
        print(f"Initialized trainer in {'PPG' if config.is_ppg_mode else 'PPO'} mode on {self.device}")
    
    def _setup_networks(self):
        """Initialize networks based on configuration."""
        model_hp = ModelHP(
            dim=self.config.model_dim,
            n_layers=self.config.model_layers,
            use_value_head=self.config.use_value_head,
            num_value_layers=self.config.num_value_layers,
            value_fork_layer=self.config.value_fork_layer,
        )
        
        if self.config.separate_networks:
            # Create separate policy and value networks
            # For PPG, policy network needs auxiliary value head for joint training
            policy_needs_value_head = self.config.is_ppg_mode
            policy_hp = ModelHP(
                dim=self.config.model_dim,
                n_layers=self.config.model_layers,
                use_value_head=policy_needs_value_head,  # PPG needs auxiliary value head
                num_value_layers=self.config.num_value_layers,
                value_fork_layer=self.config.value_fork_layer,
            )
            value_hp = ModelHP(
                dim=self.config.model_dim,
                n_layers=self.config.model_layers,
                use_value_head=True,   # Value network with value head
                num_value_layers=self.config.num_value_layers,
                value_fork_layer=self.config.value_fork_layer,
            )
            
            self.policy_net = NN(policy_hp).to(self.device)
            self.value_net = NN(value_hp).to(self.device)
            self.net = SeparateValuePolicy(self.policy_net, self.value_net)
            self.nets = (self.policy_net, self.value_net)
        else:
            # Single network with value head
            self.net = NN(model_hp).to(self.device)
            self.nets = self.net
    
    def policy_phase_update(self, experiences: List[Experience], advantages: List[float], returns: List[float]) -> Dict[str, float]:
        """Perform policy phase update (PPO training step)."""
        if not experiences:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Convert experiences to batches
        batch_data = experiences_to_batches(experiences, advantages, returns)
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            batch_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_ppo_collate_fn(),
            num_workers=2,
        )
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl_div = 0.0
        total_clipfrac = 0.0
        num_batches = 0
        
        # Set networks to training mode
        if self.config.separate_networks:
            self.policy_net.train()
            self.value_net.train()
        else:
            self.net.train()
        
        for epoch in range(self.config.num_epochs):
            for collated_batch in dataloader:
                # Move to device
                collated_batch = move_to_device(collated_batch, self.device)
                
                # Forward pass
                if self.config.separate_networks:
                    # Get policy logits from policy network
                    policy_output = self.policy_net(collated_batch)
                    if isinstance(policy_output, tuple):
                        # PPG mode: policy network has auxiliary value head
                        new_logits, _ = policy_output  # Ignore aux values in policy phase
                    else:
                        # PPO mode: policy network only outputs logits
                        new_logits = policy_output
                    
                    # Get values from value network
                    value_output = self.value_net(collated_batch)
                    if isinstance(value_output, tuple):
                        _, new_values = value_output
                    else:
                        new_values = torch.zeros(len(collated_batch['chosen_idx']), device=self.device)
                else:
                    # Single network with value head
                    output = self.net(collated_batch)
                    if isinstance(output, tuple):
                        new_logits, new_values = output
                    else:
                        new_logits = output
                        new_values = torch.zeros(len(collated_batch['chosen_idx']), device=self.device)
                
                # Get training data from batch
                old_log_probs = collated_batch['old_log_prob']
                batch_advantages = collated_batch['advantage']
                target_values = collated_batch['return']
                chosen_indices = collated_batch['chosen_idx']
                
                # Compute new log probabilities
                action_probs = F.softmax(new_logits, dim=-1)
                action_log_probs = F.log_softmax(new_logits, dim=-1)
                
                # Get log probs for chosen actions
                batch_size = len(chosen_indices)
                batch_indices_tensor = torch.arange(batch_size, device=self.device)
                new_log_probs = action_log_probs[batch_indices_tensor, chosen_indices]
                
                # Clamp for numerical stability
                new_log_probs = torch.clamp(new_log_probs, min=-20, max=20)
                old_log_probs = torch.clamp(old_log_probs, min=-20, max=20)
                
                # Compute probability ratios
                ratio = torch.exp(new_log_probs - old_log_probs)
                ratio = torch.clamp(ratio, min=1e-8, max=1e8)
                
                # PPO clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, target_values)
                
                # Entropy bonus
                valid_mask = ~torch.isinf(new_logits)
                batch_entropies = []
                for i in range(new_logits.shape[0]):
                    valid_actions = valid_mask[i]
                    if valid_actions.sum() > 1:
                        valid_probs = action_probs[i][valid_actions]
                        valid_log_probs = action_log_probs[i][valid_actions]
                        sample_entropy = -(valid_probs * valid_log_probs).sum()
                        batch_entropies.append(sample_entropy)
                    else:
                        batch_entropies.append(torch.tensor(0.0, device=self.device))
                
                entropy = torch.stack(batch_entropies).mean() if batch_entropies else torch.tensor(0.0, device=self.device)
                
                # KL divergence for monitoring
                kl_div = (old_log_probs - new_log_probs).mean()
                
                # Clipping fraction for monitoring
                clipfrac = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()
                
                # Combined loss
                total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.config.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl_div += kl_div.item()
                total_clipfrac += clipfrac.item()
                num_batches += 1
        
        return {
            'policy_loss': total_policy_loss / max(num_batches, 1),
            'value_loss': total_value_loss / max(num_batches, 1),
            'entropy': total_entropy / max(num_batches, 1),
            'kl_div': total_kl_div / max(num_batches, 1),
            'clipfrac': total_clipfrac / max(num_batches, 1),
        }
    
    def auxiliary_phase_update(self) -> Dict[str, float]:
        """Perform auxiliary phase update (PPG-specific)."""
        if not self.config.is_ppg_mode or len(self.trajectory_buffer) == 0:
            return {'aux_value_loss': 0.0, 'kl_loss': 0.0, 'bc_loss': 0.0}
        
        # Collect all trajectories from buffer for data diversity
        all_trajectories = []
        for trajectory_batch in self.trajectory_buffer:
            all_trajectories.extend(trajectory_batch)
        
        if len(all_trajectories) == 0:
            return {'aux_value_loss': 0.0, 'kl_loss': 0.0, 'bc_loss': 0.0}
        
        # Compute advantages for auxiliary phase trajectories
        all_experiences, advantages, returns = compute_advantages_for_trajectories(
            all_trajectories, self.config.gamma, self.config.gae_lambda
        )
        
        if len(all_experiences) == 0:
            return {'aux_value_loss': 0.0, 'kl_loss': 0.0, 'bc_loss': 0.0}
        
        # Convert to training format
        batch_data = experiences_to_batches(all_experiences, advantages, returns)
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            batch_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_ppo_collate_fn(),
            num_workers=2,
        )
        
        total_aux_value_loss = 0.0
        total_kl_loss = 0.0
        total_bc_loss = 0.0
        num_batches = 0
        
        # Set networks to training mode
        if self.config.separate_networks:
            self.policy_net.train()
            self.value_net.train()
        else:
            self.net.train()
        
        for epoch in range(self.config.n_aux_epochs):
            for collated_batch in dataloader:
                # Move to device
                collated_batch = move_to_device(collated_batch, self.device)
                
                # Get old policy logits for KL regularization
                with torch.no_grad():
                    if self.config.separate_networks:
                        old_logits = self.policy_net(collated_batch)
                    else:
                        output = self.net(collated_batch)
                        if isinstance(output, tuple):
                            old_logits, _ = output
                        else:
                            old_logits = output
                    old_policy_probs = F.softmax(old_logits, dim=-1)
                
                # Forward pass for auxiliary phase
                target_values = collated_batch['return']
                old_log_probs = collated_batch['old_log_prob']
                chosen_indices = collated_batch['chosen_idx']
                
                if self.config.separate_networks:
                    # PPG: Joint training of both networks
                    # Get value network output
                    value_output = self.value_net(collated_batch)
                    if isinstance(value_output, tuple):
                        _, value_net_values = value_output
                    else:
                        value_net_values = torch.zeros(len(chosen_indices), device=self.device)
                    
                    # Get policy network output (includes auxiliary value head in PPG mode)
                    policy_output = self.policy_net(collated_batch)
                    if isinstance(policy_output, tuple):
                        new_logits, policy_aux_values = policy_output
                    else:
                        new_logits = policy_output
                        policy_aux_values = torch.zeros(len(chosen_indices), device=self.device)
                    
                    # Value losses: train both value network and policy auxiliary head
                    value_net_loss = F.mse_loss(value_net_values, target_values)
                    if policy_aux_values is not None and policy_aux_values.numel() > 0:
                        policy_aux_loss = F.mse_loss(policy_aux_values, target_values)
                    else:
                        policy_aux_loss = torch.tensor(0.0, device=self.device)
                    
                    aux_value_loss = value_net_loss + policy_aux_loss
                else:
                    # Single network
                    output = self.net(collated_batch)
                    if isinstance(output, tuple):
                        new_logits, aux_values = output
                    else:
                        new_logits = output
                        aux_values = torch.zeros(len(chosen_indices), device=self.device)
                    
                    # Auxiliary value loss
                    aux_value_loss = F.mse_loss(aux_values, target_values)
                
                # KL regularization to prevent policy drift
                new_policy_probs = F.softmax(new_logits, dim=-1)
                kl_loss = F.kl_div(
                    F.log_softmax(new_logits, dim=-1),
                    old_policy_probs,
                    reduction='batchmean'
                )
                
                # PPG Reloaded: Adaptive KL regularization
                if self.config.adaptive_kl_reg:
                    # Adapt KL coefficient based on measured KL divergence
                    kl_value = kl_loss.item()
                    if kl_value > self.config.kl_target:
                        # KL too high, increase regularization
                        self.current_kl_coef *= self.config.kl_adapt_rate
                    elif kl_value < self.config.kl_target * 0.5:
                        # KL too low, decrease regularization
                        self.current_kl_coef /= self.config.kl_adapt_rate
                    
                    # Clamp to reasonable bounds
                    self.current_kl_coef = torch.clamp(torch.tensor(self.current_kl_coef), 0.1, 10.0).item()
                    
                    effective_kl_coef = self.current_kl_coef
                else:
                    effective_kl_coef = self.config.policy_reg_coef
                
                # Behavioral cloning loss (PPG component)
                # Compute log probs for chosen actions
                action_log_probs = F.log_softmax(new_logits, dim=-1)
                batch_size = len(chosen_indices)
                batch_indices_tensor = torch.arange(batch_size, device=self.device)
                new_log_probs = action_log_probs[batch_indices_tensor, chosen_indices]
                
                # BC loss: minimize negative log likelihood of old actions
                bc_loss = -new_log_probs.mean()
                
                # Combined auxiliary loss (PPG Reloaded formulation)
                total_aux_loss = (aux_value_loss + 
                                effective_kl_coef * kl_loss + 
                                self.config.behavioral_cloning_coef * bc_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_aux_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.config.max_grad_norm)
                self.optimizer.step()
                
                total_aux_value_loss += aux_value_loss.item()
                total_kl_loss += kl_loss.item()
                total_bc_loss += bc_loss.item()
                num_batches += 1
        
        metrics = {
            'aux_value_loss': total_aux_value_loss / max(num_batches, 1),
            'kl_loss': total_kl_loss / max(num_batches, 1),
            'bc_loss': total_bc_loss / max(num_batches, 1),
        }
        
        # Add adaptive KL coefficient to metrics
        if self.config.adaptive_kl_reg:
            metrics['adaptive_kl_coef'] = self.current_kl_coef
        
        return metrics
    
    def train(self):
        """Main training loop supporting both PPO and PPG."""
        print(f"Starting {'PPG' if self.config.is_ppg_mode else 'PPO'} training")
        print(f"Config: iterations={self.config.num_iterations}, games_per_step={self.config.num_games_per_step}")
        
        try:
            for iteration in range(self.config.resume_from_step, self.config.num_iterations):
                self.iteration = iteration
                print(f"\nIteration {iteration + 1}/{self.config.num_iterations}")
                
                # Collect experience
                start_time = time.time()
                self.service.update_weights(self.net)
                trajectories = collect_experience(
                    num_games=self.config.num_games_per_step,
                    num_workers=self.config.num_workers,
                    service=self.service,
                    reward_fn=self.reward_fn,
                    start_seed=iteration * 1000,
                    max_floor=self.config.max_floor
                )
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
                
                # Prepare training data
                experiences, advantages, returns = compute_advantages_for_trajectories(
                    trajectories, self.config.gamma, self.config.gae_lambda
                )
                
                if not experiences:
                    print("No experiences to train on, skipping iteration")
                    continue
                
                print(f"Training on {len(experiences)} experiences")
                
                # Policy phase update
                train_start = time.time()
                policy_metrics = self.policy_phase_update(experiences, advantages, returns)
                
                # PPG: Add trajectories to buffer and run auxiliary phase
                if self.config.is_ppg_mode:
                    # Add trajectories to buffer for data diversity (preserving trajectory structure)
                    self.trajectory_buffer.append(trajectories)
                    if len(self.trajectory_buffer) > self.config.data_diversity_buffer_size:
                        self.trajectory_buffer.pop(0)
                    
                    # Run auxiliary phase every N policy iterations
                    # PPG Reloaded: Use reduced frequency for computational efficiency
                    aux_frequency = self.config.n_policy_iterations
                    if self.config.reduced_aux_frequency:
                        # Double the frequency for efficiency (run less often)
                        aux_frequency = max(1, self.config.n_policy_iterations * 2)
                    
                    if (iteration + 1) % aux_frequency == 0:
                        aux_metrics = self.auxiliary_phase_update()
                    else:
                        aux_metrics = {'aux_value_loss': 0.0, 'kl_loss': 0.0, 'bc_loss': 0.0}
                else:
                    aux_metrics = {'aux_value_loss': 0.0, 'kl_loss': 0.0, 'bc_loss': 0.0}
                
                train_time = time.time() - train_start
                
                # Logging
                print(f"Training completed in {train_time:.1f}s")
                print(f"Policy loss: {policy_metrics['policy_loss']:.4f}, "
                      f"Value loss: {policy_metrics['value_loss']:.4f}, "
                      f"Entropy: {policy_metrics['entropy']:.4f}")
                
                if self.config.is_ppg_mode:
                    print(f"Aux value loss: {aux_metrics['aux_value_loss']:.4f}, "
                          f"KL loss: {aux_metrics['kl_loss']:.4f}, "
                          f"BC loss: {aux_metrics['bc_loss']:.4f}")
                
                # Create stats
                stats = {
                    'iteration': iteration + 1,
                    'num_trajectories': len(trajectories),
                    'collect_time': collect_time,
                    'win_rate': win_rate,
                    'avg_floor': avg_floor,
                    'avg_reward': avg_reward,
                    'num_experiences': len(experiences),
                    'train_time': train_time,
                    **policy_metrics,
                    **aux_metrics,
                }
                
                # Log to file
                log_training_stats(stats, self.config.save_path)
                
                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar('Environment/WinRate', win_rate, iteration + 1)
                    self.writer.add_scalar('Environment/AvgFloor', avg_floor, iteration + 1)
                    self.writer.add_scalar('Environment/AvgReward', avg_reward, iteration + 1)
                    self.writer.add_scalar('Loss/PolicyLoss', policy_metrics['policy_loss'], iteration + 1)
                    self.writer.add_scalar('Loss/ValueLoss', policy_metrics['value_loss'], iteration + 1)
                    self.writer.add_scalar('Loss/Entropy', policy_metrics['entropy'], iteration + 1)
                    if self.config.is_ppg_mode:
                        self.writer.add_scalar('Loss/AuxValueLoss', aux_metrics['aux_value_loss'], iteration + 1)
                        self.writer.add_scalar('Loss/KLLoss', aux_metrics['kl_loss'], iteration + 1)
                        self.writer.add_scalar('Loss/BCLoss', aux_metrics['bc_loss'], iteration + 1)
                        if 'adaptive_kl_coef' in aux_metrics:
                            self.writer.add_scalar('PPG/AdaptiveKLCoef', aux_metrics['adaptive_kl_coef'], iteration + 1)
                
                # Save checkpoint
                if (iteration + 1) % self.config.save_every == 0:
                    save_checkpoint(self.nets, self.optimizer, self.config, iteration + 1, self.config.save_path)
        
        finally:
            self.service.stop()
        
        print("Training completed!")
        save_checkpoint(self.nets, self.optimizer, self.config, self.iteration + 1, self.config.save_path)


def main():
    parser = argparse.ArgumentParser(description='Unified PPO/PPG training for Slay the Spire')
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'ppg'],
                        help='Algorithm to use: ppo or ppg')
    
    # Basic training settings
    parser.add_argument('--init-path', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--reward-function', type=str, default='victory',
                        choices=list(REWARD_FUNCTIONS.keys()),
                        help='Reward function to use')
    parser.add_argument('--torch-compile', type=str, default='default',
                        help='Torch compile mode or "no" to disable')
    
    # Automatically add all RLConfig fields as command line arguments
    config_defaults = RLConfig()
    type_hints = get_type_hints(RLConfig)
    
    for field in fields(RLConfig):
        field_name = field.name.replace('_', '-')
        default_value = getattr(config_defaults, field.name)
        field_type = type_hints[field.name]
        
        # Handle Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
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
            arg_type = type(default_value)
        
        parser.add_argument(
            f'--{field_name}',
            type=arg_type,
            default=default_value,
            help=f'Config: {field.name} (default: {default_value})'
        )
    
    args = parser.parse_args()
    
    # Setup device and optimization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.cache_size_limit = 24
    
    # Create config from parsed arguments
    config_kwargs = {}
    for field in fields(RLConfig):
        field_name = field.name.replace('_', '-')
        config_kwargs[field.name] = getattr(args, field_name.replace('-', '_'))
    
    # Set algorithm-specific defaults
    if args.algorithm == 'ppo':
        config_kwargs['n_policy_iterations'] = 0
        config_kwargs['n_aux_epochs'] = 0
        print("Configured for PPO mode (no auxiliary phase)")
    elif args.algorithm == 'ppg':
        if config_kwargs['n_policy_iterations'] == 0:
            config_kwargs['n_policy_iterations'] = 4  # Default PPG setting
        if config_kwargs['n_aux_epochs'] == 0:
            config_kwargs['n_aux_epochs'] = 2  # Reduced from original PPG
        
        # PPG Reloaded defaults
        config_kwargs['adaptive_kl_reg'] = True  # Enable adaptive KL by default
        config_kwargs['policy_reg_coef'] = 1.0   # Stronger regularization
        config_kwargs['reduced_aux_frequency'] = True  # Computational efficiency
        
        print("Configured for PPG mode with PPG Reloaded enhancements")
    
    config = RLConfig(**config_kwargs)
    
    # Select reward function
    reward_fn = REWARD_FUNCTIONS[args.reward_function]
    print(f"Using reward function: {args.reward_function}")
    
    # Create and run trainer
    trainer = UnifiedTrainer(config, reward_fn, args.torch_compile)
    
    # Load checkpoint if resuming or initializing
    if config.resume_from_step > 0:
        checkpoint_path = f"{args.save_path}.iter_{config.resume_from_step}"
        trainer.nets = load_checkpoint(trainer.nets, trainer.optimizer, checkpoint_path, device)
        print(f"Resumed from iteration {config.resume_from_step}")
    elif args.init_path:
        trainer.nets = load_checkpoint(trainer.nets, None, args.init_path, device)
        print(f"Initialized from {args.init_path}")
    
    trainer.train()


if __name__ == "__main__":
    main()