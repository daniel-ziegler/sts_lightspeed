#!/usr/bin/env python3
"""
Shared utilities for reinforcement learning training in Slay the Spire.
Common components used by both PPO and PPG training algorithms.
"""

import random
import time
import os
import json
from dataclasses import dataclass
from typing import List, NamedTuple, Dict, Any, Tuple
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import slaythespire as sts
from network import move_to_device, collate_fn
from playouts import NNServiceManager, construct_choice, Choice, path_to_action_and_desc, choice_space


@dataclass
class GameMetrics:
    """Metrics extracted from game state for reward computation."""
    floor_num: int
    cur_hp: int
    max_hp: int
    perfected_strike_count: int
    outcome: sts.GameOutcome


class Experience(NamedTuple):
    """Single step of experience from a game."""
    choice: Choice
    action_idx: int
    log_prob: float
    value: float
    metrics: GameMetrics
    action_str: str


class Trajectory(NamedTuple):
    """Complete game trajectory."""
    seed: int
    experiences: List[Experience]
    rewards: List[float]
    values: List[float]
    final_reward: float
    final_metrics: GameMetrics
    final_deck: List[sts.Card]
    final_relics: List[sts.RelicId]


# Reward Functions
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


def run_episode(seed: int, service_client, reward_fn, max_floor: int | None = None) -> Trajectory:
    """Run a complete game episode and collect experience."""
    if max_floor is None:
        max_floor = 100

    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    rng = random.Random(seed)
    
    agent = sts.Agent()
    agent.simulation_count_base = 1000
    experiences = []
    values = []
    reward_fn_vals = []
    
    # Create local battle executor for timeouts
    battle_executor = ThreadPoolExecutor(max_workers=1)
    
    try:
        while gc.outcome == sts.GameOutcome.UNDECIDED and gc.floor_num <= max_floor:
            try:
                if gc.screen_state == sts.ScreenState.BATTLE:
                    # Use MCTS agent for battles
                    future = battle_executor.submit(agent.playout_battle, gc)
                    
                    try:
                        future.result(timeout=30.0)
                    except TimeoutError:
                        break
                        
                else:
                    # Use neural network for non-battle decisions
                    obs = sts.getNNRepresentation(gc)
                    actions = sts.GameAction.getAllActionsInState(gc)
                    
                    choice = construct_choice(gc, obs, actions)
                    
                    if choice is not None:
                        total_choices = (len(choice.cards_offered) + len(choice.relics_offered) + 
                                       len(choice.potions_offered) + len(choice.fixed_actions) + len(choice.paths_offered))
                        
                        if total_choices > 1:
                            # Get network predictions
                            batch_tensors, output = service_client.get_logits(choice)
                            
                            # Handle value head output
                            if isinstance(output, tuple):
                                logits, value_output = output
                                value = float(value_output) if np.isscalar(value_output) else float(value_output[0])
                            else:
                                logits = output
                                value = 0.0
                            
                            # Convert to probabilities and sample action
                            logits_tensor = torch.tensor(logits)
                            log_probs = F.log_softmax(logits_tensor, dim=0).numpy()
                            probs = np.exp(log_probs)
                            
                            chosen_idx = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])
                            log_prob = log_probs[chosen_idx]
                            
                            # Convert back to game action
                            path = choice_space.ix_to_path(batch_tensors['choices'], chosen_idx)
                            action, action_desc = path_to_action_and_desc(choice, path, gc)
                            
                            # Extract metrics BEFORE action execution
                            perfected_strike_count = sum(1 for card in gc.deck if card.id == sts.CardId.PERFECTED_STRIKE)
                            metrics = GameMetrics(
                                floor_num=gc.floor_num,
                                cur_hp=gc.cur_hp,
                                max_hp=gc.max_hp,
                                perfected_strike_count=perfected_strike_count,
                                outcome=gc.outcome,
                            )
                            reward_fn_vals.append(reward_fn(metrics))
                            
                            assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                            action.execute(gc)
                            
                            exp = Experience(
                                choice=choice,
                                action_idx=chosen_idx,
                                log_prob=log_prob,
                                value=value,
                                metrics=metrics,
                                action_str=action_desc
                            )
                            experiences.append(exp)
                            values.append(value)
                        else:
                            action = agent.pick_gameaction(gc)
                            assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                            action.execute(gc)
                    else:
                        action = agent.pick_gameaction(gc)
                        assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                        action.execute(gc)
                
            except Exception as e:
                print(f"Error in episode {seed}: {e}")
                break
    finally:
        battle_executor.shutdown(wait=True)
    
    # Create final metrics for reward computation
    final_metrics = GameMetrics(
        floor_num=gc.floor_num,
        cur_hp=gc.cur_hp,
        max_hp=gc.max_hp,
        perfected_strike_count=sum(1 for card in gc.deck if card.id == sts.CardId.PERFECTED_STRIKE),
        outcome=gc.outcome,
    )
    
    # Compute shaped rewards
    rewards = []
    all_reward_values = reward_fn_vals + [reward_fn(final_metrics)]
    
    # Reward shaping: each step gets delta from current state to next state
    for i in range(len(experiences)):
        reward_delta = all_reward_values[i+1] - all_reward_values[i]
        rewards.append(reward_delta)
    
    # Add terminal state value (0.0) for GAE bootstrap
    values.append(0.0)
    
    return Trajectory(
        seed=seed,
        experiences=experiences,
        rewards=rewards,
        values=values,
        final_reward=all_reward_values[-1] if all_reward_values else 0.0,
        final_metrics=final_metrics,
        final_deck=list(gc.deck),
        final_relics=list(gc.relics)
    )


def _worker_run_episodes_with_queue(worker_id: int, seeds: list, request_queue, response_queue, reward_fn, max_floor: int, result_queue, worker_index: int) -> None:
    """Worker function that runs multiple episodes and puts results in a queue."""
    from playouts import NNClient
    import pickle
    
    try:
        # Create client using the shared queues
        client = NNClient(worker_id, request_queue, response_queue)
        
        trajectories = []
        
        for seed in seeds:
            trajectory = run_episode(seed, client, reward_fn, max_floor)
            trajectories.append(trajectory)
        
        # Convert trajectories to serializable format using as_dict() on Choice objects
        serializable_trajectories = []
        for traj in trajectories:
            # Create a copy of the trajectory with serializable experiences
            serializable_experiences = []
            for exp in traj.experiences:
                # Convert Choice to dict for serialization
                choice_dict = exp.choice.as_dict() if exp.choice else None
                serializable_exp = Experience(
                    choice=choice_dict,  # Use dict instead of Choice object
                    action_idx=exp.action_idx,
                    log_prob=exp.log_prob,
                    value=exp.value,
                    metrics=exp.metrics,
                    action_str=exp.action_str
                )
                serializable_experiences.append(serializable_exp)
            
            # Create serializable trajectory - convert Card and Relic objects to strings
            serializable_traj = Trajectory(
                seed=traj.seed,
                experiences=serializable_experiences,
                rewards=traj.rewards,
                values=traj.values,
                final_reward=traj.final_reward,
                final_metrics=traj.final_metrics,
                final_deck=[str(card) for card in traj.final_deck],  # Convert cards to strings
                final_relics=[relic.id for relic in traj.final_relics]  # Convert relics to IDs
            )
            serializable_trajectories.append(serializable_traj)
        
        result_data = (worker_index, serializable_trajectories)
        
        # Put results in the result queue with worker index
        result_queue.put(result_data)
    except Exception as e:
        try:
            result_queue.put((worker_index, []))
        except:
            pass


def collect_experience(num_games: int, num_workers: int, service: NNServiceManager, reward_fn, start_seed: int = 0, max_floor: int = 3) -> List[Trajectory]:
    """Collect experience from multiple game episodes using parallel workers."""
    # Distribute seeds across workers
    all_seeds = [start_seed + i for i in range(num_games)]
    seeds_per_worker = [[] for _ in range(num_workers)]
    for i, seed in enumerate(all_seeds):
        seeds_per_worker[i % num_workers].append(seed)
    
    # Create clients for multiprocessing workers
    additional_clients = []
    for i in range(num_workers):
        worker_client = service.create_client(i)
        additional_clients.append(worker_client)
    
    # Create a single shared result queue
    shared_result_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    active_workers = 0
    for i, worker_seeds in enumerate(seeds_per_worker):
        if worker_seeds:  # Only start process if it has seeds
            worker_client = additional_clients[i]
            p = mp.Process(
                target=_worker_run_episodes_with_queue,
                args=(
                    worker_client.worker_id, 
                    worker_seeds, 
                    worker_client.request_queue, 
                    worker_client.response_queue, 
                    reward_fn, 
                    max_floor,
                    shared_result_queue,
                    i  # worker index
                )
            )
            p.start()
            processes.append(p)
            active_workers += 1
    
    # Collect results
    trajectories = []
    results_received = 0
    with tqdm(total=num_games, desc="Collecting experience") as pbar:
        while results_received < active_workers:
            try:
                worker_index, worker_trajectories = shared_result_queue.get(timeout=30)  # 30 second timeout
                trajectories.extend(worker_trajectories)
                pbar.update(len(worker_trajectories))
                results_received += 1
            except Exception as e:
                break
    
    # Join all processes
    for p in processes:
        p.join()
    
    return trajectories


def print_traj(traj: Trajectory, advantages: List[float], returns: List[float]):
    print(f"=== PPO Advantage Calculation Debug (seed {traj.seed}) ===")
    print(f"Trajectory length: {len(traj.experiences)} steps")
    print(f"Step | {'State':13s} | {'Choice':20s} | {'Action':20s} | {'Prob':6s} | {'Reward':6s} | {'Pred Value':10s} | {'GAE Return':10s} | {'Raw Advantage':13s}")
    print("-" * 120)
    
    for t in range(len(traj.experiences)):
        exp = traj.experiences[t]
        
        # Get choice summary - what was offered (choices are always dicts now)
        offered_items = []
        if exp.choice:
            cards_offered = exp.choice.get('cards_offered', {})
            if isinstance(cards_offered, dict) and 'mask' in cards_offered:
                card_count = sum(cards_offered['mask']) if cards_offered['mask'] else 0
            else:
                card_count = len(cards_offered) if cards_offered else 0
            if card_count > 0:
                offered_items.append(f"{card_count}card")
            
            relics_offered = exp.choice.get('relics_offered')
            if relics_offered is not None and len(relics_offered) > 0:
                offered_items.append(f"{len(relics_offered)}rel")
            
            potions_offered = exp.choice.get('potions_offered')
            if potions_offered is not None and len(potions_offered) > 0:
                offered_items.append(f"{len(potions_offered)}pot")
            
            fixed_actions = exp.choice.get('fixed_actions')
            if fixed_actions is not None and len(fixed_actions) > 0:
                offered_items.append(f"{len(fixed_actions)}fix")
            
            paths_offered = exp.choice.get('paths_offered')
            if paths_offered is not None and len(paths_offered) > 0:
                offered_items.append(f"{len(paths_offered)}path")
        
        choice_desc = f"{'+'.join(offered_items)}" if offered_items else "none"
        
        # Get action description - use the clean description generated during experience collection
        action_desc = exp.action_str[:20] if exp.action_str else "Unknown"
        
        # Create state string: 18: 20/72hp format
        state_str = f"{exp.metrics.floor_num:>2}: {exp.metrics.cur_hp}/{exp.metrics.max_hp}hp"
        
        print(f"{t:4d} | {state_str:13s} | {choice_desc[:20]:20s} | {action_desc[:20]:20s} | {np.exp(exp.log_prob):6.3f} | {traj.rewards[t]:6.3f} | {exp.value:10.3f} | {returns[t]:10.3f} | {advantages[t]:13.3f}")
    
    print("-" * 120)
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
        print(f"  {relic}")  # Relics are now relic IDs
    
    print("=" * 80)


def compute_gae(rewards: List[float], values: List[float], gamma: float, gae_lambda: float) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    returns = []
    
    gae = 0.0
    for i in reversed(range(len(rewards))):
        # values[i+1] is next state value, values[i] is current state value
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])
    
    return advantages, returns


def compute_advantages_for_trajectories(trajectories: List[Trajectory], gamma: float, gae_lambda: float) -> Tuple[List[Experience], List[float], List[float]]:
    """Compute advantages using GAE for multiple trajectories."""
    all_experiences = []
    all_advantages = []
    all_returns = []

    print_traj_index = random.randrange(len(trajectories))
    
    for traj_idx, traj in enumerate(trajectories):
        if not traj.experiences:
            continue
            
        # Compute GAE for this trajectory
        advantages, returns = compute_gae(traj.rewards, traj.values, gamma, gae_lambda)

        if traj_idx == print_traj_index:
            print_traj(traj, advantages, returns)
        
        # Normalize advantages for this trajectory
        if len(advantages) > 1:
            advantages = np.array(advantages)
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / adv_std
            else:
                advantages = advantages - adv_mean
            advantages = advantages.tolist()
        
        all_experiences.extend(traj.experiences)
        all_advantages.extend(advantages)
        all_returns.extend(returns)
    
    return all_experiences, all_advantages, all_returns


def experiences_to_batches(experiences: List[Experience], advantages: List[float], returns: List[float]) -> List[Dict]:
    """Convert experiences to training batches."""
    batch_data = []
    
    for i, exp in enumerate(experiences):
        # Choice is already a dict from multiprocessing
        choice_dict = exp.choice if exp.choice else {}
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
        
        # Add training-specific fields
        flat_dict['chosen_idx'] = exp.action_idx
        flat_dict['old_log_prob'] = exp.log_prob
        flat_dict['advantage'] = advantages[i]
        flat_dict['return'] = returns[i]
        flat_dict['outcome'] = 1.0  # Dummy, not used in RL training
        
        batch_data.append(flat_dict)
    
    return batch_data


def save_checkpoint(nets, optimizer, config, step: int, save_path: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    if isinstance(nets, tuple):
        # Separate networks
        policy_net, value_net = nets
        torch.save(policy_net.state_dict(), f"{save_path}.policy.iter_{step}")
        torch.save(value_net.state_dict(), f"{save_path}.value.iter_{step}")
        torch.save(optimizer.state_dict(), f"{save_path}.optimizer.iter_{step}")
        print(f"Saved separate network checkpoints at iteration {step}")
    else:
        # Single network
        torch.save(nets.state_dict(), f"{save_path}.iter_{step}")
        torch.save(optimizer.state_dict(), f"{save_path}.optimizer.iter_{step}")
        print(f"Saved model checkpoint at iteration {step}")


def load_checkpoint(nets, optimizer, checkpoint_path: str, device):
    """Load model checkpoint."""
    from network import load_network_backward_compatible
    
    if isinstance(nets, tuple):
        # Separate networks
        policy_net, value_net = nets
        policy_path = f"{checkpoint_path}.policy"
        value_path = f"{checkpoint_path}.value"
        optimizer_path = f"{checkpoint_path}.optimizer"
        
        if os.path.exists(policy_path):
            policy_state = torch.load(policy_path, map_location=device, weights_only=True)
            policy_net = load_network_backward_compatible(policy_net, policy_state)
            
        if os.path.exists(value_path):
            value_state = torch.load(value_path, map_location=device, weights_only=True)
            value_net = load_network_backward_compatible(value_net, value_state)
            
        if os.path.exists(optimizer_path) and optimizer is not None:
            optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(optimizer_state)
            
        print(f"Loaded separate networks from {checkpoint_path}")
        return (policy_net, value_net)
    else:
        # Single network
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            nets = load_network_backward_compatible(nets, state)
            
        optimizer_path = f"{checkpoint_path}.optimizer"
        if os.path.exists(optimizer_path) and optimizer is not None:
            optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(optimizer_state)
            
        print(f"Loaded model from {checkpoint_path}")
        return nets


def create_ppo_collate_fn():
    """Create collate function that handles RL training fields."""
    def ppo_collate_fn(batch):
        # Extract RL-specific fields before calling main collate_fn
        old_log_probs = [x['old_log_prob'] for x in batch]
        advantage_vals = [x['advantage'] for x in batch]
        return_vals = [x['return'] for x in batch]
        
        # Call the main collate function
        collated = collate_fn(batch)
        
        # Add RL-specific fields
        collated['old_log_prob'] = torch.tensor(old_log_probs, dtype=torch.float32)
        collated['advantage'] = torch.tensor(advantage_vals, dtype=torch.float32)
        collated['return'] = torch.tensor(return_vals, dtype=torch.float32)
        
        return collated
    
    return ppo_collate_fn


def log_training_stats(stats: Dict[str, Any], save_path: str):
    """Log training statistics to JSONL file."""
    stats_path = f"{save_path}.stats.jsonl"
    with open(stats_path, 'a') as f:
        f.write(json.dumps(stats) + '\n')


# Registry of reward functions
REWARD_FUNCTIONS = {
    'smooth': compute_progress_reward,
    'perfected_strike': compute_perfected_strike_reward,
    'victory': compute_victory_reward,
    'no_pstrikes': compute_no_pstrikes_reward,
}