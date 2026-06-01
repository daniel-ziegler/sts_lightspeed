"""RL algorithm strategies (PPO, GRPO) and the loss helpers they share.

The trainer (`rl_train.py`) owns everything algorithm-agnostic: episode collection, batching,
the epoch/forward/backward/optimizer skeleton, checkpointing, and stats IO. An `Algorithm`
owns only what differs between methods: what to collect, how to turn trajectory returns/values
into per-step advantages, and the per-batch loss.

This module starts with the pure tensor helpers extracted verbatim from the original PPO loss
so both algorithms compute the policy surrogate / ratio / entropy identically.
"""
from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn.functional as F


class CollectionJob(NamedTuple):
    """One episode to collect. game_seed -> C++ GameContext (map/card RNG); sample_seed -> the
    action-sampling RNG (distinct within a group so trajectories diverge); group_id tags the
    group that trajectory belongs to (for group-relative baselines)."""
    game_seed: int
    sample_seed: int
    group_id: int

LOG_PROB_CLAMP = 20.0   # clamp log-probs to [-20, 20] for numerical stability
RATIO_CLAMP = 1e8       # clamp the importance ratio to [1/RATIO_CLAMP, RATIO_CLAMP]


def policy_log_probs(new_logits, chosen_indices):
    """Softmax/log-softmax over the action logits and gather the chosen action's log-prob.

    Returns (action_probs, action_log_probs, new_log_probs) where new_log_probs is clamped for
    numerical stability. action_probs/action_log_probs are returned for the entropy term.
    """
    action_probs = F.softmax(new_logits, dim=-1)
    action_log_probs = F.log_softmax(new_logits, dim=-1)
    batch_indices = torch.arange(new_logits.shape[0], device=new_logits.device)
    new_log_probs = action_log_probs[batch_indices, chosen_indices]
    new_log_probs = torch.clamp(new_log_probs, min=-LOG_PROB_CLAMP, max=LOG_PROB_CLAMP)
    return action_probs, action_log_probs, new_log_probs


def importance_ratio(new_log_probs, old_log_probs):
    """exp(new - old), clamped. Caller is responsible for clamping `old_log_probs` first (so the
    same clamped value feeds both this ratio and the KL estimate)."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    return torch.clamp(ratio, min=1.0 / RATIO_CLAMP, max=RATIO_CLAMP)


def approx_kl(old_log_probs, new_log_probs):
    """Behavior-vs-current approximate KL (mean log-prob difference)."""
    return (old_log_probs - new_log_probs).mean()


def clip_fraction(ratio, clip_ratio):
    """Fraction of samples whose ratio deviates from 1 by more than clip_ratio."""
    return ((ratio - 1.0).abs() > clip_ratio).float().mean()


def clipped_surrogate(ratio, advantages, clip_ratio):
    """PPO/GRPO clipped surrogate policy loss: -mean(min(r*A, clip(r)*A))."""
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    return -torch.min(surr1, surr2).mean()


def masked_entropy(new_logits, action_probs, action_log_probs):
    """Mean per-sample entropy computed only over valid (non-inf) action logits. Samples with
    <2 valid actions contribute 0 entropy."""
    valid_mask = ~torch.isinf(new_logits)
    batch_entropies = []
    for i in range(new_logits.shape[0]):
        valid_actions = valid_mask[i]
        if valid_actions.sum() > 1:  # need >=2 valid actions for meaningful entropy
            valid_probs = action_probs[i][valid_actions]
            valid_log_probs = action_log_probs[i][valid_actions]
            batch_entropies.append(-(valid_probs * valid_log_probs).sum())
        else:
            batch_entropies.append(torch.tensor(0.0, device=new_logits.device))
    if batch_entropies:
        return torch.stack(batch_entropies).mean()
    return torch.tensor(0.0, device=new_logits.device)


class Algorithm(ABC):
    """Strategy that owns the parts that differ between RL algorithms. The trainer owns the rest
    (collection, batching, the epoch/forward/backward/optimizer skeleton, checkpointing, stats)."""
    name: str = "base"
    requires_value_head: bool = True

    def needs_return_in_batch(self) -> bool:
        """Whether the collated batch must carry per-step value targets ('return')."""
        return self.requires_value_head

    @abstractmethod
    def collection_plan(self, config, iteration) -> list:
        """List of CollectionJob for this iteration's experience collection."""

    @abstractmethod
    def compute_advantages(self, trajectories, config, adv_norm, debug_traj=False):
        """-> (experiences, advantages, value_targets_or_None, meta), parallel over the
        concatenation of each trajectory's experiences."""

    @abstractmethod
    def compute_loss(self, batch, net_output, config):
        """Per-minibatch loss. batch is the collated dict (tensors); net_output is (logits, values).
        Returns (total_loss_tensor, aux) where aux holds scalar-tensor metrics
        ('policy_loss','value_loss','entropy','kl_div','clipfrac') and, for value-based algos,
        'ev_target'/'ev_pred' tensors for explained-variance accumulation."""


class PPOAlgorithm(Algorithm):
    """PPO with GAE and a value head: clipped surrogate + value MSE + entropy bonus."""
    name = "ppo"
    requires_value_head = True

    def collection_plan(self, config, iteration):
        # One game per seed, each its own group; sample_seed == game_seed reproduces the original
        # start_seed = iteration*1000 collection exactly (RNG was tied to the game seed).
        base = iteration * 1000
        return [CollectionJob(game_seed=base + i, sample_seed=base + i, group_id=base + i)
                for i in range(config.num_games_per_step)]

    def compute_advantages(self, trajectories, config, adv_norm, debug_traj=False):
        # GAE lives in rl_train (operates on the trajectory/experience structures defined there);
        # imported lazily so this module has no import-time dependency on rl_train.
        from rl_train import compute_advantages as _gae
        return _gae(trajectories, config, adv_norm, debug_traj=debug_traj)

    def compute_loss(self, batch, net_output, config):
        new_logits, new_values = net_output
        device = new_logits.device
        old_log_probs = torch.clamp(batch['old_log_prob'].to(device), min=-LOG_PROB_CLAMP, max=LOG_PROB_CLAMP)
        advantages = batch['advantage'].to(device)
        target_values = batch['return'].to(device)
        chosen = batch['chosen_idx'].to(device)

        action_probs, action_log_probs, new_log_probs = policy_log_probs(new_logits, chosen)
        ratio = importance_ratio(new_log_probs, old_log_probs)
        kl_div = approx_kl(old_log_probs, new_log_probs)
        clipfrac = clip_fraction(ratio, config.clip_ratio)
        policy_loss = clipped_surrogate(ratio, advantages, config.clip_ratio)
        value_loss = F.mse_loss(new_values, target_values)
        entropy = masked_entropy(new_logits, action_probs, action_log_probs)

        total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        aux = {
            'policy_loss': policy_loss, 'value_loss': value_loss, 'entropy': entropy,
            'kl_div': kl_div, 'clipfrac': clipfrac,
            'ev_target': target_values, 'ev_pred': new_values,
        }
        return total_loss, aux


class GRPOAlgorithm(Algorithm):
    """Critic-free GRPO with a leave-one-out (RLOO) group baseline.

    A group is `group_size` full games sharing one map seed (distinct sample seeds, so the
    trajectories diverge). Each trajectory's advantage is its total return minus the mean return
    of the *other* members of its group, broadcast to every step. No value function, no GAE, no
    KL-to-reference; the clipped surrogate only bites when num_epochs > 1 (else ratio == 1)."""
    name = "grpo"
    requires_value_head = False

    def _num_groups(self, config):
        if config.num_groups > 0:
            return config.num_groups
        return max(1, config.num_games_per_step // config.group_size)

    def collection_plan(self, config, iteration):
        G = config.group_size
        K = self._num_groups(config)
        jobs = []
        for g in range(K):
            game_seed = iteration * 1_000_000 + g  # disjoint from PPO's iteration*1000 ranges
            for m in range(G):
                jobs.append(CollectionJob(game_seed=game_seed, sample_seed=game_seed * 64 + m,
                                          group_id=g))
        return jobs

    def needs_return_in_batch(self):
        return False

    def compute_advantages(self, trajectories, config, adv_norm, debug_traj=False):
        import slaythespire as sts
        from collections import defaultdict

        returns_by_idx = {}
        groups = defaultdict(list)  # group_id -> [traj_idx]
        for i, traj in enumerate(trajectories):
            if not traj.experiences:
                continue
            returns_by_idx[i] = float(sum(traj.rewards))
            groups[traj.group_id].append(i)

        all_experiences, all_advantages, all_meta = [], [], []
        group_stds = []
        for gid, idxs in groups.items():
            Rs = [returns_by_idx[i] for i in idxs]
            n = len(idxs)
            total = sum(Rs)
            if n > 1:
                mean = total / n
                group_stds.append((sum((r - mean) ** 2 for r in Rs) / n) ** 0.5)
            for i in idxs:
                R_i = returns_by_idx[i]
                # RLOO: advantage = return minus the leave-one-out mean of the rest of the group.
                adv = (R_i - (total - R_i) / (n - 1)) if n > 1 else 0.0
                traj = trajectories[i]
                ns = len(traj.experiences)
                all_experiences.extend(traj.experiences)
                all_advantages.extend([adv] * ns)
                won = 1 if traj.final_metrics.outcome == sts.GameOutcome.PLAYER_VICTORY else 0
                all_meta.extend([
                    {'seed': traj.seed, 'group_id': gid, 'outcome': won,
                     'final_floor': traj.final_metrics.floor_num,
                     'reward': float(traj.rewards[t]), 'value': 0.0}
                    for t in range(ns)
                ])

        self.last_group_reward_std = (sum(group_stds) / len(group_stds)) if group_stds else 0.0
        if debug_traj:
            print(f"GRPO: {len(groups)} groups, mean within-group return std "
                  f"{self.last_group_reward_std:.4f}, {len(all_experiences)} experiences")
            self._print_group_debug(trajectories, groups, returns_by_idx)
        return all_experiences, all_advantages, None, all_meta

    @staticmethod
    def _print_group_debug(trajectories, groups, returns_by_idx):
        """Show the most divergent group: same map seed, different plays. Prints each member's
        outcome/return/RLOO advantage, then the full playthroughs of the best and worst members."""
        import slaythespire as sts
        from rl_train import print_trajectory  # lazy: rl_train imports this module at load time

        def rloo(idxs, i):
            total = sum(returns_by_idx[j] for j in idxs)
            n = len(idxs)
            return (returns_by_idx[i] - (total - returns_by_idx[i]) / (n - 1)) if n > 1 else 0.0

        multi = {g: idxs for g, idxs in groups.items() if len(idxs) > 1}
        if not multi:
            return
        spread = lambda idxs: max(returns_by_idx[i] for i in idxs) - min(returns_by_idx[i] for i in idxs)
        gid = max(multi, key=lambda g: spread(multi[g]))
        idxs = sorted(multi[gid], key=lambda i: -returns_by_idx[i])
        map_seed = trajectories[idxs[0]].seed

        print(f"=== GRPO group debug: group {gid} (map seed {map_seed}), "
              f"return spread {spread(idxs):.3f} ===")
        for rank, i in enumerate(idxs):
            traj = trajectories[i]
            won = traj.final_metrics.outcome == sts.GameOutcome.PLAYER_VICTORY
            print(f"  member {rank}: {'WIN ' if won else 'loss'} floor {traj.final_metrics.floor_num:>2}  "
                  f"return {returns_by_idx[i]:+.3f}  rloo_adv {rloo(idxs, i):+.3f}  "
                  f"({len(traj.experiences)} decisions)")
        # Full playthroughs of the extremes -- same map, divergent play.
        for label, i in (('BEST', idxs[0]), ('WORST', idxs[-1])):
            traj = trajectories[i]
            adv = rloo(idxs, i)
            print_trajectory(traj, [adv] * len(traj.experiences),
                             title=f"--- {label} member of group {gid} "
                                   f"(return {returns_by_idx[i]:+.3f}, rloo_adv {adv:+.3f}) ---")

    def compute_loss(self, batch, net_output, config):
        new_logits = net_output[0] if isinstance(net_output, tuple) else net_output
        device = new_logits.device
        old_log_probs = torch.clamp(batch['old_log_prob'].to(device), min=-LOG_PROB_CLAMP, max=LOG_PROB_CLAMP)
        advantages = batch['advantage'].to(device)
        chosen = batch['chosen_idx'].to(device)

        action_probs, action_log_probs, new_log_probs = policy_log_probs(new_logits, chosen)
        ratio = importance_ratio(new_log_probs, old_log_probs)
        kl_div = approx_kl(old_log_probs, new_log_probs)
        clipfrac = clip_fraction(ratio, config.clip_ratio)
        policy_loss = clipped_surrogate(ratio, advantages, config.clip_ratio)
        entropy = masked_entropy(new_logits, action_probs, action_log_probs)

        total_loss = policy_loss - config.entropy_coef * entropy
        aux = {'policy_loss': policy_loss, 'entropy': entropy, 'kl_div': kl_div, 'clipfrac': clipfrac}
        return total_loss, aux
