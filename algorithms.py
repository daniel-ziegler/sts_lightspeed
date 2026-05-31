"""RL algorithm strategies (PPO, GRPO) and the loss helpers they share.

The trainer (`rl_train.py`) owns everything algorithm-agnostic: episode collection, batching,
the epoch/forward/backward/optimizer skeleton, checkpointing, and stats IO. An `Algorithm`
owns only what differs between methods: what to collect, how to turn trajectory returns/values
into per-step advantages, and the per-batch loss.

This module starts with the pure tensor helpers extracted verbatim from the original PPO loss
so both algorithms compute the policy surrogate / ratio / entropy identically.
"""
import torch
import torch.nn.functional as F

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
