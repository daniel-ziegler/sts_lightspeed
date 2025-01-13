import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from enum import IntEnum, auto
import slaythespire as sts


from dataclasses import dataclass


@dataclass
class ModelHP:
    dim: int = 256
    ffn_dim_mult: int = 4
    n_layers: int = 4
    n_heads: int = 8
    norm_eps: float = 1e-5
    n_fixed_obs: int = len(sts.getFixedObservationMaximums())


# Constants for data processing
MAX_DECK_SIZE = 64  # Should be enough for most decks
MAX_CHOICES = 10    # Usually 3-4, but can be more in edge cases


class ActionType(IntEnum):
    INVALID = auto()
    CARD = auto()
    PATH = auto()
    EVENT_OPTION = auto()
    FIXED = auto()  # New type for fixed actions like SKIP

class InputType(IntEnum):
    # TODO maybe should unify with ActionType
    Card = 0
    Relic = auto()
    Potion = auto()
    Choice = auto()
    Fixed = auto()


class FixedAction(IntEnum):
    INVALID = -1
    SKIP = 0

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, n_features: int):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.register_buffer('inv_freq', torch.exp(torch.arange(half_dim) * -emb) * 10)
        self.out_dim = dim * n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for numerical values.
        x: tensor of integers to embed [batch_size, n_features]
        Returns: [batch_size, n_features * dim] tensor
        """
        # [batch_size, n_features, half_dim]
        emb = x.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)

        # [batch_size, n_features, dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # [batch_size, n_features * dim]
        return emb.reshape(x.shape[0], -1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float):
        super().__init__()
        self.eps = eps
        self.w = nn.parameter.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        normed = self._norm(x.float()).type_as(x)
        return normed * self.w


class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        x2 = F.silu(self.w1(x)) * self.v(x)
        return self.w2(x2)


class TransformerBlock(nn.Module):
    def __init__(self, H: ModelHP):
        super().__init__()
        self.H = H
        self.norm1 = RMSNorm(H.dim, eps=H.norm_eps)
        self.attn = nn.MultiheadAttention(H.dim, H.n_heads, batch_first=True)
        self.norm2 = RMSNorm(H.dim, eps=H.norm_eps)
        self.ffn = FFN(H.dim, H.dim * H.ffn_dim_mult)

    def forward(self, x, pos_mask):
        xn = self.norm1(x)
        xatt, _ = self.attn(xn, xn, xn, attn_mask=None, key_padding_mask=pos_mask)
        x1 = x + xatt
        x2 = x1 + self.ffn(self.norm2(x1))
        return x2


class NN(nn.Module):
    def __init__(self, H: ModelHP):
        super().__init__()
        self.H = H

        self.input_type_embed = nn.Embedding(len(InputType), H.dim)
        self.card_embed = nn.Embedding(len(sts.CardId), H.dim, padding_idx=sts.CardId.INVALID.value)
        self.upgrade_embed = nn.Embedding(21, H.dim, padding_idx=0)
        
        # Add fixed action embedding
        self.fixed_action_embed = nn.Embedding(len(FixedAction)-1, H.dim)

        # Add sinusoidal embedding and projection
        n_fixed_obs = len(sts.getFixedObservationMaximums())
        self.fixed_obs_embed = SinusoidalEmbedding(H.dim, n_fixed_obs)
        self.fixed_obs_proj = nn.Linear(self.fixed_obs_embed.out_dim, H.dim)

        self.layers = nn.ModuleList([TransformerBlock(H=H) for _ in range(H.n_layers)])

        self.norm = RMSNorm(H.dim, H.norm_eps)
        self.card_winprob = nn.Linear(H.dim, 1, bias=True)
        nn.init.uniform_(self.card_winprob.weight, -0.01, 0.01)
        nn.init.zeros_(self.card_winprob.bias)

    def forward(self, batch: dict[str, torch.Tensor]):
        """
        Process a batch of inputs through the network.
        
        Args:
            batch: Dictionary containing:
                - deck: [batch_size, MAX_DECK_SIZE] tensor of card IDs
                - deck_upgrades: [batch_size, MAX_DECK_SIZE] tensor of upgrade counts
                - choices: [batch_size, MAX_CHOICES] tensor of card IDs
                - choice_upgrades: [batch_size, MAX_CHOICES] tensor of upgrade counts
                - fixed_obs: [batch_size, n_fixed_obs] tensor of fixed observations
                - fixed_actions: [batch_size, n_fixed_actions] tensor of fixed action IDs
        """
        device = batch['deck'].device
        max_deck_len = batch['deck'].size(1)
        max_choices_len = batch['choices'].size(1)

        # Create sinusoidal embeddings for all fixed observations at once
        fixed_obs_x = self.fixed_obs_proj(self.fixed_obs_embed(batch['fixed_obs']))

        # Embed cards
        cards = torch.cat((batch['deck'], batch['choices']), dim=1)
        upgrades = torch.cat((batch['deck_upgrades'], batch['choice_upgrades']), dim=1)
        mask = cards == sts.CardId.INVALID.value

        # Combine card and upgrade embeddings
        card_x = (self.card_embed(cards) +
                 self.upgrade_embed(upgrades.clamp(max=20)) +
                 self.input_type_embed(torch.tensor([int(InputType.Card)], device=device)))
        card_x[:, max_deck_len:, :] += self.input_type_embed(torch.tensor([int(InputType.Choice)], device=device))

        # Add fixed action embeddings
        fixed_x = self.fixed_action_embed(batch['fixed_actions'])
        fixed_x = fixed_x + self.input_type_embed(torch.tensor([int(InputType.Fixed)], device=device))

        # Combine all embeddings
        x = torch.cat([
            fixed_obs_x.unsqueeze(1),  # [batch, 1, dim]
            card_x,                     # [batch, deck+choices, dim]
            fixed_x,                    # [batch, n_fixed_actions, dim]
        ], dim=1)

        # Add fixed obs and fixed action tokens to mask (False = don't mask)
        pos_mask = torch.cat([
            torch.zeros(mask.size(0), 1, device=device, dtype=mask.dtype),  # fixed obs
            mask,                                                           # cards
            batch['fixed_actions'] == FixedAction.INVALID.value,           # fixed actions
        ], dim=1)

        for l in self.layers:
            x = l(x, pos_mask)
        xn = self.norm(x)

        # Get logits for both cards and fixed actions
        choice_x = xn[:, 1+max_deck_len:1+max_deck_len+max_choices_len, :]  # card choices
        fixed_x = xn[:, -batch['fixed_actions'].size(1):, :]  # fixed actions
        
        # Get win probabilities for both
        card_logits = self.card_winprob(choice_x).squeeze(-1).float()
        fixed_action_logits = self.card_winprob(fixed_x).squeeze(-1).float()
        
        # Mask invalid card choices and fixed actions
        card_logits = card_logits.masked_fill(mask[:, max_deck_len:], float('-inf'))
        fixed_action_logits = fixed_action_logits.masked_fill(
            batch['fixed_actions'] == FixedAction.INVALID.value, 
            float('-inf')
        )

        return dict(
            card_logits=card_logits,
            fixed_logits=fixed_action_logits,
        )
    
    @property
    def device(self):
        return next(self.parameters()).device


# %%
class SlayDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'deck': np.array(row['obs.deck.cards'], dtype=np.int32),
            'deck_upgrades': np.array(row['obs.deck.upgrades'], dtype=np.int32),
            'choices': np.array(row['cards_offered.cards'], dtype=np.int32),
            'choice_upgrades': np.array(row['cards_offered.upgrades'], dtype=np.int32),
            'fixed_obs': np.array(row['obs.fixed_observation'], dtype=np.int32),
            'chosen_idx': row['chosen_idx'],
            'outcome': row['outcome'],
        }


def collate_fn(batch):
    for x in batch:
        n_card_choices = len(x['choices'])
        n_fixed_actions = len(x['fixed_actions'])
        chosen_idx = x['chosen_idx']
        # Allow indices up to n_choices + n_fixed_actions
        if x['choice_type'] == ActionType.CARD:
            assert chosen_idx < n_card_choices, f"chosen_idx {chosen_idx} >= n_choices {n_card_choices}"
            assert chosen_idx < MAX_CHOICES, f"chosen_idx {chosen_idx} >= MAX_CHOICES {MAX_CHOICES}"
        elif x['choice_type'] == ActionType.FIXED:
            assert chosen_idx < n_fixed_actions, f"chosen_idx {chosen_idx} >= n_fixed_actions {n_fixed_actions}"

    # Prepare arrays
    deck = torch.full((len(batch), MAX_DECK_SIZE), sts.CardId.INVALID.value, dtype=torch.int32)
    deck_upgrades = torch.zeros((len(batch), MAX_DECK_SIZE), dtype=torch.int32)
    choices = torch.full((len(batch), MAX_CHOICES), sts.CardId.INVALID.value, dtype=torch.int32)
    choice_upgrades = torch.zeros((len(batch), MAX_CHOICES), dtype=torch.int32)
    fixed_obs = torch.zeros((len(batch), len(sts.getFixedObservationMaximums())), dtype=torch.int32)
    fixed_actions = torch.full((len(batch), len(FixedAction)-1), FixedAction.INVALID.value, dtype=torch.int32)
    chosen_idx = torch.zeros(len(batch), dtype=torch.int64)
    choice_type = torch.zeros(len(batch), dtype=torch.int64)
    outcome = torch.zeros(len(batch), dtype=torch.float32)

    # Fill arrays
    for i, x in enumerate(batch):
        deck[i, :min(len(x['deck']), MAX_DECK_SIZE)] = torch.from_numpy(x['deck'])[:MAX_DECK_SIZE]
        deck_upgrades[i, :min(len(x['deck_upgrades']), MAX_DECK_SIZE)] = torch.from_numpy(x['deck_upgrades'])[:MAX_DECK_SIZE]
        choices[i, :min(len(x['choices']), MAX_CHOICES)] = torch.from_numpy(x['choices'])[:MAX_CHOICES]
        choice_upgrades[i, :min(len(x['choice_upgrades']), MAX_CHOICES)] = torch.from_numpy(x['choice_upgrades'])[:MAX_CHOICES]
        fixed_obs[i] = torch.from_numpy(x['fixed_obs'])
        fixed_actions[i, :len(x['fixed_actions'])] = torch.from_numpy(x['fixed_actions'])
        chosen_idx[i] = x['chosen_idx']
        choice_type[i] = x['choice_type']
        outcome[i] = x['outcome']

    return {
        'deck': deck,
        'deck_upgrades': deck_upgrades,
        'choices': choices,
        'choice_upgrades': choice_upgrades,
        'fixed_obs': fixed_obs,
        'fixed_actions': fixed_actions,
        'chosen_idx': chosen_idx,
        'choice_type': choice_type,
        'outcome': outcome,
    }

def process_batch(batch, net):
    # Move tensors to device
    device = net.device
    batch = {k: v.to(device) for k, v in batch.items()}
    return net(batch)


