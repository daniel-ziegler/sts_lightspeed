from collections import abc
from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import slaythespire as sts
from inputs import SinusoidalEmbedding, FixedVecSpace, SequenceSpace, EnumSpace, DictSpace, TupleAddSpace, IntSpace

@dataclass
class ModelHP:
    dim: int = 256
    mlp_dim_mult: int = 4
    n_layers: int = 4
    n_heads: int = 8
    norm_eps: float = 1e-5
    n_fixed_obs: int = len(sts.getFixedObservationMaximums())
    max_relics: int = 25  # Maximum number of relics a player typically has


# Constants for data processing
MAX_DECK_SIZE = 64  # Should be enough for most decks
MAX_CHOICES = 10    # Usually 3-4, but can be more in edge cases
MAX_UPGRADE = 21


class ActionType(IntEnum):
    INVALID = auto()
    CARD = auto()
    PATH = auto()
    RELIC = auto()
    POTION = auto()
    EVENT_OPTION = auto()
    FIXED = auto()  # for fixed actions like SKIP

class InputType(IntEnum):
    # TODO maybe should unify with ActionType
    Card = 0
    Relic = auto()
    Potion = auto()
    Choice = auto()
    Fixed = auto()


class FixedAction(IntEnum):
    INVALID = 0
    SKIP = auto()
    REMOVE = auto()
    SINGING_BOWL = auto()

obs_space = DictSpace({
    'deck': SequenceSpace(TupleAddSpace(EnumSpace(sts.CardId), IntSpace(MAX_UPGRADE))),
    'relics': SequenceSpace(EnumSpace(sts.RelicId)),
    'potions': SequenceSpace(EnumSpace(sts.Potion)),
    'fixed_obs': FixedVecSpace(sts.getFixedObservationMaximums()),
})

action_logit_space = DictSpace({
    'cards': SequenceSpace(TupleAddSpace(EnumSpace(sts.CardId), IntSpace(MAX_UPGRADE))),
    'relics': SequenceSpace(EnumSpace(sts.RelicId)),
    'potions': SequenceSpace(EnumSpace(sts.Potion)),
    'fixed': SequenceSpace(EnumSpace(FixedAction)),
})


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


class MLP(nn.Module):
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
        self.mlp = MLP(H.dim, H.dim * H.mlp_dim_mult)

    def forward(self, x, pos_mask):
        xn = self.norm1(x)
        xatt, _ = self.attn(xn, xn, xn, attn_mask=None, key_padding_mask=pos_mask)
        x1 = x + xatt
        x2 = x1 + self.mlp(self.norm2(x1))
        return x2


class NN(nn.Module):
    def __init__(self, H: ModelHP):
        super().__init__()
        self.H = H

        self.obs_embed = obs_space.build_embed(H.dim)
        self.action_logit_embed = action_logit_space.build_embed(H.dim)
        
        self.layers = nn.ModuleList([TransformerBlock(H=H) for _ in range(H.n_layers)])

        self.norm = RMSNorm(H.dim, H.norm_eps)

        self.choice_logits = nn.Linear(H.dim, 1, bias=True)
        nn.init.uniform_(self.choice_logits.weight, -0.01, 0.01)
        nn.init.zeros_(self.choice_logits.bias)


    def forward(self, batch: dict):
        """
        Process a batch of inputs through the network.
        
        Args:
            batch: Dictionary containing observation data and choices data.
                   The 'choices' key is popped and processed separately as action logits.
        
        Returns:
            choice_logits: [batch_size, max_action_choices] tensor of flat logits.
                          Use action_logit_space.ix_to_path to convert indices back to semantic actions.
        """
        choices = batch.pop('choices')
        obs_embed, obs_mask = self.obs_embed(batch)
        action_logit_embed, action_logit_mask = self.action_logit_embed(choices)

        assert (~obs_mask).any()
        assert (~action_logit_mask).any()

        x = torch.cat([obs_embed, action_logit_embed], dim=1)
        pos_mask = torch.cat([obs_mask, action_logit_mask], dim=1)

        for l in self.layers:
            x = l(x, pos_mask)
        xn = self.norm(x)

        action_xs = xn[:, obs_mask.size(1):, :]
        choice_logits = self.choice_logits(action_xs).squeeze(-1).float()
        choice_logits = choice_logits.masked_fill(action_logit_mask, float('-inf'))
        return choice_logits
    
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
            col: row[col]
            for col in self.df.columns
        }


def collate_fn(batch):
    # Extract observation and choices data
    obs_batch = []
    choices_batch = []
    chosen_idx_list = []
    outcome_list = []
    
    for x in batch:
        # Build observation dict that matches obs_space structure
        obs = {
            'deck': {
                'value': torch.tensor(list(zip(x['obs.deck.cards'], x['obs.deck.upgrades'])), dtype=torch.int32),
                'mask': torch.zeros(len(x['obs.deck.cards']), dtype=torch.bool)
            },
            'relics': {
                'value': torch.tensor(x['obs.relics.relics'], dtype=torch.int32),
                'mask': torch.zeros(len(x['obs.relics.relics']), dtype=torch.bool)
            },
            'potions': {
                'value': torch.tensor(x['obs.potions'], dtype=torch.int32),
                'mask': torch.zeros(len(x['obs.potions']), dtype=torch.bool)
            },
            'fixed_obs': torch.tensor(x['obs.fixed_observation'], dtype=torch.int32)
        }
        
        # Build choices dict that matches action_logit_space structure
        choices = {
            'cards': {
                'value': torch.tensor(list(zip(x['cards_offered.cards'], x['cards_offered.upgrades'])), dtype=torch.int32).reshape(-1, 2) if len(x['cards_offered.cards']) > 0 else torch.empty((0, 2), dtype=torch.int32),
                'mask': torch.zeros(len(x['cards_offered.cards']), dtype=torch.bool)
            },
            'relics': {
                'value': torch.tensor(x['relics_offered'], dtype=torch.int32),
                'mask': torch.zeros(len(x['relics_offered']), dtype=torch.bool)
            },
            'potions': {
                'value': torch.tensor(x['potions_offered'], dtype=torch.int32),
                'mask': torch.zeros(len(x['potions_offered']), dtype=torch.bool)
            },
            'fixed': {
                'value': torch.tensor(x['fixed_actions'], dtype=torch.int32),
                'mask': torch.zeros(len(x['fixed_actions']), dtype=torch.bool)
            }
        }
        
        obs_batch.append(obs)
        choices_batch.append(choices)
        chosen_idx_list.append(x['chosen_idx'])
        outcome_list.append(x['outcome'])
    
    # Create batched tensors for observation
    max_deck_len = max(len(obs['deck']['value']) for obs in obs_batch)
    max_relics_len = max(len(obs['relics']['value']) for obs in obs_batch)
    max_potions_len = max(len(obs['potions']['value']) for obs in obs_batch)
    
    batch_obs = {
        'deck': {
            'value': torch.full((len(batch), max_deck_len, 2), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), max_deck_len), dtype=torch.bool)
        },
        'relics': {
            'value': torch.full((len(batch), max_relics_len), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), max_relics_len), dtype=torch.bool)
        },
        'potions': {
            'value': torch.full((len(batch), max_potions_len), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), max_potions_len), dtype=torch.bool)
        },
        'fixed_obs': torch.zeros((len(batch), len(sts.getFixedObservationMaximums())), dtype=torch.int32)
    }
    
    # Create batched tensors for choices
    max_choice_cards_len = max(len(choices['cards']['value']) for choices in choices_batch)
    max_choice_relics_len = max(len(choices['relics']['value']) for choices in choices_batch)
    max_choice_potions_len = max(len(choices['potions']['value']) for choices in choices_batch)
    max_choice_fixed_len = max(len(choices['fixed']['value']) for choices in choices_batch)
    
    batch_choices = {
        'cards': {
            'value': torch.full((len(batch), max_choice_cards_len, 2), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), max_choice_cards_len), dtype=torch.bool)
        },
        'relics': {
            'value': torch.full((len(batch), max_choice_relics_len), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), max_choice_relics_len), dtype=torch.bool)
        },
        'potions': {
            'value': torch.full((len(batch), max_choice_potions_len), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), max_choice_potions_len), dtype=torch.bool)
        },
        'fixed': {
            'value': torch.full((len(batch), max_choice_fixed_len), FixedAction.INVALID.value, dtype=torch.int32),
            'mask': torch.ones((len(batch), max_choice_fixed_len), dtype=torch.bool)
        }
    }
    
    # Fill the batched tensors
    for i, (obs, choices) in enumerate(zip(obs_batch, choices_batch)):
        # Fill observation
        deck_len = len(obs['deck']['value'])
        batch_obs['deck']['value'][i, :deck_len] = obs['deck']['value']
        batch_obs['deck']['mask'][i, :deck_len] = obs['deck']['mask']
        
        relics_len = len(obs['relics']['value'])
        batch_obs['relics']['value'][i, :relics_len] = obs['relics']['value']
        batch_obs['relics']['mask'][i, :relics_len] = obs['relics']['mask']
        
        potions_len = len(obs['potions']['value'])
        batch_obs['potions']['value'][i, :potions_len] = obs['potions']['value']
        batch_obs['potions']['mask'][i, :potions_len] = obs['potions']['mask']
        
        batch_obs['fixed_obs'][i] = obs['fixed_obs']
        
        # Fill choices
        choice_cards_len = len(choices['cards']['value'])
        batch_choices['cards']['value'][i, :choice_cards_len] = choices['cards']['value']
        batch_choices['cards']['mask'][i, :choice_cards_len] = choices['cards']['mask']
        
        choice_relics_len = len(choices['relics']['value'])
        batch_choices['relics']['value'][i, :choice_relics_len] = choices['relics']['value']
        batch_choices['relics']['mask'][i, :choice_relics_len] = choices['relics']['mask']
        
        choice_potions_len = len(choices['potions']['value'])
        batch_choices['potions']['value'][i, :choice_potions_len] = choices['potions']['value']
        batch_choices['potions']['mask'][i, :choice_potions_len] = choices['potions']['mask']
        
        choice_fixed_len = len(choices['fixed']['value'])
        batch_choices['fixed']['value'][i, :choice_fixed_len] = choices['fixed']['value']
        batch_choices['fixed']['mask'][i, :choice_fixed_len] = choices['fixed']['mask']
    
    return {
        **batch_obs,
        'choices': batch_choices,
        'chosen_idx': torch.tensor(chosen_idx_list, dtype=torch.int64),
        'outcome': torch.tensor(outcome_list, dtype=torch.float32),
    }

def move_to_device(obj, device):
    """Recursively move tensors to device, handling nested dictionaries."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj

def process_batch(batch, net):
    # Move tensors to device recursively
    device = net.device
    batch = move_to_device(batch, device)
    return net(batch)

def output_to_cpu(output: torch.Tensor, batch: dict) -> list[np.ndarray]:
    return output.cpu().numpy()



# %%
