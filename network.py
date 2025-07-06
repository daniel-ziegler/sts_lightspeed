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
    use_value_head: bool = False  # Add value head for PPO training


# Constants for data processing
MAX_DECK_SIZE = 64  # Should be enough for most decks
MAX_CHOICES = 20    # Orrery plus Question Card
MAX_UPGRADE = 21
MAX_RELICS = 25     # Maximum number of relics a player typically has
MAX_FIXED_ACTIONS = 5  # Maximum number of fixed actions in choices


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

choice_space = DictSpace({
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
        self.choice_embed = choice_space.build_embed(H.dim)
        
        self.layers = nn.ModuleList([TransformerBlock(H=H) for _ in range(H.n_layers)])

        self.norm = RMSNorm(H.dim, H.norm_eps)

        self.choice_logits = nn.Linear(H.dim, 1, bias=True)
        nn.init.uniform_(self.choice_logits.weight, -0.01, 0.01)
        nn.init.zeros_(self.choice_logits.bias)
        
        # Add value head if enabled
        if H.use_value_head:
            self.value_head = nn.Linear(H.dim, 1, bias=True)
            nn.init.uniform_(self.value_head.weight, -0.01, 0.01)
            nn.init.zeros_(self.value_head.bias)


    def forward(self, batch: dict):
        """
        Process a batch of inputs through the network.
        
        Args:
            batch: Dictionary containing observation data and choices data.
                   The 'choices' key is popped and processed separately as action logits.
        
        Returns:
            If value_head enabled: (choice_logits, values) tuple where:
                - choice_logits: [batch_size, max_action_choices] tensor of flat logits
                - values: [batch_size] tensor of state value estimates
            If value_head disabled: choice_logits tensor only
            Use choice_space.ix_to_path to convert indices back to semantic actions.
        """
        choices = batch.pop('choices')
        
        # Debug: Check for invalid tensor values
        deck_max = batch['deck']['value'].max()
        deck_min = batch['deck']['value'].min()
        if deck_min < 0:
            print(f"DEBUG: Negative deck card ID: {deck_min}")
            print(f"DEBUG: Deck values: {batch['deck']['value']}")
            raise ValueError(f"Negative deck card ID: {deck_min}")
        if deck_max >= len(sts.CardId):
            print(f"DEBUG: Deck card ID too large: {deck_max}, max valid: {len(sts.CardId)-1}")
            print(f"DEBUG: Deck values: {batch['deck']['value']}")
            raise ValueError(f"Invalid deck card ID: {deck_max}")
        
        cards_max = choices['cards']['value'].max()
        cards_min = choices['cards']['value'].min()
        if cards_min < 0:
            print(f"DEBUG: Negative choice card ID: {cards_min}")
            print(f"DEBUG: Choice card values: {choices['cards']['value']}")
            raise ValueError(f"Negative choice card ID: {cards_min}")
        if cards_max >= len(sts.CardId):
            print(f"DEBUG: Choice card ID too large: {cards_max}, max valid: {len(sts.CardId)-1}")
            print(f"DEBUG: Choice card values: {choices['cards']['value']}")
            raise ValueError(f"Invalid choice card ID: {cards_max}")
        
        obs_embed, obs_mask = self.obs_embed(batch)
        choice_embed, choice_mask = self.choice_embed(choices)

        assert (~obs_mask).any()
        assert (~choice_mask).any()

        x = torch.cat([obs_embed, choice_embed], dim=1)
        pos_mask = torch.cat([obs_mask, choice_mask], dim=1)

        for l in self.layers:
            x = l(x, pos_mask)
        xn = self.norm(x)

        action_xs = xn[:, obs_mask.size(1):, :]
        choice_logits = self.choice_logits(action_xs).squeeze(-1).float()
        choice_logits = choice_logits.masked_fill(choice_mask, float('-inf'))
        
        # Compute value if value head is enabled
        if self.H.use_value_head:
            # Pool over non-masked elements for value prediction
            # xn: [batch_size, seq_len, dim], pos_mask: [batch_size, seq_len]
            seq_lengths = (~pos_mask).sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            pooled = xn.masked_fill(pos_mask.unsqueeze(-1), 0).sum(dim=1) / seq_lengths  # [batch_size, dim]
            values = self.value_head(pooled).squeeze(-1).float()  # [batch_size]
            return choice_logits, values
        else:
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
    chosen_idx_list = []
    outcome_list = []
    
    # Create batched tensors for observation using fixed maximum sizes
    batch_obs = {
        'deck': {
            'value': torch.full((len(batch), MAX_DECK_SIZE, 2), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), MAX_DECK_SIZE), dtype=torch.bool)
        },
        'relics': {
            'value': torch.full((len(batch), MAX_RELICS), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), MAX_RELICS), dtype=torch.bool)
        },
        'potions': {
            'value': torch.full((len(batch), sts.MAX_POTION_CAPACITY), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), sts.MAX_POTION_CAPACITY), dtype=torch.bool)
        },
        'fixed_obs': torch.zeros((len(batch), len(sts.getFixedObservationMaximums())), dtype=torch.int32)
    }
    
    # Create batched tensors for choices using fixed maximum sizes
    batch_choices = {
        'cards': {
            'value': torch.full((len(batch), MAX_CHOICES, 2), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), MAX_CHOICES), dtype=torch.bool)
        },
        'relics': {
            'value': torch.full((len(batch), MAX_CHOICES), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), MAX_CHOICES), dtype=torch.bool)
        },
        'potions': {
            'value': torch.full((len(batch), sts.MAX_POTION_CAPACITY), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), sts.MAX_POTION_CAPACITY), dtype=torch.bool)
        },
        'fixed': {
            'value': torch.full((len(batch), MAX_FIXED_ACTIONS), FixedAction.INVALID.value, dtype=torch.int32),
            'mask': torch.ones((len(batch), MAX_FIXED_ACTIONS), dtype=torch.bool)
        }
    }
    
    # Fill the batched tensors with assertions to ensure data fits
    for i, x in enumerate(batch):
        # Build observation data inline
        deck_cards = x['obs.deck.cards']
        deck_upgrades = x['obs.deck.upgrades']
        deck_len = len(deck_cards)
        assert deck_len <= MAX_DECK_SIZE, f"Deck size {deck_len} exceeds maximum {MAX_DECK_SIZE}"
        batch_obs['deck']['value'][i, :deck_len] = torch.tensor(list(zip(deck_cards, deck_upgrades)), dtype=torch.int32)
        batch_obs['deck']['mask'][i, :deck_len] = torch.zeros(deck_len, dtype=torch.bool)
        
        relics = x['obs.relics.relics']
        relics_len = len(relics)
        assert relics_len <= MAX_RELICS, f"Relics count {relics_len} exceeds maximum {MAX_RELICS}"
        batch_obs['relics']['value'][i, :relics_len] = torch.tensor(relics, dtype=torch.int32)
        batch_obs['relics']['mask'][i, :relics_len] = torch.zeros(relics_len, dtype=torch.bool)
        
        potions = x['obs.potions']
        potions_len = len(potions)
        assert potions_len <= sts.MAX_POTION_CAPACITY, f"Potions count {potions_len} exceeds maximum {sts.MAX_POTION_CAPACITY}"
        batch_obs['potions']['value'][i, :potions_len] = torch.tensor(potions, dtype=torch.int32)
        batch_obs['potions']['mask'][i, :potions_len] = torch.zeros(potions_len, dtype=torch.bool)
        
        batch_obs['fixed_obs'][i] = torch.tensor(x['obs.fixed_observation'], dtype=torch.int32)
        
        # Build choices data inline
        cards_offered = x['cards_offered.cards']
        cards_upgrades = x['cards_offered.upgrades']
        choice_cards_len = len(cards_offered)
        assert choice_cards_len <= MAX_CHOICES, f"Choice cards count {choice_cards_len} exceeds maximum {MAX_CHOICES}; {cards_offered}; {relics}"
        if choice_cards_len > 0:
            batch_choices['cards']['value'][i, :choice_cards_len] = torch.tensor(list(zip(cards_offered, cards_upgrades)), dtype=torch.int32).reshape(-1, 2)
        batch_choices['cards']['mask'][i, :choice_cards_len] = torch.zeros(choice_cards_len, dtype=torch.bool)
        
        relics_offered = x['relics_offered']
        choice_relics_len = len(relics_offered)
        assert choice_relics_len <= MAX_CHOICES, f"Choice relics count {choice_relics_len} exceeds maximum {MAX_CHOICES}"
        batch_choices['relics']['value'][i, :choice_relics_len] = torch.tensor(relics_offered, dtype=torch.int32)
        batch_choices['relics']['mask'][i, :choice_relics_len] = torch.zeros(choice_relics_len, dtype=torch.bool)
        
        potions_offered = x['potions_offered']
        choice_potions_len = len(potions_offered)
        assert choice_potions_len <= sts.MAX_POTION_CAPACITY, f"Choice potions count {choice_potions_len} exceeds maximum {sts.MAX_POTION_CAPACITY}"
        batch_choices['potions']['value'][i, :choice_potions_len] = torch.tensor(potions_offered, dtype=torch.int32)
        batch_choices['potions']['mask'][i, :choice_potions_len] = torch.zeros(choice_potions_len, dtype=torch.bool)
        
        fixed_actions = x['fixed_actions']
        choice_fixed_len = len(fixed_actions)
        assert choice_fixed_len <= MAX_FIXED_ACTIONS, f"Choice fixed actions count {choice_fixed_len} exceeds maximum {MAX_FIXED_ACTIONS}"
        batch_choices['fixed']['value'][i, :choice_fixed_len] = torch.tensor(fixed_actions, dtype=torch.int32)
        batch_choices['fixed']['mask'][i, :choice_fixed_len] = torch.zeros(choice_fixed_len, dtype=torch.bool)
        
        chosen_idx_list.append(x['chosen_idx'])
        outcome_list.append(x['outcome'])
    
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

def output_to_cpu(output, batch: dict):
    """Convert network output to CPU numpy arrays."""
    if isinstance(output, tuple):
        # Handle (choice_logits, values) tuple
        choice_logits, values = output
        return choice_logits.cpu().numpy(), values.cpu().numpy()
    else:
        # Handle single choice_logits tensor
        return output.cpu().numpy()



# %%
