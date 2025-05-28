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
    INVALID = -1
    SKIP = 0
    REMOVE = 1

obs_space = DictSpace({
    'deck': SequenceSpace(TupleAddSpace(EnumSpace(sts.CardId), IntSpace(MAX_UPGRADE))),
    'relics': SequenceSpace(EnumSpace(sts.RelicId)),
    'fixed_obs': FixedVecSpace(sts.getFixedObservationMaximums()),
})

action_logit_space = DictSpace({
    'deck': SequenceSpace(TupleAddSpace(EnumSpace(sts.CardId), IntSpace(MAX_UPGRADE))),
    'relics': SequenceSpace(EnumSpace(sts.RelicId)),
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

        self.input_type_embed = nn.Embedding(len(InputType), H.dim)
        self.num_embed = SinusoidalEmbedding(H.dim)
        self.card_embed = nn.Embedding(len(sts.CardId), H.dim, padding_idx=sts.CardId.INVALID.value)
        self.upgrade_embed = nn.Embedding(21, H.dim, padding_idx=0)
        
        # Add fixed action embedding
        self.fixed_action_embed = nn.Embedding(len(FixedAction)-1, H.dim)

        # Add sinusoidal embedding and projection
        n_fixed_obs = len(sts.getFixedObservationMaximums())
        self.fixed_obs_embed = SinusoidalEmbedding(H.dim, n_fixed_obs)
        self.fixed_obs_proj = nn.Linear(self.fixed_obs_embed.out_dim, H.dim)

        # Single relic embedding layer used for both owned and offered relics
        self.relic_embed = nn.Embedding(len(sts.RelicId), H.dim, padding_idx=sts.RelicId.INVALID.value)
        
        # Add relic winprob head
        self.relic_winprob = nn.Linear(H.dim, 1, bias=True)
        nn.init.uniform_(self.relic_winprob.weight, -0.01, 0.01)
        nn.init.zeros_(self.relic_winprob.bias)

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
                - relics: [batch_size, max_relics] tensor of relic IDs
        """
        device = batch['deck'].device
        max_deck_len = batch['deck'].size(1)
        max_choices_len = batch['choices'].size(1)

        # Create embeddings list to concatenate
        embeddings = []

        # Add fixed observations
        fixed_obs_x = self.fixed_obs_proj(self.fixed_obs_embed(batch['fixed_obs']))
        embeddings.append(fixed_obs_x.unsqueeze(1))  # [batch, 1, dim]

        # Embed cards
        cards = torch.cat((batch['deck'], batch['choices']), dim=1)
        upgrades = torch.cat((batch['deck_upgrades'], batch['choice_upgrades']), dim=1)
        card_mask = cards == sts.CardId.INVALID.value

        card_x = (self.card_embed(cards) +
                 self.upgrade_embed(upgrades.clamp(max=20)) +
                 self.input_type_embed(torch.tensor([int(InputType.Card)], device=device)))
        card_x[:, max_deck_len:, :] += self.input_type_embed(torch.tensor([int(InputType.Choice)], device=device))
        embeddings.append(card_x)

        # Add relic embeddings
        relic_x = self.relic_embed(batch['relics'])
        relic_x = relic_x + self.input_type_embed(torch.tensor([int(InputType.Relic)], device=device))
        embeddings.append(relic_x)
        relic_mask = batch['relics'] == sts.RelicId.INVALID.value

        # Add fixed action embeddings
        fixed_x = self.fixed_action_embed(batch['fixed_actions'])
        fixed_x = fixed_x + self.input_type_embed(torch.tensor([int(InputType.Fixed)], device=device))
        embeddings.append(fixed_x)

        # Add relic choice embeddings using same embedding layer
        relic_choice_x = self.relic_embed(batch['relics_offered'])
        relic_choice_x = relic_choice_x + self.input_type_embed(torch.tensor([int(InputType.Choice)], device=device))
        embeddings.append(relic_choice_x)
        relic_choice_mask = batch['relics_offered'] == sts.RelicId.INVALID.value

        # Combine all embeddings
        x = torch.cat(embeddings, dim=1)

        # Combine masks
        pos_mask = torch.cat([
            torch.zeros(card_mask.size(0), 1, device=device, dtype=card_mask.dtype),  # fixed obs
            card_mask,  # cards
            relic_mask,  # owned relics
            relic_choice_mask,  # relic choices
            batch['fixed_actions'] == FixedAction.INVALID.value,  # fixed actions
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
        card_logits = card_logits.masked_fill(card_mask[:, max_deck_len:], float('-inf'))
        fixed_action_logits = fixed_action_logits.masked_fill(
            batch['fixed_actions'] == FixedAction.INVALID.value, 
            float('-inf')
        )

        # Get logits for relics
        relic_choice_x = xn[:, 1+max_deck_len+ModelHP.max_relics:1+max_deck_len+ModelHP.max_relics+batch['relics_offered'].size(1), :]
        relic_logits = self.relic_winprob(relic_choice_x).squeeze(-1).float()
        relic_logits = relic_logits.masked_fill(relic_choice_mask, float('-inf'))

        return dict(
            card_logits=card_logits,
            fixed_logits=fixed_action_logits,
            relic_logits=relic_logits,
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
            col: row[col]
            for col in self.df.columns
        }


def collate_fn(batch):
    for x in batch:
        n_card_choices = len(x['cards_offered.cards'])
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
    relics = torch.full((len(batch), ModelHP.max_relics), sts.RelicId.INVALID.value, dtype=torch.int32)
    relics_offered = torch.full((len(batch), 3), sts.RelicId.INVALID.value, dtype=torch.int32)  # Max 3 boss relics
    chosen_idx = torch.zeros(len(batch), dtype=torch.int64)
    choice_type = torch.zeros(len(batch), dtype=torch.int64)
    outcome = torch.zeros(len(batch), dtype=torch.float32)

    # Fill arrays
    for i, x in enumerate(batch):
        deck[i, :min(len(x['obs.deck.cards']), MAX_DECK_SIZE)] = torch.tensor(x['obs.deck.cards'])[:MAX_DECK_SIZE]
        deck_upgrades[i, :min(len(x['obs.deck.upgrades']), MAX_DECK_SIZE)] = torch.tensor(x['obs.deck.upgrades'])[:MAX_DECK_SIZE]
        choices[i, :min(len(x['cards_offered.cards']), MAX_CHOICES)] = torch.tensor(x['cards_offered.cards'])[:MAX_CHOICES]
        choice_upgrades[i, :min(len(x['cards_offered.upgrades']), MAX_CHOICES)] = torch.tensor(x['cards_offered.upgrades'])[:MAX_CHOICES]
        fixed_obs[i] = torch.tensor(x['obs.fixed_observation'])
        fixed_actions[i, :len(x['fixed_actions'])] = torch.tensor(x['fixed_actions'])
        relics[i, :len(x['obs.relics.relics'])] = torch.tensor(x['obs.relics.relics'])
        relics_offered[i, :len(x['relics_offered'])] = torch.tensor(x['relics_offered'])
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
        'relics': relics,
        'relics_offered': relics_offered,
        'chosen_idx': chosen_idx,
        'choice_type': choice_type,
        'outcome': outcome,
    }

def process_batch(batch, net):
    # Move tensors to device
    device = net.device
    batch = {k: v.to(device) for k, v in batch.items()}
    return net(batch)

def output_to_cpu(output: dict[str, torch.Tensor], batch: dict) -> list[dict[str, np.ndarray]]:
    """
    Moves tensors to CPU and trims them to valid lengths.
    
    Returns:
        List of dictionaries containing trimmed numpy arrays of logits, one per batch item
    """
    batch_size = output['card_logits'].size(0)
    results = []
    
    # Move tensors to CPU once
    card_logits = output['card_logits'].cpu().numpy()
    relic_logits = output['relic_logits'].cpu().numpy()
    fixed_logits = output['fixed_logits'].cpu().numpy()
    
    for i in range(batch_size):
        # Use boolean masks to select valid entries
        card_mask = batch['choices'][i].cpu().numpy() != sts.CardId.INVALID.value
        relic_mask = batch['relics_offered'][i].cpu().numpy() != sts.RelicId.INVALID.value
        fixed_mask = batch['fixed_actions'][i].cpu().numpy() != FixedAction.INVALID.value
        
        # Trim logits to valid entries
        results.append({
            'card_logits': card_logits[i][card_mask],
            'relic_logits': relic_logits[i][relic_mask],
            'fixed_logits': fixed_logits[i][fixed_mask],
        })
    
    return results



# %%
