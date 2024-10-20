# %%
import sys
import random
from enum import IntEnum, auto
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

import slaythespire as sts

# %%

@dataclass
class ModelHP:
    dim: int = 256
    ffn_dim_mult: int = 4
    n_layers: int = 4
    n_heads: int = 8
    norm_eps: float = 1e-5

class InputType(IntEnum):
    Card = 0
    Relic = auto()
    Potion = auto()
    Choice = auto()


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

    def forward(self, x, mask):
        xn = self.norm1(x)
        xatt, _ = self.attn(xn, xn, xn, key_padding_mask=mask)
        x1 = x + xatt
        x2 = x1 + self.ffn(self.norm2(x1))
        return x2

class NN(nn.Module):
    def __init__(self, H: ModelHP):
        super().__init__()
        self.H = H

        self.input_type_embed = nn.Embedding(len(InputType), H.dim)
        self.card_embed = nn.Embedding(len(sts.CardId), H.dim, padding_idx=sts.CardId.INVALID.value)

        self.layers = nn.ModuleList([TransformerBlock(H=H) for _ in range(H.n_layers)])

        self.norm = RMSNorm(H.dim, H.norm_eps)
        self.card_out = nn.Linear(H.dim, 1, bias=True)
        nn.init.uniform_(self.card_out.weight, -0.01, 0.01)
        nn.init.zeros_(self.card_out.bias)

    def forward(self, deck, card_choices):
        max_deck_len = deck.size(1)
        max_choices_len = card_choices.size(1)

        cards = torch.cat((deck, card_choices), dim=1)
        mask = cards == sts.CardId.INVALID.value
        card_x = self.card_embed(cards) + self.input_type_embed(torch.tensor([int(InputType.Card)]))
        card_x[:, max_deck_len:, :] += self.input_type_embed(torch.tensor([int(InputType.Choice)]))

        x = card_x
        for l in self.layers:
            x = l(x, mask)
        xn = self.norm(x)
        card_choice_logits = self.card_out(xn[:, max_deck_len:, :]).squeeze(-1).float()
        card_choice_logits = card_choice_logits.masked_fill(mask[:, max_deck_len:], float('-inf'))
        return dict(
            card_choice_logits=card_choice_logits,
        )

def pad_sequence(sequences, max_len, device):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s, device=device) for s in sequences],
        batch_first=True,
        padding_value=sts.CardId.INVALID.value
    )[:, :max_len]

# %%
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# %%
H = ModelHP()

net = NN(H)
net = net.to(device)
# %%
df = pd.read_parquet("rollouts.parquet")

# %%
data_df: pd.DataFrame = df[df["cards_offered.cards"].apply(lambda c: len(c) > 0)]
len(data_df)
# %%
# Split into train and validation sets, ensuring rows with the same seed stay together
valid_seeds = data_df['seed'].drop_duplicates().sample(frac=0.1, random_state=42)
valid_df = data_df[data_df['seed'].isin(valid_seeds)]
train_df = data_df[~data_df['seed'].isin(valid_seeds)]
print(len(train_df), len(valid_df))
# %%
np.random.seed(3)

batch = train_df.sample(16)[["obs.deck.cards", "cards_offered.cards", "choice"]]


# %%
def feed_to_net(df):
    max_deck_len = max(len(d) for d in batch["obs.deck.cards"])
    max_choices_len = max(len(c) for c in batch["cards_offered.cards"])

    deck = pad_sequence(batch["obs.deck.cards"].apply(lambda x: x.astype(np.int32)), max_deck_len, device=device)
    choices = pad_sequence(batch["cards_offered.cards"].apply(lambda x: x.astype(np.int32)), max_choices_len, device=device)

    return net(deck, choices)
# %%
output = feed_to_net(batch)
print(output['card_choice_logits'])

# %%
# Test padding with longer batch element
long_batch = batch.copy()
long_batch.at[long_batch.index[0], "obs.deck.cards"] #  = np.array([7] * 100)  # Big deck
# long_batch.at[long_batch.index[0], "cards_offered.cards"] = np.array([1, 2, 3, 4, 5])  # More card choices
# %%
long_output = feed_to_net(long_batch)

# Verify that outputs for unchanged elements are the same
assert torch.allclose(output['card_choice_logits'][1:], long_output['card_choice_logits'][1:,:output['card_choice_logits'].size(1)], rtol=1e-4, atol=1e-6)
print("Padding test passed: outputs for unchanged elements are the same.")
# %%
loss = F.nll_loss(F.log_softmax(output['card_choice_logits'], dim=1), torch.tensor(batch['choice'].to_numpy(), device=device))

# %%
batch['choice'].to_numpy()
# %%
output['card_choice_logits'].shape
