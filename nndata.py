# %%
import sys
import random
from enum import IntEnum, auto
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import slaythespire as sts

# %%
# todo pyarrow everything
@dataclass
class Choice:
    obs: sts.NNRepresentation
    actions: list[sts.GameAction]
    cards_offered: list[sts.NNCardRepresentation]
    paths_offered: list[int]  # room ids (indices in NNMapRepresentation vectors)

# %%

seed = 777
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
print(gc.map)

agent = sts.Agent()
agent.simulation_count_base = 5000

# %%
choices: list[Choice] = []

# %%

while gc.outcome == sts.GameOutcome.UNDECIDED:
    if gc.screen_state == sts.ScreenState.BATTLE:
        # print(gc)
        agent.playout_battle(gc)
        obs = sts.getNNRepresentation(gc)
        # print(gc)
    else:
        chosen = None
        actions = sts.GameAction.getAllActionsInState(gc)
        obs = sts.getNNRepresentation(gc)
        cards_offered: list[sts.NNCardRepresentation] = []
        paths_offered: list[int] = []
        if len(actions) == 1:
            chosen, = actions
        elif gc.screen_state == sts.ScreenState.REWARDS:
            cards_offered = gc.screen_state_info.rewards_container.cards
            for a in actions:
                if a.rewards_action_type in (
                    sts.RewardsActionType.GOLD, sts.RewardsActionType.POTION, sts.RewardsActionType.RELIC,
                ):
                    chosen = a
                    break
        elif gc.screen_state == sts.ScreenState.EVENT_SCREEN:
            # TODO neow, events
            chosen = random.choice(actions)
        elif gc.screen_state == sts.ScreenState.MAP_SCREEN:
            def xy_to_roomid(x, y):
                roomid, = [i for i in range(len(obs.map.xs)) if obs.map.xs[i] == x and obs.map.ys[i] == y]
                return roomid
            paths_offered = [xy_to_roomid(a.idx1, gc.cur_map_node_y+1) for a in actions]
        if chosen is None:
            choices.append(Choice(obs, actions, cards_offered=cards_offered, paths_offered=paths_offered))
            chosen = random.choice(actions)
        if chosen is not None:
            print(chosen.getDesc(gc))
            chosen.execute(gc)
        else:
            break

print(gc.outcome)

# %%
for c in choices:
    if c.paths_offered:
        print(c.paths_offered)
    if c.cards_offered:
        for cp in c.cards_offered:
            print(cp.cards)

# %%
deck = choices[-1].obs.deck
list(zip(deck.cards, deck.upgrades))

# %%
gc.screen_state_info.rewards_container.relics

# %%
nnrep = sts.getNNRepresentation(gc)
nnrep.map.room_types

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
        self.attn = nn.MultiheadAttention(H.dim, H.n_heads)
        self.norm2 = RMSNorm(H.dim, eps=H.norm_eps)
        self.ffn = FFN(H.dim, H.dim * H.ffn_dim_mult)

    def forward(self, x):
        xn = self.norm1(x)
        xatt, _ = self.attn(xn, xn, xn)
        x1 = x + xatt
        x2 = x1 + self.ffn(self.norm2(x1))
        return x2

class NN(nn.Module):
    def __init__(self, H: ModelHP):
        super().__init__()
        self.H = H

        self.input_type_embed = nn.Embedding(len(InputType), H.dim)
        self.card_embed = nn.Embedding(len(sts.CardId), H.dim)

        self.layers = nn.ModuleList([TransformerBlock(H=H) for _ in range(H.n_layers)])

        self.norm = RMSNorm(H.dim, H.norm_eps)
        self.card_out = nn.Linear(H.dim, 1, bias=True)
        nn.init.uniform_(self.card_out.weight, -0.01, 0.01)
        nn.init.zeros_(self.card_out.bias)

    def forward(self, deck, card_choices):
        cards = torch.cat((deck, card_choices), dim=-1)
        card_x = self.card_embed(cards) + self.input_type_embed(torch.tensor([int(InputType.Card)]))
        card_x[..., len(deck):, :] += self.input_type_embed(torch.tensor([int(InputType.Choice)]))

        x = card_x
        for l in self.layers:
            x = l(x)
        xn = self.norm(x)
        card_choice_logits = self.card_out(xn[..., len(deck):, :]).squeeze(-1).float()
        return dict(
            card_choice_logits=card_choice_logits
        )

# %%
H = ModelHP()

net = NN(H)
# %%
rep = sts.getNNRepresentation(gc)
cards = torch.tensor([int(c) for c in rep.deck.cards], dtype=torch.int32)
card_choices = torch.tensor([int(c) for c in gc.screen_state_info.rewards_container.cards[0].cards], dtype=torch.int32)
# %%
out = net(cards, card_choices)

# %%
torch.softmax(out["card_choice_logits"], dim=-1)

# %%
