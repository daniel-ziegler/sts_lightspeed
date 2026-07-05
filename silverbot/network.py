from collections import abc
from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint  # submodule isn't auto-imported by `import torch`; forward() needs it

import slaythespire as sts
from silverbot.inputs import FixedVecSpace, SequenceSpace, EnumSpace, DictSpace, TupleAddSpace, IntSpace, DictAddSpace, ScalarToSequenceSpace, EmbedCache, FixedVecEmbedding

@dataclass
class ModelHP:
    dim: int = 256
    mlp_dim_mult: int = 4
    n_layers: int = 4
    n_heads: int = 8
    norm_eps: float = 1e-5
    n_fixed_obs: int = len(sts.getFixedObservationMaximums())
    use_value_head: bool = False  # Add value head for PPO training
    num_value_layers: int = 0  # Number of separate transformer layers for value function
    value_fork_layer: int = 0  # How many layers from the end to fork value layers (0 = after all shared layers)


# Constants for data processing
MAX_DECK_SIZE = 96  # Should be enough for most decks
MAX_UPGRADE = 21
MAX_RELICS = 40     # collate padding cap; 25 overflowed once chests started granting relics
MAX_FIXED_ACTIONS = 5  # Maximum number of fixed actions in choices
MAX_MAP_NODES = 100
MAX_GOLD = 1000
MAX_PATH_CHOICES = 7  # all possible columns from the start
# Flat-logit offset of the paths section: collate_fn pads the choice sections to fixed widths
# (cards MAX_DECK_SIZE, relics 3, potions MAX_POTION_CAPACITY, fixed MAX_FIXED_ACTIONS) and
# DictSpace consumes them in declaration order, so path-option tokens start here.
CHOICE_PATHS_OFFSET = MAX_DECK_SIZE + 3 + sts.MAX_POTION_CAPACITY + MAX_FIXED_ACTIONS


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
    REST = auto()
    SMITH = auto()
    RECALL = auto()
    LIFT = auto()
    TOKE = auto()
    DIG = auto()
    
    # Event Actions - organized by event type
    # Neow bonus
    NEOW_THREE_CARDS = auto()
    NEOW_ONE_RANDOM_RARE_CARD = auto()
    NEOW_REMOVE_CARD = auto()
    NEOW_UPGRADE_CARD = auto()
    NEOW_TRANSFORM_CARD = auto()
    NEOW_RANDOM_COLORLESS = auto()

    NEOW_THREE_SMALL_POTIONS = auto()
    NEOW_RANDOM_COMMON_RELIC = auto()
    NEOW_TEN_PERCENT_HP_BONUS = auto()
    NEOW_THREE_ENEMY_KILL = auto()
    NEOW_HUNDRED_GOLD = auto()

    NEOW_RANDOM_COLORLESS_2 = auto()
    NEOW_REMOVE_TWO = auto()
    NEOW_ONE_RARE_RELIC = auto()
    NEOW_THREE_RARE_CARDS = auto()
    NEOW_TWO_FIFTY_GOLD = auto()
    NEOW_TRANSFORM_TWO_CARDS = auto()
    NEOW_TWENTY_PERCENT_HP_BONUS = auto()

    NEOW_BOSS_RELIC = auto()
    
    # Big Fish (3 choices)
    BIG_FISH_BANANA = auto()  # Heal
    BIG_FISH_DONUT = auto()   # Max HP
    BIG_FISH_BOX = auto()     # Relic
    
    # Face Trader (3 choices)
    FACE_TRADER_LOSE_GOLD = auto()
    FACE_TRADER_LOSE_HP = auto()
    FACE_TRADER_LEAVE = auto()
    
    # Golden Shrine (3 choices)
    GOLDEN_SHRINE_PRAY = auto()
    GOLDEN_SHRINE_DESECRATE = auto()
    GOLDEN_SHRINE_LEAVE = auto()
    
    # N'loth (3 choices)
    NLOTH_OFFER_0 = auto()
    NLOTH_OFFER_1 = auto()
    NLOTH_LEAVE = auto()
    
    # Sensory Stone (3 choices)
    SENSORY_STONE_MEMORIES = auto()
    SENSORY_STONE_COLORLESS = auto()
    SENSORY_STONE_LEAVE = auto()
    
    # Winding Halls (3 choices)
    WINDING_HALLS_MADNESS = auto()
    WINDING_HALLS_WRITHE = auto()
    WINDING_HALLS_LEAVE = auto()
    
    # Knowing Skull (4 choices)
    KNOWING_SKULL_OPTION_0 = auto()
    KNOWING_SKULL_OPTION_1 = auto()
    KNOWING_SKULL_OPTION_2 = auto()
    KNOWING_SKULL_OPTION_3 = auto()
    
    # Woman in Blue (4 choices)
    WOMAN_IN_BLUE_OPTION_0 = auto()
    WOMAN_IN_BLUE_OPTION_1 = auto()
    WOMAN_IN_BLUE_OPTION_2 = auto()
    WOMAN_IN_BLUE_OPTION_3 = auto()
    
    # Two-choice events (0x3 = bits 0,1)
    ANCIENT_WRITING_ELEGANCE = auto()
    ANCIENT_WRITING_SIMPLICITY = auto()
    
    DEAD_ADVENTURER_SEARCH = auto()
    DEAD_ADVENTURER_LEAVE = auto()
    
    DUPLICATOR_DUPLICATE = auto()
    DUPLICATOR_LEAVE = auto()
    
    OLD_BEGGAR_GIVE_GOLD = auto()
    OLD_BEGGAR_REFUSE = auto()
    
    DIVINE_FOUNTAIN_HEAL = auto()
    DIVINE_FOUNTAIN_LEAVE = auto()
    
    GHOSTS_AGREE = auto()
    GHOSTS_REFUSE = auto()
    
    SSSSSERPENT_AGREE = auto()
    SSSSSERPENT_DISAGREE = auto()
    
    MASKED_BANDITS_PAY = auto()
    MASKED_BANDITS_FIGHT = auto()
    
    MUSHROOMS_HEAL = auto()
    MUSHROOMS_LEAVE = auto()
    
    MYSTERIOUS_SPHERE_OPEN = auto()
    MYSTERIOUS_SPHERE_LEAVE = auto()
    
    NEST_AGREE = auto()
    NEST_DISAGREE = auto()
    
    NOTE_FOR_YOURSELF_IGNORE = auto()
    NOTE_FOR_YOURSELF_WRITE = auto()
    
    SCRAP_OOZE_ATTACK = auto()
    SCRAP_OOZE_LEAVE = auto()
    
    SECRET_PORTAL_ENTER = auto()
    SECRET_PORTAL_LEAVE = auto()
    
    SHINING_LIGHT_ENTER = auto()
    SHINING_LIGHT_LEAVE = auto()
    
    JOUST_GIVE_GOLD = auto()
    JOUST_REFUSE = auto()
    
    LIBRARY_READ = auto()
    LIBRARY_LEAVE = auto()
    
    MAUSOLEUM_OPEN = auto()
    MAUSOLEUM_LEAVE = auto()
    
    WORLD_OF_GOOP_ENTER = auto()
    WORLD_OF_GOOP_LEAVE = auto()
    
    # Single choice events (0x1 = bit 0 only)
    LAB_OPTION = auto()
    WHEEL_OF_CHANGE_OPTION = auto()
    
    # Pleading Vagrant (conditional)
    PLEADING_VAGRANT_GIVE_GOLD = auto()
    PLEADING_VAGRANT_REFUSE = auto()
    PLEADING_VAGRANT_LEAVE = auto()
    
    # Colosseum (two phases)
    COLOSSEUM_PHASE1_PROCEED = auto()
    COLOSSEUM_PHASE2_OPTION_0 = auto()
    COLOSSEUM_PHASE2_OPTION_1 = auto()
    
    # Cursed Tome (complex multi-phase)
    CURSED_TOME_READ = auto()
    CURSED_TOME_LEAVE = auto()
    CURSED_TOME_PHASE1_OPTION = auto()
    CURSED_TOME_PHASE2_OPTION = auto()
    CURSED_TOME_PHASE3_OPTION = auto()
    CURSED_TOME_PHASE4_OPTION_0 = auto()
    CURSED_TOME_PHASE4_OPTION_1 = auto()
    
    # Designer In-Spire (complex conditional)
    DESIGNER_UPGRADE_ONE = auto()
    DESIGNER_UPGRADE_ALL = auto()
    DESIGNER_REMOVE_CARD = auto()
    DESIGNER_TRANSFORM_TWO = auto()
    DESIGNER_TRANSFORM_ONE = auto()
    DESIGNER_LEAVE = auto()
    
    # Augmenter (conditional)
    AUGMENTER_AGREE = auto()
    AUGMENTER_REFUSE = auto()
    AUGMENTER_LEAVE = auto()
    
    # Falling (conditional based on deck)
    FALLING_SKILL = auto()
    FALLING_POWER = auto()
    FALLING_ATTACK = auto()
    FALLING_LEAVE = auto()
    
    # Forgotten Altar (conditional)
    FORGOTTEN_ALTAR_PRAY = auto()
    FORGOTTEN_ALTAR_DESECRATE = auto()
    FORGOTTEN_ALTAR_LEAVE = auto()
    
    # Golden Idol (two phases)
    GOLDEN_IDOL_TAKE = auto()
    GOLDEN_IDOL_LEAVE = auto()
    GOLDEN_IDOL_PHASE2_OPTION_0 = auto()
    GOLDEN_IDOL_PHASE2_OPTION_1 = auto()
    GOLDEN_IDOL_PHASE2_OPTION_2 = auto()
    
    # Wing Statue (conditional)
    WING_STATUE_REMOVE_CARD = auto()
    WING_STATUE_LOSE_GOLD = auto()
    WING_STATUE_LEAVE = auto()
    
    # Living Wall (conditional)
    LIVING_WALL_CHANGE = auto()
    LIVING_WALL_GROW = auto()
    LIVING_WALL_LEAVE = auto()
    
    # Mindbloom (conditional based on floor)
    MINDBLOOM_ACT1_BOSS = auto()
    MINDBLOOM_UPGRADE_CARDS = auto()
    MINDBLOOM_TRANSFORM = auto()
    MINDBLOOM_HEAL = auto()
    
    # Purifier (conditional)
    PURIFIER_PURIFY = auto()
    PURIFIER_LEAVE = auto()
    
    # Transmorgrifier (conditional)
    TRANSMORGRIFIER_TRANSFORM = auto()
    TRANSMORGRIFIER_LEAVE = auto()
    
    # The Cleric (conditional)
    CLERIC_HEAL = auto()
    CLERIC_PURIFY = auto()
    CLERIC_LEAVE = auto()
    
    # Moai Head (conditional)
    MOAI_HEAD_GOLDEN_IDOL = auto()
    MOAI_HEAD_LOSE_GOLD = auto()
    MOAI_HEAD_LEAVE = auto()
    
    # Tomb of Lord Red Mask (conditional)
    TOMB_RED_MASK_DON_MASK = auto()  # Don the Red Mask (if you have it)
    TOMB_RED_MASK_OFFER_GOLD = auto()  # Offer gold for Red Mask (if you don't have it)
    TOMB_RED_MASK_LEAVE = auto()
    
    # Upgrade Shrine (conditional)
    UPGRADE_SHRINE_UPGRADE = auto()
    UPGRADE_SHRINE_LEAVE = auto()
    
    # Vampires (conditional)
    VAMPIRES_ACCEPT = auto()
    VAMPIRES_REFUSE = auto()
    VAMPIRES_BLOOD_VIAL = auto()
    
    # We Meet Again (conditional)
    WE_MEET_AGAIN_POTION = auto()
    WE_MEET_AGAIN_GOLD = auto()
    WE_MEET_AGAIN_CARD = auto()
    WE_MEET_AGAIN_LEAVE = auto()
    
    # Ominous Forge (conditional)
    OMINOUS_FORGE_UPGRADE = auto()
    OMINOUS_FORGE_LOSE_HP = auto()
    OMINOUS_FORGE_LEAVE = auto()

    # Combat/chest reward: take the sapphire or emerald key (RewardsActionType.KEY).
    # Appended last so older checkpoints' FixedAction embeddings load unchanged.
    TAKE_KEY = auto()

    # Treasure room: open or skip the chest (skip is FixedAction.SKIP).
    OPEN_CHEST = auto()


class EventFixedInfo(IntEnum):
    NONE = 0
    NEOW_TEN_PERCENT_HP_LOSS = auto()
    NEOW_NO_GOLD = auto()
    NEOW_CURSE = auto()
    NEOW_PERCENT_DAMAGE = auto()
    NEOW_LOSE_STARTER_RELIC = auto()

class IsCurrentNode(IntEnum):
    NOT_CURRENT = 0
    CURRENT = 1

card_space = EnumSpace(sts.CardId)
relic_space = EnumSpace(sts.RelicId)
potion_space = EnumSpace(sts.Potion)
upgrade_space = IntSpace(MAX_UPGRADE)


class IsReachable(IntEnum):
    """Whether a map node is reachable from the current position's forward frontier."""
    NO = 0
    YES = 1


# Cap for the dist-to-rest aggregate; also the [0,1] scale divisor for all map aggregates.
# Map aggregate features are SCALED into [0,1]: unscaled magnitudes inside the DictAdd token
# sum drown the other components (measured: unscaled aggregates cost ~40pp on unrelated tasks).
MAP_AGG_CAP = 15

# Boss id, ascension, and the act-4 key flags are categorical; the remaining
# fixed-observation scalars stay sinusoidal.
_FIXED_OBS_MAXES = list(sts.getFixedObservationMaximums())
_BOSS_OBS_IDX = 4
_ASC_OBS_IDX = 6
_KEY_OBS_IDXS = (7, 8, 9)  # ruby, emerald, sapphire
_FIXED_OBS_SCALAR_MAXES = [m for i, m in enumerate(_FIXED_OBS_MAXES)
                           if i not in (_BOSS_OBS_IDX, _ASC_OBS_IDX) + _KEY_OBS_IDXS]

obs_space = DictSpace({
    'deck': SequenceSpace(TupleAddSpace(card_space, upgrade_space)),
    'relics': SequenceSpace(relic_space),
    'potions': SequenceSpace(potion_space),
    'fixed_obs': ScalarToSequenceSpace(DictAddSpace({
        'fixed_observation': FixedVecSpace(_FIXED_OBS_SCALAR_MAXES),
        'boss': IntSpace(10),
        'ascension': IntSpace(21),
        'key_ruby': IntSpace(2),
        'key_emerald': IntSpace(2),
        'key_sapphire': IntSpace(2),
        'screen_state': EnumSpace(sts.ScreenState),
    })),
    # Per-node map features. Beyond the raw structure (room/pos/edges), nodes carry
    # ego-relative coords, a reachability flag, and scaled DAG aggregates (min/max elites
    # ahead, dist to nearest rest) -- these make path grounding and map queries learnable
    # at weak signal strength (see EXPERIMENT_LOG.md, repr lab).
    'map_nodes': SequenceSpace(DictAddSpace({
        'room': EnumSpace(sts.Room),
        'is_current': EnumSpace(IsCurrentNode),
        'pos': FixedVecSpace([7, 16]),
        'path_xs': FixedVecSpace([7, 7, 7]),  # can accept -1
        'rel': FixedVecSpace([15, 31]),       # (dx, dy) from current position
        'reachable': EnumSpace(IsReachable),
        'agg': FixedVecSpace([2, 2, 2]),      # (min_elites, max_elites, dist_rest) / MAP_AGG_CAP
        'burning': IntSpace(2),               # burning elite here (emerald key fight)
    })),
})

# Zero-initialized embedding variants: a new DictAdd component built from one of these is an
# exact no-op at init (it adds 0 to the token sum), so adding it to the obs/choice encoding
# warm-starts an existing checkpoint bit-identically -- the component only starts to matter once
# trained. Used for the per-path-choice forward-cone features added below.
class ZeroInitIntSpace(IntSpace):
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        def make():
            e = nn.Embedding(self.limit, dim)
            nn.init.zeros_(e.weight)
            return e
        return cache.build(self, dim, make)


class ZeroInitFixedVecSpace(FixedVecSpace):
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        def make():
            e = FixedVecEmbedding(dim, self.limits)
            nn.init.zeros_(e.proj.weight)
            nn.init.zeros_(e.proj.bias)
            return e
        return cache.build(self, dim, make)


choice_space = DictSpace({
    'cards': SequenceSpace(TupleAddSpace(card_space, upgrade_space, EnumSpace(sts.CardSelectScreenType))),
    'relics': SequenceSpace(relic_space),
    'potions': SequenceSpace(potion_space),
    'fixed': SequenceSpace(DictAddSpace({
        'action': EnumSpace(FixedAction),
        'gold': FixedVecSpace([MAX_GOLD]),
        'card': card_space,
        'relic': relic_space,
        'info': EnumSpace(EventFixedInfo),
    })),
    # Each path option carries its destination's room type alongside the x coordinate, so
    # option grounding is a lookup instead of a learned multi-hop attention program. It also
    # carries a forward-cone summary of what taking it leads to -- the destination node's
    # (minE, maxE, dist_rest) DAG aggregates (scaled) and a bit for whether the burning elite
    # (emerald key) is reachable down this option. Precomputing the lookahead onto the option
    # the policy scores is what makes multi-hop routing (e.g. reaching a burning elite 2-4 rows
    # ahead) learnable -- the SL repr-lab showed raw per-node reachability isn't aggregated by
    # the net at a realistic budget, while this per-choice cone solves it (EXPERIMENT_LOG
    # 2026-06-10). Zero-init so it warm-starts existing checkpoints bit-identically.
    'paths': SequenceSpace(DictAddSpace({
        'x': FixedVecSpace([7]),
        'room': EnumSpace(sts.Room),
        'cone': ZeroInitFixedVecSpace([2, 2, 2]),   # (minE, maxE, dist_rest) / MAP_AGG_CAP
        'reaches_burn': ZeroInitIntSpace(2),
    })),
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
        # need_weights=False lets MultiheadAttention dispatch to the fused
        # scaled_dot_product_attention kernel (faster; numerically equivalent output).
        xatt, _ = self.attn(xn, xn, xn, attn_mask=None, key_padding_mask=pos_mask, need_weights=False)
        x1 = x + xatt
        x2 = x1 + self.mlp(self.norm2(x1))
        return x2


class NN(nn.Module):
    def __init__(self, H: ModelHP):
        super().__init__()
        self.H = H

        embed_cache = EmbedCache()
        self.obs_embed = obs_space.build_embed(H.dim, embed_cache)
        self.choice_embed = choice_space.build_embed(H.dim, embed_cache)
        
        self.layers = nn.ModuleList([TransformerBlock(H=H) for _ in range(H.n_layers)])

        self.norm = RMSNorm(H.dim, H.norm_eps)

        self.choice_logits = nn.Linear(H.dim, 1, bias=True)
        nn.init.uniform_(self.choice_logits.weight, -0.01, 0.01)
        nn.init.zeros_(self.choice_logits.bias)

        # Auxiliary head: per-path-option destination-room classifier. Self-supervised
        # grounding scaffold (labels come free from the map obs at collate time).
        self.aux_room_head = nn.Linear(H.dim, len(sts.Room), bias=True)
        
        # Add value-specific layers if specified
        if H.num_value_layers > 0:
            self.value_layers = nn.ModuleList([TransformerBlock(H=H) for _ in range(H.num_value_layers)])
        
        # Add value head RMSNorm (always present if value head is used)
        if H.use_value_head:
            self.value_head_norm = RMSNorm(H.dim, H.norm_eps)
            self.value_head = nn.Linear(H.dim, 1, bias=True)
            nn.init.uniform_(self.value_head.weight, -0.01, 0.01)
            nn.init.zeros_(self.value_head.bias)


    def forward(self, batch: dict, return_aux: bool = False, return_pooled: bool = False):
        """
        Process a batch of inputs through the network.

        With return_aux=True, additionally returns aux_room_logits
        [batch, MAX_PATH_CHOICES, n_rooms] for the destination-room auxiliary loss.
        With return_pooled=True (value head required, exclusive with return_aux),
        additionally returns the pooled trunk embedding [batch, dim] that the value
        head reads — the attachment point for auxiliary prediction heads.

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
        choices = batch['choices']
        
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

        # Determine where to fork for value computation
        fork_point = len(self.layers) - self.H.value_fork_layer
        value_x = None
        
        # Use activation checkpointing for all but the last layer
        for i, l in enumerate(self.layers):
            # Save intermediate representation for value forking
            if self.H.use_value_head and self.H.num_value_layers > 0 and i == fork_point:
                value_x = x.clone()
            
            if i < len(self.layers) - 1:
                # Checkpoint all but the last layer
                x = torch.utils.checkpoint.checkpoint(l, x, pos_mask, use_reentrant=False)
            else:
                # Don't checkpoint the last layer
                x = l(x, pos_mask)
        xn = self.norm(x)

        action_xs = xn[:, obs_mask.size(1):, :]
        choice_logits = self.choice_logits(action_xs).squeeze(-1).float()
        choice_logits = choice_logits.masked_fill(choice_mask, float('-inf'))

        aux_room_logits = None
        if return_aux:
            path_tokens = action_xs[:, CHOICE_PATHS_OFFSET:CHOICE_PATHS_OFFSET + MAX_PATH_CHOICES, :]
            aux_room_logits = self.aux_room_head(path_tokens).float()

        # Compute value if value head is enabled
        if self.H.use_value_head:
            # Use separate value layers if specified
            if self.H.num_value_layers > 0:
                # Start from the forked representation (or final if fork_point >= n_layers)
                if value_x is None:
                    value_x = x
                    
                # Apply value-specific transformer layers
                for i, l in enumerate(self.value_layers):
                    if i < len(self.value_layers) - 1:
                        # Checkpoint all but the last value layer
                        value_x = torch.utils.checkpoint.checkpoint(l, value_x, pos_mask, use_reentrant=False)
                    else:
                        # Don't checkpoint the last value layer
                        value_x = l(value_x, pos_mask)
                # Apply value head normalization
                value_xn = self.value_head_norm(value_x)
            else:
                # Use the shared representation with value head normalization
                value_xn = self.value_head_norm(xn)
            
            # Pool over non-masked elements for value prediction
            # value_xn: [batch_size, seq_len, dim], pos_mask: [batch_size, seq_len]
            seq_lengths = (~pos_mask).sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            pooled = value_xn.masked_fill(pos_mask.unsqueeze(-1), 0).sum(dim=1) / seq_lengths  # [batch_size, dim]
            values = self.value_head(pooled).squeeze(-1).float()  # [batch_size]
            if return_aux:
                return choice_logits.clone(), values.clone(), aux_room_logits
            if return_pooled:
                return choice_logits.clone(), values.clone(), pooled
            return choice_logits.clone(), values.clone()
        else:
            if return_aux:
                return choice_logits.clone(), aux_room_logits
            return choice_logits.clone()
    
    @property
    def device(self):
        return next(self.parameters()).device


class BattleOutcomeHead(nn.Module):
    """Predicts a specific battle's ΔHP outcome (battle_buckets scheme) from the trunk's
    pooled embedding and the encounter. The encounter is a HEAD-ONLY input — the trunk never
    sees it, so the trunk representation stays identical to the policy/value net's and the
    head is forced to read combat strength out of the state embedding and cross it with the
    encounter here. out_dim = NUM_BUCKETS for the bucketed CE head, 1 for the scaled-float
    (ΔHP / maxHP) regression head."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.enc_embed = nn.Embedding(len(sts.MonsterEncounter), dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, out_dim))

    def forward(self, pooled: torch.Tensor, encounter: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([pooled, self.enc_embed(encounter)], dim=-1)).float()


def load_network_backward_compatible(net: NN, state_dict: dict) -> NN:
    """
    Load state dict into network with backward compatibility for embedding size changes.
    
    When enum-based embeddings grow (like FixedAction), old networks will have fewer
    embedding weights than the current network expects. This function handles this by:
    1. Taking the randomly initialized weights from the new network
    2. Slicing in the loaded weights for existing indices
    3. Keeping random initialization for new indices
    
    Args:
        net: Freshly initialized network with current architecture
        state_dict: State dict from saved model (potentially with smaller embeddings)
    
    Returns:
        The network with loaded weights, handling size mismatches gracefully
    """
    current_state = net.state_dict()
    
    # Create a copy of the current state dict to modify
    updated_state = current_state.copy()
    
    # Process each parameter in the loaded state dict
    for name, loaded_param in state_dict.items():
        if name in current_state:
            current_param = current_state[name]
            
            # Check if shapes match
            if loaded_param.shape == current_param.shape:
                # Shapes match - use loaded parameter directly
                updated_state[name] = loaded_param
            elif loaded_param.dim() > 0 and current_param.dim() > 0:
                # Shapes don't match - try to slice in the loaded weights
                # This handles cases where embeddings have grown
                
                if loaded_param.dim() == 2 and current_param.dim() == 2:
                    # 2D tensor (like embedding weights)
                    loaded_rows, loaded_cols = loaded_param.shape
                    current_rows, current_cols = current_param.shape
                    
                    if loaded_cols == current_cols and loaded_rows <= current_rows:
                        # Same number of features, but fewer embedding entries
                        # Slice the loaded weights into the current tensor
                        updated_param = current_param.clone()
                        updated_param[:loaded_rows, :] = loaded_param
                        updated_state[name] = updated_param
                        print(f"Resized {name}: {loaded_param.shape} -> {current_param.shape}")
                    elif loaded_rows == current_rows and loaded_cols <= current_cols:
                        # Input dimension has grown (e.g., fixed observation space expanded)
                        # Keep existing weights for old inputs, use random init for new inputs
                        updated_param = current_param.clone()
                        updated_param[:, :loaded_cols] = loaded_param
                        updated_state[name] = updated_param
                        print(f"Resized {name}: {loaded_param.shape} -> {current_param.shape}")
                    else:
                        # Can't handle this mismatch - error out
                        raise ValueError(f"Couldn't resize {name}: {loaded_param.shape} vs {current_param.shape}")
                        
                elif loaded_param.dim() == 1 and current_param.dim() == 1:
                    # 1D tensor (like bias)
                    loaded_size = loaded_param.shape[0]
                    current_size = current_param.shape[0]
                    
                    if loaded_size <= current_size:
                        # Slice the loaded bias into the current tensor
                        updated_param = current_param.clone()
                        updated_param[:loaded_size] = loaded_param
                        updated_state[name] = updated_param
                        print(f"Resized {name}: {loaded_param.shape} -> {current_param.shape}")
                    else:
                        raise ValueError(f"Couldn't resize {name}: {loaded_param.shape} vs {current_param.shape}")
                        
                else:
                    # Different dimensionality - error out
                    raise ValueError(f"Dimension mismatch for {name}: {loaded_param.shape} vs {current_param.shape}")
                    
            else:
                # Scalar or other edge case - use loaded if possible
                if loaded_param.shape == current_param.shape:
                    updated_state[name] = loaded_param
                else:
                    raise ValueError(f"Couldn't handle {name}: {loaded_param.shape} vs {current_param.shape}")
        else:
            # Parameter exists in loaded model but not current model - error out
            raise ValueError(f"Parameter {name} from loaded model not in current architecture")
    
    # Load the updated state dict
    net.load_state_dict(updated_state)
    return net


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


def map_dag_features(x):
    """Exact map-DAG features for one flattened observation: per-node (min_elites, max_elites,
    dist_rest) aggregates, reachability from the current frontier, and the destination room of
    each offered path option.

    Frontier = successors of the current node, or all y=0 roots at act start (mapY < 0).
    dist_rest counts steps from the node itself (0 if the node is a rest site) and is capped
    at MAP_AGG_CAP, which is also the [0,1] scale divisor used by the caller.
    Returns (minE, maxE, dR, reach, dest_rooms) as python lists.
    """
    xs = [int(v) for v in x['obs.map.xs']]
    ys = [int(v) for v in x['obs.map.ys']]
    rts = [int(v) for v in x['obs.map.roomTypes']]
    pxs = x['obs.map.pathXs']
    idx = {(xc, yc): j for j, (xc, yc) in enumerate(zip(xs, ys))}
    n = len(xs)
    succ = [[] for _ in range(n)]
    for j in range(n):
        for e in pxs[j]:
            e = int(e)
            k = idx.get((e, ys[j] + 1))
            if e >= 0 and k is not None:
                succ[j].append(k)
    ELITE, REST = int(sts.Room.ELITE), int(sts.Room.REST)
    minE, maxE, dR = [0] * n, [0] * n, [MAP_AGG_CAP] * n
    for j in sorted(range(n), key=lambda j: -ys[j]):
        e = 1 if rts[j] == ELITE else 0
        minE[j] = e + (min(minE[k] for k in succ[j]) if succ[j] else 0)
        maxE[j] = e + (max(maxE[k] for k in succ[j]) if succ[j] else 0)
        if rts[j] == REST:
            dR[j] = 0
        elif succ[j]:
            dR[j] = min(MAP_AGG_CAP, 1 + min(dR[k] for k in succ[j]))

    # Can each node forward-reach the burning elite (emerald key)? Reverse-y DP: a node reaches
    # it iff it IS it or any successor does. (-1,-1 burningElite => no burning elite this map.)
    bx, by = int(x.get('obs.map.burningEliteX', -1)), int(x.get('obs.map.burningEliteY', -1))
    bn = idx.get((bx, by)) if bx >= 0 else None
    reaches_burn = [False] * n
    if bn is not None:
        for j in sorted(range(n), key=lambda j: -ys[j]):
            reaches_burn[j] = (j == bn) or any(reaches_burn[k] for k in succ[j])

    cur = idx.get((int(x['obs.mapX']), int(x['obs.mapY'])))
    if cur is not None:
        frontier = list(succ[cur])
    else:  # act start: every y=0 node is enterable
        frontier = [j for j in range(n) if ys[j] == 0]
    # Offered path options can exceed the current node's edges (Winged Boots ignores them);
    # the true frontier is their union.
    ydest_f = int(x['obs.mapY']) + 1
    for px in x['paths_offered']:
        k = idx.get((int(px), ydest_f))
        if k is not None and k not in frontier:
            frontier.append(k)
    reach = [False] * n
    stack = list(frontier)
    while stack:
        j = stack.pop()
        if not reach[j]:
            reach[j] = True
            stack.extend(succ[j])

    ydest = int(x['obs.mapY']) + 1
    dest_rooms = []
    opt_cone = []   # per offered option: the destination node's (minE, maxE, dist_rest)
    opt_burn = []   # per offered option: does its forward cone reach the burning elite
    for px in x['paths_offered']:
        k = idx.get((int(px), ydest))
        dest_rooms.append(rts[k] if k is not None else 0)
        if k is not None:
            opt_cone.append((minE[k], maxE[k], dR[k]))
            opt_burn.append(reaches_burn[k])
        else:
            opt_cone.append((0, 0, MAP_AGG_CAP))
            opt_burn.append(False)
    return minE, maxE, dR, reach, dest_rooms, opt_cone, opt_burn


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
        'fixed_obs': {
            'fixed_observation': torch.zeros((len(batch), len(_FIXED_OBS_SCALAR_MAXES)), dtype=torch.int32),
            'boss': torch.zeros((len(batch),), dtype=torch.int32),
            'ascension': torch.zeros((len(batch),), dtype=torch.int32),
            'key_ruby': torch.zeros((len(batch),), dtype=torch.int32),
            'key_emerald': torch.zeros((len(batch),), dtype=torch.int32),
            'key_sapphire': torch.zeros((len(batch),), dtype=torch.int32),
            'screen_state': torch.zeros((len(batch),), dtype=torch.int32)
        },
        'map_nodes': {
            'value': {
                'room': torch.full((len(batch), MAX_MAP_NODES), 0, dtype=torch.int32),
                'is_current': torch.full((len(batch), MAX_MAP_NODES), 0, dtype=torch.int32),
                'pos': torch.full((len(batch), MAX_MAP_NODES, 2), 0, dtype=torch.int32),  # [x, y]
                'path_xs': torch.full((len(batch), MAX_MAP_NODES, 3), -1, dtype=torch.int32),   # left, straight, right
                'rel': torch.full((len(batch), MAX_MAP_NODES, 2), 0, dtype=torch.int32),  # (dx, dy) from current
                'reachable': torch.full((len(batch), MAX_MAP_NODES), 0, dtype=torch.int32),
                'agg': torch.zeros((len(batch), MAX_MAP_NODES, 3), dtype=torch.float32),  # scaled DAG aggregates
                'burning': torch.full((len(batch), MAX_MAP_NODES), 0, dtype=torch.int32),
            },
            'mask': torch.ones((len(batch), MAX_MAP_NODES), dtype=torch.bool)
        },
    }
    
    # Create batched tensors for choices using fixed maximum sizes
    batch_choices = {
        'cards': {
            'value': torch.full((len(batch), MAX_DECK_SIZE, 3), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), MAX_DECK_SIZE), dtype=torch.bool)
        },
        'relics': {
            'value': torch.full((len(batch), 3), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), 3), dtype=torch.bool)
        },
        'potions': {
            'value': torch.full((len(batch), sts.MAX_POTION_CAPACITY), 0, dtype=torch.int32),
            'mask': torch.ones((len(batch), sts.MAX_POTION_CAPACITY), dtype=torch.bool)
        },
        'fixed': {
            'value': {
                'action': torch.full((len(batch), MAX_FIXED_ACTIONS), FixedAction.INVALID.value, dtype=torch.int32),
                'gold': torch.full((len(batch), MAX_FIXED_ACTIONS, 1), 0, dtype=torch.int32),
                'card': torch.full((len(batch), MAX_FIXED_ACTIONS), sts.CardId.INVALID.value, dtype=torch.int32),
                'relic': torch.full((len(batch), MAX_FIXED_ACTIONS), sts.RelicId.INVALID.value, dtype=torch.int32),
                'info': torch.full((len(batch), MAX_FIXED_ACTIONS), EventFixedInfo.NONE.value, dtype=torch.int32),
            },
            'mask': torch.ones((len(batch), MAX_FIXED_ACTIONS), dtype=torch.bool)
        },
        'paths': {
            'value': {
                'x': torch.full((len(batch), MAX_PATH_CHOICES, 1), -1, dtype=torch.int32),
                'room': torch.full((len(batch), MAX_PATH_CHOICES), 0, dtype=torch.int32),
                'cone': torch.zeros((len(batch), MAX_PATH_CHOICES, 3), dtype=torch.float32),
                'reaches_burn': torch.zeros((len(batch), MAX_PATH_CHOICES), dtype=torch.int32),
            },
            'mask': torch.ones((len(batch), MAX_PATH_CHOICES), dtype=torch.bool)
        },
    }
    # Per-option destination room labels for the auxiliary grounding loss (-100 = no option).
    aux_dest_room = torch.full((len(batch), MAX_PATH_CHOICES), -100, dtype=torch.long)
    
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
        
        # Set fixed observation components (boss/ascension/key flags split out as
        # categoricals). Records from before an input existed default to 0 (6-entry
        # records predate ascension; 7-entry records predate the act-4 key flags).
        fo = list(x['obs.fixed_observation'])
        keys = [fo.pop(j) for j in reversed(_KEY_OBS_IDXS)][::-1] if len(fo) > _KEY_OBS_IDXS[-1] else [0, 0, 0]
        ascension = fo.pop(_ASC_OBS_IDX) if len(fo) > _ASC_OBS_IDX else 0
        boss = fo.pop(_BOSS_OBS_IDX)
        batch_obs['fixed_obs']['fixed_observation'][i] = torch.tensor(fo, dtype=torch.int32)
        batch_obs['fixed_obs']['boss'][i] = int(boss)
        batch_obs['fixed_obs']['ascension'][i] = int(ascension)
        batch_obs['fixed_obs']['key_ruby'][i] = int(keys[0])
        batch_obs['fixed_obs']['key_emerald'][i] = int(keys[1])
        batch_obs['fixed_obs']['key_sapphire'][i] = int(keys[2])
        batch_obs['fixed_obs']['screen_state'][i] = x['screen_state']
        
        # Set map observation components
        map_xs = x['obs.map.xs']
        map_ys = x['obs.map.ys']
        map_room_types = x['obs.map.roomTypes']
        map_path_xs = x['obs.map.pathXs']
        map_x_pos = x['obs.mapX']
        map_y_pos = x['obs.mapY']
        nodes_len = len(map_xs)
        assert nodes_len <= MAX_MAP_NODES, f"Map nodes count {nodes_len} exceeds maximum {MAX_MAP_NODES}"
        
        # Create node data: raw structure + ego-relative coords, reachability, DAG aggregates
        minE, maxE, dR, reach, dest_rooms, opt_cone, opt_burn = map_dag_features(x)
        # Burning-elite flag (emerald key). Records from before the field default to no flag.
        burn_x = int(x.get('obs.map.burningEliteX', -1))
        burn_y = int(x.get('obs.map.burningEliteY', -1))
        for j in range(nodes_len):
            is_current = 1 if (map_xs[j] == map_x_pos and map_ys[j] == map_y_pos) else 0
            batch_obs['map_nodes']['value']['room'][i, j] = torch.tensor(int(map_room_types[j]), dtype=torch.int32)
            batch_obs['map_nodes']['value']['is_current'][i, j] = torch.tensor(is_current, dtype=torch.int32)
            batch_obs['map_nodes']['value']['pos'][i, j] = torch.tensor([map_xs[j], map_ys[j]], dtype=torch.int32)
            batch_obs['map_nodes']['value']['path_xs'][i, j] = torch.tensor(map_path_xs[j], dtype=torch.int32)
            batch_obs['map_nodes']['value']['rel'][i, j] = torch.tensor(
                [int(map_xs[j]) - int(map_x_pos), int(map_ys[j]) - int(map_y_pos)], dtype=torch.int32)
            batch_obs['map_nodes']['value']['reachable'][i, j] = 1 if reach[j] else 0
            batch_obs['map_nodes']['value']['agg'][i, j] = torch.tensor(
                [minE[j] / MAP_AGG_CAP, maxE[j] / MAP_AGG_CAP, dR[j] / MAP_AGG_CAP], dtype=torch.float32)
            if burn_x >= 0 and map_xs[j] == burn_x and map_ys[j] == burn_y:
                batch_obs['map_nodes']['value']['burning'][i, j] = 1
        batch_obs['map_nodes']['mask'][i, :nodes_len] = torch.zeros(nodes_len, dtype=torch.bool)
        
        # Build choices data inline
        cards_offered = x['cards_offered.cards']
        cards_upgrades = x['cards_offered.upgrades']
        select_screen_type = x['select_screen_type']
        choice_cards_len = len(cards_offered)
        assert choice_cards_len <= MAX_DECK_SIZE, f"Choice cards count {choice_cards_len} exceeds maximum {MAX_DECK_SIZE}; {cards_offered}; {relics}"
        if choice_cards_len > 0:
            # Create 3-tuple: (card_id, upgrade_count, select_screen_type)
            card_tuples = [(card_id, upgrade, select_screen_type) for card_id, upgrade in zip(cards_offered, cards_upgrades)]
            batch_choices['cards']['value'][i, :choice_cards_len] = torch.tensor(card_tuples, dtype=torch.int32)
        batch_choices['cards']['mask'][i, :choice_cards_len] = torch.zeros(choice_cards_len, dtype=torch.bool)
        
        relics_offered = x['relics_offered']
        choice_relics_len = len(relics_offered)
        assert choice_relics_len <= 3, f"Choice relics count {choice_relics_len} exceeds maximum {3}"
        batch_choices['relics']['value'][i, :choice_relics_len] = torch.tensor(relics_offered, dtype=torch.int32)
        batch_choices['relics']['mask'][i, :choice_relics_len] = torch.zeros(choice_relics_len, dtype=torch.bool)
        
        potions_offered = x['potions_offered']
        choice_potions_len = len(potions_offered)
        assert choice_potions_len <= sts.MAX_POTION_CAPACITY, f"Choice potions count {choice_potions_len} exceeds maximum {sts.MAX_POTION_CAPACITY}"
        batch_choices['potions']['value'][i, :choice_potions_len] = torch.tensor(potions_offered, dtype=torch.int32)
        batch_choices['potions']['mask'][i, :choice_potions_len] = torch.zeros(choice_potions_len, dtype=torch.bool)
        
        paths_offered = x['paths_offered']
        choice_paths_len = len(paths_offered)
        assert choice_paths_len <= MAX_PATH_CHOICES, f"Choice paths count {choice_paths_len} exceeds maximum {MAX_PATH_CHOICES}"  # max 3 paths from any node
        if choice_paths_len > 0:
            batch_choices['paths']['value']['x'][i, :choice_paths_len, 0] = torch.tensor(paths_offered, dtype=torch.int32)
            batch_choices['paths']['value']['room'][i, :choice_paths_len] = torch.tensor(dest_rooms, dtype=torch.int32)
            aux_dest_room[i, :choice_paths_len] = torch.tensor(dest_rooms, dtype=torch.long)
            batch_choices['paths']['value']['cone'][i, :choice_paths_len] = torch.tensor(
                opt_cone, dtype=torch.float32) / MAP_AGG_CAP
            batch_choices['paths']['value']['reaches_burn'][i, :choice_paths_len] = torch.tensor(
                opt_burn, dtype=torch.int32)
        batch_choices['paths']['mask'][i, :choice_paths_len] = torch.zeros(choice_paths_len, dtype=torch.bool)
        
        fixed_actions = x['fixed_actions']
        choice_fixed_len = len(fixed_actions)
        assert choice_fixed_len <= MAX_FIXED_ACTIONS, f"Choice fixed actions count {choice_fixed_len} exceeds maximum {MAX_FIXED_ACTIONS}"
        
        # Manually collate the fixed actions dictionary structure
        for j, action_dict in enumerate(fixed_actions):
            # Handle the 'action' field (required)
            batch_choices['fixed']['value']['action'][i, j] = int(action_dict['action'])
            
            # Handle optional fields with defaults
            batch_choices['fixed']['value']['gold'][i, j, 0] = action_dict.get('gold', 0)
            batch_choices['fixed']['value']['card'][i, j] = int(action_dict.get('card', sts.CardId.INVALID))
            batch_choices['fixed']['value']['relic'][i, j] = int(action_dict.get('relic', sts.RelicId.INVALID))
            batch_choices['fixed']['value']['info'][i, j] = int(action_dict.get('info', EventFixedInfo.NONE))
        batch_choices['fixed']['mask'][i, :choice_fixed_len] = torch.zeros(choice_fixed_len, dtype=torch.bool)
        
        chosen_idx_list.append(x['chosen_idx'])
        outcome_list.append(x['outcome'])
    
    return {
        **batch_obs,
        'choices': batch_choices,
        'chosen_idx': torch.tensor(chosen_idx_list, dtype=torch.int64),
        'outcome': torch.tensor(outcome_list, dtype=torch.float32),
        'aux_dest_room': aux_dest_room,
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


class SeparateValuePolicy(nn.Module):
    """Wrapper that combines separate policy and value networks to look like a single network."""
    
    def __init__(self, policy_net: NN, value_net: NN):
        super().__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        
        # Ensure networks have correct configurations
        if policy_net.H.use_value_head:
            raise ValueError("Policy network should not have value head when using separate networks")
        if not value_net.H.use_value_head:
            raise ValueError("Value network should have value head when using separate networks")
    
    def forward(self, batch: dict):
        """Forward pass that combines policy and value outputs."""
        # Get policy logits
        policy_logits = self.policy_net(batch)
        
        # Get value prediction - value network returns (logits, values) tuple
        value_output = self.value_net(batch)
        _, values = value_output  # Extract values from tuple
        
        return policy_logits.clone(), values.clone()
    
    @property
    def device(self):
        return self.policy_net.device


# %%
