from collections import abc
from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import slaythespire as sts
from inputs import SinusoidalEmbedding, FixedVecSpace, SequenceSpace, EnumSpace, DictSpace, TupleAddSpace, IntSpace, DictAddSpace, ScalarToSequenceSpace

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
MAX_CHOICES = 64    # Maximum deck size for card selection screens like smithing
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
    REST = auto()
    SMITH = auto()
    RECALL = auto()
    LIFT = auto()
    TOKE = auto()
    DIG = auto()
    
    # Event Actions - organized by event type
    # NEOW (4 choices)
    NEOW_OPTION_0 = auto()
    NEOW_OPTION_1 = auto()
    NEOW_OPTION_2 = auto()
    NEOW_OPTION_3 = auto()
    
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
    NLOTH_AGREE = auto()
    NLOTH_DISAGREE = auto()
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

obs_space = DictSpace({
    'deck': SequenceSpace(TupleAddSpace(EnumSpace(sts.CardId), IntSpace(MAX_UPGRADE))),
    'relics': SequenceSpace(EnumSpace(sts.RelicId)),
    'potions': SequenceSpace(EnumSpace(sts.Potion)),
    'fixed_obs': ScalarToSequenceSpace(DictAddSpace({
        'fixed_observation': FixedVecSpace(sts.getFixedObservationMaximums()),
        'screen_state': EnumSpace(sts.ScreenState),
    })),
})

choice_space = DictSpace({
    'cards': SequenceSpace(TupleAddSpace(EnumSpace(sts.CardId), IntSpace(MAX_UPGRADE), EnumSpace(sts.CardSelectScreenType))),
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
            'fixed_observation': torch.zeros((len(batch), len(sts.getFixedObservationMaximums())), dtype=torch.int32),
            'screen_state': torch.zeros((len(batch),), dtype=torch.int32)
        }
    }
    
    # Create batched tensors for choices using fixed maximum sizes
    batch_choices = {
        'cards': {
            'value': torch.full((len(batch), MAX_CHOICES, 3), 0, dtype=torch.int32),
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
        
        # Set fixed observation components
        batch_obs['fixed_obs']['fixed_observation'][i] = torch.tensor(x['obs.fixed_observation'], dtype=torch.int32)
        batch_obs['fixed_obs']['screen_state'][i] = x['screen_state']
        
        # Build choices data inline
        cards_offered = x['cards_offered.cards']
        cards_upgrades = x['cards_offered.upgrades']
        select_screen_type = x['select_screen_type']
        choice_cards_len = len(cards_offered)
        assert choice_cards_len <= MAX_CHOICES, f"Choice cards count {choice_cards_len} exceeds maximum {MAX_CHOICES}; {cards_offered}; {relics}"
        if choice_cards_len > 0:
            # Create 3-tuple: (card_id, upgrade_count, select_screen_type)
            card_tuples = [(card_id, upgrade, select_screen_type) for card_id, upgrade in zip(cards_offered, cards_upgrades)]
            batch_choices['cards']['value'][i, :choice_cards_len] = torch.tensor(card_tuples, dtype=torch.int32)
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
