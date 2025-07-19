# %%
from __future__ import annotations

import random
import copy
from enum import IntEnum, auto
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from queue import Empty
from threading import Timer
from typing import NamedTuple, Optional, List
import time
import threading
import argparse

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch import nn
import torch.nn.functional as F

from network import NN, ActionType, FixedAction, EventFixedInfo, ModelHP, collate_fn, process_batch, output_to_cpu, choice_space, move_to_device, load_network_backward_compatible
from inputs import Path
import slaythespire as sts

# %%
def extract_event_info(gc: sts.GameContext, action: sts.GameAction, fixed_action: FixedAction) -> tuple[int, int, int, EventFixedInfo]:
    """Extract event-specific information for the neural network."""
    event = gc.cur_event
    info = gc.screen_state_info
    
    gold = 0
    card = sts.CardId.INVALID
    relic = sts.RelicId.INVALID
    event_info = EventFixedInfo.NONE
    
    # Add event-specific information based on the event type
    if event == sts.Event.NEOW:
        # Extract drawback information for Neow
        neow_options = info.neowRewards
        if action.idx1 < len(neow_options):
            drawback = neow_options[action.idx1].d
            if drawback == sts.NeowDrawback.TEN_PERCENT_HP_LOSS:
                event_info = EventFixedInfo.NEOW_TEN_PERCENT_HP_LOSS
            elif drawback == sts.NeowDrawback.NO_GOLD:
                event_info = EventFixedInfo.NEOW_NO_GOLD
            elif drawback == sts.NeowDrawback.CURSE:
                event_info = EventFixedInfo.NEOW_CURSE
            elif drawback == sts.NeowDrawback.PERCENT_DAMAGE:
                event_info = EventFixedInfo.NEOW_PERCENT_DAMAGE
            elif drawback == sts.NeowDrawback.LOSE_STARTER_RELIC:
                event_info = EventFixedInfo.NEOW_LOSE_STARTER_RELIC
    
    elif event == sts.Event.FALLING:
        # Show which card would be lost for each choice
        if fixed_action == FixedAction.FALLING_SKILL and info.skillCardDeckIdx != -1:
            card = gc.deck[info.skillCardDeckIdx].id
        elif fixed_action == FixedAction.FALLING_POWER and info.powerCardDeckIdx != -1:
            card = gc.deck[info.powerCardDeckIdx].id
        elif fixed_action == FixedAction.FALLING_ATTACK and info.attackCardDeckIdx != -1:
            card = gc.deck[info.attackCardDeckIdx].id
    
    elif event == sts.Event.WE_MEET_AGAIN:
        # Show card and potion that would be lost, and gold cost
        if info.cardIdx != -1:
            card = gc.deck[info.cardIdx].id
        gold = info.gold if info.gold != -1 else 0
    
    elif event == sts.Event.NLOTH:
        # Show which relics are offered
        if fixed_action == FixedAction.NLOTH_OFFER_0:
            if info.relicIdx0 != -1:
                relic = gc.relics[info.relicIdx0].id
        elif fixed_action == FixedAction.NLOTH_OFFER_1:
            if info.relicIdx1 != -1:
                relic = gc.relics[info.relicIdx1].id
    
    elif event == sts.Event.WORLD_OF_GOOP:
        # Show gold loss amount
        if fixed_action == FixedAction.WORLD_OF_GOOP_ENTER:
            gold = info.goldLoss
    
    elif event == sts.Event.FACE_TRADER:
        # Show HP loss amount
        if fixed_action == FixedAction.FACE_TRADER_LOSE_HP:
            gold = info.hpAmount0  # Store HP amount in gold field for display
    
    elif event == sts.Event.GOLDEN_IDOL:
        # Show HP loss amounts for different phases
        if fixed_action == FixedAction.GOLDEN_IDOL_TAKE:
            gold = info.hpAmount0  # Initial HP loss
        elif fixed_action == FixedAction.GOLDEN_IDOL_PHASE2_OPTION_0:
            gold = info.hpAmount1  # Escape HP loss
    
    elif event == sts.Event.WINDING_HALLS:
        # Show HP amounts for different choices
        if fixed_action == FixedAction.WINDING_HALLS_MADNESS:
            gold = info.hpAmount0  # Current HP loss
        elif fixed_action == FixedAction.WINDING_HALLS_WRITHE:
            gold = info.hpAmount1  # Heal amount
    
    return gold, card, relic, event_info

def map_event_action_to_fixed_action(gc: sts.GameContext, action: sts.GameAction) -> Optional[sts.FixedAction]:
    """
    Map a GameAction for an event screen to the appropriate FixedAction enum value.
    This handles the complex conditional logic for each event type.
    """
    event = gc.cur_event
    idx1 = action.idx1
    idx2 = action.idx2
    event_data = gc.screen_state_info.event_data
    
    # Single choice events
    if event == sts.Event.LAB:
        return FixedAction.LAB_OPTION
    elif event == sts.Event.WHEEL_OF_CHANGE:
        return FixedAction.WHEEL_OF_CHANGE_OPTION
    
    # Two choice events (0x3 pattern)
    elif event == sts.Event.ANCIENT_WRITING:
        return FixedAction.ANCIENT_WRITING_ELEGANCE if idx1 == 0 else FixedAction.ANCIENT_WRITING_SIMPLICITY
    elif event == sts.Event.DEAD_ADVENTURER:
        return FixedAction.DEAD_ADVENTURER_SEARCH if idx1 == 0 else FixedAction.DEAD_ADVENTURER_LEAVE
    elif event == sts.Event.DUPLICATOR:
        return FixedAction.DUPLICATOR_DUPLICATE if idx1 == 0 else FixedAction.DUPLICATOR_LEAVE
    elif event == sts.Event.OLD_BEGGAR:
        return FixedAction.OLD_BEGGAR_GIVE_GOLD if idx1 == 0 else FixedAction.OLD_BEGGAR_REFUSE
    elif event == sts.Event.THE_DIVINE_FOUNTAIN:
        return FixedAction.DIVINE_FOUNTAIN_HEAL if idx1 == 0 else FixedAction.DIVINE_FOUNTAIN_LEAVE
    elif event == sts.Event.GHOSTS:
        return FixedAction.GHOSTS_AGREE if idx1 == 0 else FixedAction.GHOSTS_REFUSE
    elif event == sts.Event.THE_SSSSSERPENT:
        return FixedAction.SSSSSERPENT_AGREE if idx1 == 0 else FixedAction.SSSSSERPENT_DISAGREE
    elif event == sts.Event.MASKED_BANDITS:
        return FixedAction.MASKED_BANDITS_PAY if idx1 == 0 else FixedAction.MASKED_BANDITS_FIGHT
    elif event == sts.Event.HYPNOTIZING_COLORED_MUSHROOMS:
        return FixedAction.MUSHROOMS_HEAL if idx1 == 0 else FixedAction.MUSHROOMS_LEAVE
    elif event == sts.Event.MYSTERIOUS_SPHERE:
        return FixedAction.MYSTERIOUS_SPHERE_OPEN if idx1 == 0 else FixedAction.MYSTERIOUS_SPHERE_LEAVE
    elif event == sts.Event.THE_NEST:
        return FixedAction.NEST_AGREE if idx1 == 0 else FixedAction.NEST_DISAGREE
    elif event == sts.Event.NOTE_FOR_YOURSELF:
        return FixedAction.NOTE_FOR_YOURSELF_IGNORE if idx1 == 0 else FixedAction.NOTE_FOR_YOURSELF_WRITE
    elif event == sts.Event.SCRAP_OOZE:
        return FixedAction.SCRAP_OOZE_ATTACK if idx1 == 0 else FixedAction.SCRAP_OOZE_LEAVE
    elif event == sts.Event.SECRET_PORTAL:
        return FixedAction.SECRET_PORTAL_ENTER if idx1 == 0 else FixedAction.SECRET_PORTAL_LEAVE
    elif event == sts.Event.SHINING_LIGHT:
        return FixedAction.SHINING_LIGHT_ENTER if idx1 == 0 else FixedAction.SHINING_LIGHT_LEAVE
    elif event == sts.Event.THE_JOUST:
        return FixedAction.JOUST_GIVE_GOLD if idx1 == 0 else FixedAction.JOUST_REFUSE
    elif event == sts.Event.THE_LIBRARY:
        return FixedAction.LIBRARY_READ if idx1 == 0 else FixedAction.LIBRARY_LEAVE
    elif event == sts.Event.THE_MAUSOLEUM:
        return FixedAction.MAUSOLEUM_OPEN if idx1 == 0 else FixedAction.MAUSOLEUM_LEAVE
    elif event == sts.Event.WORLD_OF_GOOP:
        return FixedAction.WORLD_OF_GOOP_ENTER if idx1 == 0 else FixedAction.WORLD_OF_GOOP_LEAVE
    
    # Three choice events (0x7 pattern)
    elif event == sts.Event.BIG_FISH:
        if idx1 == 0: return FixedAction.BIG_FISH_BANANA
        elif idx1 == 1: return FixedAction.BIG_FISH_DONUT
        else: return FixedAction.BIG_FISH_BOX
    elif event == sts.Event.FACE_TRADER:
        if idx1 == 0: return FixedAction.FACE_TRADER_LOSE_GOLD
        elif idx1 == 1: return FixedAction.FACE_TRADER_LOSE_HP
        else: return FixedAction.FACE_TRADER_LEAVE
    elif event == sts.Event.GOLDEN_SHRINE:
        if idx1 == 0: return FixedAction.GOLDEN_SHRINE_PRAY
        elif idx1 == 1: return FixedAction.GOLDEN_SHRINE_DESECRATE
        else: return FixedAction.GOLDEN_SHRINE_LEAVE
    elif event == sts.Event.NLOTH:
        if idx1 == 0: return FixedAction.NLOTH_OFFER_0
        elif idx1 == 1: return FixedAction.NLOTH_OFFER_1
        else: return FixedAction.NLOTH_LEAVE
    elif event == sts.Event.SENSORY_STONE:
        if idx1 == 0: return FixedAction.SENSORY_STONE_MEMORIES
        elif idx1 == 1: return FixedAction.SENSORY_STONE_COLORLESS
        else: return FixedAction.SENSORY_STONE_LEAVE
    elif event == sts.Event.WINDING_HALLS:
        if idx1 == 0: return FixedAction.WINDING_HALLS_MADNESS
        elif idx1 == 1: return FixedAction.WINDING_HALLS_WRITHE
        else: return FixedAction.WINDING_HALLS_LEAVE
    
    # Four choice events (0xF pattern)
    elif event == sts.Event.NEOW:
        # Map Neow actions to specific bonus types
        neow_options = gc.screen_state_info.neowRewards
        if idx1 >= len(neow_options):
            raise ValueError(f"Invalid Neow action idx1: {idx1}")
        
        bonus = neow_options[idx1].r
        if bonus == sts.NeowBonus.THREE_CARDS:
            return FixedAction.NEOW_THREE_CARDS
        elif bonus == sts.NeowBonus.ONE_RANDOM_RARE_CARD:
            return FixedAction.NEOW_ONE_RANDOM_RARE_CARD
        elif bonus == sts.NeowBonus.REMOVE_CARD:
            return FixedAction.NEOW_REMOVE_CARD
        elif bonus == sts.NeowBonus.UPGRADE_CARD:
            return FixedAction.NEOW_UPGRADE_CARD
        elif bonus == sts.NeowBonus.TRANSFORM_CARD:
            return FixedAction.NEOW_TRANSFORM_CARD
        elif bonus == sts.NeowBonus.RANDOM_COLORLESS:
            return FixedAction.NEOW_RANDOM_COLORLESS
        elif bonus == sts.NeowBonus.THREE_SMALL_POTIONS:
            return FixedAction.NEOW_THREE_SMALL_POTIONS
        elif bonus == sts.NeowBonus.RANDOM_COMMON_RELIC:
            return FixedAction.NEOW_RANDOM_COMMON_RELIC
        elif bonus == sts.NeowBonus.TEN_PERCENT_HP_BONUS:
            return FixedAction.NEOW_TEN_PERCENT_HP_BONUS
        elif bonus == sts.NeowBonus.THREE_ENEMY_KILL:
            return FixedAction.NEOW_THREE_ENEMY_KILL
        elif bonus == sts.NeowBonus.HUNDRED_GOLD:
            return FixedAction.NEOW_HUNDRED_GOLD
        elif bonus == sts.NeowBonus.RANDOM_COLORLESS_2:
            return FixedAction.NEOW_RANDOM_COLORLESS_2
        elif bonus == sts.NeowBonus.REMOVE_TWO:
            return FixedAction.NEOW_REMOVE_TWO
        elif bonus == sts.NeowBonus.ONE_RARE_RELIC:
            return FixedAction.NEOW_ONE_RARE_RELIC
        elif bonus == sts.NeowBonus.THREE_RARE_CARDS:
            return FixedAction.NEOW_THREE_RARE_CARDS
        elif bonus == sts.NeowBonus.TWO_FIFTY_GOLD:
            return FixedAction.NEOW_TWO_FIFTY_GOLD
        elif bonus == sts.NeowBonus.TRANSFORM_TWO_CARDS:
            return FixedAction.NEOW_TRANSFORM_TWO_CARDS
        elif bonus == sts.NeowBonus.TWENTY_PERCENT_HP_BONUS:
            return FixedAction.NEOW_TWENTY_PERCENT_HP_BONUS
        elif bonus == sts.NeowBonus.BOSS_RELIC:
            return FixedAction.NEOW_BOSS_RELIC
        else:
            raise ValueError(f"Unknown Neow bonus: {bonus}")
    elif event == sts.Event.KNOWING_SKULL:
        if idx1 == 0: return FixedAction.KNOWING_SKULL_OPTION_0
        elif idx1 == 1: return FixedAction.KNOWING_SKULL_OPTION_1
        elif idx1 == 2: return FixedAction.KNOWING_SKULL_OPTION_2
        else: return FixedAction.KNOWING_SKULL_OPTION_3
    elif event == sts.Event.THE_WOMAN_IN_BLUE:
        if idx1 == 0: return FixedAction.WOMAN_IN_BLUE_OPTION_0
        elif idx1 == 1: return FixedAction.WOMAN_IN_BLUE_OPTION_1
        elif idx1 == 2: return FixedAction.WOMAN_IN_BLUE_OPTION_2
        else: return FixedAction.WOMAN_IN_BLUE_OPTION_3
    
    # Conditional events requiring special handling
    elif event == sts.Event.PLEADING_VAGRANT:
        if idx1 == 0: return FixedAction.PLEADING_VAGRANT_GIVE_GOLD
        elif idx1 == 1: return FixedAction.PLEADING_VAGRANT_REFUSE
        else: return FixedAction.PLEADING_VAGRANT_LEAVE
    
    elif event == sts.Event.COLOSSEUM:
        if event_data == 0:
            return FixedAction.COLOSSEUM_PHASE1_PROCEED
        else:
            return FixedAction.COLOSSEUM_PHASE2_OPTION_0 if idx1 == 0 else FixedAction.COLOSSEUM_PHASE2_OPTION_1
    
    elif event == sts.Event.CURSED_TOME:
        if event_data == 0:
            return FixedAction.CURSED_TOME_READ if idx1 == 0 else FixedAction.CURSED_TOME_LEAVE
        elif event_data == 1:
            return FixedAction.CURSED_TOME_PHASE1_OPTION
        elif event_data == 2:
            return FixedAction.CURSED_TOME_PHASE2_OPTION
        elif event_data == 3:
            return FixedAction.CURSED_TOME_PHASE3_OPTION
        elif event_data == 4:
            return FixedAction.CURSED_TOME_PHASE4_OPTION_0 if idx1 == 0 else FixedAction.CURSED_TOME_PHASE4_OPTION_1
        else:
            raise ValueError(f"Unknown Cursed Tome event_data: {event_data}")
    
    elif event == sts.Event.DESIGNER_IN_SPIRE:
        if idx1 == 0: return FixedAction.DESIGNER_UPGRADE_ONE
        elif idx1 == 1: return FixedAction.DESIGNER_UPGRADE_ALL
        elif idx1 == 2: return FixedAction.DESIGNER_REMOVE_CARD
        elif idx1 == 3: return FixedAction.DESIGNER_TRANSFORM_TWO
        elif idx1 == 4: return FixedAction.DESIGNER_TRANSFORM_ONE
        elif idx1 == 5: return FixedAction.DESIGNER_LEAVE
        else:
            raise ValueError(f"Unknown Designer In-Spire action idx1: {idx1}")
    
    elif event == sts.Event.AUGMENTER:
        if idx1 == 0: return FixedAction.AUGMENTER_AGREE
        elif idx1 == 1: return FixedAction.AUGMENTER_REFUSE
        elif idx1 == 2: return FixedAction.AUGMENTER_LEAVE
        else:
            raise ValueError(f"Unknown Augmenter action idx1: {idx1}")
    
    elif event == sts.Event.FALLING:
        if idx1 == 0: return FixedAction.FALLING_SKILL
        elif idx1 == 1: return FixedAction.FALLING_POWER
        elif idx1 == 2: return FixedAction.FALLING_ATTACK
        elif idx1 == 3: return FixedAction.FALLING_LEAVE
        else:
            raise ValueError(f"Unknown Falling action idx1: {idx1}")
    
    elif event == sts.Event.FORGOTTEN_ALTAR:
        if idx1 == 0: return FixedAction.FORGOTTEN_ALTAR_PRAY
        elif idx1 == 1: return FixedAction.FORGOTTEN_ALTAR_DESECRATE
        elif idx1 == 2: return FixedAction.FORGOTTEN_ALTAR_LEAVE
        else:
            raise ValueError(f"Unknown Forgotten Altar action idx1: {idx1}")
    
    elif event == sts.Event.GOLDEN_IDOL:
        if idx1 == 0: return FixedAction.GOLDEN_IDOL_TAKE
        elif idx1 == 1: return FixedAction.GOLDEN_IDOL_LEAVE
        elif idx1 == 2: return FixedAction.GOLDEN_IDOL_PHASE2_OPTION_0
        elif idx1 == 3: return FixedAction.GOLDEN_IDOL_PHASE2_OPTION_1
        elif idx1 == 4: return FixedAction.GOLDEN_IDOL_PHASE2_OPTION_2
        else:
            raise ValueError(f"Unknown Golden Idol action idx1: {idx1}")
    
    elif event == sts.Event.WING_STATUE:
        if idx1 == 0: return FixedAction.WING_STATUE_REMOVE_CARD
        elif idx1 == 1: return FixedAction.WING_STATUE_LOSE_GOLD
        elif idx1 == 2: return FixedAction.WING_STATUE_LEAVE
        else:
            raise ValueError(f"Unknown Wing Statue action idx1: {idx1}")
    
    elif event == sts.Event.LIVING_WALL:
        if idx1 == 0: return FixedAction.LIVING_WALL_CHANGE
        elif idx1 == 1: return FixedAction.LIVING_WALL_GROW
        elif idx1 == 2: return FixedAction.LIVING_WALL_LEAVE
        else:
            raise ValueError(f"Unknown Living Wall action idx1: {idx1}")
    
    elif event == sts.Event.MINDBLOOM:
        if idx1 == 0: return FixedAction.MINDBLOOM_ACT1_BOSS
        elif idx1 == 1: return FixedAction.MINDBLOOM_UPGRADE_CARDS
        elif idx1 == 2: return FixedAction.MINDBLOOM_TRANSFORM
        elif idx1 == 3: return FixedAction.MINDBLOOM_HEAL
        else:
            raise ValueError(f"Unknown Mindbloom action idx1: {idx1}")
    
    elif event == sts.Event.PURIFIER:
        if idx1 == 0: return FixedAction.PURIFIER_PURIFY
        elif idx1 == 1: return FixedAction.PURIFIER_LEAVE
        else:
            raise ValueError(f"Unknown Purifier action idx1: {idx1}")
    
    elif event == sts.Event.TRANSMORGRIFIER:
        if idx1 == 0: return FixedAction.TRANSMORGRIFIER_TRANSFORM
        elif idx1 == 1: return FixedAction.TRANSMORGRIFIER_LEAVE
        else:
            raise ValueError(f"Unknown Transmorgrifier action idx1: {idx1}")
    
    elif event == sts.Event.THE_CLERIC:
        if idx1 == 0: return FixedAction.CLERIC_HEAL
        elif idx1 == 1: return FixedAction.CLERIC_PURIFY
        elif idx1 == 2: return FixedAction.CLERIC_LEAVE
        else:
            raise ValueError(f"Unknown Cleric action idx1: {idx1}")
    
    elif event == sts.Event.THE_MOAI_HEAD:
        if idx1 == 0: return FixedAction.MOAI_HEAD_GOLDEN_IDOL
        elif idx1 == 1: return FixedAction.MOAI_HEAD_LOSE_GOLD
        elif idx1 == 2: return FixedAction.MOAI_HEAD_LEAVE
        else:
            raise ValueError(f"Unknown Moai Head action idx1: {idx1}")
    
    elif event == sts.Event.TOMB_OF_LORD_RED_MASK:
        if idx1 == 0: return FixedAction.TOMB_RED_MASK_DON_MASK
        elif idx1 == 1: return FixedAction.TOMB_RED_MASK_OFFER_GOLD
        elif idx1 == 2: return FixedAction.TOMB_RED_MASK_LEAVE
        else:
            raise ValueError(f"Unknown Tomb of Lord Red Mask action idx1: {idx1}")
    
    elif event == sts.Event.UPGRADE_SHRINE:
        if idx1 == 0: return FixedAction.UPGRADE_SHRINE_UPGRADE
        elif idx1 == 1: return FixedAction.UPGRADE_SHRINE_LEAVE
        else:
            raise ValueError(f"Unknown Upgrade Shrine action idx1: {idx1}")
    
    elif event == sts.Event.VAMPIRES:
        if idx1 == 0: return FixedAction.VAMPIRES_ACCEPT
        elif idx1 == 1: return FixedAction.VAMPIRES_REFUSE
        elif idx1 == 2: return FixedAction.VAMPIRES_BLOOD_VIAL
        else:
            raise ValueError(f"Unknown Vampires action idx1: {idx1}")
    
    elif event == sts.Event.WE_MEET_AGAIN:
        if idx1 == 0: return FixedAction.WE_MEET_AGAIN_POTION
        elif idx1 == 1: return FixedAction.WE_MEET_AGAIN_GOLD
        elif idx1 == 2: return FixedAction.WE_MEET_AGAIN_CARD
        elif idx1 == 3: return FixedAction.WE_MEET_AGAIN_LEAVE
        else:
            raise ValueError(f"Unknown We Meet Again action idx1: {idx1}")
    
    elif event == sts.Event.OMINOUS_FORGE:
        if idx1 == 0: return FixedAction.OMINOUS_FORGE_UPGRADE
        elif idx1 == 1: return FixedAction.OMINOUS_FORGE_LOSE_HP
        elif idx1 == 2: return FixedAction.OMINOUS_FORGE_LEAVE
        else:
            raise ValueError(f"Unknown Ominous Forge action idx1: {idx1}")
    
    # Unsupported events - return None instead of throwing errors
    elif event == sts.Event.MATCH_AND_KEEP:
        return None  # Return None for unsupported events
    elif event == sts.Event.BONFIRE_SPIRITS:
        raise NotImplementedError(f"Bonfire Spirits event skips select phase - should not reach here")
    
    # Default fallback - throw error for unmapped events
    else:
        raise ValueError(f"Unmapped event {event} with idx1={idx1}, idx2={idx2}, event_data={event_data}")


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# %%
@dataclass
class Choice:
    """A set of possible actions"""
    obs: sts.NNRepresentation

    # ActionType.CARD
    cards_offered: list[sts.Card]
    card_actions: list[sts.GameAction]

    # ActionType.PATH
    paths_offered: list[int]  # x-coordinates (0-6) of valid path destinations
    path_actions: list[sts.GameAction]

    # ActionType.RELIC
    relics_offered: list[sts.RelicId]
    relic_actions: list[sts.GameAction]

    # ActionType.POTION
    potions_offered: list[sts.Potion]
    potion_actions: list[sts.GameAction]

    # ActionType.FIXED
    fixed_actions: list[FixedAction]  # Actions like SKIP
    fixed_actions_list: list[sts.GameAction]

    # Screen state information
    screen_state: sts.ScreenState
    select_screen_type: sts.CardSelectScreenType

    def as_dict(self):
        # Extract card IDs and upgrades from the Card objects
        all_card_ids = []
        all_upgrades = []
        
        for card in self.cards_offered:
            all_card_ids.append(int(card.id))
            all_upgrades.append(card.upgrade_count)
        
        return dict(
            obs=self.obs.as_dict(),
            cards_offered=dict(
                cards=np.array(all_card_ids, dtype=np.int32),
                upgrades=np.array(all_upgrades, dtype=np.int32),
            ),
            relics_offered=np.array(self.relics_offered, dtype=np.int32),
            potions_offered=np.array(self.potions_offered, dtype=np.int32),
            fixed_actions=self.fixed_actions if self.fixed_actions else [],
            paths_offered=np.array(self.paths_offered, dtype=np.int32),
            screen_state=int(self.screen_state),
            select_screen_type=int(self.select_screen_type),
        )


def process_choice(net: NN, choice: Choice) -> np.ndarray:
    """
    Process a single Choice through the neural network.
    Returns the flat logits array for this choice.
    """
    # Create a minimal batch with just this choice
    batch = [{
        **choice.as_dict(),
        'chosen_idx': 0,  # Dummy value, not used
        'outcome': 0.0,   # Dummy value, not used
    }]

    # Use existing collate_fn to create tensors
    batch_tensors = collate_fn(batch)
    
    # Process through network
    output = process_batch(batch_tensors, net)
    
    # Convert to CPU and return first (and only) item
    return output_to_cpu(output, batch_tensors)[0]


@dataclass
class Decision:
    """A Choice and what was chosen from it"""
    choice: Choice

    choice_type: ActionType  # which choice_type was chosen
    chosen_idx: int  # idx in arr/ays corresponding to choice_type

    def as_dict(self):
        return {
            **self.choice.as_dict(),
            'chosen_idx': self.chosen_idx,
            'choice_type': self.choice_type,
        }

def load_net(model_path, device=None, torch_compile_mode='default'):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = NN(ModelHP())
    net = net.to(device)
    
    if model_path is not None:
        state = torch.load(model_path, map_location=device, weights_only=True)
        net = load_network_backward_compatible(net, state)
    if torch_compile_mode != 'no':
        net = torch.compile(net, mode=torch_compile_mode)
    net.eval()
    
    return net

@dataclass
class NNRequest:
    request_id: str
    choice_data: dict  # Serialized choice data
    worker_id: int

@dataclass 
class NNResponse:
    request_id: str
    result: tuple  # (batch_tensors, output) tuple
    worker_id: int

@dataclass
class WeightUpdate:
    state_dict: dict  # CPU state dict

class NNWorkerProcess:
    """GPU process that handles neural network inference"""
    def __init__(self, request_queue: Queue, response_queues: dict, weight_queue: Queue, 
                 initial_state_dict: dict, net_constructor, batch_size: int = 32, max_wait_time: float = 0.01, 
                 batch_size_factor: int = 8, torch_compile_mode: str = 'default'):
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.weight_queue = weight_queue
        self.initial_state_dict = initial_state_dict
        self.net_constructor = net_constructor  # Function that creates the network
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.batch_size_factor = batch_size_factor
        self.torch_compile_mode = torch_compile_mode
        self.net = None
        
    def run(self):
        """Main GPU worker loop"""
        # Initialize the network in this process
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.net_constructor().to(device)
        
        # Load initial weights
        self.net.load_state_dict(self.initial_state_dict)
        
        # Apply torch.compile if needed
        if self.torch_compile_mode != 'no':
            self.net = torch.compile(self.net, mode=self.torch_compile_mode)
        self.net.eval()
        
        # Create local battle executor for timeouts
        battle_executor = ThreadPoolExecutor(max_workers=1)
        
        while True:
            # Check for weight updates (non-blocking)
            try:
                weight_update = self.weight_queue.get_nowait()
                if weight_update is None:  # Shutdown signal
                    break
                self._update_weights(weight_update, device)
            except Empty:
                pass
            
            # Collect batch of inference requests
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch, device)
        
        # Cleanup
        battle_executor.shutdown(wait=True)
    
    def _collect_batch(self):
        """Collect requests into a batch"""
        requests = []
        try:
            # Get at least one request
            first_request = self.request_queue.get(timeout=0.1)
            if first_request is None:  # Shutdown signal
                return None
            requests.append(first_request)
            
            # Try to get more requests up to next multiple of batch_size_factor
            start_time = time.time()
            target_size = ((len(requests) + self.batch_size_factor - 1) 
                         // self.batch_size_factor) * self.batch_size_factor
            target_size = min(target_size, self.batch_size)
            
            while len(requests) < target_size and time.time() - start_time < self.max_wait_time:
                try:
                    request = self.request_queue.get_nowait()
                    if request is None:  # Shutdown signal
                        break
                    requests.append(request)
                except Empty:
                    break

            unpadded_len = len(requests)
            
            # Pad batch to multiple of batch_size_factor if needed
            if len(requests) < target_size:
                target_size = ((len(requests) + self.batch_size_factor - 1) 
                             // self.batch_size_factor) * self.batch_size_factor
                # Duplicate last request to pad batch
                while len(requests) < target_size:
                    requests.append(requests[-1])
            
            return requests, unpadded_len
            
        except Empty:
            return None
    
    def _process_batch(self, batch_info, device):
        """Process a batch of requests"""
        requests, unpadded_len = batch_info
        
        try:
            # Process batch - requests now contain serialized choice data
            batch = [req.choice_data for req in requests]
            
            # Process through network
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                batch_tensors = collate_fn(batch)
                # Move batch to device
                batch_tensors = move_to_device(batch_tensors, device)
                output = process_batch(batch_tensors, self.net)
                responses = output_to_cpu(output, batch_tensors)
            
            # Send responses back to appropriate workers
            if isinstance(responses, tuple):
                # Handle (logits, values) from value head
                logits, values = responses
                for i, req in enumerate(requests[:unpadded_len]):
                    response = NNResponse(
                        request_id=req.request_id,
                        result=(batch_tensors, (logits[i], values[i])),
                        worker_id=req.worker_id
                    )
                    self.response_queues[req.worker_id].put(response)
            else:
                # Handle logits only
                for i, req in enumerate(requests[:unpadded_len]):
                    response = NNResponse(
                        request_id=req.request_id,
                        result=(batch_tensors, responses[i]),
                        worker_id=req.worker_id
                    )
                    self.response_queues[req.worker_id].put(response)
                    
        except Exception as e:
            print(f"Error in NN worker process: {type(e)} {e}")
            # Send error response to all waiting requests
            for req in requests[:unpadded_len]:
                self.response_queues[req.worker_id].put(e)
    
    def _update_weights(self, weight_update: WeightUpdate, device):
        """Update model weights"""
        # Move state dict to GPU and load
        device_state_dict = {k: v.to(device) for k, v in weight_update.state_dict.items()}
        self.net.load_state_dict(device_state_dict)

class NNClient:
    """CPU worker client for communicating with GPU process"""
    def __init__(self, worker_id: int, request_queue: Queue, response_queue: Queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.request_counter = 0
        self.pending_requests = {}
    
    def get_logits(self, choice: Choice) -> tuple[dict, np.ndarray]:
        """Get (batch_tensors, logits) from the network. Process-safe."""
        # Serialize the choice data before sending
        choice_data = {
            **flatten_dict(choice.as_dict()),
            'choice_type': 0,  # Dummy value
            'chosen_idx': 0,  # Dummy value
            'outcome': 0.0,   # Dummy value
        }
        
        # Create request
        request_id = f"worker_{self.worker_id}_req_{self.request_counter}"
        request = NNRequest(
            request_id=request_id,
            choice_data=choice_data,
            worker_id=self.worker_id
        )
        
        # Send request
        self.request_queue.put(request)
        self.request_counter += 1
        
        # Wait for response
        while True:
            response = self.response_queue.get()
            if isinstance(response, Exception):
                raise response
            if response.request_id == request_id:
                return response.result
            # Handle out-of-order responses (shouldn't happen in this simple case)
            self.pending_requests[response.request_id] = response

class NNServiceManager:
    """Manages neural network inference process and creates client interfaces"""
    def __init__(self, net: NN, net_constructor, batch_size=32, max_wait_time=0.01, batch_size_factor=8, torch_compile_mode='default', num_workers=4):
        # Round batch_size up to nearest multiple of batch_size_factor
        self.batch_size = ((batch_size + batch_size_factor - 1) // batch_size_factor) * batch_size_factor
        self.batch_size_factor = batch_size_factor
        self.max_wait_time = max_wait_time
        self.torch_compile_mode = torch_compile_mode
        self.num_workers = num_workers
        
        # Create queues
        self.request_queue = Queue(maxsize=1000)
        self.weight_queue = Queue(maxsize=10)
        self.response_queues = {i: Queue() for i in range(num_workers)}
        
        # Serialize the initial network state for the worker process
        initial_state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
        
        # Process handles
        self.gpu_process = None
        self.worker_processes = []
        
        # Start GPU process
        gpu_worker = NNWorkerProcess(
            self.request_queue,
            self.response_queues, 
            self.weight_queue,
            initial_state_dict,
            net_constructor,
            batch_size,
            max_wait_time,
            batch_size_factor,
            torch_compile_mode
        )
        self.gpu_process = Process(target=gpu_worker.run)
        self.gpu_process.start()
    
    def create_client(self) -> NNClient:
        """Create a client interface for communicating with this service"""
        # Find the next available worker ID (simple round-robin)
        worker_id = len(self.worker_processes) % self.num_workers
        return NNClient(worker_id, self.request_queue, self.response_queues[worker_id])
    
    def update_weights(self, net):
        """Update the inference network weights from the training network"""
        # Serialize the state dict and send to worker process
        state_dict = net.state_dict()
        # Move state dict to CPU for serialization
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        weight_update = WeightUpdate(state_dict=cpu_state_dict)
        self.weight_queue.put(weight_update)
    
    def stop(self):
        """Stop the service and wait for it to finish"""
        # Send shutdown signals
        self.request_queue.put(None)
        self.weight_queue.put(None)
        
        # Wait for GPU process to finish
        self.gpu_process.join(timeout=5.0)
        if self.gpu_process.is_alive():
            print("Warning: Force terminating GPU process")
            self.gpu_process.terminate()
            self.gpu_process.join()

def get_card_probs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities"""
    # Just pretend they're softmax logits (even though they're really sigmoid logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def entropy(probs: np.ndarray) -> float:
    """Calculate entropy of a probability distribution"""
    probs /= np.sum(probs)
    return -np.sum(probs * np.log(np.maximum(probs, 1e-20)))

def get_boltzmann_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Convert probabilities to Boltzmann distribution"""
    logits = np.log(np.maximum(probs, 1e-20)) / temperature
    logits = logits - np.max(logits)  # Subtract max for numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def sample_boltzmann(probs: np.ndarray, temperature: float, rng: random.Random = None) -> int:
    """Sample an index using Boltzmann distribution"""
    softmax_probs = get_boltzmann_probs(probs, temperature)
    if rng is None:
        return int(np.random.choice(len(probs), p=softmax_probs))
    return int(rng.choices(range(len(probs)), weights=softmax_probs, k=1)[0])

def pick_card_with_net(service, choice: Choice, actions: list[sts.GameAction], 
                      temperature: float = 1.0, stats: ChoiceStats = None, rng: random.Random = None) -> tuple[sts.GameAction, Path]:
    """Use neural network to pick a card/relic from the choices using Boltzmann sampling"""
    collated_input, output = service.get_logits(choice)
    
    # Handle both single logits and (logits, values) tuple
    if isinstance(output, tuple):
        logits, values = output
    else:
        logits = output
        values = None
    
    assert logits.size > 0, logits.shape
    
    # Convert logits to probabilities
    probs = get_card_probs(logits)
    
    if stats is not None:
        stats.add_choice(probs, temperature)
        
    chosen_idx = sample_boltzmann(probs, temperature, rng)
    
    # Convert flat index back to semantic path using choice_space
    path = choice_space.ix_to_path(collated_input['choices'], chosen_idx)
    
    # Special validation for card actions
    if path[0] == 'cards':
        card_index = path[1]
        if card_index >= len(choice.card_actions) or card_index < 0:
            print(f"Chosen index: {chosen_idx} from logits {logits}")
            print(f"{collated_input['choices']=}")
            raise ValueError(f"Chosen index {chosen_idx} out of bounds for {path}")
    
    action, action_desc = path_to_action_and_desc(choice, path)
    return action, path

def path_to_action_and_desc(choice: Choice, path: list, gc: Optional[sts.GameContext] = None) -> tuple[sts.GameAction, str]:
    """Convert a path to the corresponding action and description string.
    
    Args:
        choice: The Choice object containing available actions
        path: The path from choice_space.ix_to_path()
        gc: GameContext for fallback description (optional)
        
    Returns:
        tuple of (action, description_string)
    """
    if path[0] == 'cards':
        action = choice.card_actions[path[1]]
        chosen_card = choice.cards_offered[path[1]]
        action_desc = str(chosen_card)
    elif path[0] == 'relics':
        action = choice.relic_actions[path[1]]
        chosen_relic = choice.relics_offered[path[1]]
        action_desc = sts.RelicId(chosen_relic).name
    elif path[0] == 'potions':
        action = choice.potion_actions[path[1]]
        chosen_potion = choice.potions_offered[path[1]]
        action_desc = sts.Potion(chosen_potion).name
    elif path[0] == 'paths':
        action = choice.path_actions[path[1]]
        chosen_path = choice.paths_offered[path[1]]
        action_desc = f"x{chosen_path}"  # x-coordinate instead of room ID
    elif path[0] == 'fixed':
        action = choice.fixed_actions_list[path[1]]
        chosen_fixed = choice.fixed_actions[path[1]]
        # Handle both old enum format and new dict format
        if isinstance(chosen_fixed, dict):
            action_desc = str(chosen_fixed['action']).split('.')[-1]  # Remove "FixedAction." prefix
        else:
            action_desc = str(chosen_fixed).split('.')[-1]  # Remove "FixedAction." prefix
    else:
        raise ValueError(f"Unknown path type: {path[0]}")
    
    return action, action_desc

def construct_choice(gc: sts.GameContext, obs: sts.NNRepresentation, actions: list[sts.GameAction]) -> Optional[Choice]:
    """Construct a Choice object from the current game state and available actions."""
    cards_offered = []
    card_actions = []
    relics_offered = []
    relic_actions = []
    potions_offered = []
    potion_actions = []
    fixed_actions = []
    fixed_actions_list = []
    paths_offered = []
    path_actions = []
    
    # Build from available game actions, maintaining correspondence
    if gc.screen_state == sts.ScreenState.REWARDS:
        for action in actions:
            if action.rewards_action_type == sts.RewardsActionType.CARD:
                # handle singing bowl
                if action.idx2 == 5:
                    fixed_actions.append({'action': FixedAction.SINGING_BOWL})
                    fixed_actions_list.append(action)
                else:
                    cards_offered.append(gc.screen_state_info.rewards_container.cards[action.idx1][action.idx2])
                    card_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.POTION:
                potions_offered.append(gc.screen_state_info.rewards_container.potions[action.idx1])
                potion_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.SKIP:
                fixed_actions.append({'action': FixedAction.SKIP})
                fixed_actions_list.append(action)
                
    elif gc.screen_state == sts.ScreenState.SHOP_ROOM:
        # Shop cards are now returned as [card_set] where card_set contains all shop cards
        all_shop_relics = gc.screen_state_info.shop.relics
        all_shop_potions = gc.screen_state_info.shop.potions
        for action in actions:
            assert action.isValidAction(gc), f"Invalid shop action: {action.getDesc(gc)}"
            if action.rewards_action_type == sts.RewardsActionType.CARD:
                cards_offered.append(gc.screen_state_info.shop.cards[action.idx1])
                card_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.RELIC:
                relics_offered.append(all_shop_relics[action.idx1])
                relic_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.POTION:
                potions_offered.append(all_shop_potions[action.idx1])
                potion_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.SKIP:
                fixed_actions.append({'action': FixedAction.SKIP})
                fixed_actions_list.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.CARD_REMOVE:
                fixed_actions.append({'action': FixedAction.REMOVE})
                fixed_actions_list.append(action)
                
    elif gc.screen_state == sts.ScreenState.BOSS_RELIC_REWARDS:
        all_boss_relics = gc.screen_state_info.boss_relics
        for action in actions:
            if action.rewards_action_type == sts.RewardsActionType.RELIC:
                relics_offered.append(all_boss_relics[action.idx1])
                relic_actions.append(action)
            elif action.rewards_action_type == sts.RewardsActionType.SKIP:
                fixed_actions.append({'action': FixedAction.SKIP})
                fixed_actions_list.append(action)
            else:
                raise ValueError(f"Invalid boss relic reward action: {action.getDesc(gc)}")
        
    elif gc.screen_state == sts.ScreenState.REST_ROOM:
        for action in actions:
            # Rest actions use idx1 to indicate action type:
            # 0=rest, 1=smith, 2=recall, 3=lift, 4=toke, 5=dig, 6=skip
            if action.idx1 == 0:
                action_type = FixedAction.REST
            elif action.idx1 == 1:
                action_type = FixedAction.SMITH
            elif action.idx1 == 2:
                action_type = FixedAction.RECALL
            elif action.idx1 == 3:
                action_type = FixedAction.LIFT
            elif action.idx1 == 4:
                action_type = FixedAction.TOKE
            elif action.idx1 == 5:
                action_type = FixedAction.DIG
            else:  # idx1 == 6 or any other value defaults to skip
                action_type = FixedAction.SKIP
            
            fixed_actions.append({'action': action_type})
            fixed_actions_list.append(action)
                
    elif gc.screen_state == sts.ScreenState.CARD_SELECT:
        # Card selection screen (e.g., for smithing, removing cards, etc.)
        # Actions are indices into the toSelectCards list
        for action in actions:
            card_idx = action.idx1
            if card_idx < len(gc.screen_state_info.to_select_cards):
                select_card = gc.screen_state_info.to_select_cards[card_idx]
                cards_offered.append(select_card)  # select_card is already a Card object
                card_actions.append(action)
                
    elif gc.screen_state == sts.ScreenState.MAP_SCREEN:
        # Map screen actions - store x-coordinates directly from action.idx1
        for action in actions:
            paths_offered.append(action.idx1)  # x-coordinate (0-6)
            path_actions.append(action)

    elif gc.screen_state == sts.ScreenState.EVENT_SCREEN:
        # Event screen actions - map each action to appropriate FixedAction with info
        for action in actions:
            event_action = map_event_action_to_fixed_action(gc, action)
            if event_action is None:
                return None  # Return None if any action can't be mapped
            
            # Extract event-specific information
            gold, card, relic, event_info = extract_event_info(gc, action, event_action)
            
            # Create dictionary entry for this fixed action
            action_dict = {'action': event_action}
            if gold != 0:
                action_dict['gold'] = gold
            if card != sts.CardId.INVALID:
                action_dict['card'] = card
            if relic != sts.RelicId.INVALID:
                action_dict['relic'] = relic
            if event_info != EventFixedInfo.NONE:
                action_dict['info'] = event_info
            fixed_actions.append(action_dict)
            fixed_actions_list.append(action)
    else:
        # Return None for unsupported screen states
        return None

    return Choice(obs, cards_offered=cards_offered, card_actions=card_actions,
                  paths_offered=paths_offered, path_actions=path_actions,
                  fixed_actions=fixed_actions, fixed_actions_list=fixed_actions_list, 
                  relics_offered=relics_offered, relic_actions=relic_actions,
                  potions_offered=potions_offered, potion_actions=potion_actions,
                  screen_state=gc.screen_state, select_screen_type=gc.screen_state_info.select_screen_type)

def run_game(seed: int, net=None, temperature: float = 1.0, verbose: bool = False, stats: ChoiceStats = None):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    # Create seeded RNG instance for this game
    rng = random.Random(seed)

    agent = sts.Agent()
    agent.simulation_count_base = 1000
    choices: list[Decision] = []

    # Create an event to signal timeout
    timeout_event = threading.Event()
    
    def timeout_handler():
        timeout_event.set()
        print(f"Warning: Battle simulation taking too long for seed {seed}")

    while gc.outcome == sts.GameOutcome.UNDECIDED:
        try:
            if gc.screen_state == sts.ScreenState.BATTLE:
                if verbose:
                    print(gc.deck)
                
                # Start a timer before battle simulation
                timer = Timer(30.0, timeout_handler)
                timer.start()
                
                try:
                    agent.playout_battle(gc)
                finally:
                    timer.cancel()
                    
                # Check if we hit the timeout
                if timeout_event.is_set():
                    print(f"Seed {seed} did finish")
                    timeout_event.clear()
                    
                obs = sts.getNNRepresentation(gc)
            else:
                obs = sts.getNNRepresentation(gc)
                actions = sts.GameAction.getAllActionsInState(gc)

                choice = construct_choice(gc, obs, actions)
                # Pick action using either network or agent
                if net is not None and choice is not None:
                    assert choice.cards_offered or choice.paths_offered or choice.relics_offered or choice.potions_offered or choice.fixed_actions, (gc.screen_state, actions, gc.screen_state_info.boss_relics)
                    action, action_path = pick_card_with_net(net, choice, actions, temperature=temperature, stats=stats, rng=rng)
                    
                    # Use path information to determine choice_type and chosen_idx
                    if action_path[0] == 'cards':
                        choice_type = ActionType.CARD
                        chosen_idx = action_path[1]
                    elif action_path[0] == 'relics':
                        choice_type = ActionType.RELIC
                        chosen_idx = action_path[1]
                    elif action_path[0] == 'potions':
                        choice_type = ActionType.POTION
                        chosen_idx = action_path[1]
                    elif action_path[0] == 'paths':
                        choice_type = ActionType.PATH
                        chosen_idx = action_path[1]
                    elif action_path[0] == 'fixed':
                        choice_type = ActionType.FIXED
                        chosen_idx = action_path[1]
                    else:
                        choice_type = ActionType.INVALID
                        chosen_idx = -1
                else:
                    action = agent.pick_gameaction(gc)
                    
                    # For non-network actions (like map choices), use the old logic
                    choice_type = ActionType.INVALID
                    chosen_idx = -1
                
                if action not in actions:
                    print(gc)
                    print("chose", action.getDesc(gc))
                    print("options:")
                    for a in actions:
                        print(a.getDesc(gc))
                    raise ValueError("chosen action not in list of actions")

                # Record choice if valid
                if choice_type != ActionType.INVALID:
                    choices.append(Decision(choice, choice_type=choice_type, chosen_idx=chosen_idx))
                    
                if verbose:
                    print(action.getDesc(gc))
                assert action.isValidAction(gc), f"Invalid action: {action.getDesc(gc)}"
                action.execute(gc)
        except Exception:
            raise

    print(seed, gc.outcome, gc.floor_num)
    return (choices, gc.outcome, gc.floor_num)

# Global variables for multiprocessing worker initialization
_global_service_manager = None
_global_temperature = 1.0

def _init_worker_process(request_queue, response_queues, temperature):
    """Initialize worker process with communication channels"""
    global _global_service_manager, _global_temperature
    import os
    # Use process ID as worker ID for uniqueness
    worker_id = os.getpid() % len(response_queues)
    # Create client for this worker
    _global_service_manager = NNClient(worker_id, request_queue, response_queues[worker_id])
    _global_temperature = temperature

def _worker_run_game_data(seed: int):
    """Worker function for multiprocessing.Pool"""
    global _global_service_manager, _global_temperature
    try:
        choices, outcome, final_floor = run_game(seed, net=_global_service_manager, temperature=_global_temperature, verbose=False, stats=None)
    except Exception as e:
        print(f"Error in worker run_game_data for seed {seed}: {e}")
        raise

    df = pd.DataFrame([{
        **flatten_dict(c.choice.as_dict()),
        'choice_type': c.choice_type,
        'chosen_idx': c.chosen_idx,
    } for c in choices])
    df["outcome"] = {sts.GameOutcome.PLAYER_LOSS: 0, sts.GameOutcome.PLAYER_VICTORY: 1}[outcome]
    df["seed"] = seed
    df["final_floor"] = final_floor
    
    # Add pstrike count for each choice
    df["pstrike_count"] = [
        sum(1 for card_id in c.choice.obs.deck.cards if card_id == int(sts.CardId.PERFECTED_STRIKE))
        for c in choices
    ]
    
    return df

def run_game_data(seed: int, net=None, temperature: float = 1.0, stats: ChoiceStats = None):
    """Backward compatibility function for threading-based execution"""
    try:
        choices, outcome, final_floor = run_game(seed, net=net, temperature=temperature, verbose=False, stats=stats)
    except Exception as e:
        print(f"Error in run_game_data for seed {seed}: {e}")
        raise

    df = pd.DataFrame([{
        **flatten_dict(c.choice.as_dict()),
        'choice_type': c.choice_type,
        'chosen_idx': c.chosen_idx,
    } for c in choices])
    df["outcome"] = {sts.GameOutcome.PLAYER_LOSS: 0, sts.GameOutcome.PLAYER_VICTORY: 1}[outcome]
    df["seed"] = seed
    df["final_floor"] = final_floor
    
    # Add pstrike count for each choice
    df["pstrike_count"] = [
        sum(1 for card_id in c.choice.obs.deck.cards if card_id == int(sts.CardId.PERFECTED_STRIKE))
        for c in choices
    ]
    
    return df

class ChoiceStats:
    def __init__(self):
        self.entropies = []
        self.n_options = []
        self.boltzmann_entropies = []
        
    def add_choice(self, probs: np.ndarray, temperature: float):
        """Record statistics for a choice"""
        # Get raw entropy
        self.entropies.append(entropy(probs))
        
        # Count valid options
        self.n_options.append(np.sum(probs != float('-inf')))
        
        # Get Boltzmann entropy
        boltz_probs = get_boltzmann_probs(probs, temperature)
        self.boltzmann_entropies.append(entropy(boltz_probs))
    
    def plot_stats(self):
        import matplotlib.pyplot as plt
        
        # Raw entropy histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.entropies, bins=50, label='Raw')
        plt.hist(self.boltzmann_entropies, bins=50, alpha=0.5, label='After Boltzmann')
        plt.xlabel('Entropy')
        plt.ylabel('Count')
        plt.title(f'Distribution of Choice Entropies\n' +
                 f'Raw mean={np.mean(self.entropies):.3f}, ' +
                 f'Boltzmann mean={np.mean(self.boltzmann_entropies):.3f}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Entropy vs number of options scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(self.n_options, self.entropies, alpha=0.1, label='Raw')
        plt.scatter(self.n_options, self.boltzmann_entropies, alpha=0.1, label='After Boltzmann')
        plt.xlabel('Number of Options')
        plt.ylabel('Entropy')
        plt.title('Entropy vs Number of Options')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"\nChoice Statistics:")
        print(f"Total choices: {len(self.entropies)}")
        print(f"Raw entropy: mean={np.mean(self.entropies):.3f}, median={np.median(self.entropies):.3f}")
        print(f"Boltzmann entropy: mean={np.mean(self.boltzmann_entropies):.3f}, median={np.median(self.boltzmann_entropies):.3f}")
        print(f"Options: mean={np.mean(self.n_options):.1f}, median={np.median(self.n_options):.1f}")

def main(args):
    torch.set_float32_matmul_precision('high')
    
    if args.model_path == "<simple>":
        model_path = None
        print("Using SimpleAgent (no network)")
        service_manager = None
    else:
        if args.model_path in ("", "-"):
            model_path = None
        else:
            model_path = args.model_path
        # Load neural network and start service
        net = load_net(model_path, torch_compile_mode=args.torch_compile)
        
        # Create network constructor for worker process
        from network import NN, ModelHP
        net_constructor = lambda: NN(ModelHP())
        
        service_manager = NNServiceManager(
            net,
            net_constructor,
            batch_size=args.batch_size,
            batch_size_factor=min(min(8, args.batch_size), (args.num_threads + 1) // 2),
            torch_compile_mode=args.torch_compile,
        )
        print(f"Loaded neural network from {args.model_path}")

    stats = ChoiceStats()
    
    if service_manager is None:
        # Use threading for SimpleAgent
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [
                executor.submit(run_game_data, s, None, args.temperature, stats) 
                for s in range(args.start_seed, args.start_seed + args.num_games)
            ]
            df = pd.concat([
                future.result()
                for future
                in tqdm(
                    as_completed(futures),
                    total=args.num_games,
                    mininterval=5,
                    maxinterval=60,
                    miniters=args.num_threads,
                    smoothing=0.1,
                )
            ])
    else:
        # Use multiprocessing for neural network
        import functools
        
        # Create partial initializer function with the service manager's queues
        init_func = functools.partial(
            _init_worker_process,
            service_manager.request_queue,
            service_manager.response_queues,
            args.temperature
        )
        
        with mp.Pool(
            processes=args.num_threads,
            initializer=init_func
        ) as pool:
            # Map seeds to the worker function
            seeds = list(range(args.start_seed, args.start_seed + args.num_games))
            
            # Use imap_unordered for better progress tracking
            results = []
            for result in tqdm(
                pool.imap_unordered(_worker_run_game_data, seeds),
                total=args.num_games,
                mininterval=5,
                maxinterval=60,
                miniters=args.num_threads,
                smoothing=0.1,
            ):
                results.append(result)
            
            df = pd.concat(results)

    if service_manager is not None:
        service_manager.stop()

    # Shuffle the DataFrame
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if not args.no_save:
        df_path = f"rollouts_v4_{args.start_seed}_{args.start_seed+args.num_games}.parquet"
        df.to_parquet(df_path, engine="pyarrow")
        print(f"Saved to {df_path}")
    
    # Calculate and print winrate
    n_unique_seeds = df['seed'].nunique()
    n_wins = df.groupby('seed')['outcome'].last().sum()
    winrate = n_wins / n_unique_seeds
    print(f"\nResults from {n_unique_seeds} games:")
    print(f"Wins: {n_wins}")
    print(f"Winrate: {winrate:.1%}")
    
    # Plot choice statistics
    if args.plots:
        stats.plot_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Slay the Spire simulations with neural network guidance')
    parser.add_argument('--model-path', type=str, default="-",
                        help='Path to the neural network model file')
    parser.add_argument('--num-threads', type=int, default=4,
                        help='Number of parallel threads to use')
    parser.add_argument('--start-seed', type=int, default=0,
                        help='Starting seed for simulations')
    parser.add_argument('--num-games', type=int, default=1000,
                        help='Number of games to simulate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for neural network inference')
    parser.add_argument('--plots', action='store_true',
                        help='Plot statistics')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving results to parquet file')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for Boltzmann sampling (default: 1.0)')
    parser.add_argument('--torch-compile', type=str, default='default',
                        choices=['no', 'default', 'reduce-overhead', 'max-autotune'],
                        help='Torch compile mode for the neural network (default: default)')
    
    args = parser.parse_args()
    main(args)

## %%

