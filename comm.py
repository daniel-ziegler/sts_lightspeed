"""
Spirecomm to GameContext Converter

This module provides functions to convert spirecomm game state representations
into our internal C++ GameContext format for AI control of the real game.
"""

import random
import sys
import json
import argparse
import itertools
from typing import Optional, Union

import slaythespire as sts
from spirecomm.spire.game import Game
from spirecomm.spire.character import Intent, PlayerClass
from spirecomm.spire import card, relic, game, screen
from spirecomm.spire.screen import RestOption, ScreenType
from spirecomm.communication.action import *
from spirecomm.ai.priorities import *
from spirecomm.communication.coordinator import Coordinator



CHARACTER_CLASS_MAPPING = {
    PlayerClass.IRONCLAD: sts.CharacterClass.IRONCLAD,
    PlayerClass.THE_SILENT: sts.CharacterClass.SILENT,
    PlayerClass.DEFECT: sts.CharacterClass.DEFECT,
}

SCREEN_STATE_MAPPING = {
    # Map spirecomm ScreenType to our ScreenState
    ScreenType.EVENT: sts.ScreenState.EVENT_SCREEN,
    ScreenType.CHEST: sts.ScreenState.TREASURE_ROOM,
    ScreenType.SHOP_ROOM: sts.ScreenState.SHOP_ROOM,
    ScreenType.SHOP_SCREEN: sts.ScreenState.SHOP_ROOM,
    ScreenType.REST: sts.ScreenState.REST_ROOM,
    ScreenType.CARD_REWARD: sts.ScreenState.REWARDS,
    ScreenType.COMBAT_REWARD: sts.ScreenState.REWARDS,
    ScreenType.MAP: sts.ScreenState.MAP_SCREEN,
    ScreenType.BOSS_REWARD: sts.ScreenState.BOSS_RELIC_REWARDS,
    ScreenType.GRID: sts.ScreenState.CARD_SELECT,
    ScreenType.HAND_SELECT: sts.ScreenState.CARD_SELECT,
    ScreenType.GAME_OVER: sts.ScreenState.INVALID,  # Game over, no meaningful screen
    ScreenType.COMPLETE: sts.ScreenState.INVALID,   # Game complete
    ScreenType.NONE: sts.ScreenState.MAP_SCREEN,    # Default fallback
}

# Create lookup dictionaries for efficient reverse mapping
_card_string_to_id = None
_relic_name_to_id = None

def _get_card_string_to_id_map():
    """Create card string ID to CardId mapping dictionary."""
    global _card_string_to_id
    if _card_string_to_id is None:
        _card_string_to_id = {}
        for enum_idx, string_id in sts.getAllCardStringIds():
            _card_string_to_id[string_id] = enum_idx
    return _card_string_to_id

def _get_relic_name_to_id_map():
    """Create relic name to RelicId mapping dictionary."""
    global _relic_name_to_id
    if _relic_name_to_id is None:
        _relic_name_to_id = {}
        for enum_idx, name in sts.getAllRelicNames():
            _relic_name_to_id[name] = enum_idx
    return _relic_name_to_id

def map_card_id(spire_card_id: str) -> sts.CardId:
    """Map spirecomm card ID to our CardId enum using dynamic lookup."""
    card_map = _get_card_string_to_id_map()
    enum_idx = card_map.get(spire_card_id)
    if enum_idx is not None:
        return sts.CardId(enum_idx)
    return sts.CardId.INVALID


def map_relic_id(spire_relic_id: str) -> sts.RelicId:
    """Map spirecomm relic ID to our RelicId enum using dynamic lookup."""
    relic_map = _get_relic_name_to_id_map()
    enum_idx = relic_map.get(spire_relic_id)
    if enum_idx is not None:
        return sts.RelicId(enum_idx)
    return sts.RelicId.INVALID

def map_character_class(spire_class: PlayerClass) -> sts.CharacterClass:
    """Map spirecomm PlayerClass to our CharacterClass."""
    return CHARACTER_CLASS_MAPPING.get(spire_class, sts.CharacterClass.INVALID)


def map_power_id(spire_power_name: str) -> sts.PlayerStatus:
    """Map spirecomm power name to our PlayerStatus enum."""
    player_status = sts.getPlayerStatusForString(spire_power_name)
    if player_status == sts.PlayerStatus.INVALID:
        raise ValueError(f"Unknown status name: {spire_power_name}")
    return player_status


def convert_card(spire_card: card.Card) -> sts.Card:
    """Convert spirecomm Card to our Card."""
    card_id = map_card_id(spire_card.card_id)
    if card_id == sts.CardId.INVALID:
        raise ValueError(f"Unknown card ID: {spire_card.card_id}")
    
    sts_card = sts.Card(card_id, 0)
    
    # Apply upgrades (handles Searing Blow multiple upgrades correctly)
    for _ in range(spire_card.upgrades):
        sts_card.upgrade()
    
    sts_card.misc = spire_card.misc
    
    return sts_card


def convert_deck(spire_cards: list[card.Card]) -> list[sts.Card]:
    """Convert list of spirecomm Cards to our Card list."""
    if not isinstance(spire_cards, list):
        raise TypeError(f"Expected list of cards, got {type(spire_cards)}")
    
    converted_cards = []
    for i, spire_card in enumerate(spire_cards):
        try:
            converted_cards.append(convert_card(spire_card))
        except Exception as e:
            raise ValueError(f"Failed to convert card {i} ({getattr(spire_card, 'card_id', 'unknown')}): {e}")
    
    return converted_cards


def convert_relic(spire_relic: relic.Relic) -> sts.Relic:
    """Convert spirecomm Relic to our Relic."""
    relic_id = map_relic_id(spire_relic.name)  # spirecomm uses name field
    if relic_id == sts.RelicId.INVALID:
        raise ValueError(f"Unknown relic name: {spire_relic.name}")
    
    sts_relic = sts.Relic(relic_id, spire_relic.counter)
    
    return sts_relic


def convert_relics(spire_relics: list[relic.Relic]) -> list[sts.Relic]:
    """Convert list of spirecomm Relics to our Relic list."""
    if not isinstance(spire_relics, list):
        raise TypeError(f"Expected list of relics, got {type(spire_relics)}")
    
    converted_relics = []
    for i, spire_relic in enumerate(spire_relics):
        try:
            converted_relics.append(convert_relic(spire_relic))
        except Exception as e:
            raise ValueError(f"Failed to convert relic {i} ({getattr(spire_relic, 'name', 'unknown')}): {e}")
    
    return converted_relics


def convert_combat_state(spire_game: game.Game, gc: sts.GameContext) -> sts.BattleContext:
    """
    Convert spirecomm combat state to BattleContext.
    
    This function handles:
    - Player energy, HP, block, powers
    - Monster states including HP, block, intent, powers  
    - Card piles (hand, draw pile, discard pile, exhaust pile)
    - Turn information
    
    Returns:
        BattleContext initialized with spirecomm combat state
    """
    if not spire_game.in_combat:
        raise ValueError("Cannot convert combat state when not in combat")
    
    # Create battle context from GameContext
    bc = gc.empty_battle_context()
    
    # Clear the initialized cards to avoid mixing with spirecomm state
    bc.cards.clear()
    
    # Player state conversion
    player = spire_game.player
    if player:
        # Set basic player stats
        bc.player.energy = player.energy
        bc.player.block = player.block
            
        # Convert player powers/buffs/debuffs
        for power in player.powers:
            power_status = map_power_id(power.power_name)
            if is_positive_player_power(power.power_name):
                bc.player.buff(power_status, power.amount)
            else:
                bc.player.debuff(power_status, power.amount, False)
    
    # Card piles conversion - create CardInstance objects from spirecomm cards
    for spire_card in spire_game.hand:
        card_instance = convert_spire_card_to_instance(spire_card)
        print(f"card[{bc.cards.cardsInHand}]={card_instance.id}", file=sys.stderr)
        bc.cards.moveToHand(card_instance)
    
    if spire_game.draw_pile:
        for spire_card in spire_game.draw_pile:
            card_instance = convert_spire_card_to_instance(spire_card)
            bc.cards.moveToDrawPileTop(card_instance)
        
    if spire_game.discard_pile:
        for spire_card in spire_game.discard_pile:
            card_instance = convert_spire_card_to_instance(spire_card)
            bc.cards.moveToDiscardPile(card_instance)
        
    if spire_game.exhaust_pile:
        for spire_card in spire_game.exhaust_pile:
            card_instance = convert_spire_card_to_instance(spire_card)
            bc.cards.moveToExhaustPile(card_instance)
    
    # Monster states conversion - create monsters from game state
    if spire_game.monsters and len(spire_game.monsters) > 0:
        for i, monster in enumerate(spire_game.monsters):
            if i >= 5:  # MonsterGroup supports max 5 monsters
                break
            
            # Try to map spirecomm monster name to MonsterId, fallback to generic
            monster_id = map_monster_string_to_id(monster.monster_id)
            bc.monsters.createMonster(bc, monster_id)
            
            # Now set the monster properties from spirecomm data
            sts_monster = bc.monsters[i]
            
            # Basic monster stats
            sts_monster.curHp = monster.current_hp
            sts_monster.maxHp = monster.max_hp
            sts_monster.block = monster.block
            sts_monster.halfDead = monster.half_dead
            
            # Set position/index
            sts_monster.idx = i
            
            # Convert monster powers
            if hasattr(monster, 'powers') and monster.powers:
                for power in monster.powers:
                    monster_status = map_monster_power_id(power.power_name)
                    if monster_status:
                        if is_positive_monster_power(power.power_name):
                            sts_monster.buff(monster_status, power.amount)
                        else:
                            sts_monster.addDebuff(monster_status, power.amount, False)
    
    return bc


def convert_spire_card_to_instance(spire_card: card.Card) -> sts.CardInstance:
    """Convert a spirecomm Card to a CardInstance."""
    # Map the card ID using card_id field, not name
    card_id = map_card_id(spire_card.card_id)
    if card_id == sts.CardId.INVALID:
        raise ValueError(f"Unknown card: {spire_card.card_id}")
    
    # Create CardInstance
    instance = sts.CardInstance(card_id, spire_card.upgrades > 0)
    
    # Set additional properties
    if hasattr(spire_card, 'cost'):
        instance.cost = spire_card.cost
        instance.costForTurn = spire_card.cost
    
    # Handle upgrade count for Searing Blow
    if spire_card.upgrades > 1:
        # Apply additional upgrades for Searing Blow
        for _ in range(spire_card.upgrades - 1):
            instance.upgrade()
    
    return instance


def map_monster_string_to_id(monster_id: str = '') -> sts.MonsterId:
    """Map spirecomm monster id to MonsterId enum using dynamic lookup."""
    # Create lookup dictionary similar to cards and relics
    monster_map = _get_monster_string_to_id_map()
    enum_idx = monster_map.get(monster_id)
    if enum_idx is not None:
        return sts.MonsterId(enum_idx)
    
    # Fallback to a generic monster if no match found
    print(f"Warning: Unknown monster id '{monster_id}', using INVALID as fallback", file=sys.stderr)
    return sts.MonsterId.INVALID


# Create lookup dictionary for monster names
_monster_string_to_id = None

def _get_monster_string_to_id_map():
    """Create monster name to MonsterId mapping dictionary."""
    global _monster_string_to_id
    if _monster_string_to_id is None:
        _monster_string_to_id = {}
        for enum_idx, string_id in sts.getAllMonsterStringIds():
            _monster_string_to_id[string_id] = enum_idx
    return _monster_string_to_id


def map_monster_power_id(power_name: str) -> sts.MonsterStatus:
    """Map spirecomm monster power name to MonsterStatus enum."""
    monster_power_mapping = {
        # Common monster powers
        "Artifact": sts.MonsterStatus.ARTIFACT,
        "Block Return": sts.MonsterStatus.BLOCK_RETURN,
        "Choked": sts.MonsterStatus.CHOKED,
        "Corpse Explosion": sts.MonsterStatus.CORPSE_EXPLOSION,
        "Lockon": sts.MonsterStatus.LOCK_ON,
        "Lock-On": sts.MonsterStatus.LOCK_ON,
        "Malleable": sts.MonsterStatus.MALLEABLE,
        "Metallicize": sts.MonsterStatus.METALLICIZE,
        "Plated Armor": sts.MonsterStatus.PLATED_ARMOR,
        "Poison": sts.MonsterStatus.POISON,
        "Regenerate": sts.MonsterStatus.REGEN,
        "Regeneration": sts.MonsterStatus.REGEN,
        "Shackled": sts.MonsterStatus.SHACKLED,
        "Strength": sts.MonsterStatus.STRENGTH,
        "Vulnerable": sts.MonsterStatus.VULNERABLE,
        "Weakened": sts.MonsterStatus.WEAK,
        "Weak": sts.MonsterStatus.WEAK,
        
        # Specific monster powers
        "Angry": sts.MonsterStatus.ANGRY,
        "Beat of Death": sts.MonsterStatus.BEAT_OF_DEATH,
        "Curiosity": sts.MonsterStatus.CURIOSITY,
        "Curl Up": sts.MonsterStatus.CURL_UP,
        "Enrage": sts.MonsterStatus.ENRAGE,
        "Fading": sts.MonsterStatus.FADING,
        "Flight": sts.MonsterStatus.FLIGHT,
        "Generic Strength Up": sts.MonsterStatus.GENERIC_STRENGTH_UP,
        "Intangible": sts.MonsterStatus.INTANGIBLE,
        "Mode Shift": sts.MonsterStatus.MODE_SHIFT,
        "Ritual": sts.MonsterStatus.RITUAL,
        "Slow": sts.MonsterStatus.SLOW,
        "Spore Cloud": sts.MonsterStatus.SPORE_CLOUD,
        "Thievery": sts.MonsterStatus.THIEVERY,
        "Thorns": sts.MonsterStatus.THORNS,
        "Time Warp": sts.MonsterStatus.TIME_WARP,
        "Invincible": sts.MonsterStatus.INVINCIBLE,
        "Reactive": sts.MonsterStatus.REACTIVE,
        "Sharp Hide": sts.MonsterStatus.SHARP_HIDE,
    }
    
    return monster_power_mapping.get(power_name)


def is_positive_player_power(power_name: str) -> bool:
    """Determine if a player power is positive (buff) or negative (debuff)."""
    negative_powers = {
        "Double Damage", "Draw Reduction", "Frail", "Vulnerable", "Weak",
        "Bias", "Confused", "Constricted", "Entangled", "Fasting", "Hex",
        "Lose Dexterity", "Lose Strength", "No Block", "No Draw"
    }
    return power_name not in negative_powers


def is_positive_monster_power(power_name: str) -> bool:
    """Determine if a monster power is positive (buff) or negative (debuff)."""
    negative_powers = {
        "Choked", "Corpse Explosion", "Lockon", "Lock-On", "Poison", 
        "Shackled", "Vulnerable", "Weak", "Weakened"
    }
    return power_name not in negative_powers


def map_screen_state(spire_game: game.Game) -> sts.ScreenState:
    """Map spirecomm game state to our ScreenState enum."""
    
    if spire_game.in_combat:
        return sts.ScreenState.BATTLE
    
    screen_type = spire_game.screen_type
    return SCREEN_STATE_MAPPING.get(screen_type, sts.ScreenState.MAP_SCREEN)


def validate_spire_game(spire_game: game.Game) -> None:
    """
    Validate spirecomm Game state for conversion.
    
    Args:
        spire_game: Game state to validate
        
    Raises:
        ValueError: If game state is invalid or incomplete
        TypeError: If game state has wrong types
    """
    if not isinstance(spire_game, game.Game):
        raise TypeError(f"Expected Game object, got {type(spire_game)}")
    
    # Check required fields
    required_fields = ['current_hp', 'max_hp', 'gold', 'act', 'floor', 'seed', 'character']
    for field in required_fields:
        if not hasattr(spire_game, field):
            raise ValueError(f"Game missing required field: {field}")
        
        value = getattr(spire_game, field)
        if value is None:
            raise ValueError(f"Game field {field} is None")
    
    # Validate character class
    if spire_game.character not in CHARACTER_CLASS_MAPPING:
        raise ValueError(f"Unknown character class: {spire_game.character}")
    
    # Validate numeric fields
    if spire_game.current_hp < 0 or spire_game.max_hp <= 0:
        raise ValueError(f"Invalid HP values: current={spire_game.current_hp}, max={spire_game.max_hp}")
    
    if spire_game.gold < 0:
        raise ValueError(f"Invalid gold amount: {spire_game.gold}")
    
    if spire_game.act < 1 or spire_game.act > 4:
        raise ValueError(f"Invalid act: {spire_game.act}")
    
    if spire_game.floor < 0:
        raise ValueError(f"Invalid floor: {spire_game.floor}")
    
    # Validate deck if present
    for i, card_obj in enumerate(spire_game.deck):
        if not isinstance(card_obj, card.Card):
            raise TypeError(f"Deck card {i} is not a Card object: {type(card_obj)}")
    
    # Validate relics if present
    for i, relic_obj in enumerate(spire_game.relics):
        if not isinstance(relic_obj, relic.Relic):
            raise TypeError(f"Relic {i} is not a Relic object: {type(relic_obj)}")


def set_screen_state_info(gc: sts.GameContext, spire_game: game.Game) -> None:
    """
    Set ScreenStateInfo fields based on spirecomm game state.
    
    Comprehensively maps all available spirecomm data to ScreenStateInfo fields.
    """
    info = gc.screen_state_info
    
    # Set basic game state fields
    info.gold = spire_game.gold
    
    # Set potion information
    if spire_game.potions:
        info.potionIdx = len(spire_game.potions)
    
    # Set deck card indices based on deck composition  
    from spirecomm.spire.card import CardType
    attack_cards = [i for i, spire_card in enumerate(spire_game.deck) 
                   if spire_card.type == CardType.ATTACK]
    skill_cards = [i for i, spire_card in enumerate(spire_game.deck)
                  if spire_card.type == CardType.SKILL]  
    power_cards = [i for i, spire_card in enumerate(spire_game.deck)
                  if spire_card.type == CardType.POWER]
    
    if attack_cards:
        info.attackCardDeckIdx = attack_cards[0]
    if skill_cards:
        info.skillCardDeckIdx = skill_cards[0]
    if power_cards:
        info.powerCardDeckIdx = power_cards[0]
    
    # Screen-specific mappings
    if spire_game.screen_type == screen.ScreenType.BOSS_REWARD:
        # Boss reward screen - map boss relics
        boss_relics = spire_game.screen.relics
        for i, spire_relic in enumerate(boss_relics[:3]):
            relic_id = map_relic_id(spire_relic.name)
            if relic_id != sts.RelicId.INVALID:
                info.boss_relics[i] = relic_id
                
    elif spire_game.screen_type == screen.ScreenType.COMBAT_REWARD:
        # Combat reward screen - populate rewards container
        rewards = spire_game.screen.rewards
        info.rewards_container.cards.clear()
        info.rewards_container.relics.clear()
        info.rewards_container.gold.clear()
        info.rewards_container.potions.clear()
        
        for reward_item in rewards:
            if reward_item.reward_type == screen.RewardType.CARD:
                card_id = map_card_id(reward_item.card.card_id)
                if card_id != sts.CardId.INVALID:
                    info.rewards_container.cards.append(card_id)
                        
            elif reward_item.reward_type == screen.RewardType.GOLD:
                info.rewards_container.gold.append(reward_item.gold)
                    
            elif reward_item.reward_type == screen.RewardType.RELIC:
                relic_id = map_relic_id(reward_item.relic.name)
                if relic_id != sts.RelicId.INVALID:
                    info.rewards_container.relics.append(relic_id)
                        
            elif reward_item.reward_type == screen.RewardType.POTION:
                # TODO Map potion if we have potion mapping
                pass
                    
            elif reward_item.reward_type == screen.RewardType.EMERALD_KEY:
                info.rewards_container.emerald_key = True
                
            elif reward_item.reward_type == screen.RewardType.SAPPHIRE_KEY:
                info.rewards_container.sapphire_key = True
                
    elif spire_game.screen_type == screen.ScreenType.SHOP_SCREEN:
        # Shop screen - populate shop information
        shop_screen = spire_game.screen
        shop = info.shop
        
        # Clear existing shop data
        shop.cards.clear()
        shop.relics.clear()
        shop.potions.clear()
        
        # Map shop cards
        for shop_card in shop_screen.cards:
            card_id = map_card_id(shop_card.card_id)
            if card_id != sts.CardId.INVALID:
                shop.cards.append(card_id)
                
        # Map shop relics
        for shop_relic in shop_screen.relics:
            relic_id = map_relic_id(shop_relic.name)
            if relic_id != sts.RelicId.INVALID:
                shop.relics.append(relic_id)
                
        # Set purge cost
        info.goldLoss = shop_screen.purge_cost
            
    elif spire_game.screen_type == screen.ScreenType.EVENT:
        # Event screen - set event data
        event_id = spire_game.screen.event_id  
        print(f"Event: {spire_game.screen}", file=sys.stderr)
        # TODO map this once we know what the data looks like

    elif spire_game.screen_type == screen.ScreenType.GRID:
        # Grid select screen (transform, upgrade, remove, etc.)
        grid_screen = spire_game.screen
        
        # Set select screen type based on screen purpose
        if grid_screen.for_transform:
            info.select_screen_type = sts.CardSelectScreenType.TRANSFORM
        elif grid_screen.for_upgrade:
            info.select_screen_type = sts.CardSelectScreenType.UPGRADE
        elif grid_screen.for_purge:
            info.select_screen_type = sts.CardSelectScreenType.REMOVE
        else:
            info.select_screen_type = sts.CardSelectScreenType.OBTAIN
            
        # Set cards to select from
        info.to_select_cards.clear()
        for card in grid_screen.cards:
            card_id = map_card_id(card.card_id)
            if card_id != sts.CardId.INVALID:
                info.to_select_cards.append(card_id)
                    
        # Set already selected cards  
        info.have_selected_cards.clear()
        for card in grid_screen.selected_cards:
            card_id = map_card_id(card.card_id)
            if card_id != sts.CardId.INVALID:
                info.have_selected_cards.append(card_id)
                    
    elif spire_game.screen_type == screen.ScreenType.HAND_SELECT:
        # Hand select screen
        hand_screen = spire_game.screen
        
        info.select_screen_type = sts.CardSelectScreenType.DUPLICATE
        
        info.to_select_cards.clear() 
        for card in hand_screen.cards:
            card_id = map_card_id(card.card_id)
            if card_id != sts.CardId.INVALID:
                info.to_select_cards.append(card_id)
                    
    elif spire_game.screen_type == screen.ScreenType.CARD_REWARD:
        # Card reward screen
        card_screen = spire_game.screen
        
        info.to_select_cards.clear()
        for card in card_screen.cards:
            card_id = map_card_id(card.card_id)
            if card_id != sts.CardId.INVALID:
                info.to_select_cards.append(card_id)
                    
    elif spire_game.screen_type == screen.ScreenType.REST:
        # The C++ code knows what rest options are available
        pass
                    
    # Set relic indices if we have relics
    if len(spire_game.relics) > 0:
        relic_id = map_relic_id(spire_game.relics[0].name)
        info.relicIdx0 = int(relic_id)
        
    if len(spire_game.relics) > 1:
        relic_id = map_relic_id(spire_game.relics[1].name)  
        info.relicIdx1 = int(relic_id)


def spirecomm_to_gamecontext(spire_game: game.Game) -> sts.GameContext:
    """
    Convert spirecomm Game state to our GameContext.
    
    Args:
        spire_game: Game state from spirecomm
        
    Returns:
        GameContext with equivalent state
        
    Raises:
        ValueError: If game state contains unknown/unsupported elements
        TypeError: If game state has wrong types
    """
    # Validate input
    validate_spire_game(spire_game)
    
    # Create GameContext with basic parameters
    character_class = map_character_class(spire_game.character)
    gc = sts.GameContext(character_class, int(spire_game.seed), int(spire_game.ascension_level or 0))
    
    # Set basic game state
    gc.cur_hp = spire_game.current_hp
    gc.max_hp = spire_game.max_hp
    gc.gold = spire_game.gold
    gc.act = spire_game.act
    gc.floor_num = spire_game.floor
    
    # Convert and set deck
    if spire_game.deck:
        # Clear the starting deck first
        gc.clear_deck()
        sts_deck = convert_deck(spire_game.deck)
        # Add converted cards
        for card in sts_deck:
            gc.obtain_card(card)
    
    # Convert and set relics
    if spire_game.relics:
        sts_relics = convert_relics(spire_game.relics)
        # Add each relic to the GameContext
        for sts_relic in sts_relics:
            gc.obtain_relic(sts_relic.id)
    
    # Set screen state
    gc.screen_state = map_screen_state(spire_game)
    
    # Set screen state info based on current screen
    set_screen_state_info(gc, spire_game)
    
    return gc


def gamecontext_to_spirecomm_action(gc: sts.GameContext, game_action: sts.GameAction) -> str:
    """
    Convert a GameContext action to spirecomm command format.
    
    This is the reverse direction - taking actions from our AI and converting
    them to commands that spirecomm can send to the real game.
    
    Args:
        gc: Current GameContext state
        game_action: Action chosen by our AI
        
    Returns:
        String command for spirecomm to execute
        
    Raises:
        ValueError: If action cannot be converted to spirecomm format
    """
    # Get action description to understand what type of action this is
    action_desc = game_action.getDesc(gc)
    
    # Check if this is a battle action
    if gc.screen_state == sts.ScreenState.BATTLE:
        # Battle actions - playing cards, using potions, ending turn
        if "Play" in action_desc:
            # Extract card index from action - this would need more sophisticated parsing
            # For now, return a generic play command
            return f"play {game_action.idx1 + 1}"  # spirecomm uses 1-based indexing
            
        elif "End Turn" in action_desc or "end" in action_desc.lower():
            return "end"
            
        elif "Potion" in action_desc:
            # Potion usage - would need to determine use vs discard
            return f"potion use {game_action.idx1}"
            
        else:
            raise ValueError(f"Unknown battle action: {action_desc}")
    
    # Non-battle screen actions
    elif gc.screen_state == sts.ScreenState.REWARDS:
        # Reward selection
        if game_action.rewards_action_type == sts.RewardsActionType.CARD:
            # Card reward selection by name would require card mapping
            return f"choose {game_action.idx1}"
        elif game_action.rewards_action_type == sts.RewardsActionType.GOLD:
            return f"choose {game_action.idx1}"
        elif game_action.rewards_action_type == sts.RewardsActionType.RELIC:
            return f"choose {game_action.idx1}"
        elif game_action.rewards_action_type == sts.RewardsActionType.SKIP:
            return "choose skip"
        return f"choose {game_action.idx1}"
        
    elif gc.screen_state == sts.ScreenState.BOSS_RELIC_REWARDS:
        # Boss relic selection
        return f"choose {game_action.idx1}"
        
    elif gc.screen_state == sts.ScreenState.MAP_SCREEN:
        # Map navigation
        return f"choose {game_action.idx1}"
        
    elif gc.screen_state == sts.ScreenState.SHOP_ROOM:
        # Shop actions - buying cards, relics, potions, or removal
        return f"choose {game_action.idx1}"
        
    elif gc.screen_state == sts.ScreenState.REST_ROOM:
        # Rest site actions - rest, smith, dig, lift, etc.
        if "Rest" in action_desc:
            return "choose rest"
        elif "Smith" in action_desc or "Upgrade" in action_desc:
            return "choose smith"
        elif "Dig" in action_desc:
            return "choose dig"
        elif "Lift" in action_desc:
            return "choose lift"
        else:
            return f"choose {game_action.idx1}"
            
    elif gc.screen_state == sts.ScreenState.EVENT_SCREEN:
        # Event choices
        return f"choose {game_action.idx1}"
        
    elif gc.screen_state == sts.ScreenState.CARD_SELECT:
        # Card selection screens (transform, upgrade, remove, etc.)
        return f"choose {game_action.idx1}"
        
    elif gc.screen_state == sts.ScreenState.TREASURE_ROOM:
        # Treasure chest
        return "choose open"
        
    else:
        print(f"Unknown screen state: {gc.screen_state}", file=sys.stderr)
        # Fallback for unknown screen states
        return f"choose {game_action.idx1}"


def create_spirecomm_bridge(spire_game: game.Game) -> tuple[sts.GameContext, callable]:
    """
    Create a complete bridge between spirecomm and our GameContext.
    
    Returns:
        Tuple of (converted GameContext, action_converter function)
    """
    gc = spirecomm_to_gamecontext(spire_game)
    
    def action_converter(game_action: sts.GameAction) -> str:
        return gamecontext_to_spirecomm_action(gc, game_action)
    
    return gc, action_converter


def test_basic_conversion():
    """
    Test basic spirecomm to GameContext conversion with a minimal example.
    
    This function creates a mock spirecomm Game state and tests the conversion.
    """
    # Create a minimal mock spirecomm Game
    mock_game = game.Game()
    mock_game.current_hp = 70
    mock_game.max_hp = 80
    mock_game.gold = 150
    mock_game.act = 1
    mock_game.floor = 5
    mock_game.seed = 12345
    mock_game.character = character.PlayerClass.IRONCLAD
    mock_game.ascension_level = 0
    
    # Create a basic deck
    strike = card.Card("Strike_R", "Strike", card.CardType.ATTACK, card.CardRarity.BASIC, 0, False, 1)
    defend = card.Card("Defend_R", "Defend", card.CardType.SKILL, card.CardRarity.BASIC, 0, False, 1)
    bash = card.Card("Bash", "Bash", card.CardType.ATTACK, card.CardRarity.BASIC, 0, True, 2)
    
    mock_game.deck = [strike, defend, bash]
    
    # Create a basic relic
    starter_relic = relic.Relic("Burning Blood", "Burning Blood", 0)
    mock_game.relics = [starter_relic]
    
    try:
        # Test the conversion
        gc = spirecomm_to_gamecontext(mock_game)
        
        # Verify basic fields
        assert gc.cur_hp == 70
        assert gc.max_hp == 80
        assert gc.gold == 150
        assert gc.act == 1
        assert gc.floor_num == 5
        assert gc.seed == 12345
        
        print("✓ Basic conversion test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic conversion test failed: {e}")
        return False



class STSLightspeedAgent:

    def __init__(self, chosen_class=PlayerClass.THE_SILENT):
        self.game = Game()
        self.errors = 0
        self.choose_good_card = False
        self.skipped_cards = False
        self.visited_shop = False
        self.map_route = []
        self.chosen_class = chosen_class
        self.priorities = Priority()
        self.change_class(chosen_class)

    def change_class(self, new_class):
        self.chosen_class = new_class
        if self.chosen_class == PlayerClass.THE_SILENT:
            self.priorities = SilentPriority()
        elif self.chosen_class == PlayerClass.IRONCLAD:
            self.priorities = IroncladPriority()
        elif self.chosen_class == PlayerClass.DEFECT:
            self.priorities = DefectPowerPriority()
        else:
            self.priorities = random.choice(list(PlayerClass))

    def handle_error(self, error):
        raise Exception(error)

    def get_next_action_in_game(self, game_state):
        self.game = game_state
        #time.sleep(0.07)
        if self.game.choice_available:
            return self.handle_screen()
        if self.game.proceed_available:
            return ProceedAction()
        if self.game.play_available:
            return self.handle_combat()
        if self.game.end_available:
            return EndTurnAction()
        if self.game.cancel_available:
            return CancelAction()

    def get_next_action_out_of_game(self):
        return StartGameAction(self.chosen_class)

    def is_monster_attacking(self):
        for monster in self.game.monsters:
            if monster.intent.is_attack() or monster.intent == Intent.NONE:
                return True
        return False

    def get_incoming_damage(self):
        incoming_damage = 0
        for monster in self.game.monsters:
            if not monster.is_gone and not monster.half_dead:
                if monster.move_adjusted_damage is not None:
                    incoming_damage += monster.move_adjusted_damage * monster.move_hits
                elif monster.intent == Intent.NONE:
                    incoming_damage += 5 * self.game.act
        return incoming_damage

    def get_low_hp_target(self):
        available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        best_monster = min(available_monsters, key=lambda x: x.current_hp)
        return best_monster

    def get_high_hp_target(self):
        available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        best_monster = max(available_monsters, key=lambda x: x.current_hp)
        return best_monster

    def many_monsters_alive(self):
        available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        return len(available_monsters) > 1

    def handle_combat(self):
        gc = spirecomm_to_gamecontext(self.game)
        bc = convert_combat_state(self.game, gc)
        agent = sts.Agent()
        
        # TODO


    def use_next_potion(self):
        for potion in self.game.get_real_potions():
            if potion.can_use:
                if potion.requires_target:
                    return PotionAction(True, potion=potion, target_monster=self.get_low_hp_target())
                else:
                    return PotionAction(True, potion=potion)

    def handle_screen(self):
        if self.game.screen_type == ScreenType.EVENT:
            if self.game.screen.event_id in ["Vampires", "Masked Bandits", "Knowing Skull", "Ghosts", "Liars Game", "Golden Idol", "Drug Dealer", "The Library"]:
                return ChooseAction(len(self.game.screen.options) - 1)
            else:
                return ChooseAction(0)
        elif self.game.screen_type == ScreenType.CHEST:
            return OpenChestAction()
        elif self.game.screen_type == ScreenType.SHOP_ROOM:
            if not self.visited_shop:
                self.visited_shop = True
                return ChooseShopkeeperAction()
            else:
                self.visited_shop = False
                return ProceedAction()
        elif self.game.screen_type == ScreenType.REST:
            return self.choose_rest_option()
        elif self.game.screen_type == ScreenType.CARD_REWARD:
            return self.choose_card_reward()
        elif self.game.screen_type == ScreenType.COMBAT_REWARD:
            for reward_item in self.game.screen.rewards:
                if reward_item.reward_type == RewardType.POTION and self.game.are_potions_full():
                    continue
                elif reward_item.reward_type == RewardType.CARD and self.skipped_cards:
                    continue
                else:
                    return CombatRewardAction(reward_item)
            self.skipped_cards = False
            return ProceedAction()
        elif self.game.screen_type == ScreenType.MAP:
            return self.make_map_choice()
        elif self.game.screen_type == ScreenType.BOSS_REWARD:
            relics = self.game.screen.relics
            best_boss_relic = self.priorities.get_best_boss_relic(relics)
            return BossRewardAction(best_boss_relic)
        elif self.game.screen_type == ScreenType.SHOP_SCREEN:
            if self.game.screen.purge_available and self.game.gold >= self.game.screen.purge_cost:
                return ChooseAction(name="purge")
            for card in self.game.screen.cards:
                if self.game.gold >= card.price and not self.priorities.should_skip(card):
                    return BuyCardAction(card)
            for relic in self.game.screen.relics:
                if self.game.gold >= relic.price:
                    return BuyRelicAction(relic)
            return CancelAction()
        elif self.game.screen_type == ScreenType.GRID:
            if not self.game.choice_available:
                return ProceedAction()
            if self.game.screen.for_upgrade or self.choose_good_card:
                available_cards = self.priorities.get_sorted_cards(self.game.screen.cards)
            else:
                available_cards = self.priorities.get_sorted_cards(self.game.screen.cards, reverse=True)
            num_cards = self.game.screen.num_cards
            return CardSelectAction(available_cards[:num_cards])
        elif self.game.screen_type == ScreenType.HAND_SELECT:
            if not self.game.choice_available:
                return ProceedAction()
            # Usually, we don't want to choose the whole hand for a hand select. 3 seems like a good compromise.
            num_cards = min(self.game.screen.num_cards, 3)
            return CardSelectAction(self.priorities.get_cards_for_action(self.game.current_action, self.game.screen.cards, num_cards))
        else:
            return ProceedAction()

    def choose_rest_option(self):
        rest_options = self.game.screen.rest_options
        if len(rest_options) > 0 and not self.game.screen.has_rested:
            if RestOption.REST in rest_options and self.game.current_hp < self.game.max_hp / 2:
                return RestAction(RestOption.REST)
            elif RestOption.REST in rest_options and self.game.act != 1 and self.game.floor % 17 == 15 and self.game.current_hp < self.game.max_hp * 0.9:
                return RestAction(RestOption.REST)
            elif RestOption.SMITH in rest_options:
                return RestAction(RestOption.SMITH)
            elif RestOption.LIFT in rest_options:
                return RestAction(RestOption.LIFT)
            elif RestOption.DIG in rest_options:
                return RestAction(RestOption.DIG)
            elif RestOption.REST in rest_options and self.game.current_hp < self.game.max_hp:
                return RestAction(RestOption.REST)
            else:
                return ChooseAction(0)
        else:
            return ProceedAction()

    def count_copies_in_deck(self, card):
        count = 0
        for deck_card in self.game.deck:
            if deck_card.card_id == card.card_id:
                count += 1
        return count

    def choose_card_reward(self):
        reward_cards = self.game.screen.cards
        if self.game.screen.can_skip and not self.game.in_combat:
            pickable_cards = [card for card in reward_cards if self.priorities.needs_more_copies(card, self.count_copies_in_deck(card))]
        else:
            pickable_cards = reward_cards
        if len(pickable_cards) > 0:
            potential_pick = self.priorities.get_best_card(pickable_cards)
            return CardRewardAction(potential_pick)
        elif self.game.screen.can_bowl:
            return CardRewardAction(bowl=True)
        else:
            self.skipped_cards = True
            return CancelAction()

    def generate_map_route(self):
        node_rewards = self.priorities.MAP_NODE_PRIORITIES.get(self.game.act)
        best_rewards = {0: {node.x: node_rewards[node.symbol] for node in self.game.map.nodes[0].values()}}
        best_parents = {0: {node.x: 0 for node in self.game.map.nodes[0].values()}}
        min_reward = min(node_rewards.values())
        map_height = max(self.game.map.nodes.keys())
        for y in range(0, map_height):
            best_rewards[y+1] = {node.x: min_reward * 20 for node in self.game.map.nodes[y+1].values()}
            best_parents[y+1] = {node.x: -1 for node in self.game.map.nodes[y+1].values()}
            for x in best_rewards[y]:
                node = self.game.map.get_node(x, y)
                best_node_reward = best_rewards[y][x]
                for child in node.children:
                    test_child_reward = best_node_reward + node_rewards[child.symbol]
                    if test_child_reward > best_rewards[y+1][child.x]:
                        best_rewards[y+1][child.x] = test_child_reward
                        best_parents[y+1][child.x] = node.x
        best_path = [0] * (map_height + 1)
        best_path[map_height] = max(best_rewards[map_height].keys(), key=lambda x: best_rewards[map_height][x])
        for y in range(map_height, 0, -1):
            best_path[y - 1] = best_parents[y][best_path[y]]
        self.map_route = best_path

    def make_map_choice(self):
        if len(self.game.screen.next_nodes) > 0 and self.game.screen.next_nodes[0].y == 0:
            self.generate_map_route()
            self.game.screen.current_node.y = -1
        if self.game.screen.boss_available:
            return ChooseMapBossAction()
        chosen_x = self.map_route[self.game.screen.current_node.y + 1]
        for choice in self.game.screen.next_nodes:
            if choice.x == chosen_x:
                return ChooseMapNodeAction(choice)
        # This should never happen
        return ChooseAction(0)


def run_agent_cli():
    """
    Main CLI entry point for running the AI agent.
    """
    parser = argparse.ArgumentParser(description="STS Lightspeed AI Agent for CommunicationMod")
    parser.add_argument("--character", "-c", 
                       choices=["ironclad", "silent", "defect"],
                       default="ironclad",
                       help="Character class to play")
    parser.add_argument("--games", "-g", type=int, default=1,
                       help="Number of games to play (0 for infinite)")
    parser.add_argument("--test", action="store_true",
                       help="Run conversion tests instead of playing")
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing spirecomm to GameContext converter...")
        success = test_basic_conversion()
        sys.exit(0 if success else 1)
    
    # Map character name to enum
    class_mapping = {
        "ironclad": PlayerClass.IRONCLAD,
        "silent": PlayerClass.THE_SILENT, 
        "defect": PlayerClass.DEFECT,
    }
    chosen_class = class_mapping[args.character]
    
    print(f"Starting STS Lightspeed Agent for {args.character.title()}", file=sys.stderr)
    
    # Create agent and coordinator
    agent = STSLightspeedAgent(chosen_class)
    coordinator = Coordinator()
    
    # Register callbacks
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    
    # Play games
    games_played = 0
    character_classes = [chosen_class] if args.games == 1 else itertools.cycle(PlayerClass)
    
    for current_class in character_classes:
        if args.games > 0 and games_played >= args.games:
            break
            
        agent.change_class(current_class)
        print(f"Playing game {games_played + 1} as {current_class.name}", file=sys.stderr)
        
        try:
            result = coordinator.play_one_game(current_class)
            games_played += 1
            print(f"Game {games_played} completed with result: {result}", file=sys.stderr)
        except KeyboardInterrupt:
            print("Interrupted by user", file=sys.stderr)
            break
        except Exception as e:
            print(f"Game error: {e}", file=sys.stderr)
            break


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run test
        print("Testing spirecomm to GameContext converter...")
        test_basic_conversion()
    else:
        # Arguments provided - run CLI
        run_agent_cli()