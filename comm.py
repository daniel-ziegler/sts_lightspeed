"""
Spirecomm to GameContext Converter

This module provides functions to convert spirecomm game state representations
into our internal C++ GameContext format for AI control of the real game.
"""

import sys
import json
import argparse
import itertools
from typing import Optional, Union

import slaythespire as sts
from spirecomm.spire import game, card, character, relic, power, potion, screen
from spirecomm.spire.screen import RewardType, ScreenType
from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action, BossRewardAction, BuyCardAction, BuyRelicAction, CardSelectAction, ChooseShopkeeperAction, CombatRewardAction, OpenChestAction, ProceedAction, PlayCardAction, EndTurnAction, ChooseAction, RestAction, NoopAction, CancelAction
from spirecomm.spire.character import PlayerClass



CHARACTER_CLASS_MAPPING = {
    character.PlayerClass.IRONCLAD: sts.CharacterClass.IRONCLAD,
    character.PlayerClass.THE_SILENT: sts.CharacterClass.SILENT,
    character.PlayerClass.DEFECT: sts.CharacterClass.DEFECT,
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

def map_character_class(spire_class: character.PlayerClass) -> sts.CharacterClass:
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
    
    # Set misc field if available
    if hasattr(spire_card, 'misc') and spire_card.misc != 0:
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
            if hasattr(monster, 'block') and monster.block:
                sts_monster.block = monster.block
            if hasattr(monster, 'half_dead'):
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
    
    return gc


def spirecomm_to_battlecontext(spire_game: game.Game) -> tuple[sts.GameContext, sts.BattleContext]:
    """
    Convert spirecomm Game state to GameContext and BattleContext for combat.
    
    Args:
        spire_game: Game state from spirecomm (must be in combat)
        
    Returns:
        Tuple of (GameContext, BattleContext) ready for battle simulation
        
    Raises:
        ValueError: If not in combat or invalid state
    """
    if not spire_game.in_combat:
        raise ValueError("Game must be in combat to create BattleContext")
    
    # Create base GameContext
    gc = spirecomm_to_gamecontext(spire_game)
    
    # Convert combat state to BattleContext
    bc = convert_combat_state(spire_game, gc)
    
    return gc, bc


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
        if hasattr(game_action, 'rewards_action_type'):
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
    """
    AI Agent that uses our high-performance C++ STS implementation
    to make decisions for the real game via spirecomm.
    """
    
    def __init__(self, chosen_class: PlayerClass = PlayerClass.IRONCLAD):
        self.game_state = None
        self.chosen_class = chosen_class
        self.errors = 0
        self.visited_shop = False
        self.skipped_cards = False
        
    def change_class(self, new_class: PlayerClass):
        """Change the character class for the next game."""
        self.chosen_class = new_class
    
    def handle_error(self, error: str):
        """Handle errors from CommunicationMod."""
        self.errors += 1
        print(f"Error {self.errors}: {error}", file=sys.stderr)
        if self.errors > 10:
            raise Exception(f"Too many errors: {error}")
    
    def get_next_action_in_game(self, game_state: game.Game) -> Action:
        """
        Main decision function - convert game state to our format,
        run AI decision making, and return spirecomm action.
        """
        try:
            self.game_state = game_state
            
            if game_state.choice_available:
                return self.handle_choice_screen(game_state)
            if game_state.proceed_available:
                return ProceedAction()
            if game_state.play_available:
                return self.handle_combat(game_state)
            if game_state.end_available:
                return EndTurnAction()
            if game_state.cancel_available:
                return CancelAction()
            print(f"Game state: play={getattr(game_state, 'play_available', False)}, "
                    f"choice={getattr(game_state, 'choice_available', False)}, "
                    f"proceed={getattr(game_state, 'proceed_available', False)}, "
                    f"in_combat={getattr(game_state, 'in_combat', False)}", file=sys.stderr)
            
            return NoopAction()
                
        except Exception as e:
            print(f"Error in decision making: {e}", file=sys.stderr)
            return NoopAction()
    
    def handle_combat(self, game_state: game.Game) -> Action:
        """
        Handle combat decisions using our C++ BattleContext.
        """
        try:
            # Convert to our battle context
            gc, bc = spirecomm_to_battlecontext(game_state)

            print(f"Monsters: {bc.monsters}", file=sys.stderr)
            
            # Simple combat logic for now - play first playable card or end turn
            if bc.cards.cardsInHand > 0:
                hand = bc.cards.hand
                print(f"hand={hand}", file=sys.stderr)
                for i, card_instance in enumerate(hand):
                    # Check if we can afford the card
                    if card_instance.canUseOnAnyTarget(bc) and card_instance.cost <= bc.player.energy:
                        print(f"Playing, requires target={card_instance.requiresTarget()}, targetableCount={bc.monsters.getTargetableCount()}", file=sys.stderr)
                        # Try to play this card
                        if card_instance.requiresTarget():
                            # Target first targetable monster
                            target = bc.monsters.getFirstTargetable()
                            print(f"target={target}", file=sys.stderr)
                            if target >= 0 and card_instance.canUse(bc, target, False):
                                return PlayCardAction(card_index=i, target_index=target)
                        else:
                            return PlayCardAction(card_index=i)
            
            # If we can't play any cards, end turn
            return EndTurnAction()
            
        except Exception as e:
            print(f"Combat error: {e}", file=sys.stderr)
            # Fallback to end turn
            return EndTurnAction()
    
    def handle_choice_screen(self, game_state: game.Game) -> Action:
        """
        Handle non-combat choice screens with basic logic.
        """
        if game_state.screen_type == ScreenType.EVENT:
            if game_state.screen.event_id in ["Vampires", "Masked Bandits", "Knowing Skull", "Ghosts", "Liars Game", "Golden Idol", "Drug Dealer", "The Library"]:
                return ChooseAction(len(game_state.screen.options) - 1)
            else:
                return ChooseAction(0)
        elif game_state.screen_type == ScreenType.CHEST:
            return OpenChestAction()
        elif game_state.screen_type == ScreenType.SHOP_ROOM:
            if not self.visited_shop:
                self.visited_shop = True
                return ChooseShopkeeperAction()
            else:
                self.visited_shop = False
                return ProceedAction()
        elif game_state.screen_type == ScreenType.REST:
            return ChooseAction(1)
        elif game_state.screen_type == ScreenType.CARD_REWARD:
            return ChooseAction(0)
        elif game_state.screen_type == ScreenType.COMBAT_REWARD:
            for reward_item in game_state.screen.rewards:
                if reward_item.reward_type == RewardType.POTION and game_state.are_potions_full():
                    continue
                elif reward_item.reward_type == RewardType.CARD and self.skipped_cards:
                    continue
                else:
                    return CombatRewardAction(reward_item)
            self.skipped_cards = False
            return ProceedAction()
        elif game_state.screen_type == ScreenType.MAP:
            return ChooseAction(0)
        elif game_state.screen_type == ScreenType.BOSS_REWARD:
            relics = game_state.screen.relics
            return BossRewardAction(relics[0])
        elif game_state.screen_type == ScreenType.SHOP_SCREEN:
            if game_state.screen.purge_available and game_state.gold >= game_state.screen.purge_cost:
                return ChooseAction(name="purge")
            for card in game_state.screen.cards:
                if game_state.gold >= card.price:
                    return BuyCardAction(card)
            for relic in game_state.screen.relics:
                if game_state.gold >= relic.price:
                    return BuyRelicAction(relic)
            return CancelAction()
        elif game_state.screen_type == ScreenType.GRID:
            if not game_state.choice_available:
                return ProceedAction()
            num_cards = game_state.screen.num_cards
            return CardSelectAction(game_state.screen.cards[:num_cards])
        elif game_state.screen_type == ScreenType.HAND_SELECT:
            if not game_state.choice_available:
                return ProceedAction()
            return CardSelectAction(game_state.screen.cards[:3])
        else:
            return ProceedAction()

           
    def get_next_action_out_of_game(self) -> Action:
        """Handle out-of-game actions (main menu, etc.)."""
        from spirecomm.communication.action import StartGameAction
        return StartGameAction(self.chosen_class)


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