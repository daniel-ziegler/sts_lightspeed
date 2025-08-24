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
from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication.action import Action, ProceedAction, PlayCardAction, EndTurnAction, ChooseAction, RestAction
from spirecomm.spire.character import PlayerClass


# Mapping dictionaries for spirecomm string IDs to our enum values
CARD_ID_MAPPING = {
    # Basic cards - Ironclad
    "Strike_R": sts.CardId.STRIKE_RED,
    "Defend_R": sts.CardId.DEFEND_RED,
    "Bash": sts.CardId.BASH,
    
    # Common Ironclad cards
    "Anger": sts.CardId.ANGER,
    "Armaments": sts.CardId.ARMAMENTS,
    "Body Slam": sts.CardId.BODY_SLAM,
    "Clash": sts.CardId.CLASH,
    "Cleave": sts.CardId.CLEAVE,
    "Clothesline": sts.CardId.CLOTHESLINE,
    "Flex": sts.CardId.FLEX,
    "Havoc": sts.CardId.HAVOC,
    "Headbutt": sts.CardId.HEADBUTT,
    "Heavy Blade": sts.CardId.HEAVY_BLADE,
    "Iron Wave": sts.CardId.IRON_WAVE,
    "Perfected Strike": sts.CardId.PERFECTED_STRIKE,
    "Pommel Strike": sts.CardId.POMMEL_STRIKE,
    "Shrug It Off": sts.CardId.SHRUG_IT_OFF,
    "Sword Boomerang": sts.CardId.SWORD_BOOMERANG,
    "Thunderclap": sts.CardId.THUNDERCLAP,
    "True Grit": sts.CardId.TRUE_GRIT,
    "Twin Strike": sts.CardId.TWIN_STRIKE,
    "Warcry": sts.CardId.WARCRY,
    "Wild Strike": sts.CardId.WILD_STRIKE,
    
    # Uncommon Ironclad cards
    "Battle Trance": sts.CardId.BATTLE_TRANCE,
    "Bloodletting": sts.CardId.BLOODLETTING,
    "Blood for Blood": sts.CardId.BLOOD_FOR_BLOOD,
    "Burning Pact": sts.CardId.BURNING_PACT,
    "Carnage": sts.CardId.CARNAGE,
    "Combust": sts.CardId.COMBUST,
    "Dark Embrace": sts.CardId.DARK_EMBRACE,
    "Disarm": sts.CardId.DISARM,
    "Dropkick": sts.CardId.DROPKICK,
    "Dual Wield": sts.CardId.DUAL_WIELD,
    "Entrench": sts.CardId.ENTRENCH,
    "Evolve": sts.CardId.EVOLVE,
    "Feel No Pain": sts.CardId.FEEL_NO_PAIN,
    "Fire Breathing": sts.CardId.FIRE_BREATHING,
    "Flame Barrier": sts.CardId.FLAME_BARRIER,
    "Ghostly Armor": sts.CardId.GHOSTLY_ARMOR,
    "Hemokinesis": sts.CardId.HEMOKINESIS,
    "Inflame": sts.CardId.INFLAME,
    "Intimidate": sts.CardId.INTIMIDATE,
    "Metallicize": sts.CardId.METALLICIZE,
    "Power Through": sts.CardId.POWER_THROUGH,
    "Pummel": sts.CardId.PUMMEL,
    "Rage": sts.CardId.RAGE,
    "Rampage": sts.CardId.RAMPAGE,
    "Reckless Charge": sts.CardId.RECKLESS_CHARGE,
    "Rupture": sts.CardId.RUPTURE,
    "Searing Blow": sts.CardId.SEARING_BLOW,
    "Second Wind": sts.CardId.SECOND_WIND,
    "Seeing Red": sts.CardId.SEEING_RED,
    "Sentinel": sts.CardId.SENTINEL,
    "Shockwave": sts.CardId.SHOCKWAVE,
    "Spot Weakness": sts.CardId.SPOT_WEAKNESS,
    "Uppercut": sts.CardId.UPPERCUT,
    "Whirlwind": sts.CardId.WHIRLWIND,
    
    # Rare Ironclad cards
    "Barricade": sts.CardId.BARRICADE,
    "Berserk": sts.CardId.BERSERK,
    "Bludgeon": sts.CardId.BLUDGEON,
    "Brutality": sts.CardId.BRUTALITY,
    "Corruption": sts.CardId.CORRUPTION,
    "Demon Form": sts.CardId.DEMON_FORM,
    "Double Tap": sts.CardId.DOUBLE_TAP,
    "Exhume": sts.CardId.EXHUME,
    "Feed": sts.CardId.FEED,
    "Fiend Fire": sts.CardId.FIEND_FIRE,
    "Immolate": sts.CardId.IMMOLATE,
    "Impervious": sts.CardId.IMPERVIOUS,
    "Juggernaut": sts.CardId.JUGGERNAUT,
    "Limit Break": sts.CardId.LIMIT_BREAK,
    "Offering": sts.CardId.OFFERING,
    "Reaper": sts.CardId.REAPER,
    
    # Basic Silent cards  
    "Strike_G": sts.CardId.STRIKE_GREEN,
    "Defend_G": sts.CardId.DEFEND_GREEN,
    "Neutralize": sts.CardId.NEUTRALIZE,
    "Survivor": sts.CardId.SURVIVOR,
    
    # Basic Defect cards
    "Strike_B": sts.CardId.STRIKE_BLUE,
    "Defend_B": sts.CardId.DEFEND_BLUE,
    "Zap": sts.CardId.ZAP,
    "Dualcast": sts.CardId.DUALCAST,
    
    # Basic Watcher cards  
    "Strike_P": sts.CardId.STRIKE_PURPLE,
    "Defend_P": sts.CardId.DEFEND_PURPLE,
    "Eruption": sts.CardId.ERUPTION,
    "Vigilance": sts.CardId.VIGILANCE,
    
    # Status/Curse cards
    "Burn": sts.CardId.BURN,
    "Wound": sts.CardId.WOUND,
    "Dazed": sts.CardId.DAZED,
    "Slimed": sts.CardId.SLIMED,
    "Void": sts.CardId.VOID,
    "Ascender's Bane": sts.CardId.ASCENDERS_BANE,
    "Clumsy": sts.CardId.CLUMSY,
    "Curse of the Bell": sts.CardId.CURSE_OF_THE_BELL,
    "Decay": sts.CardId.DECAY,
    "Doubt": sts.CardId.DOUBT,
    "Injury": sts.CardId.INJURY,
    "Necronomicurse": sts.CardId.NECRONOMICURSE,
    "Normality": sts.CardId.NORMALITY,
    "Pain": sts.CardId.PAIN,
    "Parasite": sts.CardId.PARASITE,
    "Pride": sts.CardId.PRIDE,
    "Regret": sts.CardId.REGRET,
    "Shame": sts.CardId.SHAME,
    "Writhe": sts.CardId.WRITHE,
}

RELIC_ID_MAPPING = {
    # Starter relics
    "Burning Blood": sts.RelicId.BURNING_BLOOD,
    "Ring of the Snake": sts.RelicId.RING_OF_THE_SNAKE,
    "Cracked Core": sts.RelicId.CRACKED_CORE,
    "Pure Water": sts.RelicId.PURE_WATER,
    
    # Special relics
    "Neow's Lament": sts.RelicId.NEOWS_LAMENT,
    
    # Common relics
    "Akabeko": sts.RelicId.AKABEKO,
    "Anchor": sts.RelicId.ANCHOR,
    "Art of War": sts.RelicId.ART_OF_WAR,
    "Bag of Marbles": sts.RelicId.BAG_OF_MARBLES,
    "Bag of Preparation": sts.RelicId.BAG_OF_PREPARATION,
    "Blood Vial": sts.RelicId.BLOOD_VIAL,
    "Bronze Scales": sts.RelicId.BRONZE_SCALES,
    "Centennial Puzzle": sts.RelicId.CENTENNIAL_PUZZLE,
    "Ceramic Fish": sts.RelicId.CERAMIC_FISH,
    "Dream Catcher": sts.RelicId.DREAM_CATCHER,
    "Happy Flower": sts.RelicId.HAPPY_FLOWER,
    "Juzu Bracelet": sts.RelicId.JUZU_BRACELET,
    "Lantern": sts.RelicId.LANTERN,
    "Meal Ticket": sts.RelicId.MEAL_TICKET,
    "Nunchaku": sts.RelicId.NUNCHAKU,
    "Oddly Smooth Stone": sts.RelicId.ODDLY_SMOOTH_STONE,
    "Omamori": sts.RelicId.OMAMORI,
    "Orichalcum": sts.RelicId.ORICHALCUM,
    "Pen Nib": sts.RelicId.PEN_NIB,
    "Preserved Insect": sts.RelicId.PRESERVED_INSECT,
    "Red Skull": sts.RelicId.RED_SKULL,
    "Regal Pillow": sts.RelicId.REGAL_PILLOW,
    "Smiling Mask": sts.RelicId.SMILING_MASK,
    "Strawberry": sts.RelicId.STRAWBERRY,
    "The Boot": sts.RelicId.THE_BOOT,
    "Tiny Chest": sts.RelicId.TINY_CHEST,
    "Toy Ornithopter": sts.RelicId.TOY_ORNITHOPTER,
    "Vajra": sts.RelicId.VAJRA,
    "War Paint": sts.RelicId.WAR_PAINT,
    "Whetstone": sts.RelicId.WHETSTONE,
    
    # Uncommon relics
    "Blue Candle": sts.RelicId.BLUE_CANDLE,
    "Bottled Flame": sts.RelicId.BOTTLED_FLAME,
    "Bottled Lightning": sts.RelicId.BOTTLED_LIGHTNING,
    "Bottled Tornado": sts.RelicId.BOTTLED_TORNADO,
    "Darkstone Periapt": sts.RelicId.DARKSTONE_PERIAPT,
    "Eternal Feather": sts.RelicId.ETERNAL_FEATHER,
    "Frozen Egg": sts.RelicId.FROZEN_EGG,
    "Ginger": sts.RelicId.GINGER,
    "Gremlin Horn": sts.RelicId.GREMLIN_HORN,
    "Horn Cleat": sts.RelicId.HORN_CLEAT,
    "InkBottle": sts.RelicId.INK_BOTTLE,
    "Kunai": sts.RelicId.KUNAI,
    "Letter Opener": sts.RelicId.LETTER_OPENER,
    "Matryoshka": sts.RelicId.MATRYOSHKA,
    "Meat on the Bone": sts.RelicId.MEAT_ON_THE_BONE,
    "Mercury Hourglass": sts.RelicId.MERCURY_HOURGLASS,
    "Molten Egg": sts.RelicId.MOLTEN_EGG,
    "Mummified Hand": sts.RelicId.MUMMIFIED_HAND,
    "Ornamental Fan": sts.RelicId.ORNAMENTAL_FAN,
    "Pantograph": sts.RelicId.PANTOGRAPH,
    "Paper Krane": sts.RelicId.PAPER_KRANE,
    "Prayer Wheel": sts.RelicId.PRAYER_WHEEL,
    "Question Card": sts.RelicId.QUESTION_CARD,
    "Shuriken": sts.RelicId.SHURIKEN,
    "Singing Bowl": sts.RelicId.SINGING_BOWL,
    "Strike Dummy": sts.RelicId.STRIKE_DUMMY,
    "Sundial": sts.RelicId.SUNDIAL,
    "Teardrop Locket": sts.RelicId.TEARDROP_LOCKET,
    "The Courier": sts.RelicId.THE_COURIER,
    "Thread and Needle": sts.RelicId.THREAD_AND_NEEDLE,
    "Tingsha": sts.RelicId.TINGSHA,
    "Toolbox": sts.RelicId.TOOLBOX,
    "Toxic Egg": sts.RelicId.TOXIC_EGG,
    "Turnip": sts.RelicId.TURNIP,
    "Unceasing Top": sts.RelicId.UNCEASING_TOP,
    "White Beast Statue": sts.RelicId.WHITE_BEAST_STATUE,
    
    # Boss relics
    "Astrolabe": sts.RelicId.ASTROLABE,
    "Black Star": sts.RelicId.BLACK_STAR,
    "Busted Crown": sts.RelicId.BUSTED_CROWN,
    "Calling Bell": sts.RelicId.CALLING_BELL,
    "Coffee Dripper": sts.RelicId.COFFEE_DRIPPER,
    "Cursed Key": sts.RelicId.CURSED_KEY,
    "Ectoplasm": sts.RelicId.ECTOPLASM,
    "Empty Cage": sts.RelicId.EMPTY_CAGE,
    "Fusion Hammer": sts.RelicId.FUSION_HAMMER,
    "Pandora's Box": sts.RelicId.PANDORAS_BOX,
    "Philosopher's Stone": sts.RelicId.PHILOSOPHERS_STONE,
    "Runic Dome": sts.RelicId.RUNIC_DOME,
    "Runic Pyramid": sts.RelicId.RUNIC_PYRAMID,
    "Slavers Collar": sts.RelicId.SLAVERS_COLLAR,
    "Snecko Eye": sts.RelicId.SNECKO_EYE,
    "Sozu": sts.RelicId.SOZU,
    "Velvet Choker": sts.RelicId.VELVET_CHOKER,
}

CHARACTER_CLASS_MAPPING = {
    character.PlayerClass.IRONCLAD: sts.CharacterClass.IRONCLAD,
    character.PlayerClass.THE_SILENT: sts.CharacterClass.SILENT,
    character.PlayerClass.DEFECT: sts.CharacterClass.DEFECT,
}

SCREEN_STATE_MAPPING = {
    # Map spirecomm ScreenType to our ScreenState
    screen.ScreenType.EVENT: sts.ScreenState.EVENT_SCREEN,
    screen.ScreenType.CHEST: sts.ScreenState.TREASURE_ROOM,
    screen.ScreenType.SHOP_ROOM: sts.ScreenState.SHOP_ROOM,
    screen.ScreenType.SHOP_SCREEN: sts.ScreenState.SHOP_ROOM,
    screen.ScreenType.REST: sts.ScreenState.REST_ROOM,
    screen.ScreenType.CARD_REWARD: sts.ScreenState.REWARDS,
    screen.ScreenType.COMBAT_REWARD: sts.ScreenState.REWARDS,
    screen.ScreenType.MAP: sts.ScreenState.MAP_SCREEN,
    screen.ScreenType.BOSS_REWARD: sts.ScreenState.BOSS_RELIC_REWARDS,
    screen.ScreenType.GRID: sts.ScreenState.CARD_SELECT,
    screen.ScreenType.HAND_SELECT: sts.ScreenState.CARD_SELECT,
    screen.ScreenType.GAME_OVER: sts.ScreenState.INVALID,  # Game over, no meaningful screen
    screen.ScreenType.COMPLETE: sts.ScreenState.INVALID,   # Game complete
    screen.ScreenType.NONE: sts.ScreenState.MAP_SCREEN,    # Default fallback
}

# Power ID mapping - spirecomm power names to our PlayerStatus enum
POWER_ID_MAPPING = {
    # Status effects / debuffs
    "Frail": "FRAIL",
    "Vulnerable": "VULNERABLE", 
    "Weakened": "WEAK",
    "Intangible": "INTANGIBLE",
    "Constricted": "CONSTRICTED",
    "Entangled": "ENTANGLED",
    
    # Powers - bool types
    "Barricade": "BARRICADE",
    "Corruption": "CORRUPTION",
    "Wraith Form": "WRAITH_FORM",
    
    # Powers - counter types
    "Amplify": "AMPLIFY",
    "Blur": "BLUR", 
    "Buffer": "BUFFER",
    "Double Tap": "DOUBLE_TAP",
    "Echo Form": "ECHO_FORM",
    "Mantra": "MANTRA",
    
    # Powers - intensity types
    "Accuracy": "ACCURACY",
    "After Image": "AFTER_IMAGE",
    "Battle Hymn": "BATTLE_HYMN",
    "Brutality": "BRUTALITY",
    "Burst": "BURST",
    "Combust": "COMBUST",
    "Creative AI": "CREATIVE_AI",
    "Dark Embrace": "DARK_EMBRACE",
    "Demon Form": "DEMON_FORM",
    "Deva Form": "DEVA",
    "Devotion": "DEVOTION",
    "Energized": "ENERGIZED",
    "Envenom": "ENVENOM",
    "Establishment": "ESTABLISHMENT",
    "Evolve": "EVOLVE",
    "Feel No Pain": "FEEL_NO_PAIN",
    "Fire Breathing": "FIRE_BREATHING",
    "Flame Barrier": "FLAME_BARRIER",
    "Focus": "FOCUS",
    "Hello World": "HELLO_WORLD",
    "Infinite Blades": "INFINITE_BLADES",
    "Juggernaut": "JUGGERNAUT",
    "Like Water": "LIKE_WATER",
    "Loop": "LOOP",
    "Magnetism": "MAGNETISM",
    "Mayhem": "MAYHEM", 
    "Metallicize": "METALLICIZE",
    "Noxious Fumes": "NOXIOUS_FUMES",
    "Omega": "OMEGA",
    "Panache": "PANACHE",
    "Plated Armor": "PLATED_ARMOR",
    "Rage": "RAGE",
    "Regeneration": "REGEN",
    "Ritual": "RITUAL",
    "Rupture": "RUPTURE",
    "Static Discharge": "STATIC_DISCHARGE",
    "Thorns": "THORNS",
    "A Thousand Cuts": "THOUSAND_CUTS",
    "Tools of the Trade": "TOOLS_OF_THE_TRADE",
    "Vigor": "VIGOR",
    
    # Duration types
    "Artifact": "ARTIFACT",
    "Dexterity": "DEXTERITY",
    "Strength": "STRENGTH",
    
    # Special
    "Bomb": "THE_BOMB",
}


def map_card_id(spire_card_id: str) -> sts.CardId:
    """Map spirecomm card ID to our CardId enum."""
    return CARD_ID_MAPPING.get(spire_card_id, sts.CardId.INVALID)


def map_relic_id(spire_relic_id: str) -> sts.RelicId:
    """Map spirecomm relic ID to our RelicId enum."""  
    return RELIC_ID_MAPPING.get(spire_relic_id, sts.RelicId.INVALID)


def map_character_class(spire_class: character.PlayerClass) -> sts.CharacterClass:
    """Map spirecomm PlayerClass to our CharacterClass."""
    return CHARACTER_CLASS_MAPPING.get(spire_class, sts.CharacterClass.INVALID)


def map_power_id(spire_power_name: str) -> str:
    """Map spirecomm power name to our PlayerStatus enum string."""
    return POWER_ID_MAPPING.get(spire_power_name, "INVALID")


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


def detect_encounter_from_monsters(monsters) -> sts.MonsterEncounter:
    """
    Detect the most appropriate MonsterEncounter based on spirecomm monster data.
    
    Since we can't know the exact encounter from just monster states, we'll use
    a reasonable default based on monster count and types.
    """
    if not monsters or len(monsters) == 0:
        return sts.MonsterEncounter.INVALID
    
    # For simplicity, we'll use basic encounters based on monster count
    # In a more sophisticated implementation, we could analyze monster names/IDs
    monster_count = len([m for m in monsters if hasattr(m, 'current_hp') and m.current_hp > 0])
    
    if monster_count == 1:
        # Single monster - use a common single monster encounter
        return sts.MonsterEncounter.CULTIST
    elif monster_count == 2:
        # Two monsters - use a common two monster encounter
        return sts.MonsterEncounter.TWO_LOUSE
    elif monster_count == 3:
        # Three monsters - use a common three monster encounter  
        return sts.MonsterEncounter.THREE_LOUSE
    else:
        # More than 3 or unknown - use a default
        return sts.MonsterEncounter.GREMLIN_GANG


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
    
    # Set up a valid encounter before creating BattleContext
    # Map spirecomm monsters to a reasonable encounter
    encounter = detect_encounter_from_monsters(spire_game.monsters)
    gc.screen_state_info.encounter = encounter
    
    # Create battle context from GameContext
    bc = gc.create_battle_context()
    
    # Player state conversion
    player = spire_game.player
    if player:
        # Set basic player stats
        if hasattr(player, 'energy'):
            bc.player.energy = player.energy
        if hasattr(player, 'block'):
            bc.player.block = player.block
            
        # Convert player powers/buffs/debuffs
        if hasattr(player, 'powers') and player.powers:
            for power in player.powers:
                power_status_name = map_power_id(power.power_name)
                if power_status_name and power_status_name != "INVALID":
                    # Convert string to enum
                    power_status = getattr(sts.PlayerStatus, power_status_name)
                    # Use buff for positive effects, debuff for negative
                    if is_positive_player_power(power.power_name):
                        bc.player.buff(power_status, power.amount)
                    else:
                        bc.player.debuff(power_status, power.amount, False)
    
    # Card piles conversion - create CardInstance objects from spirecomm cards
    if spire_game.hand:
        # Clear current hand and populate with spirecomm cards
        for spire_card in spire_game.hand:
            try:
                card_instance = convert_spire_card_to_instance(spire_card)
                bc.cards.moveToHand(card_instance)
            except (ValueError, KeyError):
                # Skip unknown cards
                continue
    
    if spire_game.draw_pile:
        for spire_card in spire_game.draw_pile:
            try:
                card_instance = convert_spire_card_to_instance(spire_card)
                bc.cards.moveToDrawPileTop(card_instance)
            except (ValueError, KeyError):
                continue
        
    if spire_game.discard_pile:
        for spire_card in spire_game.discard_pile:
            try:
                card_instance = convert_spire_card_to_instance(spire_card)
                bc.cards.moveToDiscardPile(card_instance)
            except (ValueError, KeyError):
                continue
        
    if spire_game.exhaust_pile:
        for spire_card in spire_game.exhaust_pile:
            try:
                card_instance = convert_spire_card_to_instance(spire_card)
                bc.cards.moveToExhaustPile(card_instance)
            except (ValueError, KeyError):
                continue
    
    # Monster states conversion
    if spire_game.monsters and len(spire_game.monsters) > 0:
        for i, monster in enumerate(spire_game.monsters):
            if i >= len(bc.monsters):
                break  # Can't add more monsters than the MonsterGroup supports
                
            sts_monster = bc.monsters[i]
            
            # Basic monster stats
            sts_monster.curHp = monster.current_hp
            sts_monster.maxHp = monster.max_hp
            if hasattr(monster, 'block'):
                sts_monster.block = monster.block
            if hasattr(monster, 'half_dead'):
                sts_monster.halfDead = monster.half_dead
            
            # Convert monster powers
            if hasattr(monster, 'powers') and monster.powers:
                for power in monster.powers:
                    try:
                        monster_status = map_monster_power_id(power.power_name)
                        if monster_status:
                            if is_positive_monster_power(power.power_name):
                                sts_monster.buff(monster_status, power.amount)
                            else:
                                sts_monster.addDebuff(monster_status, power.amount, False)
                    except (ValueError, KeyError):
                        # Skip unknown monster powers
                        continue
    
    return bc


def convert_spire_card_to_instance(spire_card: card.Card) -> sts.CardInstance:
    """Convert a spirecomm Card to a CardInstance."""
    # Map the card ID
    card_id = CARD_ID_MAPPING.get(spire_card.name)
    if not card_id:
        raise ValueError(f"Unknown card: {spire_card.name}")
    
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
    
    # Check screen_type if available
    if hasattr(spire_game, 'screen_type') and spire_game.screen_type:
        screen_type = spire_game.screen_type
        return SCREEN_STATE_MAPPING.get(screen_type, sts.ScreenState.MAP_SCREEN)
    
    # Default fallback
    return sts.ScreenState.MAP_SCREEN


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
    if hasattr(spire_game, 'deck') and spire_game.deck:
        if not isinstance(spire_game.deck, list):
            raise TypeError(f"Deck must be a list, got {type(spire_game.deck)}")
        
        for i, card_obj in enumerate(spire_game.deck):
            if not isinstance(card_obj, card.Card):
                raise TypeError(f"Deck card {i} is not a Card object: {type(card_obj)}")
    
    # Validate relics if present
    if hasattr(spire_game, 'relics') and spire_game.relics:
        if not isinstance(spire_game.relics, list):
            raise TypeError(f"Relics must be a list, got {type(spire_game.relics)}")
        
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
        sts_deck = convert_deck(spire_game.deck)
        # Clear the starting deck and add converted cards
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
            
            # Handle different game states
            if game_state.play_available and game_state.in_combat:
                return self.handle_combat(game_state)
            elif game_state.choice_available:
                return self.handle_choice_screen(game_state)
            elif game_state.proceed_available:
                return ProceedAction()
            else:
                # Check what's actually available
                print(f"Game state: play={getattr(game_state, 'play_available', False)}, "
                      f"choice={getattr(game_state, 'choice_available', False)}, "
                      f"proceed={getattr(game_state, 'proceed_available', False)}, "
                      f"in_combat={getattr(game_state, 'in_combat', False)}", file=sys.stderr)
                
                # Try to return a reasonable default action
                if hasattr(game_state, 'choice_available') and game_state.choice_available:
                    return self.handle_choice_screen(game_state)
                elif hasattr(game_state, 'proceed_available') and game_state.proceed_available:
                    return ProceedAction()
                else:
                    # Last resort - try a basic choose action
                    return ChooseAction(0)
                
        except Exception as e:
            print(f"Error in decision making: {e}", file=sys.stderr)
            # Fallback to safe action
            if hasattr(game_state, 'choice_available') and game_state.choice_available:
                return ChooseAction(0)
            elif hasattr(game_state, 'proceed_available') and game_state.proceed_available:
                return ProceedAction()
            else:
                # Return something that won't crash
                return ChooseAction(0)
    
    def handle_combat(self, game_state: game.Game) -> Action:
        """
        Handle combat decisions using our C++ BattleContext.
        """
        try:
            # Convert to our battle context
            gc, bc = spirecomm_to_battlecontext(game_state)

            print(f"Combat state: {bc}", file=sys.stderr)
            
            # Simple combat logic for now - play first playable card or end turn
            if bc.cards.cardsInHand > 0:
                hand = bc.cards.hand
                for i, card_instance in enumerate(hand):
                    # Check if we can afford the card
                    if card_instance.cost <= bc.player.energy:
                        # Try to play this card
                        if card_instance.requiresTarget():
                            # Target first targetable monster
                            target = bc.monsters.getFirstTargetable()
                            if target >= 0:
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
        # For now, make simple choices
        # Card rewards - pick first card
        if hasattr(game_state, 'screen_type'):
            if game_state.screen_type == screen.ScreenType.CARD_REWARD:
                return ChooseAction(0)
            elif game_state.screen_type == screen.ScreenType.COMBAT_REWARD:
                return ChooseAction(0)
            elif game_state.screen_type == screen.ScreenType.MAP:
                # Choose first available path
                return ChooseAction(0)
            elif game_state.screen_type == screen.ScreenType.REST:
                # Rest to heal
                from spirecomm.spire.screen import RestOption
                return RestAction(RestOption.REST)
        
        # Default choice
        return ChooseAction(0)
    
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