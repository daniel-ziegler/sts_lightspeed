"""
Spirecomm to GameContext Converter

This module provides functions to convert spirecomm game state representations
into our internal C++ GameContext format for AI control of the real game.
"""

import os
import time
import random
import sys
import traceback
import json
import argparse
import itertools
from typing import Optional, Union

import slaythespire as sts
from spirecomm.spire.game import Game
from spirecomm.spire.character import PlayerClass
from spirecomm.spire import character
from spirecomm.spire import card, relic, game, screen
from spirecomm.spire.screen import RestOption, ScreenType
from spirecomm.communication.action import *
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

def _normalize_relic_name(name: str) -> str:
    """Casefold and drop all non-alphanumerics for relic-name matching. spirecomm's display names
    differ from the engine's only in incidental casing/spacing/punctuation (e.g. 'Bag of
    Preparation' vs 'Bag Of Preparation', or "Pandora's Box" vs 'Pandoras Box' -- the engine is
    itself inconsistent about apostrophes), so stripping them on both sides matches reliably
    without a per-relic synonym table."""
    return "".join(c for c in name.casefold() if c.isalnum())


def _get_relic_name_to_id_map():
    """Create normalized-relic-name -> RelicId mapping dictionary."""
    global _relic_name_to_id
    if _relic_name_to_id is None:
        _relic_name_to_id = {}
        for enum_idx, name in sts.getAllRelicNames():
            _relic_name_to_id[_normalize_relic_name(name)] = enum_idx
    return _relic_name_to_id

def map_card_id(spire_card_id: str) -> sts.CardId:
    """Map spirecomm card ID to our CardId enum using dynamic lookup."""
    card_map = _get_card_string_to_id_map()
    enum_idx = card_map.get(spire_card_id)
    if enum_idx is not None:
        return sts.CardId(enum_idx)
    return sts.CardId.INVALID


def map_relic_id(spire_relic_id: str) -> sts.RelicId:
    """Map spirecomm relic ID to our RelicId enum using normalized (case/space-insensitive) lookup."""
    relic_map = _get_relic_name_to_id_map()
    enum_idx = relic_map.get(_normalize_relic_name(spire_relic_id))
    if enum_idx is not None:
        return sts.RelicId(enum_idx)
    return sts.RelicId.INVALID

def map_character_class(spire_class: PlayerClass) -> sts.CharacterClass:
    """Map spirecomm PlayerClass to our CharacterClass."""
    return CHARACTER_CLASS_MAPPING.get(spire_class, sts.CharacterClass.INVALID)


# Relics whose every-Nth-card/attack/turn progress the engine tracks on bc.player, keyed by
# normalized relic id -> the writable Player counter field. Restored from the live relic.counter
# so a converted mid-fight state keeps its progress (e.g. Pen Nib's next-attack double damage).
# Letter Opener isn't here -- the engine doesn't track its counter.
_RELIC_COUNTER_ATTR = {
    _normalize_relic_name("Pen Nib"): "penNibCounter",
    _normalize_relic_name("Nunchaku"): "nunchakuCounter",
    _normalize_relic_name("Ink Bottle"): "inkBottleCounter",
    _normalize_relic_name("Happy Flower"): "happyFlowerCounter",
    _normalize_relic_name("Incense Burner"): "incenseBurnerCounter",
    _normalize_relic_name("Sundial"): "sundialCounter",
}


# Powers are keyed on the live game's stable power_id (the json "id", e.g. "DexLoss"), NOT the
# localized display name ("Dexterity Down") which drifts and forced per-power patching. The tables
# below map every StS power_id the BattleContext models to its engine status; powers the engine
# doesn't simulate are in _POWER_IDS_NOT_MODELED and dropped (the search just won't see them).
# Together the three structures cover every power_id in com/megacrit/cardcrawl/powers/*.java, so a
# power the engine modeler hasn't accounted for fails loud here instead of mid-run. The maps were
# generated by cross-referencing every power_id against the PlayerStatus/MonsterStatus enums;
# irregular ids (DexLoss->LOSE_DEXTERITY, Anger->ENRAGE, Flex->LOSE_STRENGTH, "...Power" suffixes,
# Wraith Form v2, ...) are pinned by hand. The not-modeled set is either other-class card powers
# (unreachable in an Ironclad game without Prismatic Shard) or monster behaviors the engine drives
# from move logic rather than a status (Split, Stasis, Shifting, Unawakened, Painful Stabs, ...).
_PLAYER_POWER_ID_TO_STATUS = {
    'Accuracy': sts.PlayerStatus.ACCURACY,
    'After Image': sts.PlayerStatus.AFTER_IMAGE,
    'Amplify': sts.PlayerStatus.AMPLIFY,
    'Artifact': sts.PlayerStatus.ARTIFACT,
    'Barricade': sts.PlayerStatus.BARRICADE,
    'BattleHymn': sts.PlayerStatus.BATTLE_HYMN,
    'Bias': sts.PlayerStatus.BIAS,
    'Blur': sts.PlayerStatus.BLUR,
    'Brutality': sts.PlayerStatus.BRUTALITY,
    'Buffer': sts.PlayerStatus.BUFFER,
    'Burst': sts.PlayerStatus.BURST,
    'Collect': sts.PlayerStatus.COLLECT,
    'Combust': sts.PlayerStatus.COMBUST,
    'Confusion': sts.PlayerStatus.CONFUSED,
    'Constricted': sts.PlayerStatus.CONSTRICTED,
    'Corruption': sts.PlayerStatus.CORRUPTION,
    'Creative AI': sts.PlayerStatus.CREATIVE_AI,
    'Dark Embrace': sts.PlayerStatus.DARK_EMBRACE,
    'Demon Form': sts.PlayerStatus.DEMON_FORM,
    'DevaForm': sts.PlayerStatus.DEVA,
    'DevotionPower': sts.PlayerStatus.DEVOTION,
    'DexLoss': sts.PlayerStatus.LOSE_DEXTERITY,
    'Dexterity': sts.PlayerStatus.DEXTERITY,
    'Double Damage': sts.PlayerStatus.DOUBLE_DAMAGE,
    'Double Tap': sts.PlayerStatus.DOUBLE_TAP,
    'Draw Card': sts.PlayerStatus.DRAW_CARD_NEXT_TURN,
    'Draw Reduction': sts.PlayerStatus.DRAW_REDUCTION,
    'DuplicationPower': sts.PlayerStatus.DUPLICATION,
    'Echo Form': sts.PlayerStatus.ECHO_FORM,
    'Electro': sts.PlayerStatus.ELECTRO,
    'Energized': sts.PlayerStatus.ENERGIZED,
    'EnergizedBlue': sts.PlayerStatus.ENERGIZED,
    'Entangled': sts.PlayerStatus.ENTANGLED,
    'Envenom': sts.PlayerStatus.ENVENOM,
    'Equilibrium': sts.PlayerStatus.EQUILIBRIUM,
    'EstablishmentPower': sts.PlayerStatus.ESTABLISHMENT,
    'Evolve': sts.PlayerStatus.EVOLVE,
    'Feel No Pain': sts.PlayerStatus.FEEL_NO_PAIN,
    'Fire Breathing': sts.PlayerStatus.FIRE_BREATHING,
    'Flame Barrier': sts.PlayerStatus.FLAME_BARRIER,
    'Flex': sts.PlayerStatus.LOSE_STRENGTH,
    'Focus': sts.PlayerStatus.FOCUS,
    'Frail': sts.PlayerStatus.FRAIL,
    'FreeAttackPower': sts.PlayerStatus.FREE_ATTACK_POWER,
    'Hello': sts.PlayerStatus.HELLO_WORLD,
    'Hex': sts.PlayerStatus.HEX,
    'Infinite Blades': sts.PlayerStatus.INFINITE_BLADES,
    'Intangible': sts.PlayerStatus.INTANGIBLE,
    'IntangiblePlayer': sts.PlayerStatus.INTANGIBLE,
    'Juggernaut': sts.PlayerStatus.JUGGERNAUT,
    'LikeWaterPower': sts.PlayerStatus.LIKE_WATER,
    'Loop': sts.PlayerStatus.LOOP,
    'Magnetism': sts.PlayerStatus.MAGNETISM,
    'Mantra': sts.PlayerStatus.MANTRA,
    'MasterRealityPower': sts.PlayerStatus.MASTER_REALITY,
    'Mayhem': sts.PlayerStatus.MAYHEM,
    'Metallicize': sts.PlayerStatus.METALLICIZE,
    'Next Turn Block': sts.PlayerStatus.NEXT_TURN_BLOCK,
    'No Draw': sts.PlayerStatus.NO_DRAW,
    'NoBlockPower': sts.PlayerStatus.NO_BLOCK,
    'Noxious Fumes': sts.PlayerStatus.NOXIOUS_FUMES,
    'OmegaPower': sts.PlayerStatus.OMEGA,
    'Panache': sts.PlayerStatus.PANACHE,
    'Pen Nib': sts.PlayerStatus.PEN_NIB,
    'Phantasmal': sts.PlayerStatus.PHANTASMAL,
    'Plated Armor': sts.PlayerStatus.PLATED_ARMOR,
    'Rage': sts.PlayerStatus.RAGE,
    'Rebound': sts.PlayerStatus.REBOUND,
    'Regenerate': sts.PlayerStatus.REGEN,
    'Regeneration': sts.PlayerStatus.REGEN,
    'Ritual': sts.PlayerStatus.RITUAL,
    'Rupture': sts.PlayerStatus.RUPTURE,
    'Sadistic': sts.PlayerStatus.SADISTIC,
    'StaticDischarge': sts.PlayerStatus.STATIC_DISCHARGE,
    'Strength': sts.PlayerStatus.STRENGTH,
    'Surrounded': sts.PlayerStatus.SURROUNDED,
    'TheBomb': sts.PlayerStatus.THE_BOMB,
    'Thorns': sts.PlayerStatus.THORNS,
    'Thousand Cuts': sts.PlayerStatus.THOUSAND_CUTS,
    'Tools Of The Trade': sts.PlayerStatus.TOOLS_OF_THE_TRADE,
    'Vigor': sts.PlayerStatus.VIGOR,
    'Vulnerable': sts.PlayerStatus.VULNERABLE,
    'WaveOfTheHandPower': sts.PlayerStatus.WAVE_OF_THE_HAND,
    'Weakened': sts.PlayerStatus.WEAK,
    'Wraith Form v2': sts.PlayerStatus.WRAITH_FORM,
    'WrathNextTurnPower': sts.PlayerStatus.WRATH_NEXT_TURN,
}

_MONSTER_POWER_ID_TO_STATUS = {
    'Anger': sts.MonsterStatus.ENRAGE,
    'Angry': sts.MonsterStatus.ANGRY,
    'Artifact': sts.MonsterStatus.ARTIFACT,
    'BeatOfDeath': sts.MonsterStatus.BEAT_OF_DEATH,
    'BlockReturnPower': sts.MonsterStatus.BLOCK_RETURN,
    'Choked': sts.MonsterStatus.CHOKED,
    'CorpseExplosionPower': sts.MonsterStatus.CORPSE_EXPLOSION,
    'Curiosity': sts.MonsterStatus.CURIOSITY,
    'Curl Up': sts.MonsterStatus.CURL_UP,
    'Fading': sts.MonsterStatus.FADING,
    'Flight': sts.MonsterStatus.FLIGHT,
    'Generic Strength Up Power': sts.MonsterStatus.GENERIC_STRENGTH_UP,
    'Intangible': sts.MonsterStatus.INTANGIBLE,
    'Invincible': sts.MonsterStatus.INVINCIBLE,
    'Lockon': sts.MonsterStatus.LOCK_ON,
    'Malleable': sts.MonsterStatus.MALLEABLE,
    'Metallicize': sts.MonsterStatus.METALLICIZE,
    'Mode Shift': sts.MonsterStatus.MODE_SHIFT,
    'Plated Armor': sts.MonsterStatus.PLATED_ARMOR,
    'Poison': sts.MonsterStatus.POISON,
    'Regenerate': sts.MonsterStatus.REGEN,
    'Regeneration': sts.MonsterStatus.REGEN,
    'Ritual': sts.MonsterStatus.RITUAL,
    'Shackled': sts.MonsterStatus.SHACKLED,
    'Sharp Hide': sts.MonsterStatus.SHARP_HIDE,
    'Slow': sts.MonsterStatus.SLOW,
    'Spore Cloud': sts.MonsterStatus.SPORE_CLOUD,
    'Strength': sts.MonsterStatus.STRENGTH,
    'Thievery': sts.MonsterStatus.THIEVERY,
    'Thorns': sts.MonsterStatus.THORNS,
    'Time Warp': sts.MonsterStatus.TIME_WARP,
    'Vulnerable': sts.MonsterStatus.VULNERABLE,
    'Weakened': sts.MonsterStatus.WEAK,
}

_POWER_IDS_NOT_MODELED = frozenset({
    'Adaptation', 'AlwaysMad', 'AngelForm', 'Attack Burn', 'BackAttack', 'Berserk',
    'CannotChangeStancePower', 'Compulsive', 'Conserve', 'Controlled', 'DEPRECATEDCondense',
    'DisciplinePower', 'Draw', 'EmotionalTurmoilPower', 'EndTurnDeath', 'EnergyDownPower',
    'Explosive', 'FlickPower', 'FlowPower', 'Grounded', 'GrowthPower', 'Heatsink', 'HotHot',
    'Life Link', 'Lightning Mastery', 'Mastery', 'Minion', 'Night Terror', 'Nirvana', 'NoSkills',
    'Nullify Attack', 'OmnisciencePower', 'Painful Stabs', 'PathToVictoryPower', 'RechargingCore',
    'Repair', 'Retain Cards', 'Retribution', 'Serenity', 'Shifting', 'Skill Burn', 'Split',
    'Stasis', 'Storm', 'StrikeUp', 'Study', 'TimeMazePower', 'Unawakened', 'Vault', 'Winter',
    'WireheadingPower',
})

# Engine statuses applied via .debuff()/.addDebuff() (so Artifact and the like interact correctly);
# every other modeled status is a buff. Keyed on the resolved enum, so display-name variants are
# irrelevant. These mirror the powers the live game marks as debuffs.
_DEBUFF_PLAYER_STATUSES = frozenset({
    sts.PlayerStatus.DOUBLE_DAMAGE, sts.PlayerStatus.DRAW_REDUCTION, sts.PlayerStatus.FRAIL,
    sts.PlayerStatus.VULNERABLE, sts.PlayerStatus.WEAK, sts.PlayerStatus.BIAS,
    sts.PlayerStatus.CONFUSED, sts.PlayerStatus.CONSTRICTED, sts.PlayerStatus.ENTANGLED,
    sts.PlayerStatus.FASTING, sts.PlayerStatus.HEX, sts.PlayerStatus.LOSE_DEXTERITY,
    sts.PlayerStatus.LOSE_STRENGTH, sts.PlayerStatus.NO_BLOCK, sts.PlayerStatus.NO_DRAW,
})
# Monster statuses the engine applies through Monster::addDebuff (the rest go through buff()). This
# must match addDebuff's switch exactly -- it assert(false)s on any status it doesn't handle. Note
# SHACKLED is NOT here: despite being a debuff semantically, the engine applies it via buff<SHACKLED>
# (Monster.cpp) and addDebuff has no SHACKLED case, so routing it to addDebuff aborts the process.
_DEBUFF_MONSTER_STATUSES = frozenset({
    sts.MonsterStatus.CHOKED, sts.MonsterStatus.CORPSE_EXPLOSION, sts.MonsterStatus.LOCK_ON,
    sts.MonsterStatus.POISON, sts.MonsterStatus.VULNERABLE, sts.MonsterStatus.WEAK,
})

# The exact set of statuses Monster::addDebuff handles (its switch cases in include/combat/Monster.h).
# addDebuff does `assert(false)` on anything else, which SIGABRTs the whole process -- an
# uncatchable abort that hangs live play -- so any status we route to addDebuff MUST be in here.
# (buff() silently no-ops unknown statuses instead, so the buff path can't abort.) This guard fails
# loud at import if _DEBUFF_MONSTER_STATUSES ever drifts outside addDebuff's capability.
_ADDDEBUFF_HANDLED_STATUSES = frozenset({
    sts.MonsterStatus.BLOCK_RETURN, sts.MonsterStatus.CHOKED, sts.MonsterStatus.CORPSE_EXPLOSION,
    sts.MonsterStatus.LOCK_ON, sts.MonsterStatus.MARK, sts.MonsterStatus.POISON,
    sts.MonsterStatus.STRENGTH, sts.MonsterStatus.VULNERABLE, sts.MonsterStatus.WEAK,
})
assert _DEBUFF_MONSTER_STATUSES <= _ADDDEBUFF_HANDLED_STATUSES, (
    "monster statuses routed to addDebuff but not handled by it (would assert/SIGABRT live): "
    f"{_DEBUFF_MONSTER_STATUSES - _ADDDEBUFF_HANDLED_STATUSES}")

# Every power_id the game can send: mapped (player/monster) or knowingly-unmodeled. An id outside
# this set is new/renamed -> the apply_* helpers raise rather than play a silently-wrong state.
_ALL_POWER_IDS = (set(_PLAYER_POWER_ID_TO_STATUS) | set(_MONSTER_POWER_ID_TO_STATUS)
                  | _POWER_IDS_NOT_MODELED)
_dropped_powers_logged = set()


def _drop_power_once(power_id: str, kind: str) -> None:
    if power_id not in _dropped_powers_logged:
        _dropped_powers_logged.add(power_id)
        print(f"[power] {kind} power {power_id!r} not modeled by the engine; dropping",
              file=sys.stderr)


def apply_player_power(bc: sts.BattleContext, power_id: str, amount: int) -> None:
    """Apply a live player power (by stable power_id) to the converted battle, choosing buff vs
    debuff from the resolved status. Drops powers the engine doesn't model; raises on a power_id
    unknown to StS so a new power fails loud instead of silently diverging."""
    status = _PLAYER_POWER_ID_TO_STATUS.get(power_id)
    if status is None:
        if power_id in _ALL_POWER_IDS:
            _drop_power_once(power_id, "player")
            return
        raise ValueError(f"Unknown player power id: {power_id!r}")
    if status in _DEBUFF_PLAYER_STATUSES:
        bc.player.debuff(status, amount, False)
    else:
        bc.player.buff(status, amount)


def apply_monster_power(sts_monster, power_id: str, amount: int) -> None:
    """Apply a live monster power (by stable power_id) to a converted monster. Drops powers the
    engine drives from move logic or doesn't model (Split, Stasis, Shifting, ...); raises on an id
    unknown to StS."""
    status = _MONSTER_POWER_ID_TO_STATUS.get(power_id)
    if status is None:
        if power_id in _ALL_POWER_IDS:
            _drop_power_once(power_id, "monster")
            return
        raise ValueError(f"Unknown monster power id: {power_id!r}")
    if status in _DEBUFF_MONSTER_STATUSES:
        sts_monster.addDebuff(status, amount, False)
    else:
        sts_monster.buff(status, amount)


# spirecomm forwards the game's raw AbstractPotion.ID string (CommunicationMod reads
# it straight off the live AbstractPotion). These IDs are the canonical decompiled
# values (com/megacrit/cardcrawl/potions/*.java) and match the engine's own save-file
# table (constants/SaveFileMappings.h) exactly -- including the irregular ones
# ("SteroidPotion" for Flex, "ElixirPotion", "EssenceOfSteel", and the base-game potions
# that keep a space, e.g. "Dexterity Potion"). The empty slot ("Potion Slot") is filtered
# out by spirecomm's get_real_potions() before we ever map it.
_POTION_ID_TO_ENUM = {
    "Ambrosia": sts.Potion.AMBROSIA,
    "Ancient Potion": sts.Potion.ANCIENT_POTION,
    "AttackPotion": sts.Potion.ATTACK_POTION,
    "BlessingOfTheForge": sts.Potion.BLESSING_OF_THE_FORGE,
    "Block Potion": sts.Potion.BLOCK_POTION,
    "BloodPotion": sts.Potion.BLOOD_POTION,
    "BottledMiracle": sts.Potion.BOTTLED_MIRACLE,
    "ColorlessPotion": sts.Potion.COLORLESS_POTION,
    "CultistPotion": sts.Potion.CULTIST_POTION,
    "CunningPotion": sts.Potion.CUNNING_POTION,
    "Dexterity Potion": sts.Potion.DEXTERITY_POTION,
    "DistilledChaos": sts.Potion.DISTILLED_CHAOS,
    "DuplicationPotion": sts.Potion.DUPLICATION_POTION,
    "ElixirPotion": sts.Potion.ELIXIR_POTION,
    "Energy Potion": sts.Potion.ENERGY_POTION,
    "EntropicBrew": sts.Potion.ENTROPIC_BREW,
    "EssenceOfDarkness": sts.Potion.ESSENCE_OF_DARKNESS,
    "EssenceOfSteel": sts.Potion.ESSENCE_OF_STEEL,
    "Explosive Potion": sts.Potion.EXPLOSIVE_POTION,
    "FairyPotion": sts.Potion.FAIRY_POTION,
    "FearPotion": sts.Potion.FEAR_POTION,
    "Fire Potion": sts.Potion.FIRE_POTION,
    "FocusPotion": sts.Potion.FOCUS_POTION,
    "Fruit Juice": sts.Potion.FRUIT_JUICE,
    "GamblersBrew": sts.Potion.GAMBLERS_BREW,
    "GhostInAJar": sts.Potion.GHOST_IN_A_JAR,
    "HeartOfIron": sts.Potion.HEART_OF_IRON,
    "LiquidBronze": sts.Potion.LIQUID_BRONZE,
    "LiquidMemories": sts.Potion.LIQUID_MEMORIES,
    "Poison Potion": sts.Potion.POISON_POTION,
    "PotionOfCapacity": sts.Potion.POTION_OF_CAPACITY,
    "PowerPotion": sts.Potion.POWER_POTION,
    "Regen Potion": sts.Potion.REGEN_POTION,
    "SkillPotion": sts.Potion.SKILL_POTION,
    "SmokeBomb": sts.Potion.SMOKE_BOMB,
    "SneckoOil": sts.Potion.SNECKO_OIL,
    "SpeedPotion": sts.Potion.SPEED_POTION,
    "StancePotion": sts.Potion.STANCE_POTION,
    "SteroidPotion": sts.Potion.FLEX_POTION,
    "Strength Potion": sts.Potion.STRENGTH_POTION,
    "Swift Potion": sts.Potion.SWIFT_POTION,
    "Weak Potion": sts.Potion.WEAK_POTION,
}

# Fail loud at import if the table drifts from the engine enum: every real Potion must be
# mapped exactly once (INVALID and EMPTY_POTION_SLOT are not obtainable potions).
_unmapped_potions = [
    p for p in sts.Potion.__members__.values()
    if p not in (sts.Potion.INVALID, sts.Potion.EMPTY_POTION_SLOT)
    and p not in _POTION_ID_TO_ENUM.values()
]
assert not _unmapped_potions, f"potion id table missing entries for: {_unmapped_potions}"


def map_potion_id(spire_potion_id: str) -> sts.Potion:
    """Map a spirecomm potion id (the game's AbstractPotion.ID) to a Potion enum.

    Raises on an unknown id rather than dropping the potion: a silently missing
    potion would desync the bc.potions slots from spirecomm's get_real_potions()
    ordering that map_search_action_to_spirecomm relies on."""
    potion = _POTION_ID_TO_ENUM.get(spire_potion_id)
    if potion is None:
        raise ValueError(f"Unknown potion id: {spire_potion_id}")
    return potion


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


def convert_combat_state(spire_game: game.Game, gc: sts.GameContext) -> "tuple[sts.BattleContext, dict]":
    """
    Convert spirecomm combat state to BattleContext.

    This function handles:
    - Player energy, HP, block, powers
    - Monster states including HP, block, intent, powers
    - Card piles (hand, draw pile, discard pile, exhaust pile)
    - Turn information

    Returns:
        (BattleContext initialized with spirecomm combat state, slot_to_spire mapping from sim
        MonsterGroup slot to spirecomm monster_index -- needed because the monster layout is
        repacked with reserved split slots, so sim slot != live monster_index).
    """
    if not spire_game.in_combat:
        raise ValueError("Cannot convert combat state when not in combat")

    # The potion belt is already on the gc (spirecomm_to_gamecontext sets capacity + held potions in
    # get_real_potions() order, before this is called), and empty_battle_context() copies
    # gc.potions/potionCount straight into the battle. Re-obtaining them here would double the belt
    # and desync the bc.potions slot indices map_search_action_to_spirecomm reads back.

    # Create battle context from GameContext
    bc = gc.empty_battle_context()

    # empty_battle_context() does NOT register relics, so without this the searcher plays a
    # relic-less player (no Kunai/Shuriken/Pen Nib/etc. triggers). Register ownership bits from
    # the gc (which already obtained the player's relics); no atBattleStart effects re-fire.
    bc.register_relics_from(gc)

    # Clear the initialized cards to avoid mixing with spirecomm state
    bc.cards.clear()
    
    # Set the input state to PLAYER_NORMAL so the searcher can find actions
    # InputState enum: EXECUTING_ACTIONS=0, PLAYER_NORMAL=1, CARD_SELECT=2, etc.
    bc.input_state = sts.InputState.PLAYER_NORMAL
    
    # Player state conversion
    player = spire_game.player
    if player:
        # Set basic player stats
        bc.player.energy = player.energy
        bc.player.block = player.block
            
        # Convert player powers/buffs/debuffs (keyed on the stable power_id, not the display name)
        for power in player.powers:
            apply_player_power(bc, power.power_id, power.amount)

    # Restore per-combat relic counters (progress toward the next every-Nth trigger) from the live
    # relics; register_relics_from only copies ownership bits, leaving these at zero.
    for spire_relic in spire_game.relics:
        attr = _RELIC_COUNTER_ATTR.get(_normalize_relic_name(spire_relic.relic_id))
        if attr is not None:
            setattr(bc.player, attr, spire_relic.counter)

    # Card piles conversion - create CardInstance objects from spirecomm cards.
    # Every card needs a DISTINCT uniqueId: the engine tracks cards through hand/queue/piles by id
    # (e.g. removeFromHandById on play), so leaving them all at the default -1 makes a played card
    # fail to leave the search's simulated hand -- the searcher then thinks it can replay its best
    # card every turn, massively over-valuing offense. Assign ids sequentially and leave
    # next_unique_card_id past the highest so cards generated mid-search don't collide.
    _uid = 0
    def _add(spire_card, mover):
        nonlocal _uid
        card_instance = convert_spire_card_to_instance(spire_card)
        card_instance.uniqueId = _uid
        _uid += 1
        mover(card_instance)

    for spire_card in spire_game.hand:
        _add(spire_card, bc.cards.moveToHand)

    if spire_game.draw_pile:
        # Add to the UNKNOWN region, not the known top: the player can't actually see the draw
        # order, so the searcher must draw stochastically (chance nodes) like native play.
        # moveToDrawPileTop would mark the whole pile known-order, letting the search "cheat" on
        # a reconstructed order and e.g. decline to block because it thinks a Defend is coming.
        for spire_card in spire_game.draw_pile:
            _add(spire_card, bc.cards.moveToDrawPileUnknown)

    if spire_game.discard_pile:
        for spire_card in spire_game.discard_pile:
            _add(spire_card, bc.cards.moveToDiscardPile)

    if spire_game.exhaust_pile:
        for spire_card in spire_game.exhaust_pile:
            _add(spire_card, bc.cards.moveToExhaustPile)

    bc.cards.next_unique_card_id = _uid

    # Monster states conversion. The live game keeps every monster in a fixed positional slot and
    # reports dead/split ones as is_gone entries that still occupy their slot, so a converted slime
    # fight can have a still-splittable large slime sitting directly beside a live monster. The
    # native split routines (largeSlimeSplit/slimeBossSplit) assume a large slime's next slot is a
    # free reservation; splitting into an occupied slot clobbers that monster and desyncs
    # monstersAlive/monsterCount (-> getRandomMonsterIdx walks off the 5-element array). So we don't
    # mirror the live slots: we lay out only the live monsters and reserve a free slot after each
    # one that can still split, exactly as native encounters do. slot_to_spire maps each sim slot
    # back to its spirecomm monster_index for translating the searcher's target back to a command.
    slot_to_spire = _build_monster_group(bc, spire_game.monsters)

    # Boss fights search wider + deeper (SearchAgent gates on isBossEncounter(bc.encounter)); the
    # live game doesn't report the encounter, so recover it from the boss monster on the field.
    bc.encounter = _infer_boss_encounter(spire_game.monsters)

    return bc, slot_to_spire


# Monsters whose move can still spawn two replacements, so the native split needs the slot after
# them held free. Mediums/smalls never split.
_SPLITTABLE_MONSTER_IDS = frozenset({
    sts.MonsterId.ACID_SLIME_L,
    sts.MonsterId.SPIKE_SLIME_L,
    sts.MonsterId.SLIME_BOSS,
})


# Boss fights get wider chance/end-turn search and a higher sim count (SearchAgent gates both on
# isBossEncounter(bc.encounter)). The live game doesn't report its encounter id, so we recover it
# from the boss monster on the field -- keyed on the spirecomm monster_id. Non-boss fights keep
# bc.encounter == INVALID, which is all isBossEncounter needs (it only distinguishes boss vs not).
_BOSS_ENCOUNTER_BY_MONSTER = {
    'SlimeBoss': sts.MonsterEncounter.SLIME_BOSS,
    'TheGuardian': sts.MonsterEncounter.THE_GUARDIAN,
    'Hexaghost': sts.MonsterEncounter.HEXAGHOST,
    'HexaghostBody': sts.MonsterEncounter.HEXAGHOST,
    'BronzeAutomaton': sts.MonsterEncounter.AUTOMATON,
    'TheCollector': sts.MonsterEncounter.COLLECTOR,
    'Champ': sts.MonsterEncounter.CHAMP,
    'AwakenedOne': sts.MonsterEncounter.AWAKENED_ONE,
    'TimeEater': sts.MonsterEncounter.TIME_EATER,
    'Donu': sts.MonsterEncounter.DONU_AND_DECA,
    'Deca': sts.MonsterEncounter.DONU_AND_DECA,
    'CorruptHeart': sts.MonsterEncounter.THE_HEART,
}


def _infer_boss_encounter(spire_monsters):
    """Return the MonsterEncounter for a boss fight (recognized by its signature monster), or
    INVALID for any non-boss fight."""
    for monster in spire_monsters:
        enc = _BOSS_ENCOUNTER_BY_MONSTER.get(monster.monster_id)
        if enc is not None:
            return enc
    return sts.MonsterEncounter.INVALID


def _set_sts_monster_fields(bc, sts_monster, monster, slot: int) -> None:
    """Copy a live monster's hp/block/move-history/powers onto a freshly created sim Monster.

    moveHistory[0] (the current intent) drives what the engine makes the monster do on its turn; a
    stray INVALID there hits an assert(false) in Monster::takeTurn during search -- an uncatchable
    SIGABRT. Two distinct cases produce a missing current move, handled differently:
      * move_id present but unmapped -> a real gap in our move table; raise (fail loud) so it gets
        added rather than silently mis-simulated.
      * move_id absent (CommunicationMod reports intent NONE with no move_id, e.g. a flying Byrd
        between turns) -> the live game itself hasn't committed the move, so there's no ground truth
        to mirror; let the engine's own AI roll a legal move via rollMove, exactly as the game will
        when the monster's turn comes."""
    sts_monster.curHp = monster.current_hp
    sts_monster.maxHp = monster.max_hp
    sts_monster.block = monster.block
    sts_monster.halfDead = monster.half_dead
    sts_monster.idx = slot

    # spirecomm reports an absent move_id as -1 (None for last_move_id); a real move byte is >= 0.
    invalid = int(sts.MonsterMoveId.INVALID)
    move_known = monster.move_id is not None and monster.move_id >= 0
    move_history = [0, 0]
    if move_known:
        mapped = int(map_move_id(monster.monster_id, monster.move_id))
        if mapped == invalid:
            raise ValueError(f"unmapped current move for {monster.monster_id} (move_id="
                             f"{monster.move_id}); add (monster, move_id) to map_move_id")
        move_history[0] = mapped
    if monster.last_move_id is not None and monster.last_move_id >= 0:
        move_history[1] = int(map_move_id(monster.monster_id, monster.last_move_id))
    sts_monster.moveHistory = move_history

    for power in monster.powers:
        apply_monster_power(sts_monster, power.power_id, power.amount)

    if not move_known:
        sts_monster.rollMove(bc)
        if sts_monster.isAlive() and sts_monster.moveHistory[0] == invalid \
                and sts_monster.pending_move_rolls == 0:
            raise ValueError(f"rollMove left {monster.monster_id} with no move "
                             f"(intent {monster.intent})")


# Encounters whose summon/respawn actions write to hardcoded monster slots, so a converted group
# must reproduce the engine's native slot layout instead of a dense pack -- otherwise the summon's
# slot math (which assumes the summoner sits at a fixed slot with specific reserved minion slots)
# overwrites a live monster or runs off the array and assert(false)s during search (an uncatchable
# SIGABRT). Maps summoner MonsterId -> (summoner_slot, [minion slots in the summon's fill order]).
# Slots/orders are read straight from MonsterGroup.cpp inits + the summon helpers:
#   GremlinLeader  _SummonGremlins   gremlins {1,2,0}, leader @3   (MonsterGroup GREMLIN_LEADER)
#   BronzeAutomaton spawnBronzeOrbs  orbs {0,2},       automaton @1 (AUTOMATON)
#   TheCollector   _SpawnTorchHeads  torchheads {1,0}, collector @2
#   Reptomancer    reptoSummonHelper daggers {4,1,3,0},reptomancer @2 (REPTOMANCER)
def _fixed_slot_layouts():
    return {
        sts.MonsterId.GREMLIN_LEADER: (3, [1, 2, 0]),
        sts.MonsterId.BRONZE_AUTOMATON: (1, [0, 2]),
        sts.MonsterId.THE_COLLECTOR: (2, [1, 0]),
        sts.MonsterId.REPTOMANCER: (2, [4, 1, 3, 0]),
    }


def _build_fixed_layout_group(bc, live, summoner_entry, summoner_slot, minion_slots) -> dict:
    """Build a fixed-slot summoner encounter (see _fixed_slot_layouts). The summoner is pinned to
    summoner_slot; live minions fill minion_slots in order; every other slot up to the highest used
    one is left as a dying buffer so the summon finds open slots exactly where it looks. Returns
    slot_to_spire."""
    minions = [e for e in live if e is not summoner_entry]
    if len(minions) > len(minion_slots):
        raise ValueError(f"summoner fight has {len(minions)} minions (> {len(minion_slots)} slots): "
                         f"{[m[1].monster_id for m in minions]}")
    by_slot = {summoner_slot: summoner_entry}
    for minion, slot in zip(minions, minion_slots):
        by_slot[slot] = minion
    n_slots = max([summoner_slot] + minion_slots) + 1

    slot_to_spire = {}
    for slot in range(n_slots):
        entry = by_slot.get(slot)
        if entry is None:
            bc.monsters.skipMonsterSlot()
            continue
        spire_idx, monster, monster_id = entry
        slot_to_spire[slot] = spire_idx
        bc.monsters.createMonster(bc, monster_id)
        _set_sts_monster_fields(bc, bc.monsters[slot], monster, slot)
    return slot_to_spire


def _build_monster_group(bc: sts.BattleContext, spire_monsters) -> dict:
    """Populate bc.monsters from the live monster list and return slot_to_spire (sim slot ->
    spirecomm monster_index). Live monsters are packed in order; a free slot is reserved after each
    splittable monster so largeSlimeSplit/slimeBossSplit have the slot they assume. The Gremlin
    Leader fight uses its own fixed layout (see _build_gremlin_leader_group). Raises if the reserved
    layout would exceed the 5-slot group (never happens for a legal slime lineage: at most two large
    slimes -> four slots with reservations)."""
    if not spire_monsters:
        return {}

    # Resolve the live combatants once (alive, non-prop). map_monster_string_to_id raises on unknown.
    live = []
    for spire_idx, monster in enumerate(spire_monsters):
        if monster.current_hp <= 0 or monster.is_gone:
            continue
        # Hexaghost orbs etc. are part of another engine monster, not standalone combatants.
        if monster.monster_id in _MONSTER_IDS_SKIP_IN_COMBAT:
            continue
        monster_id = map_monster_string_to_id(monster.monster_id)
        live.append((spire_idx, monster, monster_id))

    # Fixed-slot summoner encounters (Gremlin Leader, Bronze Automaton, Collector, Reptomancer)
    # must reproduce the engine's native layout so their hardcoded summon slots stay valid.
    fixed_layouts = _fixed_slot_layouts()
    for entry in live:
        layout = fixed_layouts.get(entry[2])
        if layout is not None:
            return _build_fixed_layout_group(bc, live, entry, layout[0], layout[1])

    # Default layout: pack in order, reserving a free slot after each splittable monster.
    plan = []
    for entry in live:
        plan.append(('mon',) + entry)
        if entry[2] in _SPLITTABLE_MONSTER_IDS:
            plan.append(('gap',))

    if len(plan) > 5:
        raise ValueError(f"Converted monster layout needs {len(plan)} slots (>5): "
                         f"{[m.monster_id for m in spire_monsters]}")

    slot_to_spire = {}
    for slot, entry in enumerate(plan):
        if entry[0] == 'gap':
            bc.monsters.skipMonsterSlot()
            continue
        _, spire_idx, monster, monster_id = entry
        slot_to_spire[slot] = spire_idx
        bc.monsters.createMonster(bc, monster_id)
        _set_sts_monster_fields(bc, bc.monsters[slot], monster, slot)

    return slot_to_spire


def convert_spire_card_to_instance(spire_card: card.Card) -> sts.CardInstance:
    """Convert a spirecomm Card to a CardInstance."""
    # Map the card ID using card_id field, not name
    card_id = map_card_id(spire_card.card_id)
    if card_id == sts.CardId.INVALID:
        raise ValueError(f"Unknown card: {spire_card.card_id}")
    
    # Create CardInstance
    instance = sts.CardInstance(card_id, spire_card.upgrades > 0)
    
    # Set additional properties
    instance.cost = spire_card.cost
    instance.costForTurn = spire_card.cost
    
    # Handle upgrade count for Searing Blow
    if spire_card.upgrades > 1:
        # Apply additional upgrades for Searing Blow
        for _ in range(spire_card.upgrades - 1):
            instance.upgrade()
    
    return instance


# Live monster ids the engine keys under a different string. The engine names the Masked Bandits
# by their in-game display names (Romeo/Bear/Pointy) and drops the "The"/"Body" affixes, so we pin
# the live id -> engine string here. Checked against the engine at import (below).
_MONSTER_ID_SYNONYMS = {
    "TheCollector": "Collector",
    "Maw": "TheMaw",
    "BanditLeader": "Romeo",
    "BanditBear": "Bear",
    "BanditChild": "Pointy",
    # The engine models Hexaghost as a single monster; its 6 orbs (skipped below) are folded in.
    "HexaghostBody": "Hexaghost",
    # The act-2 "Healer" combatant is the engine's Mystic (its moves are already keyed "Healer").
    "Healer": "Mystic",
}

# Entries that appear in a combat's monster list but are not separate engine combatants: the
# Hexaghost's orbs are part of the single engine Hexaghost. Skipped during conversion (like is_gone).
_MONSTER_IDS_SKIP_IN_COMBAT = frozenset({"HexaghostOrb"})

# Monster ids that are non-combat event props (Apology Slime, the Serpent's merchant) -- they never
# appear in a real battle's monster list, so reaching one in combat is a real bug we want to fail
# loud on rather than silently mishandle. (The act-2 "Healer" is NOT here: it is a real combatant,
# the engine's Mystic -- see _MONSTER_ID_SYNONYMS.)
_MONSTER_IDS_NON_COMBAT = frozenset({"Apology Slime", "Serpent"})


# (engine Event enum name, eventIdStrings value, eventGameNames value) for every ? -room event the
# sim models. The live game's screen.event_id matches the id-string and screen.event_name matches
# the game-name; we accept either so the lookup is robust to which one CommunicationMod sends.
# Generated from include/constants/Events.h (kept in lockstep with the engine enum).
_EVENT_ENUM_NAME_ID_GAME = [
    ('NEOW', 'NEOW', 'NEOW'),
    ('OMINOUS_FORGE', 'Accursed Blacksmith', 'Ominous Forge'),
    ('PLEADING_VAGRANT', 'Addict', 'Pleading Vagrant'),
    ('ANCIENT_WRITING', 'Back to Basics', 'Ancient Writing'),
    ('OLD_BEGGAR', 'Beggar', 'Old Beggar'),
    ('BIG_FISH', 'Big Fish', 'Big Fish'),
    ('BONFIRE_SPIRITS', 'Bonfire Elementals', 'Bonfire Spirits'),
    ('COLOSSEUM', 'Colosseum', 'The Colosseum'),
    ('CURSED_TOME', 'Cursed Tome', 'Cursed Tome'),
    ('DEAD_ADVENTURER', 'Dead Adventurer', 'Dead Adventurer'),
    ('DESIGNER_IN_SPIRE', 'Designer', 'Designer In-Spire'),
    ('AUGMENTER', 'Drug Dealer', 'Augmenter'),
    ('DUPLICATOR', 'Duplicator', 'Duplicator'),
    ('FACE_TRADER', 'Face Trader', 'Face Trader'),
    ('FALLING', 'Falling', 'Falling'),
    ('FORGOTTEN_ALTAR', 'Forgotten Altar', 'Forgotten Altar'),
    ('THE_DIVINE_FOUNTAIN', 'Fountain of Cleansing', 'The Divine Fountain'),
    ('GHOSTS', 'Ghosts', 'Council of Ghosts'),
    ('GOLDEN_IDOL', 'Golden Idol', 'Golden Idol'),
    ('GOLDEN_SHRINE', 'Golden Shrine', 'Golden Shrine'),
    ('WING_STATUE', 'Golden Wing', 'Wing Statue'),
    ('KNOWING_SKULL', 'Knowing Skull', 'Knowing Skull'),
    ('LAB', 'Lab', 'Lab'),
    ('THE_SSSSSERPENT', 'Liars Game', 'The Ssssserpent'),
    ('LIVING_WALL', 'Living Wall', 'Living Wall'),
    ('MASKED_BANDITS', 'Masked Bandits', 'Masked Bandits'),
    ('MATCH_AND_KEEP', 'Match and Keep', 'Match and Keep'),
    ('MINDBLOOM', 'Mindbloom', 'Mindbloom'),
    ('HYPNOTIZING_COLORED_MUSHROOMS', 'Mushrooms', 'Hypnotizing Colored Mushrooms'),
    ('MYSTERIOUS_SPHERE', 'Mysterious Sphere', 'Mysterious Sphere'),
    ('THE_NEST', 'Nest', 'The Nest'),
    ('NLOTH', 'Nloth', "N'loth"),
    ('NOTE_FOR_YOURSELF', 'Note For Yourself', 'Note For Yourself'),
    ('PURIFIER', 'Purifier', 'Purifier'),
    ('SCRAP_OOZE', 'Scrap Ooze', 'Scrap Ooze'),
    ('SECRET_PORTAL', 'Secret Portal', 'Secret Portal'),
    ('SENSORY_STONE', 'Sensory Stone', 'Sensory Stone'),
    ('SHINING_LIGHT', 'Shining Light', 'Shining Light'),
    ('THE_CLERIC', 'The Cleric', 'The Cleric'),
    ('THE_JOUST', 'The Joust', 'The Joust'),
    ('THE_LIBRARY', 'The Library', 'The Library'),
    ('THE_MAUSOLEUM', 'The Mausoleum', 'The Mausoleum'),
    ('THE_MOAI_HEAD', 'The Moai Head', 'The Moai Head'),
    ('THE_WOMAN_IN_BLUE', 'The Woman in Blue', 'The Woman in Blue'),
    ('TOMB_OF_LORD_RED_MASK', 'Tomb of Lord Red Mask', 'Tomb of Lord Red Mask'),
    ('TRANSMORGRIFIER', 'Transmorgrifier', 'Transmorgrifier'),
    ('UPGRADE_SHRINE', 'Upgrade Shrine', 'Upgrade Shrine'),
    ('VAMPIRES', 'Vampires', 'Vampires(?)'),
    ('WE_MEET_AGAIN', 'WeMeetAgain', 'We Meet Again!'),
    ('WHEEL_OF_CHANGE', 'Wheel of Change', 'Wheel of Change'),
    ('WINDING_HALLS', 'Winding Halls', 'Winding Halls'),
    ('WORLD_OF_GOOP', 'World of Goop', 'World of Goop'),
]

def _normalize_event_name(name: str) -> str:
    """Strip all non-alphanumerics and casefold, so the live event_id/event_name matches the table
    regardless of spacing/punctuation/case drift -- CommunicationMod sends the Java event id
    ('NoteForYourself', 'Match and Keep!') while the table carries the spaced game name."""
    return "".join(ch for ch in name if ch.isalnum()).casefold()


def _build_event_name_to_enum():
    m = {}
    for enum_name, id_str, game_name in _EVENT_ENUM_NAME_ID_GAME:
        ev = getattr(sts.Event, enum_name)
        m[_normalize_event_name(id_str)] = ev
        m[_normalize_event_name(game_name)] = ev
    # CommunicationMod labels the start-of-run blessing screen "Neow Event"; the engine id is NEOW.
    m[_normalize_event_name("Neow Event")] = sts.Event.NEOW
    return m

_EVENT_NAME_TO_ENUM = _build_event_name_to_enum()

# Events whose option choice depends on which specific player relic/card/potion is offered. The
# engine's setup_event picks those items via the gc's eventRng, which doesn't match the live game's
# pick. Both events we know of (N'loth, We Meet Again) are now reconstructed by injecting the
# live-observed items in net_event_action (_inject_nloth_offers / _inject_wemeetagain), so this set
# is empty -- any future such event would fail loud here until it gets an injector.
_EVENTS_NOT_FAITHFULLY_RECONSTRUCTED = frozenset()


def map_event_to_enum(spire_event_screen) -> "sts.Event":
    """Resolve a spirecomm event screen to the engine Event enum, trying both the id-string
    (screen.event_id) and game-name (screen.event_name). Returns Event.INVALID if unknown so the
    caller can fail loud rather than net-drive an unmapped event."""
    for key in (getattr(spire_event_screen, "event_id", None),
                getattr(spire_event_screen, "event_name", None)):
        if key:
            ev = _EVENT_NAME_TO_ENUM.get(_normalize_event_name(key))
            if ev is not None:
                return ev
    return sts.Event.INVALID


def _is_mini_neow(spire_game) -> bool:
    """True when the live game is showing the 2-option Neow miniBlessing (Neow's Lament / Max HP).
    The real game presents it instead of the 4-option blessing when the previous run beat no boss
    (NeowEvent.bossCount == 0). The GameContext must be built in that mode so its neowRewards and
    option count match the screen, letting net_event_action drive it instead of falling back."""
    if spire_game.screen_type != screen.ScreenType.EVENT:
        return False
    if map_event_to_enum(spire_game.screen) != sts.Event.NEOW:
        return False
    enabled = [o for o in spire_game.screen.options if not o.disabled]
    return len(enabled) == 2


def _inject_nloth_offers(gc, spire_game) -> bool:
    """N'loth offers to take one of two of the player's relics (chosen by an RNG shuffle the live
    snapshot doesn't expose), so setup_event's relicIdx0/relicIdx1 -- rolled from the gc's RNG --
    won't match the live offer. Read the two offered relics off the live option labels (each offer
    option's text names the relic it takes) and point relicIdx0/relicIdx1 at those relics' positions
    in gc.relics, so the net's reasoning (extract_event_info reads gc.relics[relicIdx*]) matches the
    real game. The live options are ordered offer-choice1, offer-choice2, leave -- the same ascending
    idx1 order the engine emits -- so offers[0]->relicIdx0, offers[1]->relicIdx1. Returns True only if
    both offers resolved to held relics."""
    gc_relic_index = {}
    for i, r in enumerate(gc.relics):
        gc_relic_index.setdefault(r.id, i)
    enabled = [o for o in spire_game.screen.options if not o.disabled]
    offers = []
    for opt in enabled:
        text = getattr(opt, "text", "") or ""
        # Longest relic name first so a longer name ("Bag of Marbles") isn't shadowed by a shorter
        # substring of it. The leave option names no relic and is skipped.
        match = None
        for spire_relic in sorted(spire_game.relics, key=lambda r: -len(r.name or "")):
            name = spire_relic.name or ""
            if name and name in text:
                match = spire_relic
                break
        if match is None:
            continue
        rid = map_relic_id(match.name)
        if rid == sts.RelicId.INVALID or rid not in gc_relic_index:
            return False
        offers.append(gc_relic_index[rid])
    if len(offers) < 2:
        return False
    gc.screen_state_info.relicIdx0 = offers[0]
    gc.screen_state_info.relicIdx1 = offers[1]
    return True


def _inject_wemeetagain(gc, spire_game) -> bool:
    """We Meet Again offers a relic back for one of the player's items -- a card, a potion, or gold --
    chosen by the gc's eventRng, which a live snapshot can't match. The engine's option bitmask keys
    on info.potionIdx/gold/cardIdx (each -1 = that offer absent; bit 8 = leave is always present), and
    extract_event_info reads gc.deck[cardIdx] + info.gold. Read the offered items off the live option
    labels and set those fields so both the option count and the net's reasoning match live. Live
    options are ordered potion, gold, card, leave -- the same ascending bit/idx1 order the engine
    emits. Returns True if every ENABLED give-option resolved."""
    import re
    info = gc.screen_state_info
    info.potionIdx = -1
    info.gold = -1
    info.cardIdx = -1
    gc_card_index = {}
    for i, c in enumerate(gc.deck):
        gc_card_index.setdefault(c.id, i)
    real_potions = spire_game.get_real_potions()
    for opt in spire_game.screen.options:
        if opt.disabled:
            continue
        text = getattr(opt, "text", "") or ""
        if "Gold" in text:
            m = re.search(r"Lose\s+(\d+)\s+Gold", text)
            if not m:
                return False
            info.gold = int(m.group(1))
        elif "Card" in text:
            # "[Give Card] Lose <CardName>. Obtain a Relic." -- match the named card to a held deck
            # card by CardId (copies are interchangeable, so first index is fine).
            m = re.search(r"Lose\s+(.+?)\.\s", text)
            if not m:
                return False
            cid = _card_name_to_id(m.group(1).strip(), spire_game)
            if cid is None or cid not in gc_card_index:
                return False
            info.cardIdx = gc_card_index[cid]
        elif "Potion" in text:
            m = re.search(r"Lose\s+(.+?)\.\s", text)
            if not m:
                return False
            pname = m.group(1).strip()
            pidx = next((i for i, p in enumerate(real_potions) if p.name == pname), None)
            if pidx is None:
                return False
            info.potionIdx = pidx
        # the "[Attack]" / leave option names no item
    return info.gold != -1 or info.cardIdx != -1 or info.potionIdx != -1


def _card_name_to_id(name: str, spire_game) -> "sts.CardId | None":
    """Resolve a card display name (from an event option label) to a CardId via the live deck's
    stable card_id, falling back to the normalized-name card table."""
    for c in spire_game.deck:
        if c.name == name:
            cid = map_card_id(c.card_id)
            return cid if cid != sts.CardId.INVALID else None
    cid = map_card_id(name)
    return cid if cid != sts.CardId.INVALID else None


def _normalize_monster_id(monster_id: str) -> str:
    """Casefold + strip ALL whitespace for monster-id matching. spirecomm sometimes sends a
    spaced display form ('Shelled Parasite') where the engine's id is the class name
    ('ShelledParasite'); removing spaces matches them. Underscored variants (SpikeSlime_M vs
    _S) stay distinct since only spaces are removed."""
    return monster_id.replace(" ", "").casefold()


def map_monster_string_to_id(monster_id: str = '') -> sts.MonsterId:
    """Map a spirecomm monster id to a MonsterId enum (synonym-aware, space/case-insensitive).

    Raises on an unknown monster rather than returning INVALID: createMonster(INVALID) aborts the
    process (SIGABRT in initHp), so a clean exception is both safer and debuggable."""
    canonical = _MONSTER_ID_SYNONYMS.get(monster_id, monster_id)
    monster_map = _get_monster_string_to_id_map()
    enum_idx = monster_map.get(_normalize_monster_id(canonical))
    if enum_idx is not None:
        return sts.MonsterId(enum_idx)
    raise ValueError(f"Unknown monster id: {monster_id!r}")


# Create lookup dictionary for monster names
_monster_string_to_id = None

def _get_monster_string_to_id_map():
    """Create normalized-monster-id -> MonsterId mapping dictionary."""
    global _monster_string_to_id
    if _monster_string_to_id is None:
        _monster_string_to_id = {}
        for enum_idx, string_id in sts.getAllMonsterStringIds():
            _monster_string_to_id[_normalize_monster_id(string_id)] = enum_idx
    return _monster_string_to_id


# Every card id reachable in an Ironclad game (cards/{red,colorless,curses,status}/*.java -- the
# only colors obtainable without Prismatic Shard, which is out of scope). Validated at import to
# all resolve through the engine's card-id strings, so a card the engine doesn't model surfaces at
# startup instead of mid-run. (All 133 currently match the engine exactly -- no synonyms needed.)
_IRONCLAD_REACHABLE_CARD_IDS = frozenset({
    'Anger', 'Apotheosis', 'Armaments', 'AscendersBane', 'Bandage Up', 'Barricade',
    'Bash', 'Battle Trance', 'Berserk', 'Bite', 'Blind', 'Blood for Blood',
    'Bloodletting', 'Bludgeon', 'Body Slam', 'Brutality', 'Burn', 'Burning Pact',
    'Carnage', 'Chrysalis', 'Clash', 'Cleave', 'Clothesline', 'Clumsy',
    'Combust', 'Corruption', 'CurseOfTheBell', 'Dark Embrace', 'Dark Shackles', 'Dazed',
    'Decay', 'Deep Breath', 'Defend_R', 'Demon Form', 'Disarm', 'Discovery',
    'Double Tap', 'Doubt', 'Dramatic Entrance', 'Dropkick', 'Dual Wield', 'Enlightenment',
    'Entrench', 'Evolve', 'Exhume', 'Feed', 'Feel No Pain', 'Fiend Fire',
    'Finesse', 'Fire Breathing', 'Flame Barrier', 'Flash of Steel', 'Flex', 'Forethought',
    'Ghostly', 'Ghostly Armor', 'Good Instincts', 'HandOfGreed', 'Havoc', 'Headbutt',
    'Heavy Blade', 'Hemokinesis', 'Immolate', 'Impatience', 'Impervious', 'Infernal Blade',
    'Inflame', 'Injury', 'Intimidate', 'Iron Wave', 'J.A.X.', 'Jack Of All Trades',
    'Juggernaut', 'Limit Break', 'Madness', 'Magnetism', 'Master of Strategy', 'Mayhem',
    'Metallicize', 'Metamorphosis', 'Mind Blast', 'Necronomicurse', 'Normality', 'Offering',
    'Pain', 'Panacea', 'Panache', 'PanicButton', 'Parasite', 'Perfected Strike',
    'Pommel Strike', 'Power Through', 'Pride', 'Pummel', 'Purity', 'Rage',
    'Rampage', 'Reaper', 'Reckless Charge', 'Regret', 'RitualDagger', 'Rupture',
    'Sadistic Nature', 'Searing Blow', 'Second Wind', 'Secret Technique', 'Secret Weapon', 'Seeing Red',
    'Sentinel', 'Sever Soul', 'Shame', 'Shockwave', 'Shrug It Off', 'Slimed',
    'Spot Weakness', 'Strike_R', 'Swift Strike', 'Sword Boomerang', 'The Bomb', 'Thinking Ahead',
    'Thunderclap', 'Transmutation', 'Trip', 'True Grit', 'Twin Strike', 'Uppercut',
    'Violence', 'Void', 'Warcry', 'Whirlwind', 'Wild Strike', 'Wound',
    'Writhe',
})

# Every monster id in the game (monsters/*.java). Each must resolve to an engine MonsterId
# (directly, via _normalize_monster_id, or via _MONSTER_ID_SYNONYMS), be a non-combatant skipped in
# conversion (_MONSTER_IDS_SKIP_IN_COMBAT), or be a non-combat event prop (_MONSTER_IDS_NON_COMBAT).
# Validated at import so an unhandled monster the bot could face fails loud at startup.
_ALL_MONSTER_IDS = frozenset({
    'AcidSlime_L', 'AcidSlime_M', 'AcidSlime_S', 'Apology Slime', 'AwakenedOne', 'BanditBear',
    'BanditChild', 'BanditLeader', 'BookOfStabbing', 'BronzeAutomaton', 'BronzeOrb', 'Byrd',
    'Centurion', 'Champ', 'Chosen', 'CorruptHeart', 'Cultist', 'Dagger',
    'Darkling', 'Deca', 'Donu', 'Exploder', 'FungiBeast', 'FuzzyLouseDefensive',
    'FuzzyLouseNormal', 'GiantHead', 'GremlinFat', 'GremlinLeader', 'GremlinNob', 'GremlinThief',
    'GremlinTsundere', 'GremlinWarrior', 'GremlinWizard', 'Healer', 'Hexaghost', 'HexaghostBody',
    'HexaghostOrb', 'JawWorm', 'Lagavulin', 'Looter', 'Maw', 'Mugger',
    'Nemesis', 'Orb Walker', 'Reptomancer', 'Repulsor', 'Sentry', 'Serpent',
    'Shelled Parasite', 'SlaverBlue', 'SlaverBoss', 'SlaverRed', 'SlimeBoss', 'SnakePlant',
    'Snecko', 'SphericGuardian', 'SpikeSlime_L', 'SpikeSlime_M', 'SpikeSlime_S', 'Spiker',
    'SpireShield', 'SpireSpear', 'TheCollector', 'TheGuardian', 'TimeEater', 'TorchHead',
    'Transient', 'WrithingMass',
})


def _validate_card_and_monster_coverage():
    """Fail loud at import if a card or monster the bot could meet isn't accounted for -- the same
    completeness guarantee the power tables have, so coverage gaps surface at startup not mid-run."""
    bad_cards = sorted(c for c in _IRONCLAD_REACHABLE_CARD_IDS if map_card_id(c) == sts.CardId.INVALID)
    assert not bad_cards, f"engine card table missing Ironclad-reachable cards: {bad_cards}"

    monster_map = _get_monster_string_to_id_map()
    unresolved = []
    for mid in _ALL_MONSTER_IDS:
        if mid in _MONSTER_IDS_SKIP_IN_COMBAT or mid in _MONSTER_IDS_NON_COMBAT:
            continue
        canonical = _MONSTER_ID_SYNONYMS.get(mid, mid)
        if _normalize_monster_id(canonical) not in monster_map:
            unresolved.append(mid)
    assert not unresolved, f"monster ids with no engine mapping/classification: {unresolved}"


_validate_card_and_monster_coverage()


def map_move_id(monster_string: str, move_id: int) -> sts.MonsterMoveId:
    """
    Map spirecomm monster string and move_id to MonsterMoveId enum.
    
    Args:
        monster_string: Monster identifier string (e.g., "Cultist", "JawWorm")
        move_id: Java monster move ID (the byte constants from Java files)
        
    Returns:
        Corresponding MonsterMoveId enum value
    """
    # Create a tuple key for efficient lookup
    key = (monster_string, move_id)
    
    # Comprehensive mapping based on Java source analysis
    move_mapping = {
        # Cultist
        ("Cultist", 1): sts.MonsterMoveId.CULTIST_DARK_STRIKE,
        ("Cultist", 3): sts.MonsterMoveId.CULTIST_INCANTATION,
        
        # JawWorm  
        ("JawWorm", 1): sts.MonsterMoveId.JAW_WORM_CHOMP,
        ("JawWorm", 2): sts.MonsterMoveId.JAW_WORM_BELLOW, 
        ("JawWorm", 3): sts.MonsterMoveId.JAW_WORM_THRASH,
        
        # Red Louse (FuzzyLouseNormal)
        ("FuzzyLouseNormal", 3): sts.MonsterMoveId.RED_LOUSE_BITE,
        ("FuzzyLouseNormal", 4): sts.MonsterMoveId.RED_LOUSE_GROW,
        
        # Green Louse (FuzzyLouseDefensive) 
        ("FuzzyLouseDefensive", 1): sts.MonsterMoveId.GREEN_LOUSE_BITE,
        ("FuzzyLouseDefensive", 2): sts.MonsterMoveId.GREEN_LOUSE_SPIT_WEB,
        
        # Gremlin Nob
        # byte ids from GremlinNob.java: BULL_RUSH=1, SKULL_BASH=2, BELLOW=3
        ("GremlinNob", 1): sts.MonsterMoveId.GREMLIN_NOB_RUSH,
        ("GremlinNob", 2): sts.MonsterMoveId.GREMLIN_NOB_SKULL_BASH,
        ("GremlinNob", 3): sts.MonsterMoveId.GREMLIN_NOB_BELLOW,
        
        # Fat Gremlin
        ("FatGremlin", 1): sts.MonsterMoveId.FAT_GREMLIN_SMASH,
        
        # Mad Gremlin 
        ("MadGremlin", 1): sts.MonsterMoveId.MAD_GREMLIN_SCRATCH,
        
        # Shield Gremlin
        ("ShieldGremlin", 1): sts.MonsterMoveId.SHIELD_GREMLIN_PROTECT,
        ("ShieldGremlin", 2): sts.MonsterMoveId.SHIELD_GREMLIN_SHIELD_BASH,
        
        # Sneaky Gremlin  
        ("SneakyGremlin", 1): sts.MonsterMoveId.SNEAKY_GREMLIN_PUNCTURE,
        
        # Gremlin Wizard
        ("GremlinWizard", 1): sts.MonsterMoveId.GREMLIN_WIZARD_ULTIMATE_BLAST,  # DOPE_MAGIC = 1
        ("GremlinWizard", 2): sts.MonsterMoveId.GREMLIN_WIZARD_CHARGING,        # CHARGE = 2
        ("GremlinWizard", 99): sts.MonsterMoveId.GENERIC_ESCAPE_MOVE,           # Escape
        
        # Slaver (Blue)
        ("SlaverBlue", 1): sts.MonsterMoveId.BLUE_SLAVER_STAB,
        ("SlaverBlue", 4): sts.MonsterMoveId.BLUE_SLAVER_RAKE,  # Gap at 2, 3
        
        # Slaver (Red)
        # byte ids from SlaverRed.java: STAB=1, ENTANGLE=2, SCRAPE=3
        ("SlaverRed", 1): sts.MonsterMoveId.RED_SLAVER_STAB,
        ("SlaverRed", 2): sts.MonsterMoveId.RED_SLAVER_ENTANGLE,
        ("SlaverRed", 3): sts.MonsterMoveId.RED_SLAVER_SCRAPE,
        
        # Lagavulin
        ("Lagavulin", 1): sts.MonsterMoveId.LAGAVULIN_SIPHON_SOUL,  # DEBUFF
        ("Lagavulin", 3): sts.MonsterMoveId.LAGAVULIN_ATTACK,       # STRONG_ATK
        ("Lagavulin", 4): sts.MonsterMoveId.LAGAVULIN_ATTACK,       # OPEN (stun/wake up)
        ("Lagavulin", 5): sts.MonsterMoveId.LAGAVULIN_SLEEP,        # IDLE (sleep)
        ("Lagavulin", 6): sts.MonsterMoveId.LAGAVULIN_ATTACK,       # OPEN_NATURAL
        
        # Sentry
        ("Sentry", 3): sts.MonsterMoveId.SENTRY_BOLT,  # BOLT - adds Dazed cards
        ("Sentry", 4): sts.MonsterMoveId.SENTRY_BEAM,  # BEAM - attack move
        
        # Looter  
        # byte ids from Looter.java: MUG=1, SMOKE_BOMB=2, ESCAPE=3, LUNGE=4
        ("Looter", 1): sts.MonsterMoveId.LOOTER_MUG,
        ("Looter", 2): sts.MonsterMoveId.LOOTER_SMOKE_BOMB,
        ("Looter", 3): sts.MonsterMoveId.LOOTER_ESCAPE,
        ("Looter", 4): sts.MonsterMoveId.LOOTER_LUNGE,
        
        # Fungi Beast
        ("FungiBeast", 1): sts.MonsterMoveId.FUNGI_BEAST_BITE,
        ("FungiBeast", 2): sts.MonsterMoveId.FUNGI_BEAST_GROW,
        
        # Hexaghost (boss) - corrected based on Java constants
        ("Hexaghost", 1): sts.MonsterMoveId.HEXAGHOST_DIVIDER,   # DIVIDER
        ("Hexaghost", 2): sts.MonsterMoveId.HEXAGHOST_TACKLE,    # TACKLE  
        ("Hexaghost", 3): sts.MonsterMoveId.HEXAGHOST_INFLAME,   # INFLAME
        ("Hexaghost", 4): sts.MonsterMoveId.HEXAGHOST_SEAR,      # SEAR
        ("Hexaghost", 5): sts.MonsterMoveId.HEXAGHOST_ACTIVATE,  # ACTIVATE
        ("Hexaghost", 6): sts.MonsterMoveId.HEXAGHOST_INFERNO,   # INFERNO
        
        # SlimeBoss - corrected based on Java constants
        ("SlimeBoss", 1): sts.MonsterMoveId.SLIME_BOSS_SLAM,       # SLAM
        ("SlimeBoss", 2): sts.MonsterMoveId.SLIME_BOSS_PREPARING,  # PREP_SLAM
        ("SlimeBoss", 3): sts.MonsterMoveId.SLIME_BOSS_SPLIT,      # SPLIT
        ("SlimeBoss", 4): sts.MonsterMoveId.SLIME_BOSS_GOOP_SPRAY, # STICKY
        
        # The Guardian (boss) - corrected based on Java constants  
        ("TheGuardian", 1): sts.MonsterMoveId.THE_GUARDIAN_DEFENSIVE_MODE,  # CLOSE_UP
        ("TheGuardian", 2): sts.MonsterMoveId.THE_GUARDIAN_FIERCE_BASH,     # FIERCE_BASH
        ("TheGuardian", 3): sts.MonsterMoveId.THE_GUARDIAN_ROLL_ATTACK,     # ROLL_ATTACK
        ("TheGuardian", 4): sts.MonsterMoveId.THE_GUARDIAN_TWIN_SLAM,       # TWIN_SLAM
        ("TheGuardian", 5): sts.MonsterMoveId.THE_GUARDIAN_WHIRLWIND,       # WHIRLWIND
        ("TheGuardian", 6): sts.MonsterMoveId.THE_GUARDIAN_CHARGING_UP,     # CHARGE_UP
        ("TheGuardian", 7): sts.MonsterMoveId.THE_GUARDIAN_VENT_STEAM,      # VENT_STEAM
        
        # Acid Slimes  
        ("AcidSlime_L", 1): sts.MonsterMoveId.ACID_SLIME_L_CORROSIVE_SPIT,  # SLIME_TACKLE
        ("AcidSlime_L", 2): sts.MonsterMoveId.ACID_SLIME_L_TACKLE,          # NORMAL_TACKLE 
        ("AcidSlime_L", 3): sts.MonsterMoveId.ACID_SLIME_L_SPLIT,           # SPLIT
        ("AcidSlime_L", 4): sts.MonsterMoveId.ACID_SLIME_L_LICK,            # WEAK_LICK
        
        ("AcidSlime_M", 1): sts.MonsterMoveId.ACID_SLIME_M_CORROSIVE_SPIT,  # WOUND_TACKLE
        ("AcidSlime_M", 2): sts.MonsterMoveId.ACID_SLIME_M_TACKLE,          # NORMAL_TACKLE
        ("AcidSlime_M", 4): sts.MonsterMoveId.ACID_SLIME_M_LICK,            # WEAK_LICK
        
        ("AcidSlime_S", 1): sts.MonsterMoveId.ACID_SLIME_S_TACKLE,  # TACKLE
        ("AcidSlime_S", 2): sts.MonsterMoveId.ACID_SLIME_S_LICK,    # DEBUFF (Weak)
        
        # Spike Slimes
        ("SpikeSlime_L", 1): sts.MonsterMoveId.SPIKE_SLIME_L_FLAME_TACKLE,  # FLAME_TACKLE
        ("SpikeSlime_L", 3): sts.MonsterMoveId.SPIKE_SLIME_L_SPLIT,         # SPLIT  
        ("SpikeSlime_L", 4): sts.MonsterMoveId.SPIKE_SLIME_L_LICK,          # FRAIL_LICK
        
        ("SpikeSlime_M", 1): sts.MonsterMoveId.SPIKE_SLIME_M_FLAME_TACKLE,  # FLAME_TACKLE
        ("SpikeSlime_M", 4): sts.MonsterMoveId.SPIKE_SLIME_M_LICK,          # FRAIL_LICK
        
        ("SpikeSlime_S", 1): sts.MonsterMoveId.SPIKE_SLIME_S_TACKLE,        # TACKLE
        
        # Gremlins (additional ones)
        ("GremlinFat", 2): sts.MonsterMoveId.FAT_GREMLIN_SMASH,         # BLUNT
        ("GremlinFat", 99): sts.MonsterMoveId.GENERIC_ESCAPE_MOVE,      # Escape
        
        ("GremlinThief", 1): sts.MonsterMoveId.SNEAKY_GREMLIN_PUNCTURE,  # PUNCTURE
        ("GremlinThief", 99): sts.MonsterMoveId.GENERIC_ESCAPE_MOVE,     # Escape
        
        ("GremlinTsundere", 1): sts.MonsterMoveId.SHIELD_GREMLIN_PROTECT,     # PROTECT 
        ("GremlinTsundere", 2): sts.MonsterMoveId.SHIELD_GREMLIN_SHIELD_BASH, # BASH
        ("GremlinTsundere", 99): sts.MonsterMoveId.GENERIC_ESCAPE_MOVE,        # Escape
        
        ("GremlinWarrior", 1): sts.MonsterMoveId.MAD_GREMLIN_SCRATCH,    # SCRATCH
        ("GremlinWarrior", 99): sts.MonsterMoveId.GENERIC_ESCAPE_MOVE,   # Escape
        
        # Green Louse (FuzzyLouseDefensive) 
        ("FuzzyLouseDefensive", 3): sts.MonsterMoveId.GREEN_LOUSE_BITE,     # BITE
        ("FuzzyLouseDefensive", 4): sts.MonsterMoveId.GREEN_LOUSE_SPIT_WEB, # WEAKEN
        
        # City Monsters
        # Chosen (already seen above)
        ("Chosen", 1): sts.MonsterMoveId.CHOSEN_ZAP,
        ("Chosen", 2): sts.MonsterMoveId.CHOSEN_DRAIN,
        ("Chosen", 3): sts.MonsterMoveId.CHOSEN_DEBILITATE,
        ("Chosen", 4): sts.MonsterMoveId.CHOSEN_HEX,
        ("Chosen", 5): sts.MonsterMoveId.CHOSEN_POKE,
        
        # Byrd 
        ("Byrd", 1): sts.MonsterMoveId.BYRD_PECK,
        ("Byrd", 2): sts.MonsterMoveId.BYRD_FLY,
        ("Byrd", 3): sts.MonsterMoveId.BYRD_SWOOP,
        ("Byrd", 4): sts.MonsterMoveId.BYRD_STUNNED,
        ("Byrd", 5): sts.MonsterMoveId.BYRD_HEADBUTT,
        ("Byrd", 6): sts.MonsterMoveId.BYRD_CAW,
        
        # Bronze Automaton (boss)
        ("BronzeAutomaton", 1): sts.MonsterMoveId.BRONZE_AUTOMATON_FLAIL,
        ("BronzeAutomaton", 2): sts.MonsterMoveId.BRONZE_AUTOMATON_HYPER_BEAM,
        ("BronzeAutomaton", 3): sts.MonsterMoveId.BRONZE_AUTOMATON_STUNNED,
        ("BronzeAutomaton", 4): sts.MonsterMoveId.BRONZE_AUTOMATON_SPAWN_ORBS,
        ("BronzeAutomaton", 5): sts.MonsterMoveId.BRONZE_AUTOMATON_BOOST,
        
        # Bronze Orb
        ("BronzeOrb", 1): sts.MonsterMoveId.BRONZE_ORB_BEAM,
        ("BronzeOrb", 2): sts.MonsterMoveId.BRONZE_ORB_SUPPORT_BEAM,
        ("BronzeOrb", 3): sts.MonsterMoveId.BRONZE_ORB_STASIS,
        
        # Centurion
        ("Centurion", 1): sts.MonsterMoveId.CENTURION_SLASH,
        ("Centurion", 2): sts.MonsterMoveId.CENTURION_DEFEND,
        ("Centurion", 3): sts.MonsterMoveId.CENTURION_FURY,
        
        # The Champ (boss)
        ("Champ", 1): sts.MonsterMoveId.THE_CHAMP_HEAVY_SLASH,
        ("Champ", 2): sts.MonsterMoveId.THE_CHAMP_DEFENSIVE_STANCE,
        ("Champ", 3): sts.MonsterMoveId.THE_CHAMP_EXECUTE,
        ("Champ", 4): sts.MonsterMoveId.THE_CHAMP_FACE_SLAP,
        ("Champ", 5): sts.MonsterMoveId.THE_CHAMP_GLOAT,
        ("Champ", 6): sts.MonsterMoveId.THE_CHAMP_TAUNT,
        ("Champ", 7): sts.MonsterMoveId.THE_CHAMP_ANGER,
        
        # Snecko
        ("Snecko", 1): sts.MonsterMoveId.SNECKO_PERPLEXING_GLARE,
        ("Snecko", 2): sts.MonsterMoveId.SNECKO_BITE,
        ("Snecko", 3): sts.MonsterMoveId.SNECKO_TAIL_WHIP,
        
        # The Collector (boss)
        ("TheCollector", 1): sts.MonsterMoveId.THE_COLLECTOR_SPAWN,
        ("TheCollector", 2): sts.MonsterMoveId.THE_COLLECTOR_FIREBALL,
        ("TheCollector", 3): sts.MonsterMoveId.THE_COLLECTOR_BUFF,
        ("TheCollector", 4): sts.MonsterMoveId.THE_COLLECTOR_MEGA_DEBUFF,
        
        # Shelled Parasite
        ("ShelledParasite", 1): sts.MonsterMoveId.SHELLED_PARASITE_FELL,
        ("ShelledParasite", 2): sts.MonsterMoveId.SHELLED_PARASITE_DOUBLE_STRIKE,
        ("ShelledParasite", 3): sts.MonsterMoveId.SHELLED_PARASITE_SUCK,
        ("ShelledParasite", 4): sts.MonsterMoveId.SHELLED_PARASITE_STUNNED,
        
        # Book Of Stabbing
        ("BookOfStabbing", 1): sts.MonsterMoveId.BOOK_OF_STABBING_SINGLE_STAB,
        ("BookOfStabbing", 2): sts.MonsterMoveId.BOOK_OF_STABBING_MULTI_STAB,
        
        # Healer (Mystic)
        ("Healer", 1): sts.MonsterMoveId.MYSTIC_ATTACK_DEBUFF,
        ("Healer", 2): sts.MonsterMoveId.MYSTIC_HEAL,
        ("Healer", 3): sts.MonsterMoveId.MYSTIC_BUFF,
        
        # Spheric Guardian
        ("SphericGuardian", 1): sts.MonsterMoveId.SPHERIC_GUARDIAN_SLAM,
        ("SphericGuardian", 2): sts.MonsterMoveId.SPHERIC_GUARDIAN_ACTIVATE,
        ("SphericGuardian", 3): sts.MonsterMoveId.SPHERIC_GUARDIAN_HARDEN,
        ("SphericGuardian", 4): sts.MonsterMoveId.SPHERIC_GUARDIAN_ATTACK_DEBUFF,
        
        # Taskmaster
        ("SlaverBoss", 2): sts.MonsterMoveId.TASKMASTER_SCOURING_WHIP,  # Gap at 1
        
        # Torch Head
        ("TorchHead", 1): sts.MonsterMoveId.TORCH_HEAD_TACKLE,
        
        # Snake Plant
        ("SnakePlant", 1): sts.MonsterMoveId.SNAKE_PLANT_CHOMP,
        ("SnakePlant", 2): sts.MonsterMoveId.SNAKE_PLANT_ENFEEBLING_SPORES,
        
        # Mugger
        # byte ids from Mugger.java: MUG=1, SMOKE_BOMB=2, ESCAPE=3, BIGSWIPE(lunge)=4
        ("Mugger", 1): sts.MonsterMoveId.MUGGER_MUG,
        ("Mugger", 2): sts.MonsterMoveId.MUGGER_SMOKE_BOMB,
        ("Mugger", 3): sts.MonsterMoveId.MUGGER_ESCAPE,
        ("Mugger", 4): sts.MonsterMoveId.MUGGER_LUNGE,
        
        # Bandit Bear
        # byte ids from BanditBear.java: MAUL=1, BEAR_HUG=2, LUNGE=3
        ("BanditBear", 1): sts.MonsterMoveId.BEAR_MAUL,
        ("BanditBear", 2): sts.MonsterMoveId.BEAR_BEAR_HUG,
        ("BanditBear", 3): sts.MonsterMoveId.BEAR_LUNGE,
        
        # Bandit Pointy
        ("BanditPointy", 1): sts.MonsterMoveId.POINTY_ATTACK,
        
        # Gremlin Leader (byte ids from GremlinLeader.java: RALLY=2, ENCOURAGE=3, STAB=4)
        ("GremlinLeader", 2): sts.MonsterMoveId.GREMLIN_LEADER_RALLY,
        ("GremlinLeader", 3): sts.MonsterMoveId.GREMLIN_LEADER_ENCOURAGE,
        ("GremlinLeader", 4): sts.MonsterMoveId.GREMLIN_LEADER_STAB,
        
        # Beyond Monsters
        # Awakened One (boss)
        ("AwakenedOne", 1): sts.MonsterMoveId.AWAKENED_ONE_SLASH,
        ("AwakenedOne", 2): sts.MonsterMoveId.AWAKENED_ONE_SOUL_STRIKE,
        ("AwakenedOne", 3): sts.MonsterMoveId.AWAKENED_ONE_REBIRTH,
        ("AwakenedOne", 5): sts.MonsterMoveId.AWAKENED_ONE_DARK_ECHO,
        ("AwakenedOne", 6): sts.MonsterMoveId.AWAKENED_ONE_SLUDGE,
        ("AwakenedOne", 8): sts.MonsterMoveId.AWAKENED_ONE_TACKLE,
        
        # Time Eater (boss)
        ("TimeEater", 2): sts.MonsterMoveId.TIME_EATER_REVERBERATE,
        ("TimeEater", 3): sts.MonsterMoveId.TIME_EATER_RIPPLE,
        ("TimeEater", 4): sts.MonsterMoveId.TIME_EATER_HEAD_SLAM,
        ("TimeEater", 5): sts.MonsterMoveId.TIME_EATER_HASTE,
        
        # Donu 
        ("Donu", 0): sts.MonsterMoveId.DONU_BEAM,
        ("Donu", 2): sts.MonsterMoveId.DONU_CIRCLE_OF_POWER,
        
        # Deca
        ("Deca", 0): sts.MonsterMoveId.DECA_BEAM,
        ("Deca", 2): sts.MonsterMoveId.DECA_SQUARE_OF_PROTECTION,
        
        # Darkling  
        ("Darkling", 1): sts.MonsterMoveId.DARKLING_CHOMP,
        ("Darkling", 2): sts.MonsterMoveId.DARKLING_HARDEN,
        ("Darkling", 3): sts.MonsterMoveId.DARKLING_NIP,
        ("Darkling", 4): sts.MonsterMoveId.DARKLING_REGROW,      # COUNT = 4
        ("Darkling", 5): sts.MonsterMoveId.DARKLING_REINCARNATE,
        
        # Repulsor
        ("Repulsor", 1): sts.MonsterMoveId.REPULSOR_REPULSE,
        ("Repulsor", 2): sts.MonsterMoveId.REPULSOR_BASH,
        
        # Exploder
        ("Exploder", 1): sts.MonsterMoveId.EXPLODER_SLAM,
        ("Exploder", 2): sts.MonsterMoveId.EXPLODER_EXPLODE,
        
        # Writhing Mass
        ("WrithingMass", 0): sts.MonsterMoveId.WRITHING_MASS_STRONG_STRIKE,
        ("WrithingMass", 1): sts.MonsterMoveId.WRITHING_MASS_MULTI_STRIKE,
        ("WrithingMass", 2): sts.MonsterMoveId.WRITHING_MASS_WITHER,
        ("WrithingMass", 3): sts.MonsterMoveId.WRITHING_MASS_FLAIL,
        ("WrithingMass", 4): sts.MonsterMoveId.WRITHING_MASS_IMPLANT,
        
        # Nemesis
        ("Nemesis", 2): sts.MonsterMoveId.NEMESIS_ATTACK,
        ("Nemesis", 3): sts.MonsterMoveId.NEMESIS_SCYTHE,
        ("Nemesis", 4): sts.MonsterMoveId.NEMESIS_DEBUFF,
        
        # Reptomancer
        ("Reptomancer", 1): sts.MonsterMoveId.REPTOMANCER_SNAKE_STRIKE,
        ("Reptomancer", 2): sts.MonsterMoveId.REPTOMANCER_SUMMON,
        ("Reptomancer", 3): sts.MonsterMoveId.REPTOMANCER_BIG_BITE,
        
        # Snake Dagger
        ("SnakeDagger", 1): sts.MonsterMoveId.DAGGER_STAB,
        ("SnakeDagger", 2): sts.MonsterMoveId.DAGGER_EXPLODE,
        
        # Spiker
        ("Spiker", 1): sts.MonsterMoveId.SPIKER_CUT,
        ("Spiker", 2): sts.MonsterMoveId.SPIKER_SPIKE,
        
        # Transient
        ("Transient", 1): sts.MonsterMoveId.TRANSIENT_ATTACK,
        
        # Orb Walker
        ("OrbWalker", 1): sts.MonsterMoveId.ORB_WALKER_LASER,
        ("OrbWalker", 2): sts.MonsterMoveId.ORB_WALKER_CLAW,
        
        # Giant Head
        ("GiantHead", 1): sts.MonsterMoveId.GIANT_HEAD_GLARE,        # GLARE = 1
        ("GiantHead", 2): sts.MonsterMoveId.GIANT_HEAD_IT_IS_TIME,   # IT_IS_TIME = 2
        ("GiantHead", 3): sts.MonsterMoveId.GIANT_HEAD_COUNT,        # COUNT = 3
        
        # Maw
        ("Maw", 2): sts.MonsterMoveId.THE_MAW_ROAR,     # ROAR = 2 (gap at 0, 1)
        ("Maw", 3): sts.MonsterMoveId.THE_MAW_SLAM,     # SLAM = 3
        ("Maw", 4): sts.MonsterMoveId.THE_MAW_DROOL,    # DROOL = 4
        ("Maw", 5): sts.MonsterMoveId.THE_MAW_NOM,      # NOMNOMNOM = 5
        
        # Spire Growth
        # byte ids from SpireGrowth.java: QUICK_TACKLE=1, CONSTRICT=2, SMASH=3
        ("SpireGrowth", 1): sts.MonsterMoveId.SPIRE_GROWTH_QUICK_TACKLE,
        ("SpireGrowth", 2): sts.MonsterMoveId.SPIRE_GROWTH_CONSTRICT,
        ("SpireGrowth", 3): sts.MonsterMoveId.SPIRE_GROWTH_SMASH,
        
        # Ending Monsters
        # Corrupt Heart (final boss)
        ("CorruptHeart", 1): sts.MonsterMoveId.CORRUPT_HEART_BLOOD_SHOTS,
        ("CorruptHeart", 2): sts.MonsterMoveId.CORRUPT_HEART_ECHO,
        ("CorruptHeart", 3): sts.MonsterMoveId.CORRUPT_HEART_DEBILITATE,
        ("CorruptHeart", 4): sts.MonsterMoveId.CORRUPT_HEART_BUFF,
        
        # Spire Shield
        ("SpireShield", 1): sts.MonsterMoveId.SPIRE_SHIELD_BASH,
        ("SpireShield", 2): sts.MonsterMoveId.SPIRE_SHIELD_FORTIFY,
        ("SpireShield", 3): sts.MonsterMoveId.SPIRE_SHIELD_SMASH,
        
        # Spire Spear
        ("SpireSpear", 1): sts.MonsterMoveId.SPIRE_SPEAR_BURN_STRIKE,
        ("SpireSpear", 2): sts.MonsterMoveId.SPIRE_SPEAR_PIERCER,
        ("SpireSpear", 3): sts.MonsterMoveId.SPIRE_SPEAR_SKEWER,
    }
    
    # Look up the move. The mapping keys use the engine class names, which are space-free; some live
    # ids are the spaced display form ("Shelled Parasite", "Orb Walker") whose engine name just drops
    # the spaces. Try the raw id first (it preserves names the map intentionally keeps in live form,
    # e.g. "TheCollector"/"Maw"/"BanditBear"), then the space-stripped form.
    move_id_enum = move_mapping.get(key)
    if move_id_enum is not None:
        return move_id_enum
    move_id_enum = move_mapping.get((monster_string.replace(" ", ""), move_id))
    if move_id_enum is not None:
        return move_id_enum

    print(f"Warning: Unknown monster move mapping for '{monster_string}' move_id={move_id}, using INVALID", file=sys.stderr)
    return sts.MonsterMoveId.INVALID


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
        # Boss reward screen: inject the three offered boss relics. set_boss_relic writes the
        # live array (the boss_relics property returns a copy, so `[i] =` would no-op).
        for i, spire_relic in enumerate(spire_game.screen.relics[:3]):
            relic_id = map_relic_id(spire_relic.name)
            if relic_id == sts.RelicId.INVALID:
                raise ValueError(f"Unknown boss relic: {spire_relic.name}")
            info.set_boss_relic(i, relic_id)
                
    elif spire_game.screen_type == screen.ScreenType.COMBAT_REWARD:
        # Combat reward screen. Gold/relic/potion/keys are known and injected here. A CARD
        # reward is opaque on this screen -- spirecomm's CombatReward carries no card data; the
        # three cards are only revealed on the CARD_REWARD screen -- so we do NOT inject a card
        # group here. The network makes the card decision when it reaches CARD_REWARD.
        rc = info.rewards_container
        rc.clear()
        for reward_item in spire_game.screen.rewards:
            rtype = reward_item.reward_type
            if rtype in (screen.RewardType.GOLD, screen.RewardType.STOLEN_GOLD):
                rc.add_gold(reward_item.gold)
            elif rtype == screen.RewardType.RELIC:
                relic_id = map_relic_id(reward_item.relic.name)
                if relic_id == sts.RelicId.INVALID:
                    raise ValueError(f"Unknown relic in combat reward: {reward_item.relic.name}")
                rc.add_relic(relic_id)
            elif rtype == screen.RewardType.POTION:
                rc.add_potion(map_potion_id(reward_item.potion.potion_id))
            elif rtype == screen.RewardType.EMERALD_KEY:
                rc.emerald_key = True
            elif rtype == screen.RewardType.SAPPHIRE_KEY:
                rc.sapphire_key = True
            # CARD (and other opaque markers): cards not revealed yet; handled at CARD_REWARD.
                
    elif spire_game.screen_type == screen.ScreenType.SHOP_SCREEN:
        # Shop screen: reconstruct the merchant's stock into the engine Shop so getAllActionsInState
        # offers exactly the affordable buys (it checks price != -1 and gold >= price). The Shop
        # getters return copies, so we go through the set_* mutators with the live prices.
        shop_screen = spire_game.screen
        shop = info.shop
        shop.clear()
        for i, shop_card in enumerate(shop_screen.cards[:7]):
            card_id = map_card_id(shop_card.card_id)
            if card_id == sts.CardId.INVALID:
                raise ValueError(f"Unknown shop card: {shop_card.card_id}")
            shop.set_card(i, sts.Card(card_id, shop_card.upgrades), shop_card.price)
        for i, shop_relic in enumerate(shop_screen.relics[:3]):
            relic_id = map_relic_id(shop_relic.name)
            if relic_id == sts.RelicId.INVALID:
                raise ValueError(f"Unknown shop relic: {shop_relic.name}")
            shop.set_relic(i, relic_id, shop_relic.price)
        for i, shop_potion in enumerate(shop_screen.potions[:3]):
            shop.set_potion(i, map_potion_id(shop_potion.potion_id), shop_potion.price)
        if shop_screen.purge_available:
            shop.set_remove_cost(shop_screen.purge_cost)
            
    elif spire_game.screen_type == screen.ScreenType.EVENT:
        # Put the GameContext into the live event's choice state so getAllActionsInState offers the
        # event's options and the NN (construct_choice) can encode them. setup_event regenerates the
        # event's info fields from the gc's RNG; for the start-of-run NEOW the constructor already
        # rolled info.neowRewards. cur_event drives setup_event's per-event branch. Unknown events
        # leave the gc as-is (the net handler fails loud).
        ev = map_event_to_enum(spire_game.screen)
        if ev != sts.Event.INVALID:
            gc.cur_event = ev
            gc.screen_state = sts.ScreenState.EVENT_SCREEN
            gc.setup_event()
            if ev == sts.Event.COLOSSEUM:
                # Colosseum is a multi-phase combat event the engine doesn't simulate forward (the
                # live game runs the fights). Its option count is phase-keyed on eventData: the
                # intro/fight phase has 1 option, the post-combat phase (leave / fight the Nobs) has
                # 2. Reconstruct the phase from the live screen so getValidEventSelectBits matches and
                # the net drives the post-combat choice (idx1 0 = leave, 1 = fight the Nobs).
                enabled = [o for o in spire_game.screen.options if not o.disabled]
                gc.screen_state_info.event_data = 0 if len(enabled) <= 1 else 1

    elif spire_game.screen_type == screen.ScreenType.GRID:
        # Grid select screen (transform/upgrade/remove/obtain). The engine builds one select action
        # per to_select_card; the getters return copies, so we go through the add_* mutators. Order
        # is preserved so to_select_cards[i] == grid_screen.cards[i] for translating the pick back.
        grid_screen = spire_game.screen
        if grid_screen.for_transform:
            info.select_screen_type = sts.CardSelectScreenType.TRANSFORM
        elif grid_screen.for_upgrade:
            info.select_screen_type = sts.CardSelectScreenType.UPGRADE
        elif grid_screen.for_purge:
            info.select_screen_type = sts.CardSelectScreenType.REMOVE
        else:
            info.select_screen_type = sts.CardSelectScreenType.OBTAIN
        info.to_select_count = grid_screen.num_cards

        info.clear_to_select_cards()
        for i, grid_card in enumerate(grid_screen.cards):
            card_id = map_card_id(grid_card.card_id)
            if card_id == sts.CardId.INVALID:
                raise ValueError(f"Unknown card in grid select: {grid_card.card_id}")
            info.add_to_select_card(sts.Card(card_id, grid_card.upgrades), i)

        info.clear_have_selected_cards()
        for sel_card in grid_screen.selected_cards:
            card_id = map_card_id(sel_card.card_id)
            if card_id == sts.CardId.INVALID:
                raise ValueError(f"Unknown selected card: {sel_card.card_id}")
            info.add_have_selected_card(sts.Card(card_id, sel_card.upgrades))

    elif spire_game.screen_type == screen.ScreenType.HAND_SELECT:
        # Hand select screen (in-combat: Warcry/Headbutt/etc.). Same to_select_cards reconstruction.
        hand_screen = spire_game.screen
        info.select_screen_type = sts.CardSelectScreenType.DUPLICATE
        info.to_select_count = hand_screen.num_cards
        info.clear_to_select_cards()
        for i, hand_card in enumerate(hand_screen.cards):
            card_id = map_card_id(hand_card.card_id)
            if card_id == sts.CardId.INVALID:
                raise ValueError(f"Unknown card in hand select: {hand_card.card_id}")
            info.add_to_select_card(sts.Card(card_id, hand_card.upgrades), i)
                    
    elif spire_game.screen_type == screen.ScreenType.CARD_REWARD:
        # Card reward screen: the offered cards are revealed here, so inject them as a single
        # reward group. getAllActionsInState/construct_choice read them from
        # rewards_container.cards[group][i] (a REWARDS screen); to_select_cards is for the
        # CARD_SELECT screens (transform/upgrade/remove), not card rewards.
        rc = info.rewards_container
        rc.clear()
        offered = []
        for spire_card in spire_game.screen.cards:
            card_id = map_card_id(spire_card.card_id)
            if card_id == sts.CardId.INVALID:
                raise ValueError(f"Unknown card in card reward: {spire_card.card_id}")
            offered.append(sts.Card(card_id, spire_card.upgrades))
        rc.add_card_reward(offered)
                    
    elif spire_game.screen_type == screen.ScreenType.REST:
        # The C++ code knows what rest options are available
        pass

    elif spire_game.screen_type == screen.ScreenType.MAP:
        # The GameContext regenerates this seed's map (RNG-accurate, so it matches the live map);
        # we only need to place the player on their current node so getAllActionsInState offers the
        # real next-row choices. Valid in-act rows are 0..14. At game start current_node.y is -1, and
        # at the START of a new act the live game reports the just-cleared boss as y=15 (x=-1); both
        # must leave curMapNodeY at the engine's default -1, which yields the act's first row.
        # Copying y=15 would index a 15-row map array out of bounds (getNode -> array::at(15)).
        cur = spire_game.screen.current_node
        if cur is not None and 0 <= cur.y <= 14:
            gc.cur_map_node_x = cur.x
            gc.cur_map_node_y = cur.y

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
    gc = sts.GameContext(character_class, int(spire_game.seed), int(spire_game.ascension_level or 0),
                         _is_mini_neow(spire_game))
    
    # Set basic game state
    gc.cur_hp = spire_game.current_hp
    gc.max_hp = spire_game.max_hp
    gc.gold = spire_game.gold
    gc.act = spire_game.act
    gc.floor_num = spire_game.floor

    # The GameContext constructor builds the act-1 map. Regenerate it for the live act so map
    # navigation (getAllActionsInState path choices) and the NN's map features match the real game;
    # an act-1 map left in place yields zero/wrong next-node actions on every act-2+ map screen and
    # corrupts the map features the net sees for all act-2+ decisions. transitionToAct() also does
    # this but heals the player and advances RNG -- side effects we must not apply to a state we are
    # only reconstructing. assignBurningElite mirrors transitionToAct's !hasKey(EMERALD_KEY): a
    # reconstructed gc holds no keys, and the flag affects only which elite is burning, not topology.
    if spire_game.act == 2 or spire_game.act == 3:
        # SpireMap's seed arg is uint64_t; the Java/spirecomm seed is signed and is often negative,
        # which pybind11 refuses to convert. Pass its unsigned 64-bit bit pattern -- the same value
        # GameContext stores internally -- so the regenerated map matches the engine's own map.
        seed_u64 = int(spire_game.seed) & 0xFFFFFFFFFFFFFFFF
        gc.map = sts.SpireMap(seed_u64, int(spire_game.ascension_level or 0),
                              int(spire_game.act), True)
    elif spire_game.act >= 4:
        gc.map = sts.SpireMap.act4()


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

    # Set the potion belt (capacity + held). Without this the gc has an empty belt, so the sim
    # offers a buy-potion action in the shop even when the live belt is full -- the net then picks
    # it and the live game rejects it ("potion slots are full"). It also lets the net see the real
    # potion state for every out-of-combat decision. Mirrors convert_combat_state.
    if spire_game.potions:
        gc.potion_capacity = len(spire_game.potions)
    for spire_potion in spire_game.get_real_potions():
        gc.obtain_potion(map_potion_id(spire_potion.potion_id))

    # Set screen state
    gc.screen_state = map_screen_state(spire_game)
    
    # Set screen state info based on current screen
    set_screen_state_info(gc, spire_game)
    
    return gc


def _sim_target_to_spire_index(target_idx: int, slot_to_spire: dict) -> int:
    """Translate a searcher monster-target (a sim MonsterGroup slot) back to the spirecomm
    monster_index the live game targets by. The sim group is repacked (reserved gaps after
    splittable monsters), so slot != spire index in general. Returns the raw target_idx when there
    is no target (<0). Raises if the search targeted a slot with no live monster -- that would mean
    it aimed at a reserved gap, which it must never do."""
    if target_idx < 0:
        return target_idx
    if target_idx not in slot_to_spire:
        raise ValueError(f"Search targeted sim slot {target_idx} with no spirecomm monster "
                         f"(slot_to_spire={slot_to_spire})")
    return slot_to_spire[target_idx]


def map_search_action_to_spirecomm(action: "sts.Action", bc: "sts.BattleContext", game: game.Game,
                                   slot_to_spire: dict) -> "Action":
    """
    Map a sim/search/Action to a spirecomm Action.

    Args:
        action: The search Action from BattleSearcher
        bc: The BattleContext for reference
        game: The spirecomm Game state for monster/card references
        slot_to_spire: sim MonsterGroup slot -> spirecomm monster_index (from convert_combat_state)

    Returns:
        Corresponding spirecomm Action object
    """
    action_type = action.get_action_type()

    if action_type == sts.ActionType.CARD:
        # Only a targeted card carries a real monster target. For a non-targeted card the search's
        # target_idx is a meaningless default (often 0) and must not be mapped: with reserved/empty
        # monster slots (the summoner layouts) slot 0 holds no live monster, so mapping it would
        # raise. target_index=None tells the live game to play the card untargeted.
        card_index = action.get_source_idx()
        if 0 <= card_index < len(game.hand) and game.hand[card_index].has_target:
            target_idx = _sim_target_to_spire_index(action.get_target_idx(), slot_to_spire)
        else:
            target_idx = None
        return PlayCardAction(card_index=card_index, target_index=target_idx)

    elif action_type == sts.ActionType.POTION:
        # Using a potion - only a target-requiring potion carries a monster target (see CARD above).
        potion_idx = action.get_source_idx()
        potions = game.get_real_potions()
        if not (0 <= potion_idx < len(potions)):
            raise ValueError(f"Invalid potion index: {potion_idx}")
        potion = potions[potion_idx]
        # The engine encodes a potion DISCARD as target_idx > 5 (Action.cpp): a passive Fairy in a
        # Bottle (never drinkable -- it auto-revives on lethal), or a target-requiring potion with no
        # valid target. Send a discard, not a use, or the live game rejects "potion use" with
        # "Selected potion cannot be used."
        if action.get_target_idx() > 5:
            return PotionAction(False, potion=potion)
        target_idx = (_sim_target_to_spire_index(action.get_target_idx(), slot_to_spire)
                      if potion.requires_target else None)
        return PotionAction(True, potion=potion, target_index=target_idx)
            
    elif action_type == sts.ActionType.END_TURN:
        return EndTurnAction()
        
    elif action_type == sts.ActionType.SINGLE_CARD_SELECT:
        # Card selection action (for cases like Warcry, etc.)
        select_idx = action.get_select_idx()
        # This would need more context to handle properly
        # For now, return a generic choice action
        return ChooseAction(select_idx)
        
    elif action_type == sts.ActionType.MULTI_CARD_SELECT:
        # Multiple card selection (like for Dual Wield, etc.)
        selected_idxs = action.get_selected_idxs()
        # Convert to list and return as card select action
        # This would need proper implementation based on the specific scenario
        return ChooseAction(0)  # Placeholder
        
    else:
        raise ValueError(f"Unknown action type: {action_type}")


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


# In-combat card-select: the live game_state names the resolving Java AbstractGameAction in
# `current_action`; map it to the engine CardSelectTask so the combat MCTS resolves the pick from
# the reconstructed piles. Names confirmed against live captures where noted; the rest are the
# decompiled card actions (com/megacrit/cardcrawl/cards/**). An unmapped action fails loud (logging
# the name) so this table is filled from real games rather than guessed.
_CARD_SELECT_TASK_BY_ACTION = {
    "ArmamentsAction": sts.CardSelectTask.ARMAMENTS,                # Armaments (confirmed live)
    "DiscardPileToTopOfDeckAction": sts.CardSelectTask.HEADBUTT,    # Headbutt (confirmed live)
    "BetterDiscardPileToHandAction": sts.CardSelectTask.HEADBUTT,   # discard-retrieval (confirmed live)
    "DualWieldAction": sts.CardSelectTask.DUAL_WIELD,               # Dual Wield
    "ExhumeAction": sts.CardSelectTask.EXHUME,                      # Exhume
    "ForethoughtAction": sts.CardSelectTask.FORETHOUGHT,           # Forethought
    "PutOnDeckAction": sts.CardSelectTask.WARCRY,                   # Warcry (a hand card to draw top)
    "ExhaustAction": sts.CardSelectTask.EXHAUST_ONE,               # True Grit+ etc. (exhaust a chosen hand card)
    "DiscoveryAction": sts.CardSelectTask.DISCOVERY,               # Discovery / Attack-Skill-Power Potion (confirmed live)
    "ChooseOneColorless": sts.CardSelectTask.DISCOVERY,           # Colorless Potion: 1 of 3 colorless (confirmed live)
}

# Tasks whose candidates are freshly-generated cards offered on the screen (not drawn from a pile):
# the offered cards are injected into the select and the chosen index maps straight back to them.
_DISCOVERY_TASKS = frozenset({sts.CardSelectTask.DISCOVERY})


# StS seed alphabet (base-35, no 'O'); matches SeedHelper.CHARACTERS.
_SEED_CHARS = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"


def seed_long_to_string(seed: int) -> str:
    """Convert a numeric game seed (the live game_state's `seed`, a signed int64) to the base-35
    string the game's `start`/`--seed` command expects -- mirrors SeedHelper.getString (unsigned
    base-35). Lets a captured game be replayed deterministically with `comm.py --seed <string>`."""
    n = seed & 0xFFFFFFFFFFFFFFFF  # interpret as unsigned 64-bit, like Long.toUnsignedString
    if n == 0:
        return "0"
    out = []
    while n:
        out.append(_SEED_CHARS[n % 35])
        n //= 35
    return "".join(reversed(out))

# Engine pile a task's getSelectIdx() indexes, used to translate the search's chosen index back to
# the live screen card. Mirrors isValidSingleCardSelectAction's per-task pile (Action.cpp).
_CARD_SELECT_POOL_BY_TASK = {
    sts.CardSelectTask.ARMAMENTS: "hand",
    sts.CardSelectTask.DUAL_WIELD: "hand",
    sts.CardSelectTask.FORETHOUGHT: "hand",
    sts.CardSelectTask.WARCRY: "hand",
    sts.CardSelectTask.EXHAUST_ONE: "hand",
    sts.CardSelectTask.HEADBUTT: "discard",
    sts.CardSelectTask.HOLOGRAM: "discard",
    sts.CardSelectTask.MEDITATE: "discard",
    sts.CardSelectTask.EXHUME: "exhaust",
    sts.CardSelectTask.SEEK: "draw",
    sts.CardSelectTask.SECRET_TECHNIQUE: "draw",
    sts.CardSelectTask.SECRET_WEAPON: "draw",
}


class STSLightspeedAgent:

    def __init__(self, chosen_class=PlayerClass.THE_SILENT, net=None, temperature=0.0, net_seed=0,
                 start_seed=None):
        self.game = Game()
        self.errors = 0
        # When set (a base-35 StS seed string, e.g. "54FYPZX13RLTT"), new runs start on this exact
        # seed -- used to replay a specific game (e.g. the captured slime-boss crash seed).
        self.start_seed = start_seed
        # Set when heart1 skips the combat card reward, so _collect_combat_reward doesn't re-open it.
        self.skipped_cards = False
        # Toggles the two-step SHOP_ROOM transition (approach merchant, then leave).
        self.visited_shop = False
        self.chosen_class = chosen_class
        self.change_class(chosen_class)
        self.choice_count = 0
        # Set by run_agent_cli so out-of-combat decision states can be captured for replay.
        self.coordinator = None
        # heart1 policy (an NNService) driving every out-of-combat choice; required (no heuristic
        # fallback). temperature<=0 picks greedily (argmax); >0 samples (Boltzmann) with net_rng.
        self.net = net
        self.temperature = temperature
        self.net_rng = random.Random(net_seed)
        # Reference SearchAgent whose tuned knobs configure each per-decision BattleSearcher.
        # simulation_count_base=1000 matches training/eval (run_game / --mcts-simulations 1000);
        # the agent's defaults supply the jointly-tuned exploration/widening/eval weights.
        self.search_agent = sts.Agent()
        self.search_agent.simulation_count_base = 1000

    def change_class(self, new_class):
        self.chosen_class = new_class

    def _log_seed_once(self):
        """Print the replayable base-35 seed string the first time we see each game, so any later
        crash leaves the exact seed to deterministically reproduce it with `comm.py --seed <s>`."""
        seed = getattr(self.game, "seed", None)
        if seed is None or seed == getattr(self, "_logged_seed", None):
            return
        self._logged_seed = seed
        print(f"[seed] game seed {seed} = {seed_long_to_string(int(seed))!r} "
              f"(replay: --seed {seed_long_to_string(int(seed))})", file=sys.stderr)

    def handle_error(self, error):
        raise Exception(error)

    def get_next_action_in_game(self, game_state):
        self.choice_count += 1
        self.game = game_state
        self._log_seed_once()
        if self.game.choice_available:
            # nchoice = min(4, len(self.game.choice_list))
            # if self.choice_count < 6:
            #     time.sleep(3 * nchoice)
            # else:
            #     time.sleep(0.5 * nchoice)
            return self.handle_screen()
        if self.game.proceed_available:
            return ProceedAction()
        if self.game.play_available:
            return self.handle_combat()
        if self.game.end_available:
            # time.sleep(4)
            return EndTurnAction()
        if self.game.cancel_available:
            return CancelAction()

    def get_next_action_out_of_game(self):
        return StartGameAction(self.chosen_class, seed=self.start_seed)

    def handle_combat(self):
        self.capture_battle_state()
        # Convert spirecomm game state to our internal format
        gc = spirecomm_to_gamecontext(self.game)
        bc, slot_to_spire = convert_combat_state(self.game, gc)
        print(bc, file=sys.stderr)

        # Configure the searcher with heart1's exact training/eval battle-search knobs
        # (exploration / chance + end-turn widening / eval weights, boss variants) and matching
        # per-decision sim count, via the shared SearchAgent config -- so live play uses the same
        # search heart1 was tuned around rather than a mistuned standalone BattleSearcher.
        searcher = sts.BattleSearcher(bc)
        simulation_count = self.search_agent.configure_searcher(searcher, bc)

        print("=" * 80, file=sys.stderr)
        print(f"Running {simulation_count} simulations for combat decision...", file=sys.stderr)

        # Get the best action (most visited child of root). A RuntimeError here is a C++ battle-
        # search throw on this converted state -- e.g. a splitting monster (Slimes) overflowing the
        # 5-slot MonsterGroup, a known conversion edge case. This is NOT a silent swallow: dump the
        # full crashing state (stderr + runs/battle_search_crashes.jsonl) for root-causing, then
        # end the turn so one bad fight doesn't kill the whole run.
        try:
            searcher.search(simulation_count)
            first_action = searcher.get_best_action()
        except Exception as e:
            print(f"!!! BATTLE SEARCH CRASH ({type(e).__name__}: {e}) -- ending turn; state dumped",
                  file=sys.stderr)
            print(bc, file=sys.stderr)
            try:
                crash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "runs", "battle_search_crashes.jsonl")
                with open(crash_path, "a") as f:
                    raw = self.coordinator.last_raw_communication_state if self.coordinator else None
                    f.write(json.dumps({"error": str(e), "raw": raw}) + "\n")
            except Exception:
                pass
            return EndTurnAction()

        # Map the search action to a spirecomm action
        spirecomm_action = map_search_action_to_spirecomm(first_action, bc, self.game, slot_to_spire)

        print(f"Chosen action: {spirecomm_action}", file=sys.stderr)

        # Print top 5 moves and their visit counts
        edges = searcher.get_root_edges()
        if edges:
            # Sort edges by visit count (descending)
            sorted_edges = sorted(edges, key=lambda e: e.node.simulation_count, reverse=True)
            print("Top 5 moves by visit count:", file=sys.stderr)
            for i, edge in enumerate(sorted_edges[:5]):
                action_desc = edge.action.print_desc(bc)
                visits = edge.node.simulation_count
                avg_value = edge.node.evaluation_sum / visits if visits > 0 else 0
                print(f"  {i+1}. {action_desc} - visits: {visits}, avg_value: {avg_value:.2f}", file=sys.stderr)

        return spirecomm_action

    def mcts_card_select_action(self):
        """Resolve an in-combat card-select (Armaments/Headbutt/Warcry/Dual Wield/Exhume/...) with
        the combat MCTS -- the same way the search resolves it in-sim. Reconstruct the bc at the
        mid-resolution state (the live piles already reflect the triggering card being played), put
        it into the CARD_SELECT input state for that action's task, search, and translate the chosen
        pile index back to the live screen card. Fails loud on an unmapped action or a select the
        search can't place on the live screen."""
        action_name = self.game.current_action
        task = _CARD_SELECT_TASK_BY_ACTION.get(action_name)
        if task is None:
            cards = [c.name for c in self.game.screen.cards]
            raise NotImplementedError(
                f"in-combat card-select current_action {action_name!r} unmapped "
                f"(screen {self.game.screen_type}, {len(cards)} cards: {cards}); "
                f"add it to _CARD_SELECT_TASK_BY_ACTION")
        # CardRewardScreen (the in-combat Discovery/potion choice) has no num_cards; it always picks 1.
        num = getattr(self.game.screen, "num_cards", None) or 1
        if num != 1:
            raise NotImplementedError(
                f"multi-card in-combat select (num_cards={num}, {action_name!r}) not yet supported")
        offered = self.game.screen.cards

        gc = spirecomm_to_gamecontext(self.game)
        bc, _ = convert_combat_state(self.game, gc)

        if task in _DISCOVERY_TASKS:
            # Generated-card choice: the candidates are the offered cards themselves. Inject them and
            # let the search pick; the chosen index maps straight back to the live screen card.
            ids = []
            for c in offered:
                cid = map_card_id(c.card_id)
                if cid == sts.CardId.INVALID:
                    raise ValueError(f"unknown offered card in discovery select: {c.card_id}")
                ids.append(cid)
            bc.open_discovery_select(ids, 1, True)
            searcher = sts.BattleSearcher(bc)
            searcher.search(self.search_agent.configure_searcher(searcher, bc))
            sel_idx = searcher.get_best_action().get_select_idx()
            if not (0 <= sel_idx < len(offered)):
                raise RuntimeError(f"MCTS discovery idx {sel_idx} out of range "
                                   f"({len(offered)} offered, {action_name})")
            chosen = offered[sel_idx]
            print(f"[mcts] discovery ({action_name}) -> {chosen.card_id} (idx {sel_idx})",
                  file=sys.stderr)
            # A Discovery/potion choice is delivered on a CARD_REWARD screen, where the pick is a
            # "choose <index>" command (CardSelectAction only works on HAND_SELECT/GRID). The
            # choice_list / screen.cards order matches sel_idx.
            if self.game.screen_type == ScreenType.CARD_REWARD:
                return ChooseAction(sel_idx)
            return CardSelectAction([chosen])

        bc.open_card_select(task, num)
        searcher = sts.BattleSearcher(bc)
        searcher.search(self.search_agent.configure_searcher(searcher, bc))
        sel_idx = searcher.get_best_action().get_select_idx()

        pool_name = _CARD_SELECT_POOL_BY_TASK[task]
        pool = {"hand": bc.cards.hand, "discard": bc.cards.discardPile,
                "exhaust": bc.cards.exhaustPile, "draw": bc.cards.drawPile}[pool_name]
        if not (0 <= sel_idx < len(pool)):
            raise RuntimeError(f"MCTS card-select idx {sel_idx} out of range for the {pool_name} "
                               f"pile (size {len(pool)}, task {task})")
        chosen = pool[sel_idx]
        live_card = self._match_live_select_card(chosen)
        print(f"[mcts] card-select ({action_name}) -> {chosen.getName()}"
              f"{'+' if chosen.upgraded else ''} ({pool_name} idx {sel_idx})", file=sys.stderr)
        return CardSelectAction([live_card])

    def _match_live_select_card(self, engine_card):
        """Find the live select-screen candidate matching the engine-chosen card by stable CardId +
        upgrade count (robust to display-name drift). Duplicate instances (same id/upgrade) are
        interchangeable -- they resolve identically -- so first match is correct. Raises if the
        search picked a card not offered live."""
        want_id, want_upg = engine_card.id, engine_card.upgrade_count
        for c in self.game.screen.cards:
            if map_card_id(c.card_id) == want_id and c.upgrades == want_upg:
                return c
        for c in self.game.screen.cards:  # tolerate an upgrade-count mismatch (e.g. Searing Blow)
            if map_card_id(c.card_id) == want_id:
                return c
        raise RuntimeError(
            f"MCTS-selected card {engine_card.getName()} (id {want_id}, +{want_upg}) is not on the "
            f"live select screen ({[(c.card_id, c.upgrades) for c in self.game.screen.cards]})")

    def capture_decision_state(self):
        """Append the raw CommunicationMod message for this out-of-combat decision to the
        capture file named by $STS_COMM_CAPTURE (one JSON object per line), so real screens
        can be replayed offline when building/validating the GameContext bindings. No-op
        unless the env var is set."""
        path = os.environ.get("STS_COMM_CAPTURE")
        if not path or self.coordinator is None:
            return
        raw = self.coordinator.last_raw_communication_state
        if raw is None:
            return
        record = {
            "choice_count": self.choice_count,
            "screen_type": str(self.game.screen_type),
            "raw": raw,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def capture_battle_state(self):
        """Append the raw combat message to a sibling '<path>.battle.jsonl' file (same
        $STS_COMM_CAPTURE gate). One record per battle decision -- carries combat_state with
        each monster's move_id/last_move_id/intent, the ground truth for auditing whether the
        converted BattleContext predicts the monsters' next moves correctly. No-op unless set."""
        path = os.environ.get("STS_COMM_CAPTURE")
        if not path or self.coordinator is None:
            return
        raw = self.coordinator.last_raw_communication_state
        if raw is None:
            return
        cs = (raw.get("game_state") or {}).get("combat_state") or {}
        record = {
            "choice_count": self.choice_count,
            "turn": cs.get("turn"),
            "raw": raw,
        }
        with open(path + ".battle.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def net_pick_action(self, gc):
        """Run heart1 on gc's current choice screen and return the chosen sts.GameAction (in
        GameContext space), or None if construct_choice can't represent this screen (so the
        caller fails loud). Real errors propagate -- we don't play on a guessed state."""
        from playouts import construct_choice, pick_card_with_net, get_card_probs, path_to_action_and_desc
        from network import choice_space
        import numpy as np

        obs = sts.getNNRepresentation(gc)
        actions = sts.GameAction.getAllActionsInState(gc)
        choice = construct_choice(gc, obs, actions)
        if choice is None:
            return None
        if self.temperature and self.temperature > 0:
            action, _path = pick_card_with_net(self.net, choice, actions,
                                               temperature=self.temperature, rng=self.net_rng)
            return action
        # Greedy: pick_card_with_net's Boltzmann path divides by temperature, so argmax directly.
        collated_input, output = self.net.get_logits(choice)
        logits = output[0] if isinstance(output, tuple) else output
        probs = get_card_probs(logits)
        idx = int(np.argmax(probs))
        path = choice_space.ix_to_path(collated_input["choices"], idx)
        action, _desc = path_to_action_and_desc(choice, path)
        return action

    def net_card_reward_action(self):
        """heart1's pick for a single (already-revealed) card reward screen: take a card, take
        the Singing Bowl, or skip. Returns a spirecomm Action, or None to fail loud. The
        multi-group Prayer Wheel reveal flow layers on top of this later."""
        gc = spirecomm_to_gamecontext(self.game)
        action = self.net_pick_action(gc)
        if action is None:
            return None
        rtype = action.rewards_action_type
        if rtype == sts.RewardsActionType.CARD:
            if action.idx2 == 5:  # Singing Bowl pseudo-option (+2 max HP instead of a card)
                print("[net] card reward -> Singing Bowl", file=sys.stderr)
                return CardRewardAction(bowl=True)
            chosen = self.game.screen.cards[action.idx2]
            print(f"[net] card reward -> take {chosen.card_id} (idx {action.idx2})", file=sys.stderr)
            return CardRewardAction(chosen)
        if rtype == sts.RewardsActionType.SKIP:
            print("[net] card reward -> skip", file=sys.stderr)
            self.skipped_cards = True
            return CancelAction()
        print(f"[net] unexpected card-reward action {rtype}; failing loud", file=sys.stderr)
        return None

    def net_boss_relic_action(self):
        """heart1's pick among the three boss relics. Returns a spirecomm Action, or None to fail
        loud. Vanilla can't skip a boss relic, so a SKIP pick shouldn't occur -> fail loud."""
        gc = spirecomm_to_gamecontext(self.game)
        action = self.net_pick_action(gc)
        if action is None:
            return None
        if action.rewards_action_type == sts.RewardsActionType.RELIC:
            chosen = self.game.screen.relics[action.idx1]
            print(f"[net] boss relic -> take {chosen.name} (idx {action.idx1})", file=sys.stderr)
            return BossRewardAction(chosen)
        print(f"[net] boss relic -> {action.rewards_action_type}; failing loud", file=sys.stderr)
        return None

    def net_card_select_action(self):
        """heart1's pick on an out-of-combat single-card grid select (transform/upgrade/remove/
        obtain -- e.g. shop card removal, rest-site smith, event transforms). Returns None (-> fail
        loud) for in-combat selects (those are the combat MCTS's job) and multi-card selects."""
        scr = self.game.screen
        if self.game.in_combat:
            return None
        if getattr(scr, "num_cards", 1) != 1 or getattr(scr, "any_number", False):
            return None
        gc = spirecomm_to_gamecontext(self.game)
        action = self.net_pick_action(gc)
        if action is None:
            return None
        idx = action.idx1
        if not (0 <= idx < len(scr.cards)):
            return None
        chosen = scr.cards[idx]
        print(f"[net] grid select -> {chosen.card_id} (idx {idx})", file=sys.stderr)
        return CardSelectAction([chosen])

    def net_rest_action(self):
        """heart1's campfire choice (rest / smith / and any relic options like dig/lift/recall).
        The engine offers the same options the live site does (it reads the player's relics), so we
        map the picked action back by its option name. Returns a spirecomm Action, or None to fall
        back. Smith's which-card-to-upgrade is a follow-up card-select screen."""
        gc = spirecomm_to_gamecontext(self.game)
        action = self.net_pick_action(gc)
        if action is None:
            return None
        desc = action.getDesc(gc).strip().lower()
        rest_by_key = {
            "rest": RestOption.REST, "smith": RestOption.SMITH, "recall": RestOption.RECALL,
            "dig": RestOption.DIG, "lift": RestOption.LIFT, "toke": RestOption.TOKE,
        }
        for key, opt in rest_by_key.items():
            if desc.startswith(key) and opt in self.game.screen.rest_options:
                print(f"[net] rest -> {opt}", file=sys.stderr)
                return RestAction(opt)
        print(f"[net] rest pick {desc!r} not an available option; failing loud", file=sys.stderr)
        return None

    def net_shop_action(self):
        """heart1's shop decision: buy a card/relic/potion, start a card removal, or leave. The
        engine Shop (injected with live prices) makes getAllActionsInState offer exactly the
        affordable buys, so the net only ever picks something we can afford. Returns a spirecomm
        Action, or None to fail loud. One purchase per call; the shop screen re-opens for the next."""
        gc = spirecomm_to_gamecontext(self.game)
        action = self.net_pick_action(gc)
        if action is None:
            return None
        rtype = action.rewards_action_type
        shop = self.game.screen
        if rtype == sts.RewardsActionType.CARD:
            chosen = shop.cards[action.idx1]
            print(f"[net] shop -> buy card {chosen.card_id} ({chosen.price}g)", file=sys.stderr)
            return BuyCardAction(chosen)
        if rtype == sts.RewardsActionType.RELIC:
            chosen = shop.relics[action.idx1]
            print(f"[net] shop -> buy relic {chosen.name} ({chosen.price}g)", file=sys.stderr)
            return BuyRelicAction(chosen)
        if rtype == sts.RewardsActionType.POTION:
            # The gc now carries the real potion belt so the sim shouldn't offer this when full;
            # guard anyway, since BuyPotionAction raises (kills the run) on a full belt.
            if self.game.are_potions_full():
                print("[net] shop -> buy potion skipped (belt full); failing loud", file=sys.stderr)
                return None
            chosen = shop.potions[action.idx1]
            print(f"[net] shop -> buy potion {chosen.potion_id} ({chosen.price}g)", file=sys.stderr)
            return BuyPotionAction(chosen)
        if rtype == sts.RewardsActionType.CARD_REMOVE:
            # Initiate the purge; the card to remove is chosen on the following card-select screen.
            print("[net] shop -> card removal", file=sys.stderr)
            return ChooseAction(name="purge")
        if rtype == sts.RewardsActionType.SKIP:
            print("[net] shop -> leave", file=sys.stderr)
            return CancelAction()
        print(f"[net] shop -> {rtype}; failing loud", file=sys.stderr)
        return None

    def net_map_action(self):
        """heart1's pick of the next map node. The GameContext regenerates this seed's map and is
        placed on the player's current node, so getAllActionsInState offers the real next-row
        nodes as path choices (idx1 == node x). Returns a spirecomm Action, or None to fail loud."""
        gc = spirecomm_to_gamecontext(self.game)
        action = self.net_pick_action(gc)
        if action is None:
            return None
        chosen_x = action.idx1
        # When the boss is the only thing ahead, the live game exposes no next_nodes and the move
        # is a dedicated boss choice; the net still sees a single map-node action for it.
        if self.game.screen.boss_available and not self.game.screen.next_nodes:
            print("[net] map -> boss", file=sys.stderr)
            return ChooseMapBossAction()
        for node in self.game.screen.next_nodes:
            if node.x == chosen_x:
                print(f"[net] map -> node x={chosen_x} (y={node.y}, {node.symbol})", file=sys.stderr)
                return ChooseMapNodeAction(node)
        print(f"[net] map pick x={chosen_x} not in next_nodes; failing loud", file=sys.stderr)
        return None

    def net_event_action(self):
        """heart1's pick among an event's options. Single-option screens (Talk/Continue/Leave/etc.)
        are forced, so we take option 0 without consulting the net. For real choices we reconstruct
        the event in the GameContext (set_screen_state_info ran setup_event), let the net pick, and
        translate the chosen engine option back to the live choice index.

        The engine returns event options in ascending bit/idx1 order, exactly matching the live
        game's enabled options in order, so the chosen action's rank among the valid engine actions
        IS the live choice index. We only net-drive when the engine and the live game agree on the
        number of available options -- otherwise the gc reconstruction diverged from live (e.g. a
        sub-phase or RNG-dependent option set we can't mirror) and we fail loud (return None) rather
        than risk picking the wrong option."""
        options = self.game.screen.options
        enabled = [o for o in options if not o.disabled]
        if len(enabled) <= 1:
            # Forced acknowledgement / single path: no decision to make.
            return ChooseAction(0)

        # Match and Keep is a blind matching game: CommunicationMod serializes the grid as bare
        # position labels (card0..card11) with NO card identities (no 'cards' field, empty body),
        # so the bot has zero observable signal -- there is nothing for the net or MCTS to reason
        # about, and every position is equivalent in expectation. Play it out mechanically (flip the
        # first available card each step). This is forced participation, not a value choice.
        if (getattr(self.game.screen, "event_id", "") or "").startswith("Match and Keep"):
            print("[event] Match and Keep (blind: no card identities exposed) -> flip card 0",
                  file=sys.stderr)
            return ChooseAction(0)

        ev = map_event_to_enum(self.game.screen)
        if ev == sts.Event.INVALID:
            print(f"[net] event {self.game.screen.event_id!r} unmapped; failing loud", file=sys.stderr)
            return None
        if ev in _EVENTS_NOT_FAITHFULLY_RECONSTRUCTED:
            # The choice hinges on which specific player relic/item is offered, but setup_event picks
            # those via the gc's eventRng, which doesn't match the live game's pick -- so the net
            # would (and extract_event_info does, crashing on an out-of-range index) reason about the
            # wrong item. Cannot reconstruct faithfully -> fail loud.
            print(f"[net] event {self.game.screen.event_id!r} not faithfully reconstructable; "
                  f"failing loud", file=sys.stderr)
            return None

        gc = spirecomm_to_gamecontext(self.game)
        if gc.screen_state != sts.ScreenState.EVENT_SCREEN:
            # setup_event routed into a card-select / combat-reward sub-screen the live screen
            # doesn't match; fail loud.
            return None
        if ev == sts.Event.NLOTH and not _inject_nloth_offers(gc, self.game):
            # Couldn't match both offered relics off the live labels; the gc's RNG-rolled relicIdxs
            # would point the net at the wrong relics, so fail loud rather than choose blind.
            print(f"[net] N'loth offered relics unresolved; failing loud", file=sys.stderr)
            return None
        if ev == sts.Event.WE_MEET_AGAIN and not _inject_wemeetagain(gc, self.game):
            # Couldn't parse the offered card/potion/gold off the live labels; the RNG-rolled items
            # would diverge from live, so fail loud rather than reason about the wrong items.
            print(f"[net] We Meet Again offered items unresolved; failing loud", file=sys.stderr)
            return None
        actions = sts.GameAction.getAllActionsInState(gc)
        if len(actions) != len(enabled):
            print(f"[net] event {self.game.screen.event_id!r}: engine {len(actions)} vs live "
                  f"{len(enabled)} options; failing loud", file=sys.stderr)
            return None

        action = self.net_pick_action(gc)
        if action is None:
            return None
        # Rank of the chosen option among the valid engine options (ascending idx1) == live index.
        sorted_idx1 = sorted(a.idx1 for a in actions)
        try:
            rank = sorted_idx1.index(action.idx1)
        except ValueError:
            print(f"[net] event pick idx1={action.idx1} not among options {sorted_idx1}; failing loud",
                  file=sys.stderr)
            return None
        chosen = enabled[rank]
        print(f"[net] event {self.game.screen.event_id!r} -> [{chosen.choice_index}] {chosen.label!r}",
              file=sys.stderr)
        return ChooseAction(chosen.choice_index)

    def net_chest_action(self):
        """heart1's open-or-skip decision for a treasure chest. Opening isn't free -- Cursed Key
        adds a curse on open, and act 4's sapphire key sits behind a chest -- so it's a real policy
        choice, not a mechanical 'always open'. The reconstructed gc is on the TREASURE_ROOM screen
        (open == idx1 0, skip == idx1 1)."""
        gc = spirecomm_to_gamecontext(self.game)
        action = self.net_pick_action(gc)
        if action is None:
            return None
        if action.idx1 == 0:
            print("[net] chest -> open", file=sys.stderr)
            return OpenChestAction()
        print("[net] chest -> skip", file=sys.stderr)
        return ProceedAction()

    def handle_screen(self):
        """Drive an out-of-combat decision screen exactly as the RL training loop drives the
        engine's equivalent: heart1 for every value choice, the combat MCTS for in-combat selects.
        There is NO heuristic fallback -- if a screen can't be net-reconstructed/represented we fail
        loud rather than play a guessed move (a diverged reconstruction must never silently pick)."""
        self.capture_decision_state()
        st = self.game.screen_type
        if self.net is None:
            raise RuntimeError("no policy loaded: comm.py drives every decision with heart1/MCTS "
                               "and keeps no heuristic fallback")

        # Mechanical, non-decision transitions: collect the (free) combat rewards, and walk up to or
        # away from the merchant. These make no value judgment -- the real choices (which card to
        # take, which item to buy, whether to leave) are separate net-driven screens -- so they
        # mirror the engine auto-advancing, not a heuristic.
        if st == ScreenType.COMBAT_REWARD:
            return self._collect_combat_reward()
        if st == ScreenType.SHOP_ROOM:
            if not self.visited_shop:
                self.visited_shop = True
                return ChooseShopkeeperAction()
            self.visited_shop = False
            return ProceedAction()

        # In-combat card selects -- pile-based (Warcry/Headbutt/Armaments/Dual Wield/Exhume) on
        # HAND_SELECT/GRID, and generated-card choices (Discovery / Attack-Skill-Power Potion) that
        # arrive as an in-combat CARD_REWARD -- are the combat MCTS's job, not the policy's.
        if (st == ScreenType.HAND_SELECT
                or (st in (ScreenType.GRID, ScreenType.CARD_REWARD) and self.game.in_combat)):
            return self.mcts_card_select_action()

        net_handlers = {
            ScreenType.CARD_REWARD: self.net_card_reward_action,
            ScreenType.BOSS_REWARD: self.net_boss_relic_action,
            ScreenType.MAP: self.net_map_action,
            ScreenType.SHOP_SCREEN: self.net_shop_action,
            ScreenType.REST: self.net_rest_action,
            ScreenType.GRID: self.net_card_select_action,
            ScreenType.EVENT: self.net_event_action,
            ScreenType.CHEST: self.net_chest_action,
        }
        handler = net_handlers.get(st)
        if handler is None:
            raise RuntimeError(f"no net handler for decision screen {st}; failing loud rather than "
                               f"guessing an action")
        action = handler()
        if action is None:
            raise RuntimeError(f"heart1 could not drive the {st} screen (unrepresentable or diverged "
                               f"reconstruction); failing loud rather than playing a heuristic")
        return action

    def _collect_combat_reward(self):
        """Take the post-combat rewards. Gold/relic/potion/keys are free and always taken (the same
        the engine's own pick does); a CARD reward opens the CARD_REWARD screen where heart1 chooses
        the card. skipped_cards (set when heart1 skipped the card) stops us re-opening it."""
        for reward_item in self.game.screen.rewards:
            if reward_item.reward_type == RewardType.POTION and self.game.are_potions_full():
                continue
            if reward_item.reward_type == RewardType.CARD and self.skipped_cards:
                continue
            return CombatRewardAction(reward_item)
        self.skipped_cards = False
        return ProceedAction()


DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "heart1.pt.iter_1295")


def load_policy_service(ckpt_path, device=None):
    """Load the heart1 policy checkpoint into an NNService for inference.

    Imports torch/network/playouts lazily (kept out of module import so the conversion tests and
    offline replay tooling load without torch). Mirrors eval_hero.py's loader: single net, value
    head, default ModelHP (the architecture rl_train.py used for the heart1 run)."""
    import torch
    from network import NN, ModelHP, load_network_backward_compatible
    from playouts import NNService

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    hp = ModelHP(use_value_head=True, dim=256, n_layers=4)
    net = NN(hp).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    net = load_network_backward_compatible(net, state)
    net.eval()
    # torch_compile_mode='no': avoid compile/cudagraph latency+warmup for live single-request play.
    service = NNService(net, batch_size=8, max_wait_time=0.005, torch_compile_mode="no")
    service.update_weights(net)
    print(f"[net] loaded heart1 policy from {ckpt_path} on {device}", file=sys.stderr)
    return service


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
    parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                       help="heart1 policy checkpoint for out-of-combat decisions")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Network action-sampling temperature (0 = greedy/argmax)")
    parser.add_argument("--seed", default=None,
                       help="Start runs on this exact base-35 StS seed string (e.g. 54FYPZX13RLTT) "
                            "to replay a specific game")

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

    net = load_policy_service(args.ckpt)

    # Create agent and coordinator
    agent = STSLightspeedAgent(chosen_class, net=net, temperature=args.temperature,
                               start_seed=args.seed)
    coordinator = Coordinator()
    agent.coordinator = coordinator  # lets the agent capture raw decision states for replay

    # Register callbacks
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    
    # Play games. Always play the chosen character every game -- the policy net is
    # character-specific (heart1 is Ironclad), so cycling classes would run it off-distribution.
    games_played = 0
    character_classes = itertools.repeat(chosen_class)

    for current_class in character_classes:
        if args.games > 0 and games_played >= args.games:
            break
            
        agent.change_class(current_class)
        print(f"Playing game {games_played + 1} as {current_class.name}", file=sys.stderr)
        
        try:
            # Pass --seed through so play_one_game's StartGameAction uses it (it defaults to a random
            # seed otherwise). With a seed set, every game replays the same run -- intended for
            # deterministic crash repro (--seed <s> --games 1).
            result = coordinator.play_one_game(current_class, seed=args.seed)
            games_played += 1
            print(f"Game {games_played} completed with result: {result}", file=sys.stderr)
        except KeyboardInterrupt:
            print("Interrupted by user", file=sys.stderr)
            break
        except Exception as e:
            print(f"Game error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            break


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run test
        print("Testing spirecomm to GameContext converter...")
        test_basic_conversion()
    else:
        # Arguments provided - run CLI
        run_agent_cli()