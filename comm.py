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
# Per-combat relic counters (progress toward an every-Nth trigger): synced onto bc.player so the
# search models WHEN the relic next fires (Nunchaku/Pen Nib energy & double-damage, Happy Flower
# energy every 3 turns, Incense Burner Intangible every 6, Ink Bottle draw, Sundial energy). These
# are the engine's persistent Player counter fields; inserterCounter is omitted (Defect orb slots,
# inert for Ironclad). The per-TURN counts (attacksPlayedThisTurn etc. -- Kunai/Shuriken/Ornamental
# Fan/Letter Opener/Velvet Choker) are synced separately in convert_combat_state from the forked
# CommunicationMod's per-turn fields.
_RELIC_COUNTER_ATTR = {
    _normalize_relic_name("Pen Nib"): "penNibCounter",
    _normalize_relic_name("Nunchaku"): "nunchakuCounter",
    _normalize_relic_name("Ink Bottle"): "inkBottleCounter",
    _normalize_relic_name("Happy Flower"): "happyFlowerCounter",
    _normalize_relic_name("Incense Burner"): "incenseBurnerCounter",
    _normalize_relic_name("Sundial"): "sundialCounter",
}

# Relics whose persistent stored VALUE gates an out-of-combat option the net can pick; their live
# counter is synced onto the gc (see spirecomm_to_gamecontext) so the engine offers exactly the live
# choices. GIRYA -> LIFT at a campfire; WING_BOOTS -> map "bypass to any next node" routes.
_OPTION_GATING_RELIC_VALUES = frozenset({sts.RelicId.GIRYA, sts.RelicId.WING_BOOTS})


# Powers are keyed on the live game's stable power_id (the json "id", e.g. "DexLoss"), NOT the
# localized display name ("Dexterity Down") which drifts and forced per-power patching. The tables
# below map every StS power_id the BattleContext models to its engine status; the few monster powers
# the engine drives intrinsically from move logic are in _POWER_IDS_ENGINE_INTRINSIC and skipped.
# Anything else fails loud (apply_*_power asserts) -- a silently-dropped power mis-simulates invisibly.
# Together they cover every power_id a real Ironclad game can produce. The maps were
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
    # 'TheBomb' is NOT routed through the generic buff() path: the live power id carries a per-bomb
    # index suffix ('TheBomb0'/'TheBomb1'/...) and reports countdown-vs-damage separately, so it's
    # handled specially in convert_combat_state (see apply_the_bomb).
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
    # Tag summoned minions (Bronze Orbs, Reptomancer Daggers, Torch Heads, spawned Gremlins) so the
    # engine's minion-gated logic is correct on a reconstructed fight -- notably Feed not raising max
    # HP on a minion kill (caught by the persistent-bc shadow: Feed->BronzeOrb gave +3 HP), plus
    # combat-end / minion-leader checks. Was silently dropped before.
    'Minion': sts.MonsterStatus.MINION,
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
    # Engine-modeled monster powers that were previously dropped (the enum existed but wasn't bound).
    'Barricade': sts.MonsterStatus.BARRICADE,     # Spheric Guardian: block doesn't expire each turn
    'Life Link': sts.MonsterStatus.REGROW,        # Darkling: gates Feed's max-HP gain on a revivable kill
    'Painful Stabs': sts.MonsterStatus.PAINFUL_STABS,  # Book of Stabbing: adds a Wound on hit
    'Stasis': sts.MonsterStatus.STASIS,           # Bronze Orb: returns a stolen card on death
    'Shifting': sts.MonsterStatus.SHIFTING,       # Transient: strength scales with turn
    'Compulsive': sts.MonsterStatus.REACTIVE,     # Writhing Mass: re-rolls intent on taking attack damage (display name "Reactive", internal id "Compulsive")
}

# Monster powers the engine drives INTRINSICALLY from move/encounter logic -- there is no status to
# set, so skipping them on reconstruction is correct, not a gap. Keep this list tiny and justified:
# every other power must be mapped above, and an unmapped power now ASSERTS (see apply_*_power). A
# silently-dropped power mis-simulates invisibly -- that is exactly how the Minion and Life Link
# (Regrow) bugs hid. Non-Ironclad powers (Watcher/Defect/Silent) can never reach an Ironclad game,
# so they are deliberately absent here: if one appears it's a real bug and should fail loud.
_POWER_IDS_ENGINE_INTRINSIC = frozenset({
    'Split',       # slimes split by HP threshold inside the monster move logic (Monster.cpp split moves)
    'Explosive',   # Exploder detonates on a scripted turn; engine drives it by monster id
    'AlwaysMad',   # Mad Gremlin's permanent Angry is baked into its move set
    'Unawakened',  # Awakened One phase-1 -> awaken transition is move-driven
    'BackAttack',  # act-4 Spire elites: the +50% back-attack is driven by the player's SURROUNDED
                   # status (mapped, set from live powers) + facing in calculateDamageToPlayer, not a
                   # monster status -- the monster flag itself is a passive with nothing to reconstruct
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

_intrinsic_powers_logged = set()


def _note_intrinsic_once(power_id: str, kind: str) -> None:
    if power_id not in _intrinsic_powers_logged:
        _intrinsic_powers_logged.add(power_id)
        print(f"[power] {kind} power {power_id!r} handled intrinsically by the engine; skipping",
              file=sys.stderr)


def apply_player_power(bc: sts.BattleContext, power_id: str, amount: int) -> None:
    """Apply a live player power (by stable power_id) to the converted battle, choosing buff vs debuff
    from the resolved status. ASSERTS on any unmapped power -- a silently-dropped power mis-simulates
    invisibly. Ironclad has no engine-intrinsic player powers, so every player power must map; a
    non-Ironclad power (Watcher/Defect/Silent) can never reach an Ironclad game, so it failing loud
    here flags a real bug."""
    if power_id == 'Berserk':
        # BerserkPower has no engine status: it grants +amount energy at the start of each turn, which
        # the engine folds into energyPerTurn (the Berserk card does ++energyPerTurn, BattleContext.cpp).
        # Bump it here so the search budgets every future turn correctly (same gap as the energy relics);
        # the Vulnerable Berserk also applies rides in as its own power.
        bc.player.energyPerTurn += amount
        return
    status = _PLAYER_POWER_ID_TO_STATUS.get(power_id)
    if status is None:
        raise ValueError(
            f"Unmapped player power {power_id!r}: add it to _PLAYER_POWER_ID_TO_STATUS. If it has no "
            f"engine status but a modeled effect (e.g. Berserk -> energyPerTurn), handle it explicitly. "
            f"Non-Ironclad powers should never reach an Ironclad game.")
    if status in _DEBUFF_PLAYER_STATUSES:
        bc.player.debuff(status, amount, False)
    else:
        bc.player.buff(status, amount)


def apply_the_bomb(bc: sts.BattleContext, countdown: int, damage: int) -> None:
    """Place a live The Bomb power into the engine's countdown slots. The engine holds each bomb's
    explosion damage in bomb1/bomb2/bomb3, where bombN fires N end-of-turns from now (bomb1 at the
    end of this turn, then the slots shift down). The live power reports the turns-until-explosion in
    `amount` and the explosion damage in `damage`, so damage goes into bomb{countdown}. Multiple
    bombs at the same countdown stack in the same slot, matching the engine's single DamageAllEnemy."""
    if not (1 <= countdown <= 3):
        raise ValueError(f"The Bomb countdown {countdown} outside the engine's 3 slots "
                         f"(damage={damage}); live power desync")
    if countdown == 1:
        bc.player.bomb1 += damage
    elif countdown == 2:
        bc.player.bomb2 += damage
    else:
        bc.player.bomb3 += damage


def apply_monster_power(sts_monster, power_id: str, amount: int) -> None:
    """Apply a live monster power (by stable power_id) to a converted monster. Skips the few powers
    the engine drives intrinsically from move logic (_POWER_IDS_ENGINE_INTRINSIC, e.g. Split); ASSERTS
    on any other unmapped power -- a silently-dropped monster power mis-simulates invisibly (the Minion
    and Life Link/Regrow bugs both hid this way)."""
    status = _MONSTER_POWER_ID_TO_STATUS.get(power_id)
    if status is None:
        if power_id in _POWER_IDS_ENGINE_INTRINSIC:
            _note_intrinsic_once(power_id, "monster")
            return
        raise ValueError(
            f"Unmapped monster power {power_id!r}: add it to _MONSTER_POWER_ID_TO_STATUS (binding the "
            f"MonsterStatus enum if needed), or to _POWER_IDS_ENGINE_INTRINSIC if the engine drives it "
            f"from move logic.")
    if status in _DEBUFF_MONSTER_STATUSES:
        sts_monster.addDebuff(status, amount, False)
    else:
        sts_monster.buff(status, amount)
    # buff()/addDebuff() flag the status "just applied" (skipFirst), so its FIRST end-of-round effect is
    # skipped -- RitualPower/WeakPower semantics. A mid-fight reconstruction always observes a power
    # applied on a PRIOR turn (the round it was cast has ended), so clear the flag. Without this the
    # engine skips Ritual's strength gain EVERY simulated turn (confirmed: a live Cultist with Ritual 4
    # gained 0 strength per END_TURN instead of 4 -> the search under-models its escalating damage).
    sts_monster.set_just_applied(status, False)


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

    # energyPerTurn and cardDrawPerTurn are persistent per-turn scalars the engine sets ONCE at
    # atBattleStart (from relics), which register_relics_from() deliberately skips -- so they stay at
    # their base and the search, which runs END_TURN internally to look ahead, mis-budgets EVERY future
    # turn. (Current-turn energy/hand come straight from the snapshot, so only the lookahead degrades.)
    # The persistent-bc END_TURN shadow caught the energy case: "energy pred 3 vs live 4" on ~all
    # end-turns. Restore both here, exactly mirroring the engine's atBattleStart bumps.

    # +1 energy/turn relics. Base energy is 3 for every character.
    _energy_relics = (sts.RelicId.MARK_OF_PAIN, sts.RelicId.ECTOPLASM, sts.RelicId.PHILOSOPHERS_STONE,
                      sts.RelicId.RUNIC_DOME, sts.RelicId.SOZU, sts.RelicId.VELVET_CHOKER,
                      sts.RelicId.BUSTED_CROWN, sts.RelicId.COFFEE_DRIPPER, sts.RelicId.CURSED_KEY,
                      sts.RelicId.FUSION_HAMMER)
    energy_per_turn = 3 + sum(1 for r in _energy_relics if bc.player.hasRelic(r))
    # Slaver's Collar grants +1 energy/turn when the room's eliteTrigger is set OR any monster's
    # type == BOSS (SlaversCollar.java onEnergyRecharge). A boss fight reached via an EVENT (Mind
    # Bloom spawns The Guardian in an EventRoom) still satisfies the boss-monster clause, so gating
    # on room_type == MonsterRoomBoss misses it -- detect the boss by its encounter signature
    # instead (same keys as the Java BOSS-type check). Elites only ever appear in MonsterRoomElite,
    # so room_type is a faithful proxy for eliteTrigger there.
    is_boss_fight = _infer_boss_encounter(spire_game.monsters) != sts.MonsterEncounter.INVALID
    if (bc.player.hasRelic(sts.RelicId.SLAVERS_COLLAR)
            and (spire_game.room_type == "MonsterRoomElite" or is_boss_fight)):
        energy_per_turn += 1
    bc.player.energyPerTurn = energy_per_turn

    # +cards/turn relics: Snecko Eye (+2), Ring of the Serpent (+1, Silent-only). Base draw is 5.
    bc.player.cardDrawPerTurn = (5
                                 + 2 * bc.player.hasRelic(sts.RelicId.SNECKO_EYE)
                                 + 1 * bc.player.hasRelic(sts.RelicId.RING_OF_THE_SERPENT))

    # Clear the initialized cards to avoid mixing with spirecomm state
    bc.cards.clear()
    
    # Set the input state to PLAYER_NORMAL so the searcher can find actions
    # InputState enum: EXECUTING_ACTIONS=0, PLAYER_NORMAL=1, CARD_SELECT=2, etc.
    bc.input_state = sts.InputState.PLAYER_NORMAL

    # Carry the live turn number onto the bc. The engine's bc.turn is 0-BASED (0 during the first
    # player turn; getMonsterTurnNumber() returns turn+1), while the live combat_state.turn is
    # 1-based, so subtract one. Several monster behaviours read this: turn-scaling attack damage
    # (Transient = 30+10*(turn), Giant Head's It Is Time, The Maw's Nom hit count all via
    # getMonsterTurnNumber), Hexaghost's Burn upgrade (turn>8), and the search's per-turn eval
    # terms. Getting it wrong makes the search mis-predict those monsters' incoming damage.
    assert spire_game.turn >= 1, f"live combat_state.turn {spire_game.turn} is not 1-based (expected >= 1)"
    bc.turn = spire_game.turn - 1

    # Player state conversion
    player = spire_game.player
    if player:
        # Set basic player stats
        bc.player.energy = player.energy
        bc.player.block = player.block
            
        # Convert player powers/buffs/debuffs (keyed on the stable power_id, not the display name).
        # The Bomb is special-cased: its live id carries a per-bomb index ('TheBomb0', 'TheBomb1',
        # ...) and it reports countdown and damage separately, so it can't go through the generic
        # buff() path (which would treat the countdown as the buff amount and always land slot 3).
        for power in player.powers:
            if power.power_id.startswith("TheBomb"):
                apply_the_bomb(bc, power.amount, power.damage)
                continue
            apply_player_power(bc, power.power_id, power.amount)

        # Per-turn play counts (exposed by the forked CommunicationMod). The engine reads these
        # absolutely -- Kunai/Shuriken/Ornamental Fan fire on every 3rd attack, Letter Opener on
        # every 3rd skill, Velvet Choker/Normality lock at 6 cards -- so a mid-turn reconstruction
        # must restore them or the search mis-times those triggers.
        bc.player.cardsPlayedThisTurn = spire_game.cards_played_this_turn
        bc.player.attacksPlayedThisTurn = spire_game.attacks_played_this_turn
        bc.player.skillsPlayedThisTurn = spire_game.skills_played_this_turn

    # Restore per-combat relic counters (progress toward the next every-Nth trigger) from the live
    # relics; register_relics_from only copies ownership bits, leaving these at zero. Also correct the
    # combat bit of spent one-shot relics: the engine's atBattleStart (skipped by register_relics_from)
    # runs setHasRelic<r>(r.data) for Lizard Tail and Omamori, clearing the bit once the charge is used.
    # Without this the search over-values an already-spent revive (BattleSearcher scores Lizard Tail at
    # 0.5*maxHp of survival) or curse-negate. Counter conventions (from the live game's relic.counter):
    #   Lizard Tail -- -1 available, -2 used (LizardTail.setCounter(-2) calls usedUp()).
    #   Omamori     -- charges remaining (starts 2, grays out / inert at 0).
    for spire_relic in spire_game.relics:
        norm = _normalize_relic_name(spire_relic.relic_id)
        attr = _RELIC_COUNTER_ATTR.get(norm)
        if attr is not None:
            setattr(bc.player, attr, spire_relic.counter)
        rid = map_relic_id(spire_relic.name)
        if rid == sts.RelicId.LIZARD_TAIL:
            bc.player.setHasRelic(sts.RelicId.LIZARD_TAIL, spire_relic.counter != -2)
        elif rid == sts.RelicId.OMAMORI:
            bc.player.setHasRelic(sts.RelicId.OMAMORI, spire_relic.counter > 0)

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
        # Mirror native deck-load: every card entering combat is registered so combat-wide counters
        # (strikeCount, read by Perfect Strike) are maintained. The pile movers only touch
        # pile-local counts; moveToExhaustPile's notifyRemoveFromCombat then nets exhausted strikes
        # back out, leaving strikeCount = strikes in hand+draw+discard, as Perfect Strike expects.
        bc.cards.notify_add_card_to_combat(card_instance)
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
    _reconstruct_stasis_cards(bc, spire_game, slot_to_spire)

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


# Monsters whose move is assigned by a summoner (not their own getMoveForRoll) -- rolling them
# assert(false)s in the engine. When one is converted with no committed intent (just spawned, no
# live move_id), set this fixed move directly instead of rolling. TorchHead only ever tackles.
_UNKNOWN_INTENT_DEFAULT_MOVE = {
    sts.MonsterId.TORCH_HEAD: sts.MonsterMoveId.TORCH_HEAD_TACKLE,
}

# Monsters whose next move is a DETERMINISTIC function of their last move (a fixed alternation set via
# setMove in takeTurn, no roll). When the live snapshot omits their current intent, getMoveForRoll
# returns only their fixed OPENER (Donu->Circle, Deca->Beam) regardless of cycle phase -- so the search
# forces Donu to buff (+3 strength) and Deca to beam (0 block) every lookahead turn, drifting the boss's
# strength/block out of phase. Infer the real next move from last_move instead. Successors mirror the
# setMove calls in MonsterSpecific.cpp (DONU_BEAM<->DONU_CIRCLE_OF_POWER, DECA_BEAM<->DECA_SQUARE).
_DETERMINISTIC_MOVE_SUCCESSOR = {
    int(sts.MonsterMoveId.DONU_BEAM): sts.MonsterMoveId.DONU_CIRCLE_OF_POWER,
    int(sts.MonsterMoveId.DONU_CIRCLE_OF_POWER): sts.MonsterMoveId.DONU_BEAM,
    int(sts.MonsterMoveId.DECA_BEAM): sts.MonsterMoveId.DECA_SQUARE_OF_PROTECTION,
    int(sts.MonsterMoveId.DECA_SQUARE_OF_PROTECTION): sts.MonsterMoveId.DECA_BEAM,
}


# Moves whose damage the engine reads from the monster's `miscInfo` (rolled at battle start or set
# mid-move), rather than a hardcoded constant in takeTurn. A mid-fight reconstruction can't recover
# miscInfo, so it stays 0 and these attacks simulate as 0 damage -- the search then never blocks
# them (Louse bites, Orb Walker lasers, Hexaghost's Divider). The live intent carries the real
# per-hit base damage, so set miscInfo from it. BOOK_OF_STABBING reads miscInfo as the STAB COUNT
# (a multiplier on a fixed 6/7 base), so it's restored from the live hit count instead.
_MISCINFO_DAMAGE_MOVE_INTS = frozenset(int(m) for m in (
    sts.MonsterMoveId.GREEN_LOUSE_BITE,
    sts.MonsterMoveId.RED_LOUSE_BITE,
    sts.MonsterMoveId.ORB_WALKER_LASER,
    sts.MonsterMoveId.HEXAGHOST_DIVIDER,
    sts.MonsterMoveId.DARKLING_NIP,
    # Giant Head's It Is Time escalates +5 per cast (40..70). The engine reads miscInfo as the
    # current slam damage; the live turn can't recover the cast count (it desyncs from the player
    # turn -- two turn-5 states can be the 1st vs 2nd slam), so restore the damage from the intent.
    sts.MonsterMoveId.GIANT_HEAD_IT_IS_TIME,
))
_MISCINFO_HITS_MOVE_INTS = frozenset(int(m) for m in (
    sts.MonsterMoveId.BOOK_OF_STABBING_MULTI_STAB,
))

# Under Runic Dome the intent is hidden, so move_base_damage/move_hits aren't reported and the
# per-hit damage / stab count these monsters read from their hidden `miscInfo` stays 0 -> the search
# predicts 0 incoming damage and NEVER BLOCKS -> compounding chip death (observed: the agent walked
# into Book of Stabbing at 8 HP thinking its multi-stab did 0). Estimate miscInfo from the monster
# (and turn for the escalating ones) so the deferred move roll produces a realistic attack. Fixed
# values are the asc0 means of Monster::construct's rolls (the deployed run is asc0); slightly
# approximate, but vastly better than 0. Hexaghost Divider (curHp/12) and Gremlin Wizard (charge)
# are dynamic/rare under RD and left as-is.
_RD_HIDDEN_MISCINFO_FIXED = {
    sts.MonsterId.GREEN_LOUSE: 6,   # bite damage, mean of rng(5,7)
    sts.MonsterId.RED_LOUSE: 6,     # bite damage, mean of rng(5,7)
    sts.MonsterId.DARKLING: 9,      # nip damage, mean of rng(7,11)
}


def _estimate_hidden_miscinfo(monster_id, turn0):
    """A reasonable miscInfo when Runic Dome hides the real value (turn0 = bc.turn, 0-based)."""
    if monster_id == sts.MonsterId.BOOK_OF_STABBING:
        return max(1, turn0 + 1)    # stab count: 1 at battle start (++miscInfo), grows ~1/turn
    return _RD_HIDDEN_MISCINFO_FIXED.get(monster_id)


def assert_intent_damage_matches(bc, spire_game, slot_to_spire) -> None:
    """Fail loud if the engine's predicted attack damage for any monster disagrees with the live
    game's displayed intent. The live snapshot can't restore some moves' damage (it lives in hidden
    per-monster state -- miscInfo-parameterized attacks like Louse bite, Orb Walker laser, Hexaghost
    Divider), so a reconstruction gap makes the search simulate 0 incoming damage and never block.
    This catches any such gap the moment it would mis-simulate, comparing BASE damage (pre
    strength/vulnerable; both sides exclude those) and hit count. Skips monsters whose intent isn't
    observably an attack and Runic Dome's deferred (hidden) intents."""
    monsters = spire_game.monsters
    for slot, spire_idx in slot_to_spire.items():
        if not (0 <= spire_idx < len(monsters)):
            continue
        live = monsters[spire_idx]
        sim = bc.monsters[slot]
        if not sim.isAlive() or live.is_gone or live.half_dead:
            continue
        if sim.pending_move_rolls > 0:   # Runic Dome: intent deferred/hidden, engine can't predict
            continue
        live_dmg = live.move_base_damage
        live_hits = live.move_hits or 0
        if live_dmg is None or live_dmg < 0 or live_hits < 1:   # not an observable attack intent
            continue
        eng_dmg, eng_hits = sim.get_move_base_damage(bc)
        if (eng_dmg, eng_hits) != (live_dmg, live_hits):
            raise AssertionError(
                f"intent-damage mismatch for {live.monster_id} (move_id={live.move_id}): engine "
                f"predicts {eng_dmg}x{eng_hits} base but the live intent shows {live_dmg}x{live_hits}"
                f" -- this move's damage is being mis-reconstructed (likely hidden miscInfo); the "
                f"search would mis-judge blocking. Add it to the miscInfo-restore table or fix the "
                f"conversion.")


def assert_card_damage_matches(bc, spire_game) -> None:
    """Fail loud if the engine's in-hand displayed damage for a reconstructed attack card disagrees
    with the live game's displayed damage. Catches card-damage reconstruction gaps where a card's
    damage depends on combat-accumulated state the live snapshot can't directly restore (Perfect
    Strike's Strike count -> strikeCount; Body Slam's block; permanently-buffed Searing Blow /
    Rampage / Genetic Algorithm via specialData). Compares the DISPLAYED value (live AbstractCard.
    damage vs engine getCardDamageDisplay) -- base + player-side modifiers (strength/weak/stance),
    neither side applying a target's vulnerable -- so it's target-independent and catches the strike
    bonus (which the live game folds into `damage`, not `base_damage`). Hand cards convert in order,
    so bc.cards.hand[i] corresponds to spire_game.hand[i]."""
    hand = spire_game.hand
    for i, live in enumerate(hand):
        dmg = live.damage
        if dmg is None or dmg < 0 or i >= bc.cards.cardsInHand:
            continue
        eng = bc.get_card_damage_display(bc.cards.hand[i])
        if eng < 0:    # engine says non-attack (live reports 0 for these, not -1) -- nothing to check
            continue
        if eng != dmg:
            raise AssertionError(
                f"card-damage mismatch for {live.card_id} (hand idx {i}): engine displays {eng} vs "
                f"live {dmg} -- this card's damage is being mis-reconstructed (hidden combat state "
                f"like strikeCount / specialData / strength); the search would mis-judge playing it.")


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

    # Restore miscInfo for moves whose damage/hit-count the engine reads from it (see the table
    # above). Without this the search predicts 0 incoming damage for these attacks and never blocks
    # them -- the dominant live-combat HP bleed (Louses are everywhere; Hexaghost's Divider is ~36).
    if move_known:
        cur = move_history[0]
        if cur in _MISCINFO_DAMAGE_MOVE_INTS and monster.move_base_damage:
            sts_monster.miscInfo = int(monster.move_base_damage)
        elif cur in _MISCINFO_HITS_MOVE_INTS and monster.move_hits:
            sts_monster.miscInfo = int(monster.move_hits)

    for power in monster.powers:
        apply_monster_power(sts_monster, power.power_id, power.amount)

    # A sleeping Lagavulin reports only its Metallicize power (no "Asleep" status), so without this the
    # reconstruction is an awake attacker that keeps Metallicize 8 -- and the engine, having no ASLEEP
    # to remove, regains 8 block every turn forever (the search then badly over-estimates its bulk, and
    # it "attacks" while it should be sleeping). The engine drops Metallicize only when it wakes FROM
    # ASLEEP (Monster::damageUnblockedHelper on a block break, or the turn-timeout wake), so seed ASLEEP
    # from the sleep intent and let the engine model the wake.
    if move_known and move_history[0] == int(sts.MonsterMoveId.LAGAVULIN_SLEEP):
        sts_monster.buff(sts.MonsterStatus.ASLEEP, 1)

    if not move_known:
        # A deterministic alternator (Donu/Deca) with a known last move: infer the real next move from
        # the fixed cycle rather than rolling to its phase-blind opener (see _DETERMINISTIC_MOVE_SUCCESSOR).
        successor = _DETERMINISTIC_MOVE_SUCCESSOR.get(move_history[1]) if move_history[1] else None
        if successor is not None:
            sts_monster.moveHistory = [int(successor), move_history[1]]
            return
        # A summoned minion whose move is set by its summoner (TorchHead via the Collector spawn)
        # has no getMoveForRoll case -- rolling it assert(false)s in the engine (uncatchable). When
        # it appears with no committed intent (just spawned), set its fixed move directly instead.
        default_move = _UNKNOWN_INTENT_DEFAULT_MOVE.get(sts_monster.id)
        if default_move is not None:
            sts_monster.moveHistory = [int(default_move), sts_monster.moveHistory[1]]
        else:
            # Runic Dome hides this monster's miscInfo too (the per-hit damage / stab count), leaving it
            # 0 so the deferred roll would predict a 0-damage attack. Seed a realistic estimate first so
            # the search blocks. Only when unset (a restored value from move_hits wins).
            if sts_monster.miscInfo == 0:
                est = _estimate_hidden_miscinfo(sts_monster.id, bc.turn)
                if est is not None:
                    sts_monster.miscInfo = est
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


def _reconstruct_stasis_cards(bc: sts.BattleContext, spire_game, slot_to_spire: dict) -> None:
    """Restore the card each Bronze Orb holds in Stasis. The orb's Stasis move steals a card from the
    draw pile and holds it; the engine returns it to hand when the orb dies (returnStasisCard), reading
    it from cards.stasisCards[min(orbMonsterIdx, 1)]. A reconstructed mid-fight state sets the orb's
    STASIS status (so the engine WILL try to return a card) but the stolen card is in NO reported pile,
    so returnStasisCard would assert on an INVALID card. CommunicationMod exposes the held card on the
    Stasis power itself (StasisPower's private `card` field, reflected into the power json -> Power.card),
    so place that exact card into the matching stasis slot (engine slot = min(sim monster idx, 1))."""
    spire_to_slot = {spire_idx: slot for slot, spire_idx in slot_to_spire.items()}
    next_uid = bc.cards.next_unique_card_id
    for spire_idx, monster in enumerate(spire_game.monsters):
        sim_slot = spire_to_slot.get(spire_idx)
        if sim_slot is None:
            continue
        for power in monster.powers:
            if power.power_id != 'Stasis':
                continue
            if power.card is None:
                raise ValueError(f"{monster.monster_id} has Stasis but no exposed card (slot {sim_slot}); "
                                 f"returnStasisCard would assert -- check CommunicationMod power json")
            instance = convert_spire_card_to_instance(power.card)
            instance.uniqueId = next_uid
            next_uid += 1
            bc.cards.set_stasis_card(min(sim_slot, 1), instance)
    bc.cards.next_unique_card_id = next_uid


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

    # Restore per-instance accumulated state (the engine's specialData == the live card's misc):
    # Rampage's growing damage, Ritual Dagger / Genetic Algorithm bonuses, Glass Knife, etc. Without
    # this the search plays these cards at their printed base. Harmless for cards that don't use it
    # (misc == 0).
    instance.specialData = spire_card.misc

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
    # Spire Growth's live id is "Serpent" (its StS class id); the engine names it SpireGrowth.
    "Serpent": "SpireGrowth",
}

# Entries that appear in a combat's monster list but are not separate engine combatants: the
# Hexaghost's orbs are part of the single engine Hexaghost. Skipped during conversion (like is_gone).
_MONSTER_IDS_SKIP_IN_COMBAT = frozenset({"HexaghostOrb"})

# Monster ids that are non-combat event props (Apology Slime) -- they never appear in a real
# battle's monster list, so reaching one in combat is a real bug we want to fail loud on rather than
# silently mishandle. (The act-2 "Healer" is NOT here: it is a real combatant, the engine's Mystic
# -- see _MONSTER_ID_SYNONYMS. "Serpent" is NOT here either: it is Spire Growth's live id -- the
# act-3 combatant SpireGrowth -- not the "The Ssssserpent" event prop.)
_MONSTER_IDS_NON_COMBAT = frozenset({"Apology Slime"})


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
        ("TheCollector", 5): sts.MonsterMoveId.THE_COLLECTOR_SPAWN,   # re-summon (engine: "5 spawn")
        
        # Shelled Parasite
        ("ShelledParasite", 1): sts.MonsterMoveId.SHELLED_PARASITE_FELL,
        ("ShelledParasite", 2): sts.MonsterMoveId.SHELLED_PARASITE_DOUBLE_STRIKE,
        ("ShelledParasite", 3): sts.MonsterMoveId.SHELLED_PARASITE_SUCK,
        ("ShelledParasite", 4): sts.MonsterMoveId.SHELLED_PARASITE_STUNNED,
        
        # Book Of Stabbing
        ("BookOfStabbing", 1): sts.MonsterMoveId.BOOK_OF_STABBING_MULTI_STAB,   # Java STAB=1 (the stabCount multi-hit)
        ("BookOfStabbing", 2): sts.MonsterMoveId.BOOK_OF_STABBING_SINGLE_STAB,  # Java BIG_STAB=2 (single 21/24)
        
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
        
        # Masked Bandits (live ids BanditChild=Pointy, BanditLeader=Romeo; keyed to the live name
        # map_move_id looks up). Byte ids read from live data + matched to engine damage:
        #   Pointy ATTACK 5x2 = POINTY_ATTACK
        #   Romeo CROSS_SLASH=1 (15 dmg), MOCK=2 (no dmg, turn 1), AGONIZING_SLASH=3 (10 + Weak)
        ("BanditChild", 1): sts.MonsterMoveId.POINTY_ATTACK,
        ("BanditLeader", 1): sts.MonsterMoveId.ROMEO_CROSS_SLASH,
        ("BanditLeader", 2): sts.MonsterMoveId.ROMEO_MOCK,
        ("BanditLeader", 3): sts.MonsterMoveId.ROMEO_AGONIZING_SLASH,

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
        ("WrithingMass", 2): sts.MonsterMoveId.WRITHING_MASS_FLAIL,    # Java ATTACK_BLOCK=2 (engine FLAIL=2)
        ("WrithingMass", 3): sts.MonsterMoveId.WRITHING_MASS_WITHER,   # Java ATTACK_DEBUFF=3 (engine WITHER=3)
        ("WrithingMass", 4): sts.MonsterMoveId.WRITHING_MASS_IMPLANT,
        
        # Nemesis
        ("Nemesis", 2): sts.MonsterMoveId.NEMESIS_ATTACK,
        ("Nemesis", 3): sts.MonsterMoveId.NEMESIS_SCYTHE,
        ("Nemesis", 4): sts.MonsterMoveId.NEMESIS_DEBUFF,
        
        # Reptomancer
        ("Reptomancer", 1): sts.MonsterMoveId.REPTOMANCER_SNAKE_STRIKE,
        ("Reptomancer", 2): sts.MonsterMoveId.REPTOMANCER_SUMMON,
        ("Reptomancer", 3): sts.MonsterMoveId.REPTOMANCER_BIG_BITE,
        
        # Reptomancer's Daggers (live id "Dagger"; keyed to the live name map_move_id looks up).
        # move 1 = STAB (ATTACK_DEBUFF 9), 2 = EXPLODE.
        ("Dagger", 1): sts.MonsterMoveId.DAGGER_STAB,
        ("Dagger", 2): sts.MonsterMoveId.DAGGER_EXPLODE,
        
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
        
        # Spire Growth (live id "Serpent" -- the StS class id -- keyed to match map_move_id's
        # raw-name lookup). byte ids from SpireGrowth.java: QUICK_TACKLE=1, CONSTRICT=2, SMASH=3
        ("Serpent", 1): sts.MonsterMoveId.SPIRE_GROWTH_QUICK_TACKLE,
        ("Serpent", 2): sts.MonsterMoveId.SPIRE_GROWTH_CONSTRICT,
        ("Serpent", 3): sts.MonsterMoveId.SPIRE_GROWTH_SMASH,
        
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

        # to_select_cards holds only the still-UNSELECTED grid cards: getGridScreenCards() returns the
        # full target group (selected cards stay in it), and the engine offers one action per
        # to_select_card -- so a multi-card select (Astrolabe: pick 3) must drop already-picked cards
        # or the net could re-pick one (the live game would toggle it back off). Order is preserved so
        # to_select_cards[i] lines up with the same-filtered live list in net_card_select_action.
        selected_uuids = {c.uuid for c in grid_screen.selected_cards}
        info.clear_to_select_cards()
        for i, grid_card in enumerate(grid_screen.cards):
            if grid_card.uuid in selected_uuids:
                continue
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
        # num_cards is the screen's max_cards: 99 for an any-number select, and sometimes absent
        # (None) on the combat-start frame of one (e.g. Gambling Chip). Either way you can never
        # select more than the hand holds, so clamp; None falls back to the whole hand.
        info.to_select_count = min(hand_screen.num_cards or len(hand_screen.cards),
                                   len(hand_screen.cards))
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
        # obtain_relic re-fires one-time onEquip effects that the LIVE snapshot ALREADY reflects --
        # Pear/Strawberry/Mango/Lee's Waffle (+maxHP), Blood Vial (+curHP), Maw Bank/Old Coin (gold).
        # Re-applying them double-counts: Pear adds +10 maxHP on top of the live 90 -> 100, so the
        # search plays the whole game with phantom HP and under-estimates danger (was the dominant ET
        # shadow divergence: php pred consistently +10 vs live). Overwrite HP/gold with live truth.
        gc.cur_hp = spire_game.current_hp
        gc.max_hp = spire_game.max_hp
        gc.gold = spire_game.gold
        # Sync the stored value of relics whose value GATES AN OUT-OF-COMBAT OPTION the net could
        # pick -- otherwise the engine offers a choice the live game no longer does and the net may
        # pick it (fail-loud). Both mirror the game's own counter, so the live counter maps straight
        # to the engine value:
        #   GIRYA       lift count (0-3); LIFT offered at a campfire only while value != 3
        #   WING_BOOTS  bypass charges remaining (3->0); "go to any next node" offered while value>0
        for spire_relic in spire_game.relics:
            rid = map_relic_id(spire_relic.name)
            if rid in _OPTION_GATING_RELIC_VALUES:
                gc.set_relic_value(rid, spire_relic.counter)
            elif rid == sts.RelicId.OMAMORI:
                # obtain_relic defaults Omamori to 2 charges; the combat search reads the charge from
                # gc (bc.gameContext->relics.getRelicValue(OMAMORI), e.g. BattleSearcher / Writhing
                # Mass), so sync the live remaining charges or it over-counts curse-negates.
                gc.set_relic_value(rid, spire_relic.counter)

    # Act-4 keys (forked CommunicationMod exposes them on every screen). They gate out-of-combat
    # options the engine derives from gc state -- the Ruby (red) key's RECALL at a campfire is the
    # most visible -- and feed the net's representation. Set from the live values so the engine offers
    # exactly the live choices.
    gc.red_key = spire_game.has_ruby_key
    gc.green_key = spire_game.has_emerald_key
    gc.blue_key = spire_game.has_sapphire_key

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
        # Use the ENGINE's notion of whether the potion targets, not spirecomm's requires_target:
        # spirecomm mis-flags AOE potions (e.g. Explosive Potion) as requires_target, which would
        # send us mapping the search's meaningless default target_idx (often 0, an empty/reserved
        # sim slot) and raise. Only Fear/Fire/Poison/Weak truly target. spirecomm's PotionAction
        # appends a target only when target_index is not None, so a target-less AOE potion is fine.
        needs_target = sts.potion_requires_target(map_potion_id(potion.potion_id))
        target_idx = (_sim_target_to_spire_index(action.get_target_idx(), slot_to_spire)
                      if needs_target else None)
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
    "SkillFromDeckToHandAction": sts.CardSelectTask.SECRET_TECHNIQUE,  # Secret Technique (a Skill from the draw pile)
    "AttackFromDeckToHandAction": sts.CardSelectTask.SECRET_WEAPON,    # Secret Weapon (an Attack from the draw pile)
    "CodexAction": sts.CardSelectTask.CODEX,                       # Nilry's Codex: 1 of 3 cards at end of turn
}

# Tasks whose candidates are freshly-generated cards offered on the screen (not drawn from a pile):
# the offered cards are injected into the select and the chosen index maps straight back to them.
_DISCOVERY_TASKS = frozenset({sts.CardSelectTask.DISCOVERY})

# Multi-card "choose any number" selects (the live screen reports num_cards>1 / max 99). The same
# spirecomm action name can be single (True Grit -> one card) or multi (Elixir Potion / Purity ->
# exhaust any number), so these are keyed separately and chosen by num_cards in the handler.
_MULTI_CARD_SELECT_TASK_BY_ACTION = {
    "ExhaustAction": sts.CardSelectTask.EXHAUST_MANY,    # Elixir Potion / Purity (exhaust any number)
    "GamblingChipAction": sts.CardSelectTask.GAMBLE,     # Gambling Chip / Gambler's Brew (discard any number, redraw)
}


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

# Single-card-select tasks the persistent-bc drive resolves directly on the parked pbc -- deterministic
# pile picks (a known pool, mapped from a real action) the search reads straight off the carried piles.
_PBC_THROUGH_SELECT_TASKS = (
    (frozenset(_CARD_SELECT_TASK_BY_ACTION.values()) & frozenset(_CARD_SELECT_POOL_BY_TASK.keys()))
    - _DISCOVERY_TASKS - {sts.CardSelectTask.CODEX}
)

# RNG-generated single selects: the pbc rolls its OWN candidates from a desynced RNG, so the drive
# injects the live-observed offered cards onto the parked pbc before searching (M4). The per-decision
# reconcile masks the RNG-state desync (every observable is overwritten from reality each decision),
# so this only needs the right candidates, not a matching roll.
_PBC_DISCOVERY_SELECT_TASKS = _DISCOVERY_TASKS | {sts.CardSelectTask.CODEX}

# "Choose any number" selects (Gamble / Exhaust-many): resolved by looping search+execute on the
# parked pbc (each pick sets a bit and re-opens; a confirm applies the set) until it leaves CARD_SELECT.
_PBC_MULTI_SELECT_TASKS = frozenset(_MULTI_CARD_SELECT_TASK_BY_ACTION.values())

# Every card-select task the drive can advance THROUGH (park the pbc instead of dropping + re-seeding).
# Anything outside this set (the engine-unimplemented Meditate/Nightmare/Recycle/Setup/Seek) re-seeds.
_PBC_PARK_SELECT_TASKS = (
    _PBC_THROUGH_SELECT_TASKS | _PBC_DISCOVERY_SELECT_TASKS | _PBC_MULTI_SELECT_TASKS
)


class STSLightspeedAgent:

    def __init__(self, chosen_class=PlayerClass.THE_SILENT, net=None, temperature=0.0, net_seed=0,
                 start_seed=None, ascension=0, sims=1000, watch=False, watch_pre_ms=1000, watch_post_ms=500):
        self.game = Game()
        self.errors = 0
        # Watch mode: when enabled, each net decision pauses watch_pre_ms, moves the cursor onto its
        # intended pick (hovering it where supported), pauses watch_post_ms, then commits -- so a
        # human can follow the play. Disabled = full speed.
        self.watch = watch
        self.watch_pre_ms = watch_pre_ms
        self.watch_post_ms = watch_post_ms
        # When set (a base-35 StS seed string, e.g. "54FYPZX13RLTT"), new runs start on this exact
        # seed -- used to replay a specific game (e.g. the captured slime-boss crash seed).
        self.start_seed = start_seed
        # Ascension level new runs start on (passed to StartGameAction).
        self.ascension = ascension
        # Set when heart1 skips the combat card reward, so _collect_combat_reward doesn't re-open it.
        self.skipped_cards = False
        # Toggles the two-step SHOP_ROOM transition (approach merchant, then leave).
        self.visited_shop = False
        # Persistent-bc shadow carry-over between combat decisions: the prior decision's reconstructed
        # bc, the action taken, and the live draw-pile top (id, upgrade) for forcing a Havoc draw. All
        # None between decisions or when an intervening screen invalidates the one-step prediction.
        self._shadow_prev_bc = None
        self._shadow_prev_action = None
        self._shadow_prev_draw_top = None
        self._shadow_prev_floor = None
        # Persistent-bc bridge (PERSISTENT_BC_PHASE2.md), gated behind STS_PBC_DRIVE (default OFF).
        # When on, one engine-advanced BattleContext is carried across a combat's decisions and the
        # live combat decision is searched on the reconciled pbc instead of a fresh per-decision
        # reconstruction -- so the search sees the engine-evolved hidden state the reconstruction can't
        # restore (Hexaghost's uniquePower0 move sequence, Book of Stabbing's stab count, escalating-
        # damage counters). The pbc's observables equal the fresh reconstruction, so its chosen action
        # is valid live exactly as a reconstruction action would be. Off => master (reconstruction only).
        self._pbc_drive = (
            os.environ.get("STS_PBC_DRIVE", "") not in ("", "0", "false", "False", "no"))
        self._pbc = None
        self._pbc_slots = None
        self._pbc_floor = None
        # Live turn we last advanced the pbc through via the out-of-handle_combat end-turn path
        # (get_next_action_in_game), to dedup a repeated end-turn emit (that path has no transient
        # guard). Reset per combat seed.
        self._pbc_last_end_turn = None
        # Short description of the action the pbc was last advanced by, tagged onto the next DESYNC so
        # a divergence is attributable to a card mis-sim vs a monster-turn mis-sim.
        self._pbc_prev_action_desc = None
        # Combat-decision signature of the state we last issued an action for. CommunicationMod can
        # emit a transient combat_state mid-resolution (e.g. the *_played_this_turn counters reset
        # while hand/energy/monsters still show the pre-play values) with ready_for_command=true; the
        # bridge would otherwise re-decide on it and fire a second command into the still-busy game
        # ("Invalid command: play, ready_for_command:false" -> fatal). We skip a decision whose sig
        # matches the last action's, waiting for the state to actually change.
        self._last_acted_combat_sig = None
        # When the dedup above first started skipping the current (unchanged) position. If it stays
        # unchanged far longer than any real resolution transient, the last command didn't take and we
        # re-decide rather than wait out the hard watchdog. None whenever the position is progressing.
        self._dedup_stuck_since = None
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
        # simulation_count_base defaults to 1000 (matches training/eval --mcts-simulations 1000);
        # the agent's defaults supply the jointly-tuned exploration/widening/eval weights.
        self.search_agent = sts.Agent()
        self.search_agent.simulation_count_base = sims
        # Live-sweepable override for the victory turn penalty (per-turn score cost of a win). The
        # compiled default already finishes winnable fights promptly; set STS_VICTORY_TURN_PENALTY
        # to retune without a C++ rebuild. Read-modify-write the whole EvalWeights struct so the
        # change sticks regardless of pybind's nested-member return policy.
        _vtp = os.environ.get("STS_VICTORY_TURN_PENALTY")
        if _vtp is not None:
            ew = self.search_agent.eval_weights
            ew.victory_turn_penalty = float(_vtp)
            self.search_agent.eval_weights = ew
            print(f"[search] victory_turn_penalty override = {ew.victory_turn_penalty}", file=sys.stderr)

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
        # Persist the raw incoming state BEFORE we touch it, so a silent C++ segfault during
        # processing (no Python traceback) still leaves the triggering state on disk for offline
        # repro. Overwrites each decision; the file is the last state we started to handle.
        try:
            if self.coordinator is not None and self.coordinator.last_raw_communication_state is not None:
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "last_instate.json")
                with open(path, "w") as f:
                    json.dump(self.coordinator.last_raw_communication_state, f)
        except Exception:
            pass
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
        # Not a card-play decision point -> clear the combat dedup baseline so a fresh fight's first
        # decision is never mistaken for a duplicate of the previous fight's last action.
        self._last_acted_combat_sig = None
        if self.game.end_available:
            # time.sleep(4)
            # The bot ends a turn two ways: the search picking END_TURN inside handle_combat, or here
            # when no card is playable (out of energy / nothing affordable). This path bypasses
            # handle_combat, so the persistent bc must be advanced through the END_TURN (and its
            # monster turn) here too -- otherwise it falls a full turn behind reality and its hidden
            # state never evolves through monster turns. Guarded; drops the pbc if it isn't cleanly at
            # a player decision.
            if self._pbc_drive and self._pbc is not None:
                live_turn = getattr(self.game, "turn", None)
                if live_turn != self._pbc_last_end_turn:
                    self._pbc_last_end_turn = live_turn
                    self._pbc_advance(sts.Action(sts.ActionType.END_TURN))
                    self._pbc_prev_action_desc = "END_TURN(auto)"
            return EndTurnAction()
        if self.game.cancel_available:
            return CancelAction()

    def get_next_action_out_of_game(self):
        return StartGameAction(self.chosen_class, ascension_level=self.ascension, seed=self.start_seed)

    def _bc_observe(self, bc):
        """Deterministic observable scalars of a bc, for the persistent-bc shadow check. Excludes
        hand/draw CONTENTS (draw order is RNG, expected to diverge) -- only the values a faithful
        engine must reproduce exactly after a card play: player hp/block/energy, hand size, per-monster
        hp/block, plus turn for a boundary guard."""
        o = {"php": getattr(bc.player, "curHp", None),
             "pblock": getattr(bc.player, "block", None),
             "energy": getattr(bc.player, "energy", None),
             "hand": getattr(bc.cards, "cardsInHand", None),
             "turn": getattr(bc, "turn", None),
             "mon": []}
        for i in range(bc.monsters.monsterCount):
            m = bc.monsters[i]
            o["mon"].append((m.curHp, m.block))
        return o

    def _force_observed_draw(self, prev_bc, want):
        """Make the shadow deterministic for cards that play off the top of the draw pile (Havoc): the
        engine draws a RANDOM card from the reconstructed (unknown-order) pile, but live drew a specific
        one, so a naive replay falsely diverges. `want` is the real top card (id, upgrade) captured from
        the live draw pile before the play, so just move that exact card to prev_bc's known draw-top and
        execute() replays the real card. Forcing only the single top card is correct for Havoc (plays one
        top card); harmless for any other play (an un-drawn known-top card is never popped). Not wrapped:
        a bug here should surface as a [shadow ERR] (the caller's handler), not be swallowed."""
        if want is None:
            return "no-top"   # draw pile empty at the prior decision: Havoc reshuffles, top is unforceable
        want_id, want_upg = want
        played = next((c for c in prev_bc.cards.drawPile
                       if c.id == want_id and c.upgrade_count == want_upg), None)
        if played is None:
            return "not-in-draw"
        # C++ encapsulates the (meaningless) pile order: match by id+upgrade, remove, return the card.
        removed = prev_bc.cards.removeFromDrawPile(played)
        if removed.id != sts.CardId.INVALID:
            prev_bc.cards.moveToDrawPileTop(removed)          # known-top -> drawTop pops it next
            return "forced"
        return "remove-failed"

    def _force_observed_monster_moves(self, prev_bc, truth_bc):
        """ET shadow: before advancing prev_bc by END_TURN, inject each monster's ACTUAL move so the
        engine replays it instead of rolling a (hidden) guess. A monster's move is only hidden when the
        live game defers its intent (Runic Dome -> rollMove leaves it INVALID with pending_move_rolls>0);
        the move it then makes this turn is its last_move at the NEXT decision, which the reconstruction
        already mapped into truth_bc.monsters[slot].moveHistory[1]. Commit that (setMove + drop the
        deferred roll) so the prediction uses the real move, not bc.rng. Only touches hidden-move
        monsters (a visible intent is already committed correctly). Returns (hidden, forced): how many
        monsters had a deferred/unset move, and how many of those we could fill from the observed last
        move. A still-divergent end-turn with hidden==forced is then a REAL signal (the move was right);
        hidden>forced stays unverifiable (the move itself was never observed)."""
        invalid = int(sts.MonsterMoveId.INVALID)
        n = prev_bc.monsters.monsterCount
        if n != truth_bc.monsters.monsterCount:
            return 0, 0   # layout changed (a monster died/spawned): slots may not align
        hidden = forced = 0
        for slot in range(n):
            pm = prev_bc.monsters[slot]
            if pm.pending_move_rolls > 0 or int(pm.moveHistory[0]) == invalid:
                hidden += 1
                observed = int(truth_bc.monsters[slot].moveHistory[1])
                if observed != invalid:
                    pm.commit_observed_move(observed)
                    forced += 1
        return hidden, forced

    def _shadow_card_play_check(self, truth_bc):
        """Phase-1 persistent-bc shadow (logging only, never affects live play). When the bot's last
        decision was a CARD play, advance the PRIOR reconstructed bc by that card via the engine
        (Action.execute) and diff the predicted deterministic state against this decision's freshly
        reconstructed (ground-truth) bc. A mismatch in player/monster hp/block/energy means the engine
        mis-simulated the card's effect vs the real game -- a fidelity bug contributing to the
        live<->offline gap. See PERSISTENT_BC_PLAN.md.

        Phase 1b also covers END_TURN: the reconstruction sets each monster's CURRENT move from the
        visible intent, so executing END_TURN runs the *real* monster moves -- any divergence in the
        post-monster-turn player/monster hp/block is a genuine monster-turn fidelity bug (the boss
        concern). The engine's roll of the NEXT intent during END_TURN diverges by RNG, but we don't
        compare intents so it's harmless here."""
        prev_bc = self._shadow_prev_bc
        prev_action = self._shadow_prev_action
        prev_draw_top = self._shadow_prev_draw_top
        prev_floor = self._shadow_prev_floor
        self._shadow_prev_bc = None
        self._shadow_prev_action = None
        self._shadow_prev_draw_top = None
        self._shadow_prev_floor = None
        if prev_bc is None or prev_action is None:
            return
        # A combat lives on exactly one floor; the next fight is a higher floor. If the floor changed,
        # prev_bc is the PREVIOUS combat's final state (the bot won/lost and moved on) and comparing it
        # to this fresh fight is a measurement artifact -- e.g. a dead-player end state (php 0) diffed
        # against a full-HP turn 1. The monster-count guard below misses this when both fights happen
        # to have the same number of monsters, so gate on the floor explicitly.
        if prev_floor != self.game.floor:
            return
        try:
            atype = prev_action.get_action_type()
            is_card = atype == sts.ActionType.CARD
            is_end_turn = atype == sts.ActionType.END_TURN
            if not (is_card or is_end_turn):
                return
            tag = "CARD" if is_card else "ET"
            desc = prev_action.print_desc(prev_bc)
            # Snapshot the block-relevant player modifiers BEFORE execute() mutates prev_bc, so a
            # pblock divergence reveals whether it's a real engine miscompute (dex/frail/vigor not
            # folded into the card) or pre-card reconstruction drift (preblk already wrong).
            pre_ctx = None
            mayhem = 0
            force_status = None
            mon_hidden = mon_forced = 0
            if is_card:
                p = prev_bc.player
                pre_ctx = (f"dex={p.dexterity} str={p.strength} "
                           f"frail={p.getStatus(sts.PlayerStatus.FRAIL)} "
                           f"preblk={p.block} prehp={p.curHp}")
                force_status = self._force_observed_draw(prev_bc, prev_draw_top)
                pre_ctx += f" force={force_status}"
            elif is_end_turn:
                # Energy/block carry-over context for ET divergences. Ice Cream conserves energy, so a
                # single 1-energy mis-sim recurs every subsequent turn -- knowing preE/energyPerTurn and
                # whether Ice Cream is held tells real-relic-reconstruction-gap from one-off mis-sim.
                p = prev_bc.player
                mayhem = p.getStatus(sts.PlayerStatus.MAYHEM)
                # draw/hand context: cardDrawPerTurn + the leftover (pre-end-turn) hand size + relics
                # pin a hand-size divergence to its cause (draw-bonus relic vs over-retain vs deck size).
                pre_ctx = (f"preE={p.energy} ept={p.energyPerTurn} preblk={p.block} "
                           f"prehp={p.curHp} icecream={int(p.hasRelic(sts.RelicId.ICE_CREAM))} "
                           f"mayhem={mayhem} draw={p.cardDrawPerTurn} prevhand={prev_bc.cards.cardsInHand} "
                           f"drawpile={len(prev_bc.cards.drawPile)} discard={len(prev_bc.cards.discardPile)} "
                           f"relics={[r.name for r in (self.game.relics or [])]}")
                # Mayhem plays the top of the draw pile at the START of the next turn, before the draw
                # (atStartOfTurn, engine BattleContext applyStartOfTurnPowers precedes DrawCards) -- the
                # same top we observe now (nothing touches the draw pile between end-turn and that play).
                # Force the live top onto prev_bc so the engine's Mayhem plays the real card, exactly like
                # the Havoc forcing. A single stack is then verifiable; an empty pile (reshuffle) or
                # stacked Mayhem (only the first play is forceable) stays unverifiable below.
                if mayhem > 0:
                    force_status = self._force_observed_draw(prev_bc, prev_draw_top)
                    pre_ctx += f" force={force_status}"
                # Inject each monster's actual move (observed via the next turn's last_move) so the
                # engine replays it instead of rolling a hidden Runic Dome guess. No-op for a normal
                # visible-intent fight (nothing is hidden). See _force_observed_monster_moves.
                mon_hidden, mon_forced = self._force_observed_monster_moves(prev_bc, truth_bc)
                if mon_hidden:
                    pre_ctx += f" rdmoves={mon_forced}/{mon_hidden}"
            # Per-monster state BEFORE executing the action, to tell a card-execute mis-sim (pred
            # differs from pre-card) from pre-existing reconstruction drift (already off pre-card),
            # plus identity/poison to spot the cause of a monster-hp divergence.
            mon_before = [(prev_bc.monsters[i].curHp, prev_bc.monsters[i].block,
                           prev_bc.monsters[i].getName(), prev_bc.monsters[i].poison)
                          for i in range(prev_bc.monsters.monsterCount)]
            # execute() asserts(false) (uncatchable SIGABRT) on an action invalid for prev_bc -- e.g. a
            # target slot that existed when the action was chosen but not in this prev_bc (a split/death
            # mismatch). Gate on validity; an invalid shadow replay is unverifiable, never fatal.
            if not prev_action.is_valid_action(prev_bc):
                print(f"[shadow unverifiable] {desc}: action invalid on prev_bc", file=sys.stderr)
                return
            prev_action.execute(prev_bc)          # advance the prediction (one card, or the monster turn)
            pred = self._bc_observe(prev_bc)
            truth = self._bc_observe(truth_bc)
            # Different combat slipped in -> can't compare. A CARD play must stay within the turn (a
            # turn boundary means an intervening end-turn we didn't attribute); END_TURN legitimately
            # crosses the boundary, so don't turn-guard it.
            if len(pred["mon"]) != len(truth["mon"]):
                return
            if is_card and pred.get("turn") != truth.get("turn"):
                return
            diffs = []
            for k in ("php", "pblock", "energy", "hand"):
                if pred[k] != truth[k]:
                    diffs.append(f"{k} pred {pred[k]} vs live {truth[k]}")
            for i in range(len(pred["mon"])):
                if pred["mon"][i] != truth["mon"][i]:
                    b = mon_before[i] if i < len(mon_before) else (None, None, "?", 0)
                    diffs.append(f"mon{i}={b[2]}(hp,blk) pred {pred['mon'][i]} vs live {truth['mon'][i]} "
                                 f"(pre-card ({b[0]},{b[1]}) poison={b[3]})")
            if diffs:
                ctx = f" [{pre_ctx}]" if pre_ctx else ""
                # Havoc on an EMPTY draw pile (force=no-top) reshuffles the discard with the live RNG
                # and plays the resulting top -- a card the shadow can't reproduce from the pre-play
                # state (the reshuffle order is unknowable, and exhaust[-1] misses a Power played off
                # the top). So it's not an engine mis-sim, just unverifiable; don't count it as a
                # divergence. A NON-Havoc empty-pile divergence is real (no top-of-pile draw) and stays.
                if is_card and force_status == "no-top" and "Havoc" in desc:
                    print(f"[shadow unverifiable] (CARD) after {desc} (Havoc reshuffled an empty "
                          f"draw pile -- live RNG): " + "; ".join(diffs) + ctx, file=sys.stderr)
                elif is_end_turn and mayhem > 0:
                    # Best-effort forcing the observed top makes the stable-top case verifiable (it then
                    # shows as [shadow ok]); but the draw-pile top at Mayhem-time often differs from the
                    # end-turn snapshot (the monster turn shuffles in status cards, the pile reshuffles,
                    # stacked Mayhem plays more than the one observed card). A residual diff there can't be
                    # attributed to a real engine mis-sim vs Mayhem's draw uncertainty, so don't count it.
                    print(f"[shadow unverifiable] (ET) after {desc} (Mayhem plays the draw-pile top at "
                          f"turn start -- live draw order): " + "; ".join(diffs) + ctx, file=sys.stderr)
                elif is_end_turn and mon_hidden > mon_forced:
                    # A monster's move was hidden (Runic Dome) and we couldn't recover it from the next
                    # turn's last_move (e.g. the monster died, or its own last move was also unobserved),
                    # so the engine had to roll it -- the divergence is move uncertainty, not a mis-sim.
                    print(f"[shadow unverifiable] (ET) after {desc} (Runic Dome -- "
                          f"{mon_hidden - mon_forced} hidden move(s) unobserved): "
                          + "; ".join(diffs) + ctx, file=sys.stderr)
                else:
                    # For a Runic Dome end-turn where every hidden move WAS forced from the observed
                    # last_move, this is now a real signal (the move was right), not move uncertainty.
                    print(f"[shadow DIVERGE] ({tag}) after {desc}: " + "; ".join(diffs) + ctx,
                          file=sys.stderr)
            else:
                print(f"[shadow ok] ({tag}) after {desc}", file=sys.stderr)
        except Exception as e:
            print(f"[shadow ERR] {type(e).__name__}: {e}", file=sys.stderr)

    def _check_attack_intent_target(self, first_action, spirecomm_action, slot_to_spire):
        """Catch a mis-targeted Spot Weakness (the live game rejects it unless the target intends to
        attack). Non-fatal: if the resolved live target isn't an attacking monster, dump the search
        target slot, the slot->spire mapping, and EVERY monster's live intent to stderr +
        runs/spot_weakness_mistarget.jsonl so the exact reconstruction/targeting fault is visible."""
        try:
            if not isinstance(spirecomm_action, PlayCardAction):
                return
            ci = spirecomm_action.card_index
            ti = spirecomm_action.target_index
            if ti is None or not (0 <= ci < len(self.game.hand)):
                return
            if self.game.hand[ci].card_id != "Spot Weakness":
                return
            mons = self.game.monsters
            tgt = mons[ti] if 0 <= ti < len(mons) else None
            attacking = (tgt is not None and not tgt.is_gone
                         and tgt.move_base_damage is not None and tgt.move_base_damage >= 0)
            if attacking:
                return
            sim_target = first_action.get_target_idx()
            info = {
                "card_index": ci, "target_index": ti,
                "sim_target_slot": sim_target,
                "slot_to_spire": {int(k): int(v) for k, v in slot_to_spire.items()},
                "resolved_target": (None if tgt is None else
                                    {"name": tgt.name, "id": tgt.monster_id, "is_gone": tgt.is_gone,
                                     "intent": str(getattr(tgt, "intent", None)),
                                     "move_id": tgt.move_id, "move_base_damage": tgt.move_base_damage}),
                "all_monsters": [{"idx": i, "name": m.name, "is_gone": m.is_gone,
                                  "intent": str(getattr(m, "intent", None)), "move_id": m.move_id,
                                  "move_base_damage": m.move_base_damage} for i, m in enumerate(mons)],
                "floor": self.game.floor,
            }
            print(f"[spot-weakness MISTARGET] target idx {ti} is not attacking: {info['resolved_target']}; "
                  f"all intents: {[(m['idx'], m['name'], m['intent']) for m in info['all_monsters']]}",
                  file=sys.stderr)
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                                "spot_weakness_mistarget.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(info) + "\n")
        except Exception as e:
            print(f"[spot-weakness check error] {e}", file=sys.stderr)

    def _combat_decision_sig(self):
        """A fingerprint of the live combat position that changes on any real player progress but NOT
        on a transient mid-resolution re-emit. Deliberately excludes the *_played_this_turn counters
        (the only fields the glitch transient disturbs -- they reset mid-resolution and every turn).
        Every action the combat search can issue through handle_combat moves at least one field here:
          - play a card  -> the card leaves the hand (hand ids); also energy/block/powers/monsters
          - drink a potion -> the slot empties (potions), so even a pure-stat buff potion (Strength,
            Dexterity, Regen, Ancient, Cultist...) that touches nothing else is still caught
          - a buff card/potion with no hand/energy footprint -> player powers
          - end the turn -> turn
        so a genuine new decision point always differs from the one before it, while the duplicate
        emit (identical everywhere except the excluded counters) matches and is skipped."""
        g = self.game
        p = getattr(g, "player", None)
        hand = tuple((c.card_id, c.upgrades) for c in (g.hand or []))
        mons = tuple((m.current_hp, m.block, str(m.intent), m.is_gone) for m in (g.monsters or []))
        potions = tuple(pot.potion_id for pot in (getattr(g, "potions", None) or []))
        powers = tuple(sorted((pw.power_id, pw.amount) for pw in (getattr(p, "powers", None) or [])))
        return (getattr(g, "turn", None),
                getattr(p, "energy", None), getattr(p, "block", None),
                hand, mons, potions, powers)

    def _pbc_reconcile_build(self, fresh_bc, fresh_slots):
        """M5 (drive): reconcile/seed self._pbc from the fresh reconstruction (observables + slot
        layout from reality, hidden monster state transplanted from the carried pbc) and RETURN it to
        search on. Does not advance -- the advance happens after the search picks the action. A failure
        here is a genuine reconcile bug, so it propagates (crash) rather than silently degrading to a
        fresh-reconstruction search -- which would defeat the whole point of driving on the pbc."""
        floor = self.game.floor
        if self._pbc is not None and floor != self._pbc_floor:
            self._pbc = None
        if self._pbc is None:
            self._pbc = fresh_bc.copy()
            self._pbc_last_end_turn = None
            print(f"[pbc] seeded persistent bc at floor {floor}", file=sys.stderr)
        else:
            self._pbc = self._pbc_reconcile(fresh_bc, fresh_slots)
        self._pbc_slots = dict(fresh_slots)
        self._pbc_floor = floor
        return self._pbc

    def _describe_action(self, action, bc):
        """Short tag for the action the pbc was advanced by, for DESYNC attribution."""
        try:
            at = action.get_action_type()
            if at == sts.ActionType.CARD:
                return f"CARD(src{action.get_source_idx()},tgt{action.get_target_idx()})"
            return str(at).split(".")[-1]
        except Exception:
            return "?"

    def _pbc_reconcile(self, fresh_bc, fresh_slots):
        """M2 reconcile (transplant form). Return a NEW bc: the faithful per-decision reconstruction
        `fresh_bc` (so every OBSERVABLE field -- hp/block/energy/piles/powers/move intents -- and the
        slot layout come from reality) with the engine-evolved HIDDEN monster state transplanted from
        the carried pbc. This is correct-by-construction: one bc, observables from the live snapshot,
        only the few counters the snapshot can't see carried forward.

        Hidden fields transplanted per monster (matched by stable live monster_index -- spirecomm keeps
        dead monsters listed, so indices survive deaths/repacks): `uniquePower0/1` always (pure hidden
        counters the reconstruction never sets), and `miscInfo` EXCEPT where the reconstruction already
        restored it from an observable intent (current move in the _MISCINFO damage/hits tables, e.g.
        Giant Head's It Is Time slam damage) -- there the observed value wins. A monster absent from
        the carried pbc (a split/summon child) keeps the reconstruction's values (no carry).

        Also emits `[pbc DESYNC]`: where the carried pbc's one-step prediction missed reality
        (player/monster hp/block/energy, monster intent) -- the artifact-free fidelity signal."""
        old = self._pbc
        old_live_to_slot = {live: slot for slot, live in self._pbc_slots.items()}
        new = fresh_bc.copy()
        d = []
        op, fp = old.player, fresh_bc.player
        for f in ("curHp", "block", "energy", "strength", "dexterity", "focus"):
            ov, nv = getattr(op, f), getattr(fp, f)
            if ov != nv:
                d.append(f"p.{f} {ov}->{nv}")
        # Deterministic monster fields -- a divergence here is a genuine engine mis-simulation (not the
        # RNG that makes .move diverge): strength/vuln/weak/poison drive incoming damage and ticks.
        mon_fields = ("curHp", "block", "strength", "vulnerable", "weak", "poison",
                      "metallicize", "platedArmor", "artifact")
        for s_b, live in fresh_slots.items():
            nm = new.monsters[s_b]
            os_ = old_live_to_slot.get(live)
            if os_ is None:
                continue                      # split/summon child: no carried hidden state
            om = old.monsters[os_]
            name = nm.getName()
            for f in mon_fields:
                ov, nv = getattr(om, f), getattr(nm, f)
                if ov != nv:
                    d.append(f"{name}.{f} {ov}->{nv}")
            omv, nmv = int(om.moveHistory[0]), int(nm.moveHistory[0])
            if omv != nmv:
                d.append(f"{name}.move {omv}->{nmv}")     # usually RNG roll divergence; reconcile keeps fresh
            nm.uniquePower0 = om.uniquePower0
            nm.uniquePower1 = om.uniquePower1
            # Keep the reconstruction's miscInfo only when it was restored from an observable intent;
            # otherwise carry the engine-evolved counter (Book of Stabbing stab count, Champ phase, ...).
            if nmv not in _MISCINFO_DAMAGE_MOVE_INTS and nmv not in _MISCINFO_HITS_MOVE_INTS:
                nm.miscInfo = om.miscInfo
        if d:
            print(f"[pbc DESYNC after {self._pbc_prev_action_desc}] {', '.join(d)}", file=sys.stderr)
        return new

    def _pbc_advance(self, action):
        """Carry the persistent bc forward through the action just committed live, to the next input
        point. Drop the pbc (re-seed next decision) when the action leaves the engine awaiting a
        sub-input (CARD_SELECT, handled by M4) or ends the combat, and on any engine error.

        execute() asserts(false) -- an uncatchable SIGABRT, not a Python exception -- on an action
        that isn't valid in the bc, so we MUST gate on is_valid_action first (raising a clean Python
        error instead of aborting the process). When DRIVING, the action was chosen by searching this
        exact pbc, so it MUST be valid -- an invalid action is a genuine divergence and we crash to
        surface it. When carrying in parallel (non-drive measurement), the action was chosen on a
        separate reconstruction, so drift can legitimately invalidate it: drop the pbc, never crash."""
        if not action.is_valid_action(self._pbc):
            if self._pbc_drive:
                raise RuntimeError("[pbc] chosen action invalid on driven persistent bc: "
                                   f"{self._describe_action(action, self._pbc)}")
            print("[pbc] chosen action invalid on persistent bc; dropping it (no execute)",
                  file=sys.stderr)
            self._pbc = None
            return
        try:
            action.execute(self._pbc)
        except Exception as e:
            if self._pbc_drive:
                raise
            print(f"[pbc] execute failed ({type(e).__name__}: {e}); dropping persistent bc",
                  file=sys.stderr)
            self._pbc = None
            return
        if self._pbc.outcome != sts.BattleOutcome.UNDECIDED:
            self._pbc = None            # combat ended -- nothing left to carry
            return
        ist = self._pbc.input_state
        if ist == sts.InputState.PLAYER_NORMAL:
            return                      # clean decision point
        # A played card opened a card-select sub-input. When driving, park the pbc at the select so
        # the in-combat card-select handler resolves it on this SAME bc and advances through (M4) --
        # preserving the carried hidden monster state across the whole card+select sequence rather
        # than dropping and re-seeding. Covers single (pile), Discovery/Codex (inject live candidates)
        # and multi (loop) selects; only the engine-unimplemented tasks fall through to re-seed.
        if (ist == sts.InputState.CARD_SELECT and self._pbc_drive
                and self._pbc.card_select_task in _PBC_PARK_SELECT_TASKS):
            print(f"[pbc] parked at card-select ({self._pbc.card_select_task}); "
                  f"resolving on the persistent bc", file=sys.stderr)
            return
        print(f"[pbc] not at a clean decision after execute "
              f"(input_state={ist}, outcome={self._pbc.outcome}); re-seeding next decision",
              file=sys.stderr)
        self._pbc = None

    def _pbc_advance_through_select(self, select_action):
        """Advance the parked pbc through an in-combat card-select by executing the chosen select
        action on it, resolving the CARD_SELECT sub-input back to the next player decision. The action
        was chosen by searching this same pbc, so it MUST be valid and MUST land on a clean player
        turn; a failure is a genuine divergence, so we crash to surface it. is_valid_action gates the
        execute (an invalid action would SIGABRT), turning that case into a clean Python error."""
        if not select_action.is_valid_action(self._pbc):
            raise RuntimeError("[pbc] select action invalid on driven persistent bc: "
                               f"{self._describe_action(select_action, self._pbc)}")
        select_action.execute(self._pbc)
        if (self._pbc.input_state != sts.InputState.PLAYER_NORMAL
                or self._pbc.outcome != sts.BattleOutcome.UNDECIDED):
            raise RuntimeError(f"[pbc] select left driven pbc unclean "
                               f"(input_state={self._pbc.input_state}, outcome={self._pbc.outcome})")
        self._pbc_prev_action_desc = "CARD_SELECT"   # DESYNC attribution for the next reconcile

    def _pbc_reconcile_at_select(self, fresh_bc, fresh_slots):
        """Adopt, as the new self._pbc for resolving an in-combat card-select, a fresh LIVE
        reconstruction (piles/observables + slot layout from reality) with the carried hidden monster
        state transplanted from the parked pbc. The parked pbc's OWN piles can diverge from the live
        select screen -- the select-opening card (Warcry, etc.) draws cards off the pbc's desynced RNG,
        so its resulting hand/discard differs and its pick may be a card not offered live. Rebuilding
        the pool from reality makes every pick live-valid (no fallback) while keeping the hidden-state
        carry. Returns the new bc (== self._pbc); the caller opens the select on it and searches."""
        new = self._pbc_reconcile(fresh_bc, fresh_slots)
        self._pbc = new
        self._pbc_slots = dict(fresh_slots)
        return new

    def handle_combat(self):
        self.capture_battle_state()
        # Step marker (see handle_screen): pinpoints a hang inside convert_combat_state / the search,
        # which otherwise leaves no clue (the "Running N simulations" log comes only after conversion).
        print(f"[step] handle_combat floor={self.game.floor} act={self.game.act} "
              f"turn={getattr(self.game, 'combat_round', '?')}", file=sys.stderr)
        # Drop a transient/duplicate emit: if the position hasn't changed since our last action, that
        # action is still resolving in the live game. Re-deciding now would send a second command into
        # a busy game (ready_for_command=false) and get a fatal "Invalid command". Return None so the
        # coordinator waits for the next state instead. Real progress changes the sig and we act again.
        sig = self._combat_decision_sig()
        if sig == self._last_acted_combat_sig:
            now = time.monotonic()
            if self._dedup_stuck_since is None:
                self._dedup_stuck_since = now
            if now - self._dedup_stuck_since < 8.0:
                print("[combat] position unchanged since last action (transient/duplicate emit); "
                      "waiting for the prior action to resolve", file=sys.stderr)
                return None
            # Unchanged for far longer than any real resolution transient: the last command didn't take
            # (dropped mid-emit, or an action that doesn't advance the live game), so waiting will only
            # burn out the 150s watchdog. Release the dedup and re-decide -- re-issuing the command
            # rescues a dropped send; a genuinely stuck position just re-loops here until the watchdog.
            print(f"[combat] position unchanged for {now - self._dedup_stuck_since:.0f}s -- prior action "
                  f"did not advance the game; re-deciding (dedup released)", file=sys.stderr)
        self._dedup_stuck_since = None
        self._last_acted_combat_sig = sig
        # Convert spirecomm game state to our internal format
        gc = spirecomm_to_gamecontext(self.game)
        bc, slot_to_spire = convert_combat_state(self.game, gc)
        self._shadow_card_play_check(bc)
        # Sanity-check the reconstruction against the live displayed intents before searching: any
        # monster whose engine-predicted attack damage disagrees with the live intent is being
        # mis-simulated (a wrong move-byte mapping or unrestored damage state), so the search would
        # mis-judge blocking. Fail loud -- a mis-simulated fight is worse than a stopped run.
        assert_intent_damage_matches(bc, self.game, slot_to_spire)
        # Card base-damage check. Warn-first (not hard) for now: unlike the monster intent check, the
        # captures predate the forked card base_damage field so this can't be validated offline, and
        # special-data cards (Searing Blow / Rampage / Genetic Algorithm) may need handling in
        # get_card_base_damage. Once a fork-live run shows it clean, promote to a hard assert.
        try:
            assert_card_damage_matches(bc, self.game)
        except AssertionError as e:
            print(f"[card-damage WARN] {e}", file=sys.stderr)
        print(bc, file=sys.stderr)

        # M5: when driving, search the reconciled persistent bc (observables == this reconstruction,
        # plus the engine-evolved hidden state the reconstruction can't restore). Its slot layout is
        # the reconstruction's (pbc is a copy of it), so search_slots == slot_to_spire and the chosen
        # action maps to live exactly as a fresh-reconstruction action would. When not driving, live
        # runs on the fresh reconstruction (the pbc is carried in parallel for measurement only).
        search_bc, search_slots = bc, slot_to_spire
        if self._pbc_drive:
            search_bc = self._pbc_reconcile_build(bc, slot_to_spire)

        # Configure the searcher with heart1's exact training/eval battle-search knobs
        # (exploration / chance + end-turn widening / eval weights, boss variants) and matching
        # per-decision sim count, via the shared SearchAgent config -- so live play uses the same
        # search heart1 was tuned around rather than a mistuned standalone BattleSearcher.
        searcher = sts.BattleSearcher(search_bc)
        simulation_count = self.search_agent.configure_searcher(searcher, search_bc)

        print("=" * 80, file=sys.stderr)
        print(f"Running {simulation_count} simulations for combat decision...", file=sys.stderr)

        # Get the best action (most visited child of root). A RuntimeError here is a C++ battle-
        # search throw on this converted state -- e.g. a splitting monster (Slimes) overflowing the
        # 5-slot MonsterGroup, a known conversion edge case. Dump the full crashing state (stderr +
        # runs/battle_search_crashes.jsonl) for root-causing, then re-raise (crash the run) rather than
        # limp on with a guessed EndTurn -- a mis-simulated fight that silently ends the turn is worse
        # than a stopped run, and the crash makes the conversion gap debuggable.
        try:
            searcher.search(simulation_count)
            first_action = searcher.get_best_action()
        except Exception as e:
            print(f"!!! BATTLE SEARCH CRASH ({type(e).__name__}: {e}) -- state dumped", file=sys.stderr)
            print(search_bc, file=sys.stderr)
            try:
                crash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "runs", "battle_search_crashes.jsonl")
                with open(crash_path, "a") as f:
                    raw = self.coordinator.last_raw_communication_state if self.coordinator else None
                    f.write(json.dumps({"error": str(e), "raw": raw}) + "\n")
            except Exception:
                pass
            raise

        # Map the search action to a spirecomm action (interpreted against the bc we searched on)
        spirecomm_action = map_search_action_to_spirecomm(first_action, search_bc, self.game, search_slots)

        print(f"Chosen action: {spirecomm_action}", file=sys.stderr)

        # Diagnostic: Spot Weakness is rejected live if its target doesn't intend to attack. If the
        # search aimed it at a monster the live game shows as non-attacking (or gone / out of range),
        # the play wastes the card -- capture the full target picture (non-fatal) to root-cause it.
        self._check_attack_intent_target(first_action, spirecomm_action, search_slots)

        # Print top 5 moves and their visit counts
        edges = searcher.get_root_edges()
        if edges:
            # Sort edges by visit count (descending)
            sorted_edges = sorted(edges, key=lambda e: e.node.simulation_count, reverse=True)
            print("Top 5 moves by visit count:", file=sys.stderr)
            for i, edge in enumerate(sorted_edges[:5]):
                action_desc = edge.action.print_desc(search_bc)
                visits = edge.node.simulation_count
                avg_value = edge.node.evaluation_sum / visits if visits > 0 else 0
                print(f"  {i+1}. {action_desc} - visits: {visits}, avg_value: {avg_value:.2f}", file=sys.stderr)

        # Persistent-bc shadow: remember this decision's reconstructed bc + chosen action so the next
        # handle_combat can check whether the engine advances it the same way the real game did. bc is
        # unmutated here (the searcher works on an internal clone) and first_action was chosen on it,
        # so it stays valid to replay next decision. Logging only.
        self._shadow_prev_bc = bc
        self._shadow_prev_action = first_action
        self._shadow_prev_floor = self.game.floor
        # Capture the TRUE top of the live draw pile so the shadow can force the exact card a top-of-deck
        # play (Havoc) draws. The reconstructed pile is deliberately in unknown order, but the raw live
        # order is real -- observing the top card up front is direct and robust to any side-effect of the
        # play (extra draws, a power card that never hits a pile), unlike a post-hoc pile diff.
        self._shadow_prev_draw_top = self._live_draw_top()

        # Advance the persistent bc by the action we committed live.
        if self._pbc_drive:
            # The pbc was already reconciled/built before the search (search_bc is self._pbc), so only
            # advance it here through the chosen action.
            if self._pbc is search_bc:
                # A card played off the top of the draw pile (Havoc) draws whatever is on top; the pbc's
                # reconstructed draw order is arbitrary, so without help its Havoc plays a DIFFERENT card
                # than live -- and if live's top is a select-opener (Headbutt), the pbc never opens the
                # matching select and the parked-select drive diverges. Force the observed live draw-top
                # onto the pbc first (same mechanism the shadow uses) so a top-of-deck play resolves to
                # the real card. No-op when live's top isn't in the pbc's draw pile or the pile is empty.
                if first_action.get_action_type() == sts.ActionType.CARD:
                    self._force_observed_draw(self._pbc, self._shadow_prev_draw_top)
                self._pbc_advance(first_action)
                self._pbc_prev_action_desc = self._describe_action(first_action, bc)
                if first_action.get_action_type() == sts.ActionType.END_TURN:
                    self._pbc_last_end_turn = getattr(self.game, "turn", None)

        return spirecomm_action

    def _live_draw_top(self):
        """The live draw pile's top card as (CardId, upgrades), or None if the pile is empty or the top
        card doesn't map. draw_pile[-1] is AbstractCard.getTopCard() -- the card a top-of-deck play
        (Havoc) or a start-of-turn Mayhem draws next."""
        if not self.game.draw_pile:
            return None
        top = self.game.draw_pile[-1]
        cid = map_card_id(top.card_id)
        if cid == sts.CardId.INVALID:
            return None
        return (cid, top.upgrades)

    def _pbc_driving_at_select(self, task):
        """True if the driven persistent bc is parked at the expected card-select `task`, so the select
        resolves on it (carrying the hidden monster state through the pick). False -> resolve the select
        on the fresh reconstruction instead, which is correct whenever the pbc isn't parked at this
        select:
          - not driving, or the pbc is unseeded (None) at a combat-start select (e.g. Gambling Chip
            before the first decision seeds the pbc); or
          - the pbc is at PLAYER_NORMAL because the select was opened by something the drive doesn't
            advance the pbc through -- a POTION played via the net path (Attack/Skill/Colorless Potion
            Discovery, Elixir/Gambler exhaust-many), or an unforceable Havoc top-of-deck play. Potions
            don't touch monster hidden state, so the pbc stays valid for the next decision's reconcile.
        Raises ONLY on a true contradiction: the pbc opened a card-select for a DIFFERENT task than live
        (same triggering card, divergent select) -- a genuine sim divergence to surface, not mask.

        (Havoc that plays a select-opener off the top DOES park correctly: _pbc_advance forces the
        observed live draw-top first, so the pbc plays the same card and opens the same select.)"""
        if not self._pbc_drive or self._pbc is None:
            return False
        if self._pbc.input_state == sts.InputState.CARD_SELECT:
            if self._pbc.card_select_task == task:
                return True
            raise RuntimeError(f"[pbc] driving: pbc parked at card-select {self._pbc.card_select_task} "
                               f"but live opened {task} (same play, divergent select)")
        return False   # PLAYER_NORMAL: select opened by a potion / combat-start / untracked action

    def mcts_card_select_action(self):
        """Resolve an in-combat card-select (Armaments/Headbutt/Warcry/Dual Wield/Exhume/...) with
        the combat MCTS -- the same way the search resolves it in-sim. Reconstruct the bc at the
        mid-resolution state (the live piles already reflect the triggering card being played), put
        it into the CARD_SELECT input state for that action's task, search, and translate the chosen
        pile index back to the live screen card. Fails loud on an unmapped action or a select the
        search can't place on the live screen."""
        action_name = self.game.current_action
        # A card-select screen intervened between combat decisions -- the prior card play didn't lead
        # directly to the next handle_combat, so the shadow's one-step prediction would be invalid.
        self._shadow_prev_bc = None
        self._shadow_prev_action = None
        self._shadow_prev_draw_top = None
        self._shadow_prev_floor = None
        # Invalidate the combat duplicate-emit signature: resolving this select is an action that
        # changes the position, but it's committed here (not through handle_combat), so the sig was
        # never updated for it. Without this reset, if the post-select combat position happens to match
        # the pre-select-card signature, handle_combat mis-reads it as a still-resolving duplicate and
        # never sends the next command -- a hang (observed after a Warcry put-back in act 3).
        self._last_acted_combat_sig = None
        # CardRewardScreen (the in-combat Discovery/potion choice) has no num_cards; it always picks 1.
        num = getattr(self.game.screen, "num_cards", None) or 1
        single_task = _CARD_SELECT_TASK_BY_ACTION.get(action_name)
        multi_task = _MULTI_CARD_SELECT_TASK_BY_ACTION.get(action_name)
        # Route to the multi-card path for a "choose any number" select. GamblingChip is ONLY ever
        # multi (discard any number at combat start), so route it by name -- the screen's max_cards
        # is sometimes absent on the combat-start frame, which would otherwise misroute it to the
        # single path. ExhaustAction is in BOTH tables (True Grit = one card; Elixir/Purity = any
        # number), so for it we disambiguate on num.
        if multi_task is not None and (single_task is None or num != 1):
            # The battle search does not enumerate these subsets -- it resolves them to "select
            # nothing" -- so playout_battle (and thus RL training) always picks zero. Drive the
            # search the same way and forward whatever it selects (empty => confirm nothing).
            gc = spirecomm_to_gamecontext(self.game)
            bc, slot_to_spire = convert_combat_state(self.game, gc)
            mnum = min(num, bc.cards.cardsInHand)
            # When driving, resolve on a bc reconciled from LIVE at the select (hand == the live screen)
            # with the carried hidden monster state, advancing the pbc through the whole select. NO
            # fallback: if it can't resolve, raise (crash) so the divergence is debuggable, not masked.
            if self._pbc_driving_at_select(multi_task):
                sel_bc = self._pbc_reconcile_at_select(bc, slot_to_spire)
                sel_bc.open_card_select(multi_task, mnum)
                return CardSelectAction(self._pbc_resolve_multi_select(multi_task, action_name))

            bc.open_card_select(multi_task, mnum)
            searcher = sts.BattleSearcher(bc)
            searcher.search(self.search_agent.configure_searcher(searcher, bc))
            sel_idxs = searcher.get_best_action().get_selected_idxs()
            hand = bc.cards.hand
            chosen = []
            for i in sel_idxs:
                if not (0 <= i < len(hand)):
                    raise RuntimeError(f"MCTS multi-select idx {i} out of range "
                                       f"(hand {len(hand)}, {multi_task})")
                chosen.append(self._match_live_select_card(hand[i]))
            print(f"[mcts] multi-select ({action_name}, {multi_task}) -> {len(chosen)} card(s)",
                  file=sys.stderr)
            return CardSelectAction(chosen)

        task = single_task
        if task is None:
            cards = [c.name for c in self.game.screen.cards]
            raise NotImplementedError(
                f"in-combat card-select current_action {action_name!r} unmapped "
                f"(screen {self.game.screen_type}, {len(cards)} cards: {cards}); "
                f"add it to _CARD_SELECT_TASK_BY_ACTION")
        offered = self.game.screen.cards

        gc = spirecomm_to_gamecontext(self.game)
        bc, slot_to_spire = convert_combat_state(self.game, gc)

        if task in _DISCOVERY_TASKS or task == sts.CardSelectTask.CODEX:
            # Generated-card choice (Discovery / Nilry's Codex): the candidates are the offered cards
            # themselves. Inject them and let the search pick; the chosen index maps straight back to
            # the live screen card. Codex uses its own task (added to the draw pile, not made free).
            ids = []
            for c in offered:
                cid = map_card_id(c.card_id)
                if cid == sts.CardId.INVALID:
                    raise ValueError(f"unknown offered card in discovery select: {c.card_id}")
                ids.append(cid)
            # When driving, resolve on a bc reconciled from LIVE at the select (a fresh, clean action
            # queue -- the parked pbc can still have start-of-turn effects pending, e.g. a Nilry's Codex
            # that fires mid start-of-turn, so executing the pick there doesn't drain to a clean player
            # turn) with the carried hidden monster state, then inject the live candidates and advance
            # through the pick. NO fallback: raise (crash) on failure so the divergence is debuggable.
            if self._pbc_driving_at_select(task):
                self._pbc_reconcile_at_select(bc, slot_to_spire)
                return self._pbc_resolve_discovery(task, ids, offered, action_name)
            if task == sts.CardSelectTask.CODEX:
                bc.open_codex_select(ids)
            else:
                bc.open_discovery_select(ids, 1, True)
            searcher = sts.BattleSearcher(bc)
            searcher.search(self.search_agent.configure_searcher(searcher, bc))
            sel_idx = searcher.get_best_action().get_select_idx()
            # Nilry's Codex is skippable: the engine's CODEX select validates idx in [0,4) and treats
            # idx == 3 (one past the 3 offered cards) as "skip" (Action.cpp / BattleSimulator); the live
            # CARD_REWARD advertises this as skip_available, and the mod routes skip == cancel. Discovery
            # (open_discovery_select) is not skippable, so this only fires for CODEX.
            if task == sts.CardSelectTask.CODEX and sel_idx == len(offered):
                print(f"[mcts] discovery ({action_name}) -> skip (idx {sel_idx})", file=sys.stderr)
                return CancelAction()
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

        # When driving, resolve on a bc reconciled from LIVE at the select (pool == the live screen)
        # with the carried hidden monster state transplanted, then advance the pbc through it. This
        # keeps the fidelity carry while guaranteeing the pick is live-valid. NO fallback: raise (crash)
        # on failure so any residual divergence is debuggable rather than masked.
        if self._pbc_driving_at_select(task):
            sel_bc = self._pbc_reconcile_at_select(bc, slot_to_spire)
            sel_bc.open_card_select(task, num)
            live_card, chosen_action = self._search_single_select(sel_bc, task, action_name,
                                                                  driven=True)
            self._pbc_advance_through_select(chosen_action)
            return CardSelectAction([live_card])

        bc.open_card_select(task, num)
        live_card, _ = self._search_single_select(bc, task, action_name, driven=False)
        return CardSelectAction([live_card])

    def _search_single_select(self, select_bc, task, action_name, driven):
        """Run the combat MCTS on `select_bc` (parked in CARD_SELECT for `task`) and return
        (live_card, chosen_action): the live screen card to commit, and the engine Action that made
        the pick (used to advance a driven pbc through the select). Raises if the search's index is
        out of range or its card isn't on the live screen."""
        searcher = sts.BattleSearcher(select_bc)
        searcher.search(self.search_agent.configure_searcher(searcher, select_bc))
        chosen_action = searcher.get_best_action()
        sel_idx = chosen_action.get_select_idx()

        pool_name = _CARD_SELECT_POOL_BY_TASK[task]
        pool = {"hand": select_bc.cards.hand, "discard": select_bc.cards.discardPile,
                "exhaust": select_bc.cards.exhaustPile, "draw": select_bc.cards.drawPile}[pool_name]
        if not (0 <= sel_idx < len(pool)):
            raise RuntimeError(f"MCTS card-select idx {sel_idx} out of range for the {pool_name} "
                               f"pile (size {len(pool)}, task {task})")
        chosen = pool[sel_idx]
        live_card = self._match_live_select_card(chosen)
        print(f"[mcts] card-select ({action_name}) -> {chosen.getName()}"
              f"{'+' if chosen.upgraded else ''} ({pool_name} idx {sel_idx}"
              f"{', pbc-driven' if driven else ''})", file=sys.stderr)
        return live_card, chosen_action

    def _pbc_resolve_discovery(self, task, ids, offered, action_name):
        """Drive an in-combat Discovery/Codex select on the parked pbc: OVERWRITE its rolled
        candidates with the live-observed offered cards (the pbc rolled its own from a desynced RNG;
        reality's offered set is observable), search, and advance the pbc through the pick. The chosen
        index lines up with `offered` (ids were built from it in order). Returns the live action;
        raises (crashes -- no fallback) on an invalid pick or an unclean resolution."""
        if task == sts.CardSelectTask.CODEX:
            self._pbc.open_codex_select(ids)
        else:
            self._pbc.open_discovery_select(ids, 1, True)
        searcher = sts.BattleSearcher(self._pbc)
        searcher.search(self.search_agent.configure_searcher(searcher, self._pbc))
        chosen_action = searcher.get_best_action()
        sel_idx = chosen_action.get_select_idx()
        if not chosen_action.is_valid_action(self._pbc):
            raise RuntimeError(f"discovery pick invalid on persistent bc ({task})")
        chosen_action.execute(self._pbc)          # advance the pbc through the select (CODEX idx==3 = skip)
        if (self._pbc.input_state != sts.InputState.PLAYER_NORMAL
                or self._pbc.outcome != sts.BattleOutcome.UNDECIDED):
            raise RuntimeError(f"discovery left pbc unclean (input_state={self._pbc.input_state})")
        self._pbc_prev_action_desc = "DISCOVERY"
        if task == sts.CardSelectTask.CODEX and sel_idx == len(offered):
            print(f"[mcts] discovery ({action_name}) -> skip (idx {sel_idx}, pbc-driven)",
                  file=sys.stderr)
            return CancelAction()
        if not (0 <= sel_idx < len(offered)):
            raise RuntimeError(f"MCTS discovery idx {sel_idx} out of range "
                               f"({len(offered)} offered, {action_name})")
        chosen = offered[sel_idx]
        print(f"[mcts] discovery ({action_name}) -> {chosen.card_id} (idx {sel_idx}, pbc-driven)",
              file=sys.stderr)
        if self.game.screen_type == ScreenType.CARD_REWARD:
            return ChooseAction(sel_idx)
        return CardSelectAction([chosen])

    def _pbc_resolve_multi_select(self, multi_task, action_name):
        """Drive an in-combat multi-card-select (Gamble / Exhaust-many) on the parked pbc. The engine
        models these sequentially: each SINGLE pick sets a bit and re-opens the screen; a MULTI confirm
        applies the running set. Loop search+execute on the pbc until it leaves CARD_SELECT, collecting
        the picked hand cards to mirror live (the search resolves these to "select nothing" in
        practice, so the common result is an empty confirm). Raises (crashes -- no fallback) on a
        pick that can't be matched live, non-convergence, or an unclean resolution."""
        chosen = []
        for _ in range(64):                       # bound: hand size is <= ~10; 64 is a safe backstop
            if (self._pbc.input_state != sts.InputState.CARD_SELECT
                    or self._pbc.card_select_task != multi_task):
                break
            searcher = sts.BattleSearcher(self._pbc)
            searcher.search(self.search_agent.configure_searcher(searcher, self._pbc))
            best = searcher.get_best_action()
            if best.get_action_type() == sts.ActionType.SINGLE_CARD_SELECT:
                idx = best.get_select_idx()
                hand = self._pbc.cards.hand
                if not (0 <= idx < len(hand)):
                    raise RuntimeError(f"multi-select idx {idx} out of hand range "
                                       f"({len(hand)}, {multi_task})")
                chosen.append(self._match_live_select_card(hand[idx]))
            if not best.is_valid_action(self._pbc):
                raise RuntimeError(f"multi-select action invalid on persistent bc ({multi_task})")
            best.execute(self._pbc)               # SINGLE re-opens (loop continues); MULTI confirms (loop exits)
        else:
            raise RuntimeError(f"multi-select did not converge ({multi_task})")
        if (self._pbc.input_state != sts.InputState.PLAYER_NORMAL
                or self._pbc.outcome != sts.BattleOutcome.UNDECIDED):
            raise RuntimeError(f"multi-select left pbc unclean (input_state={self._pbc.input_state})")
        self._pbc_prev_action_desc = "MULTI_CARD_SELECT"
        print(f"[mcts] multi-select ({action_name}, {multi_task}) -> {len(chosen)} card(s), pbc-driven",
              file=sys.stderr)
        return chosen

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

    def _watch_hover_index(self, action, actions=None, gc=None):
        """Live choice index (matching CommunicationMod's choice list) to hover for this net pick --
        card reward (take, or -1 = skip button), boss relic, any shop buy (card/relic/potion/removal),
        an event option (incl. Neow), a campfire option, or a map node -- else None (delay only)."""
        st = getattr(self.game, "screen_type", None)
        rt = getattr(action, "rewards_action_type", None)
        try:
            if st == ScreenType.CARD_REWARD:
                if rt == sts.RewardsActionType.CARD:
                    if 0 <= action.idx2 < len(getattr(self.game.screen, "cards", []) or []):
                        return action.idx2
                elif rt == sts.RewardsActionType.SKIP:
                    return -1   # sentinel: hover the skip button
            elif st == ScreenType.BOSS_REWARD and rt == sts.RewardsActionType.RELIC:
                if 0 <= action.idx1 < len(getattr(self.game.screen, "relics", []) or []):
                    return action.idx1
            elif st == ScreenType.SHOP_SCREEN:
                if rt == sts.RewardsActionType.SKIP:
                    return "leave"   # hover the Leave button (CancelAction exits the shop)
                # _shop_choice_index maps (type, idx) -> position in getAvailableShopItems order
                # (purge, then affordable cards, relics, potions) -- the mod's shop choice list.
                return self._shop_choice_index(rt, action.idx1)
            elif st == ScreenType.EVENT and actions:
                # The engine returns event options in ascending idx1 order, matching the live enabled
                # options (the mod's getActiveEventButtons order), so the chosen action's rank in the
                # action list IS the live choice index.
                if action in actions:
                    return actions.index(action)
            elif st == ScreenType.REST and gc is not None:
                # Map the chosen rest action (by its description, as net_rest_action does) to its
                # position in the live rest_options list (the mod's getValidRestRoomButtons order).
                desc = (action.getDesc(gc) or "").strip().lower()
                opts = list(getattr(self.game.screen, "rest_options", []) or [])
                rest_by_key = {"rest": RestOption.REST, "smith": RestOption.SMITH,
                               "recall": RestOption.RECALL, "dig": RestOption.DIG,
                               "lift": RestOption.LIFT, "toke": RestOption.TOKE}
                for key, opt in rest_by_key.items():
                    if desc.startswith(key) and opt in opts:
                        return opts.index(opt)
            elif st == ScreenType.MAP:
                # Path choice: hover the chosen next node (idx1 == node x). The mod lists next nodes in
                # x order, so the hover index is the chosen x's rank. No hover for a lone boss choice.
                if getattr(self.game.screen, "boss_available", False) and not getattr(self.game.screen, "next_nodes", None):
                    return None
                xs = sorted(n.x for n in (getattr(self.game.screen, "next_nodes", None) or []))
                if action.idx1 in xs:
                    return xs.index(action.idx1)
            elif st == ScreenType.GRID:
                # Card-select grid (remove/upgrade/transform): idx1 is the card's index in the grid,
                # which lines up with the mod's getGridScreenCards order.
                if 0 <= action.idx1 < len(getattr(self.game.screen, "cards", []) or []):
                    return action.idx1
        except Exception:
            pass
        return None

    def _watch_pause(self, desc, hover_idx=None):
        """Watch mode: pause `watch_pre_ms`, move the cursor onto the intended net choice (hover it,
        where the screen supports it), pause `watch_post_ms`, then return so the caller commits -- so
        a human can follow the play. No-op at full speed (watch disabled)."""
        if not self.watch:
            return
        # Pause BEFORE the cursor moves -- the screen sits a beat before the cursor travels to the pick.
        if self.watch_pre_ms > 0:
            time.sleep(self.watch_pre_ms / 1000.0)
        if hover_idx is not None and self.coordinator is not None:
            self.coordinator.send_message(f"hover {hover_idx}")
            # `hover` is a fire-and-forget on-screen signal: it warps the cursor but does NOT consume
            # the game's command-readiness (the choice screen is still waiting for the real pick) and
            # the mod replies with no state. send_message just cleared game_is_ready, so restore it --
            # otherwise the real pick can't execute and the run stalls until the 30s silence-nudge.
            self.coordinator.game_is_ready = True
        print(f"[watch] {desc}{'' if hover_idx is None else f' [hover {hover_idx}]'} -- "
              f"pre {self.watch_pre_ms}ms / post {self.watch_post_ms}ms", file=sys.stderr)
        # Pause AFTER the cursor moves, before the caller commits the pick.
        if self.watch_post_ms > 0:
            time.sleep(self.watch_post_ms / 1000.0)

    def net_pick_action(self, gc, action_filter=None):
        """Run heart1 on gc's current choice screen and return the chosen sts.GameAction (in
        GameContext space), or None if construct_choice can't represent this screen (so the
        caller fails loud). Real errors propagate -- we don't play on a guessed state.

        action_filter, if given, is a predicate over sts.GameAction; actions it rejects are
        masked out of the choice set so the net never selects them (used to hide potion buys
        when the belt is full). If filtering leaves no actions, returns None.

        Uses playouts.choose_overworld_action -- the SAME decision core rl_train.run_episode uses
        for training/eval -- so heart1 makes the same choice live as it did in training. temperature
        <= 0 (the deploy default) picks greedily; > 0 samples with net_rng."""
        from playouts import construct_choice, choose_overworld_action

        obs = sts.getNNRepresentation(gc)
        actions = sts.GameAction.getAllActionsInState(gc)
        if action_filter is not None:
            actions = [a for a in actions if action_filter(a)]
            if not actions:
                return None
        choice = construct_choice(gc, obs, actions)
        if choice is None:
            return None
        action, desc, _path, _idx, _logp, _val = choose_overworld_action(
            self.net, choice, gc, self.net_rng, temperature=self.temperature)
        self._watch_pause(desc or str(action), self._watch_hover_index(action, actions, gc))
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
        """heart1's pick on an out-of-combat grid select (transform/upgrade/remove/obtain -- shop card
        removal, rest-site smith, event transforms, and fixed-count multi-card picks like Astrolabe's
        transform-3). For a fixed-count grid spirecomm's CardSelectAction requires ALL still-needed
        cards in one call (a partial selection raises), so pick num_remaining distinct cards here:
        query the net once per card, excluding the indices already picked, then submit them together
        (CardSelectAction clicks each and confirms). Returns None (-> fail loud) for in-combat selects
        (the combat MCTS's job) and 'choose any number' selects."""
        scr = self.game.screen
        if self.game.in_combat:
            return None
        if getattr(scr, "any_number", False):
            return None
        # Pick only from the unselected cards (same filter as the gc reconstruction), so an index from
        # the net maps to a card not yet chosen -- re-choosing a selected card would toggle it off.
        selected_uuids = {c.uuid for c in scr.selected_cards}
        selectable = [c for c in scr.cards if c.uuid not in selected_uuids]
        num_remaining = scr.num_cards - len(scr.selected_cards)
        if not selectable or num_remaining <= 0:
            return None
        gc = spirecomm_to_gamecontext(self.game)
        chosen = []
        chosen_idxs = set()
        while len(chosen) < num_remaining:
            action = self.net_pick_action(
                gc, action_filter=lambda a: a.idx1 not in chosen_idxs)
            if action is None:
                break
            idx = action.idx1
            if not (0 <= idx < len(selectable)) or idx in chosen_idxs:
                break
            chosen_idxs.add(idx)
            chosen.append(selectable[idx])
        if len(chosen) != num_remaining:
            return None
        print(f"[net] grid select -> {[c.card_id for c in chosen]} "
              f"({num_remaining} of {scr.num_cards})", file=sys.stderr)
        return CardSelectAction(chosen)

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

    def _shop_item_name(self, rtype, idx1):
        """The chosen shop item's name as CommunicationMod stores it in the shop choice_list
        (getShopScreenChoices: card.name.toLowerCase()/relic.name/potion.name/'purge')."""
        scr = self.game.screen
        if rtype == sts.RewardsActionType.CARD_REMOVE:
            return "purge"
        pool = {sts.RewardsActionType.CARD: getattr(scr, "cards", None),
                sts.RewardsActionType.RELIC: getattr(scr, "relics", None),
                sts.RewardsActionType.POTION: getattr(scr, "potions", None)}.get(rtype)
        if pool is not None and 0 <= idx1 < len(pool):
            return getattr(pool[idx1], "name", None)
        return None

    def _shop_choice_index(self, rtype, idx1):
        """Index of the chosen shop item in CommunicationMod's LIVE choice_list -- the ground truth for
        'choose <N>' (getShopScreenChoices: 'purge', then affordable cards, relics, potions, BY NAME).
        Match the chosen item's NAME against the live list rather than reconstructing the affordable
        set (a positional reconstruction can desync from the mod, and an out-of-range 'choose N'
        silently no-ops and WEDGES the shop -- even a following leave then hangs). Duplicate names (two
        same-named cards/potions, possibly at DIFFERENT costs) are disambiguated by the chosen item's
        rank among affordable same-named items of its pool -- the live slots are exactly those, in pool
        order -- bounded by the live slots so it can never go out of range. RAISES with full context if
        the pick can't be resolved (unaffordable/absent pick, or a rank mismatch): we fail loud to debug
        the desync rather than mask it with a hang-prone by-name buy."""
        cl = [str(c) for c in (self.game.choice_list or [])]
        gold = self.game.gold
        scr = self.game.screen
        want = self._shop_item_name(rtype, idx1)
        pool = {sts.RewardsActionType.CARD: getattr(scr, "cards", None),
                sts.RewardsActionType.RELIC: getattr(scr, "relics", None),
                sts.RewardsActionType.POTION: getattr(scr, "potions", None)}.get(rtype)
        price = pool[idx1].price if (pool is not None and 0 <= idx1 < len(pool)) else None

        def fail(why):
            raise RuntimeError(f"shop choice unresolved ({why}): {rtype} idx {idx1} name={want!r} "
                               f"price={price} gold={gold} choice_list={cl}")

        if want is None:
            fail("no item name")
        w = want.lower()
        matches = [i for i, c in enumerate(cl) if c.lower() == w]
        if not matches:
            fail("name absent from live choice_list")
        # Purge is a unique named service with no pool/price.
        if rtype == sts.RewardsActionType.CARD_REMOVE:
            return matches[0]
        # Map the chosen item to its live slot by its rank among AFFORDABLE same-named items of its
        # pool that precede it -- the live slots are exactly those, in pool order. This handles
        # same-named items at DIFFERENT costs (where the pool has more entries than live slots because
        # some are unaffordable); the unique case is just rank 0. Bounded by the live slots.
        if price is None:
            fail("bad pool/idx")
        if price > gold:
            fail("chosen item is not affordable (net picked an unbuyable item)")
        rank = sum(1 for j in range(idx1)
                   if getattr(pool[j], "name", None) and pool[j].name.lower() == w
                   and pool[j].price <= gold)
        if rank >= len(matches):
            fail(f"rank {rank} >= {len(matches)} live slots (affordability desync)")
        return matches[rank]

    def net_shop_action(self):
        """heart1's shop decision: buy a card/relic/potion, start a card removal, or leave. The
        engine Shop (injected with live prices) makes getAllActionsInState offer exactly the
        affordable buys, so the net only ever picks something we can afford. Returns a spirecomm
        Action, or None to fail loud. One purchase per call; the shop screen re-opens for the next."""
        gc = spirecomm_to_gamecontext(self.game)
        # A full belt makes BuyPotionAction raise (kills the run), and the engine sim can still
        # offer potion buys, so mask them out of the choice set rather than failing loud later.
        # TODO: instead of masking, offer a potion-discard-then-buy option.
        # Sozu blocks ALL potion obtaining, so the live game silently rejects a shop potion buy
        # (purchasePotion -> obtainPotion returns false) and the net would re-pick it forever -- mask
        # potion buys just like a full belt. (getAllActionsInState ignores belt capacity, so this
        # decision-time mask is the only gate.)
        has_sozu = any(map_relic_id(r.name) == sts.RelicId.SOZU for r in (self.game.relics or []))
        mask_potions = self.game.are_potions_full() or has_sozu
        pot_filter = ((lambda a: a.rewards_action_type != sts.RewardsActionType.POTION)
                      if mask_potions else None)
        action = self.net_pick_action(gc, action_filter=pot_filter)
        if action is None:
            return None
        rtype = action.rewards_action_type
        shop = self.game.screen
        # Buy by the choice-list INDEX (resolved against the live choice_list by name, duplicate-safe);
        # _shop_choice_index raises with full context if the pick can't be resolved rather than falling
        # back to a hang-prone by-name buy, so a desync fails loud for debugging.
        if rtype == sts.RewardsActionType.CARD:
            chosen = shop.cards[action.idx1]
            ci = self._shop_choice_index(rtype, action.idx1)
            print(f"[net] shop -> buy card {chosen.card_id} ({chosen.price}g) [choice {ci}]", file=sys.stderr)
            return ChooseAction(choice_index=ci)
        if rtype == sts.RewardsActionType.RELIC:
            chosen = shop.relics[action.idx1]
            ci = self._shop_choice_index(rtype, action.idx1)
            print(f"[net] shop -> buy relic {chosen.name} ({chosen.price}g) [choice {ci}]", file=sys.stderr)
            return ChooseAction(choice_index=ci)
        if rtype == sts.RewardsActionType.POTION:
            # Potion buys are masked out above when the belt is full or Sozu blocks obtaining, so
            # reaching here means the buy is actually possible; assert to catch any masking
            # regression before BuyPotionAction can raise / the live buy silently no-ops.
            assert not mask_potions, "potion buy reached net_shop_action despite mask"
            chosen = shop.potions[action.idx1]
            ci = self._shop_choice_index(rtype, action.idx1)
            print(f"[net] shop -> buy potion {chosen.potion_id} ({chosen.price}g) [choice {ci}]", file=sys.stderr)
            return ChooseAction(choice_index=ci)
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
        # Step marker: the last one printed before a freeze names the exact decision that hung (the
        # main capture only flushes per-decision, so a mid-decision C++ spin leaves no capture clue).
        print(f"[step] handle_screen {st} floor={self.game.floor} act={self.game.act}", file=sys.stderr)
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
            # An unmapped in-combat card-select raises NotImplementedError with the live action name +
            # offered cards. We let it propagate (crash the run) rather than blindly pick card 0 -- a
            # silent wrong pick masks the missing mapping; the crash surfaces exactly what to add to
            # _CARD_SELECT_TASK_BY_ACTION.
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
        """Take the post-combat rewards. Gold (always) and potions (when the belt has room) are
        free, no-decision pickups, as are relics UNLESS a key shares the screen. A sapphire key and
        the relic are mutually exclusive (taking the relic clears the key, executeRewardsAction in
        GameAction.cpp), so when both are present the relic-vs-key choice is a real value decision --
        heart1 makes it (relic identity visible via construct_choice), the same as run_episode.

        A CARD reward opens the separate CARD_REWARD screen where heart1 chooses the card (its
        identities are opaque here), so on this screen the net decides relic-vs-key WITHOUT the card
        in view -- an unavoidable live-play split run_episode doesn't have. skipped_cards (set when
        heart1 skipped the card) stops us re-opening it."""
        rewards = self.game.screen.rewards
        has_key = any(r.reward_type in (RewardType.EMERALD_KEY, RewardType.SAPPHIRE_KEY)
                      for r in rewards)

        # In watch mode, hover the reward-list item being taken before committing it.
        def take(i):
            self._watch_pause(f"reward {rewards[i].reward_type}", i)
            return CombatRewardAction(rewards[i])

        # Free pickups, one per call (the screen re-opens for the next).
        for i, reward_item in enumerate(rewards):
            if reward_item.reward_type in (RewardType.GOLD, RewardType.STOLEN_GOLD):
                return take(i)
            if reward_item.reward_type == RewardType.POTION and not self.game.are_potions_full():
                return take(i)
            if reward_item.reward_type == RewardType.RELIC and not has_key:
                return take(i)

        # Relic and key both on the screen: heart1 decides which to take.
        if has_key:
            decided = self._net_relic_or_key_action(rewards)
            if decided is not None:
                return decided

        # Only the card reward (and/or skip) remains.
        for i, reward_item in enumerate(rewards):
            if reward_item.reward_type == RewardType.POTION and self.game.are_potions_full():
                continue
            if reward_item.reward_type == RewardType.CARD and self.skipped_cards:
                continue
            return take(i)
        self.skipped_cards = False
        self._watch_pause("proceed", "proceed")   # hover the Proceed button before leaving the screen
        return ProceedAction()

    def _net_relic_or_key_action(self, rewards):
        """heart1's relic-vs-key pick when a chest/elite offers both (mutually exclusive for a
        sapphire key). The reconstructed gc is on the REWARDS screen with the relic and key injected,
        so net_pick_action -> construct_choice exposes the relic alongside TAKE_KEY. Returns a
        CombatRewardAction or None to fall through. Defaults to the KEY on any ambiguity (net chose
        skip / an unrepresentable action / reconstruction diverged) -- a heart agent never forfeits a
        free key, so this can't regress heart access even if the net pick is off."""
        key_item = next((r for r in rewards
                         if r.reward_type in (RewardType.EMERALD_KEY, RewardType.SAPPHIRE_KEY)), None)
        relic_items = [r for r in rewards if r.reward_type == RewardType.RELIC]
        gc = spirecomm_to_gamecontext(self.game)
        if gc.screen_state == sts.ScreenState.REWARDS:
            action = self.net_pick_action(gc)
            if action is not None and action.rewards_action_type == sts.RewardsActionType.RELIC:
                idx = action.idx1
                if 0 <= idx < len(relic_items):
                    print(f"[net] reward relic-vs-key -> relic {relic_items[idx].relic.name}",
                          file=sys.stderr)
                    return CombatRewardAction(relic_items[idx])
        if key_item is not None:
            print("[net] reward relic-vs-key -> key", file=sys.stderr)
            return CombatRewardAction(key_item)
        return None


DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "heart1.pt.iter_2575")


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
    # Determinism: the combat MCTS is already seeded (BattleSearcher uses bc.seed+floor) and runs in
    # pure C++, but the policy net's CUDA forward is not -- cuBLAS/cuDNN can return slightly different
    # floats across processes, occasionally flipping an argmax on a near-tie in an out-of-combat
    # decision, which cascades into a divergent run. Pin every RNG and force deterministic GPU kernels
    # so a given seed replays identically (needed to reproduce a specific loss/crash for debugging).
    # cuBLAS determinism additionally requires CUBLAS_WORKSPACE_CONFIG; set it here (read lazily when
    # the cuBLAS handle is first created, which is after this) so it survives the mod's config rewrite.
    # warn_only keeps a rare kernel-less op from aborting a live run.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
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
    # Each knob defaults from an env var when present. ModTheSpire/CommunicationMod re-normalizes
    # config.properties at startup and drops appended CLI flags, but preserves env vars set via the
    # command's `/usr/bin/env VAR=val ...` prefix (like STS_COMM_CAPTURE) -- so run_live.sh passes
    # these as env vars, and explicit CLI flags still override for manual invocations.
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("STS_TEMPERATURE", 0.0)),
                       help="Network action-sampling temperature (0 = greedy/argmax)")
    parser.add_argument("--seed", default=os.environ.get("STS_START_SEED") or None,
                       help="Start runs on this exact base-35 StS seed string (e.g. 54FYPZX13RLTT) "
                            "to replay a specific game")
    parser.add_argument("--ascension", type=int, default=int(os.environ.get("STS_ASCENSION", 0)),
                       help="Ascension level to start new runs on (0-20)")
    parser.add_argument("--sims", type=int, default=int(os.environ.get("STS_SIMS", 1000)),
                       help="Combat MCTS simulations per decision (simulation_count_base)")
    parser.add_argument("--watch", action="store_true",
                       default=("STS_WATCH_PRE_MS" in os.environ or "STS_WATCH_POST_MS" in os.environ),
                       help="Enable watch mode (also auto-enabled by setting either watch delay / its "
                            "env var): at each net decision pause, move the cursor onto the intended "
                            "pick, pause, then commit. Off = full speed.")
    parser.add_argument("--watch-pre-ms", type=int, default=int(os.environ.get("STS_WATCH_PRE_MS", 1000)),
                       help="Watch mode: ms to wait BEFORE moving the cursor to the pick (default 1000).")
    parser.add_argument("--watch-post-ms", type=int, default=int(os.environ.get("STS_WATCH_POST_MS", 500)),
                       help="Watch mode: ms to wait AFTER moving the cursor, before committing (default 500).")

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
                               start_seed=args.seed, ascension=args.ascension, sims=args.sims,
                               watch=args.watch, watch_pre_ms=args.watch_pre_ms,
                               watch_post_ms=args.watch_post_ms)
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
            result = coordinator.play_one_game(current_class, ascension_level=args.ascension,
                                               seed=args.seed)
            games_played += 1
            # Split a victory into a heart kill (reached act 4) vs an act-3-only win -- a heart-run
            # agent's act-3 wins mean it failed to collect the keys, so they are NOT heart wins.
            max_act = getattr(coordinator, "last_game_max_act", 0)
            max_floor = getattr(coordinator, "last_game_max_floor", 0)
            kind = "heart" if (result and max_act >= 4) else "act3" if result else "loss"
            print(f"Game {games_played} completed with result: {result} "
                  f"(max_act={max_act} max_floor={max_floor} kind={kind})", file=sys.stderr)
        except KeyboardInterrupt:
            print("Interrupted by user", file=sys.stderr)
            break
        except Exception as e:
            print(f"Game error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Dump the last raw communication state so a hang/crash is debuggable after the fact -- the
            # errlog otherwise keeps only [step]/[net] summaries, not the raw screen (choice_list,
            # available_commands, ready_for_command, prices) needed to root-cause e.g. a wedged shop buy.
            try:
                raw = getattr(coordinator, "last_raw_communication_state", None)
                if raw is not None:
                    dump = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                                        "game_error_states.jsonl")
                    with open(dump, "a") as f:
                        f.write(json.dumps({"error": str(e), "raw": raw}) + "\n")
                    print(f"[dump] last raw state -> {dump}", file=sys.stderr)
            except Exception as de:
                print(f"[dump] failed to write last raw state: {de}", file=sys.stderr)
            break


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run test
        print("Testing spirecomm to GameContext converter...")
        test_basic_conversion()
    else:
        # Arguments provided - run CLI
        run_agent_cli()