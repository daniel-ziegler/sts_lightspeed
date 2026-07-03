"""Spirecomm-string <-> engine-enum mappings (cards, relics, potions, monsters, classes,
screens) plus live-power application, validated at import for full coverage."""

import sys

import slaythespire as sts
from spirecomm.spire.character import PlayerClass
from spirecomm.spire import card, relic
from spirecomm.spire.screen import ScreenType



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
    # engine's minion-gated logic is correct on a reconstructed fight: Feed's max-HP gain is denied
    # on a minion kill, and combat-end / minion-leader checks key on it.
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
# silently-dropped power mis-simulates invisibly. Non-Ironclad powers (Watcher/Defect/Silent) can
# never reach an Ironclad game,
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
    on any other unmapped power -- a silently-dropped monster power mis-simulates invisibly."""
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
    # applied on a PRIOR turn (the round it was cast has ended), so clear the flag; leaving it set makes
    # the engine skip the status's first end-of-round tick on EVERY reconstruction (e.g. a Cultist's
    # Ritual never gains strength in the search's lookahead).
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


