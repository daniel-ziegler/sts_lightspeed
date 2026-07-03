"""Engine action -> spirecomm command translation, and the live-action -> CardSelectTask
tables the in-combat select handling keys on."""

import sys

import slaythespire as sts
from spirecomm.spire import character
from spirecomm.spire import card, relic, game
from lightspeed.bridge.mappings import map_potion_id
from lightspeed.bridge.overworld import spirecomm_to_gamecontext
from spirecomm.communication.action import (
    Action, ChooseAction, EndTurnAction, PlayCardAction, PotionAction,
)




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

