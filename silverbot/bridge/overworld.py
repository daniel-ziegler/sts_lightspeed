"""Live out-of-combat state -> engine GameContext conversion: events, screen state, deck,
relics, shops, rewards."""


import slaythespire as sts
from spirecomm.spire import card, relic, game, screen
from silverbot.bridge.mappings import (
    CHARACTER_CLASS_MAPPING, SCREEN_STATE_MAPPING, _OPTION_GATING_RELIC_VALUES, convert_deck,
    convert_relics, map_card_id, map_character_class, map_potion_id, map_relic_id,
)




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


# Live AbstractRoom class name (CommunicationMod's room_type) -> engine Room. Unlisted rooms
# (Neow/debug rooms) fall back to NONE at the lookup site.
_ROOM_BY_LIVE_TYPE = {
    "MonsterRoom": sts.Room.MONSTER,
    "MonsterRoomElite": sts.Room.ELITE,
    "MonsterRoomBoss": sts.Room.BOSS,
    "EventRoom": sts.Room.EVENT,
    "ShopRoom": sts.Room.SHOP,
    "RestRoom": sts.Room.REST,
    "TreasureRoom": sts.Room.TREASURE,
    "TreasureRoomBoss": sts.Room.BOSS_TREASURE,
}


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
    # Room kind, from the live AbstractRoom class name. Consumers must not treat an unknown room as
    # anything specific, so default to NONE. The one room/encounter distinction that matters in
    # combat: postBattleHealedHp only credits the act-transition heal for a boss ENCOUNTER fought in
    # the boss ROOM -- Mind Bloom's I Am War is the same encounter in an EventRoom, with no heal.
    gc.cur_room = _ROOM_BY_LIVE_TYPE.get(spire_game.room_type, sts.Room.NONE)

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
        # Re-applying them double-counts (a Pear adds +10 maxHP on top of the live value, so the
        # search plays with phantom HP and under-estimates danger). Overwrite HP/gold with live truth.
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


