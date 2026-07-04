"""Live combat state -> engine BattleContext conversion: monster groups/moves/hidden state,
card piles and instances, and the reconstruction-fidelity asserts."""

import sys

import slaythespire as sts
from spirecomm.spire import card, game
from lightspeed.bridge.mappings import (
    _MONSTER_IDS_SKIP_IN_COMBAT, _RELIC_COUNTER_ATTR, _normalize_relic_name,
    apply_monster_power, apply_player_power, apply_the_bomb, map_card_id,
    map_monster_string_to_id, map_relic_id,
)




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
    # type == BOSS (SlaversCollar.java onEnergyRecharge). Both boss and elite fights can be reached
    # via an EVENT (Mind Bloom's boss, Dead Adventurer / Colosseum elites), where room_type is
    # EventRoom -- so detect both by their encounter signature (elite/boss encounters never appear
    # as normal fights), mirroring the engine's isEliteEncounter/isBossEncounter gate.
    is_boss_fight = _infer_boss_encounter(spire_game.monsters) != sts.MonsterEncounter.INVALID
    is_elite_fight = _infer_elite_encounter(spire_game.monsters) != sts.MonsterEncounter.INVALID
    if bc.player.hasRelic(sts.RelicId.SLAVERS_COLLAR) and (is_elite_fight or is_boss_fight):
        energy_per_turn += 1
    bc.player.energyPerTurn = energy_per_turn

    # +cards/turn relics: Snecko Eye (+2), Ring of the Serpent (+1, Silent-only). Base draw is 5.
    bc.player.cardDrawPerTurn = (5
                                 + 2 * bc.player.hasRelic(sts.RelicId.SNECKO_EYE)
                                 + 1 * bc.player.hasRelic(sts.RelicId.RING_OF_THE_SERPENT))

    # Clear the initialized cards to avoid mixing with spirecomm state
    bc.cards.clear()

    # Frozen Eye: the player genuinely sees the full draw pile order, so convert the pile in
    # order-observed mode -- every card definitely placed in the live order, deterministic pops,
    # concrete reshuffles: the same dynamics a native battle with the relic gets from
    # CardManager::init. Must be set while the pile is still empty.
    frozen_eye = bc.player.hasRelic(sts.RelicId.FROZEN_EYE)
    if frozen_eye:
        bc.set_draw_order_observed(True)
    
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
            if power.power_id == "Panache":
                # PanachePower reports the 5..1 card countdown in `amount` and the per-proc damage
                # in `damage`; the engine keeps them the other way around (status amount = damage,
                # panacheCounter = countdown). Routed through the generic buff() the countdown
                # would land in the damage slot and the counter would stay 0 -- which fires a
                # phantom AoE on EVERY card play.
                bc.player.buff(sts.PlayerStatus.PANACHE, power.damage or 10)
                bc.player.panacheCounter = power.amount    # after buff(), which starts it at 5
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
        elif rid == sts.RelicId.CENTENNIAL_PUZZLE and getattr(spire_relic, "grayscale", False):
            # The engine models the once-per-combat draw-3 by clearing the relic bit when it fires
            # (Player::hpWasLost); the live game grays the relic out at the same moment. Without
            # this every reconstructed HP loss re-fires it. grayscale needs the forked
            # CommunicationMod; absent it the relic converts as always-fresh.
            bc.player.setHasRelic(sts.RelicId.CENTENNIAL_PUZZLE, False)
        elif rid == sts.RelicId.NECRONOMICON and getattr(spire_relic, "activated", None) is False:
            # Necronomicon's `activated` latch is True at turn start and False once the free
            # duplication fired this turn -- so False mid-turn means spent. Without this the
            # search double-plays every 2+-cost attack for the rest of the turn (engine flag
            # resets with each fresh reconstruction). Needs the forked mod; None = not exposed.
            bc.player.haveUsedNecronomiconThisTurn = True

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
        # Without Frozen Eye, add to the UNKNOWN region, not the known top: the live list order is
        # real but the player can't see it, so the searcher must draw stochastically (chance nodes)
        # like native play. moveToDrawPileTop would mark the whole pile known-order, letting the
        # search "cheat" on a reconstructed order and e.g. decline to block because it thinks a
        # Defend is coming. With Frozen Eye the order IS the player's information: place each card
        # definitely, live list order (draw_pile[-1] ends on top).
        mover = bc.cards.moveToDrawPileTop if frozen_eye else bc.cards.moveToDrawPileUnknown
        for spire_card in spire_game.draw_pile:
            _add(spire_card, mover)

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
    # createMonster counted every converted monster as alive; a half-dead one is not (the engine's
    # die() decremented it -- its revival move re-increments), and the victory check reads this.
    for slot, spire_idx in slot_to_spire.items():
        if spire_game.monsters[spire_idx].half_dead:
            bc.monsters.monstersAlive -= 1
    _reconstruct_stasis_cards(bc, spire_game, slot_to_spire)

    # Boss fights search wider + deeper (SearchAgent gates on isBossEncounter(bc.encounter)); the
    # live game doesn't report the encounter, so recover it from the boss monster on the field.
    # Boss encounters drive the search's wider boss config (SearchAgent gates on isBossEncounter)
    # and the eval's act-transition heal; elite encounters drive isEliteEncounter consumers. All
    # other fights keep INVALID, which both predicates treat as "neither".
    enc = _infer_boss_encounter(spire_game.monsters)
    if enc == sts.MonsterEncounter.INVALID:
        enc = _infer_elite_encounter(spire_game.monsters)
    bc.encounter = enc

    # Facing (act-4 Surrounded): live marks the monster BEHIND the player with the BackAttack
    # power, but the engine derives the +50% from player.lastTargetedMonster in
    # calculateDamageToPlayer -- which nothing else here sets, so a fresh conversion keeps the
    # struct default (slot 1) regardless of live facing. Face an alive monster that does NOT
    # carry BackAttack. Live flips the power between the elites as the player retargets, so the
    # per-decision reconcile tracks facing exactly.
    back_slots = {slot for slot, si in slot_to_spire.items()
                  if any(p.power_id == 'BackAttack' for p in spire_game.monsters[si].powers)}
    if back_slots:
        for slot in sorted(slot_to_spire):
            m = spire_game.monsters[slot_to_spire[slot]]
            if slot not in back_slots and not m.is_gone and not m.half_dead:
                bc.player.lastTargetedMonster = slot
                break

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


# Elite fights recovered from their signature monster, same scheme as _BOSS_ENCOUNTER_BY_MONSTER.
# Every key appears ONLY in elite encounters (map elites or the event elites -- Dead Adventurer,
# Colosseum round 2), so a hit is exact; all values satisfy the engine's isEliteEncounter.
_ELITE_ENCOUNTER_BY_MONSTER = {
    'GremlinNob': sts.MonsterEncounter.GREMLIN_NOB,
    'Lagavulin': sts.MonsterEncounter.LAGAVULIN,
    'Sentry': sts.MonsterEncounter.THREE_SENTRIES,
    'GremlinLeader': sts.MonsterEncounter.GREMLIN_LEADER,
    'SlaverBoss': sts.MonsterEncounter.SLAVERS,           # Taskmaster
    'BookOfStabbing': sts.MonsterEncounter.BOOK_OF_STABBING,
    'GiantHead': sts.MonsterEncounter.GIANT_HEAD,
    'Nemesis': sts.MonsterEncounter.NEMESIS,
    'Reptomancer': sts.MonsterEncounter.REPTOMANCER,
    'SpireShield': sts.MonsterEncounter.SHIELD_AND_SPEAR,
    'SpireSpear': sts.MonsterEncounter.SHIELD_AND_SPEAR,
}


def _infer_elite_encounter(spire_monsters):
    """Return the MonsterEncounter for an elite fight (recognized by its signature monster), or
    INVALID for any non-elite fight. Colosseum round 2 (Taskmaster + Gremlin Nob together) maps to
    its own encounter so the inferred id matches the engine's native one."""
    ids = {m.monster_id for m in spire_monsters}
    if 'SlaverBoss' in ids and 'GremlinNob' in ids:
        return sts.MonsterEncounter.COLOSSEUM_EVENT_NOBS
    for monster in spire_monsters:
        enc = _ELITE_ENCOUNTER_BY_MONSTER.get(monster.monster_id)
        if enc is not None:
            return enc
    return sts.MonsterEncounter.INVALID


# Monsters whose move is assigned by a summoner (not their own getMoveForRoll) -- rolling them
# assert(false)s in the engine. When one is converted with no committed intent (just spawned, no
# live move_id), set this fixed move directly instead of rolling. TorchHead only ever tackles.
_UNKNOWN_INTENT_DEFAULT_MOVE = {
    sts.MonsterId.TORCH_HEAD: sts.MonsterMoveId.TORCH_HEAD_TACKLE,
}

# Fallback move for a half-dead monster whose committed move the snapshot doesn't report (it
# normally does): the move the engine's die() parks it on. The only monsters the live game
# reports half_dead: Awakened One (stage-1 death -> Rebirth) and Darkling (any death while
# another darkling stands -> the do-nothing idle turn; Reincarnate follows from its own roll).
_HALF_DEAD_REVIVAL_MOVE = {
    sts.MonsterId.AWAKENED_ONE: sts.MonsterMoveId.AWAKENED_ONE_REBIRTH,
    sts.MonsterId.DARKLING: sts.MonsterMoveId.DARKLING_REGROW,
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
# (and turn for the escalating ones) so the deferred move roll produces a realistic attack. Values
# are the means of Monster::construct's rolls at the fight's ascension (the rolls step up at A2+);
# slightly approximate, but vastly better than 0. Hexaghost Divider (curHp/12) and Gremlin Wizard
# (charge) are dynamic/rare under RD and left as-is.
_RD_HIDDEN_MISCINFO_FIXED = {
    # monster -> (mean below A2, mean at A2+), from the engine's construct rolls
    sts.MonsterId.GREEN_LOUSE: (6, 7),   # bite damage: rng(5,7) / rng(6,8)
    sts.MonsterId.RED_LOUSE: (6, 7),     # bite damage: rng(5,7) / rng(6,8)
    sts.MonsterId.DARKLING: (9, 13),     # nip damage: rng(7,11) / rng(9,13)+2
}


def _estimate_hidden_miscinfo(monster_id, turn0, ascension):
    """A reasonable miscInfo when Runic Dome hides the real value (turn0 = bc.turn, 0-based)."""
    if monster_id == sts.MonsterId.BOOK_OF_STABBING:
        return max(1, turn0 + 1)    # stab count: 1 at battle start (++miscInfo), grows ~1/turn
    pair = _RD_HIDDEN_MISCINFO_FIXED.get(monster_id)
    if pair is None:
        return None
    return pair[1] if ascension >= 2 else pair[0]


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


# Cards whose base damage derives from OBSERVABLE state the live display recomputes lazily --
# Body Slam (current block), Mind Blast (draw pile size) -- so the live base_damage can lag a
# mid-turn change while the reconstruction (same snapshot) is current: a mismatch there is live
# display staleness, not a reconstruction bug, and there is no hidden state to validate anyway.
# Perfected Strike is skipped because the live game keeps its strike bonus OUT of base_damage
# (it lands in `damage` only), so base comparison has no ground truth; its strikeCount input is
# derived from the observable piles during conversion.
_CARD_BASE_CHECK_SKIP_IDS = frozenset({'Body Slam', 'Mind Blast', 'Perfected Strike'})


def assert_card_damage_matches(bc, spire_game) -> None:
    """Fail loud if a reconstructed attack card's BASE damage disagrees with the live card's
    base_damage (exposed by the forked CommunicationMod). The base captures exactly the hidden
    per-instance state the snapshot must restore -- Rampage's growth and Ritual Dagger's bonus
    (specialData), Searing Blow's multi-upgrade count -- while excluding player-side modifiers
    (strength/weak), whose live displayed `damage` can lag a mid-turn change (the game refreshes
    card displays lazily) and so cannot be asserted against. Hand cards convert in order, so
    bc.cards.hand[i] corresponds to spire_game.hand[i]."""
    hand = spire_game.hand
    for i, live in enumerate(hand):
        bd = getattr(live, "base_damage", None)
        if (bd is None or bd < 0 or i >= bc.cards.cardsInHand
                or live.card_id in _CARD_BASE_CHECK_SKIP_IDS):
            continue
        eng = bc.get_card_base_damage(bc.cards.hand[i])
        if eng < 0:    # engine says non-attack -- nothing to check
            continue
        if eng != bd:
            raise AssertionError(
                f"card-damage mismatch for {live.card_id} (hand idx {i}): engine base {eng} vs "
                f"live base_damage {bd} -- this card's per-instance state (specialData / upgrade "
                f"count) is being mis-reconstructed; the search would mis-value playing it.")


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

    # A half-dead monster (Awakened One stage 1, Darkling) is mid-revive. Its committed move IS
    # exposed live even under an UNKNOWN intent (AO: Rebirth=3; Darkling: idle move 4 the enemy
    # phase after it fell, THEN Reincarnate=5 with a visible BUFF intent), so the mapped live move
    # above already parks it in the right phase -- overriding it here would rewind a Darkling
    # that has already spent its idle turn (it revives THIS enemy phase, not next; drive52 g10's
    # "pred 0 vs live 29" ET divergence). Only an unreported move falls back to the move die()
    # parks on. Nothing below applies to a half-dead monster (no damage/hits miscInfo, no rolls).
    if monster.half_dead:
        for power in monster.powers:
            apply_monster_power(sts_monster, power.power_id, power.amount)
        if not move_known:
            revival = _HALF_DEAD_REVIVAL_MOVE.get(sts_monster.id)
            if revival is None:
                raise ValueError(f"half-dead {monster.monster_id} has no revival move; "
                                 f"add it to _HALF_DEAD_REVIVAL_MOVE")
            sts_monster.moveHistory = [int(revival), move_history[1]]
        return

    # Awakened One's stage (miscInfo: 0 = unawakened -> die() parks it half-dead for Rebirth;
    # 1 = awakened -> death is final) isn't in the snapshot, but stage 2 is observable from its
    # moves: Dark Echo / Sludge / Tackle exist only awakened, and a seen Rebirth means it revived.
    if sts_monster.id == sts.MonsterId.AWAKENED_ONE:
        stage2_moves = {int(sts.MonsterMoveId.AWAKENED_ONE_DARK_ECHO),
                        int(sts.MonsterMoveId.AWAKENED_ONE_SLUDGE),
                        int(sts.MonsterMoveId.AWAKENED_ONE_TACKLE),
                        int(sts.MonsterMoveId.AWAKENED_ONE_REBIRTH)}
        if any(mv in stage2_moves for mv in sts_monster.moveHistory):
            sts_monster.miscInfo = 1

    # Restore miscInfo for moves whose damage/hit-count the engine reads from it (see the table
    # above). Without this the search predicts 0 incoming damage for these attacks and never blocks
    # them -- the dominant live-combat HP bleed (Louses are everywhere; Hexaghost's Divider is ~36).
    if move_known:
        cur = move_history[0]
        if cur in _MISCINFO_DAMAGE_MOVE_INTS and monster.move_base_damage:
            sts_monster.miscInfo = int(monster.move_base_damage)
        elif cur in _MISCINFO_HITS_MOVE_INTS and monster.move_hits:
            sts_monster.miscInfo = int(monster.move_hits)

    # Time Eater's one-shot Haste (heal to 50% max HP) is gated by miscInfo (usedHaste in
    # getMoveForRoll: it hastes whenever hp < half and the flag is unset). A single snapshot can't
    # observe the flag, but below half HP with a committed non-Haste intent it must already have
    # hasted -- an unspent Haste below half would BE the intent. Without this the search re-heals
    # it +50% in every line (observed: pred hp jumped 153 -> 228 = maxHp/2 on a Time Warp turn).
    # The residual error (it fell below half only this turn, roll pending) suppresses one heal for
    # one decision; the opposite error mis-heals for the rest of the fight.
    if (sts_monster.id == sts.MonsterId.TIME_EATER and move_known
            and move_history[0] != int(sts.MonsterMoveId.TIME_EATER_HASTE)
            and monster.current_hp * 2 < monster.max_hp):
        sts_monster.miscInfo = 1

    for power in monster.powers:
        apply_monster_power(sts_monster, power.power_id, power.amount)

    # A sleeping Lagavulin reports only its Metallicize power (no "Asleep" status), so without this the
    # reconstruction is an awake attacker that keeps Metallicize 8 -- and the engine, having no ASLEEP
    # to remove, regains 8 block every turn forever (the search then badly over-estimates its bulk, and
    # it "attacks" while it should be sleeping). The engine drops Metallicize only when it wakes FROM
    # ASLEEP (Monster::damageUnblockedHelper on a block break, or the turn-timeout wake), so seed ASLEEP
    # from the sleep intent and let the engine model the wake. Only the live IDLE byte (5) means
    # genuinely asleep: the do-nothing wake turns (STUNNED 4 / OPEN_NATURAL 6) also park on
    # LAGAVULIN_SLEEP but must NOT get ASLEEP (the engine's woken branch models their
    # idle-then-attack turn). The byte is the only reliable sleep signal -- a burning elite's
    # Metallicize buff survives the wake, so powers can't distinguish sleeping from woken.
    if (move_known and move_history[0] == int(sts.MonsterMoveId.LAGAVULIN_SLEEP)
            and monster.move_id == 5):
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
                est = _estimate_hidden_miscinfo(sts_monster.id, bc.turn, bc.ascension)
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
        # A half-dead monster (Awakened One stage-1 death, Darkling) reports hp 0 / is_gone but is
        # still a combatant mid-revive -- dropping it makes the engine end the fight when the last
        # ordinary monster dies, while live plays the revival (the drive52 g3 phantom victory).
        if (monster.current_hp <= 0 or monster.is_gone) and not monster.half_dead:
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
    
    instance = sts.CardInstance(card_id, False)

    # Restore per-instance accumulated state (the engine's specialData == the live card's misc):
    # Ritual Dagger / Genetic Algorithm bonuses etc. Without this the search plays these cards at
    # their printed base. Harmless for cards that don't use it (misc == 0). Must precede the
    # upgrade loop: Searing Blow's upgrade() counts its upgrades IN specialData, which a later
    # blanket write would erase.
    instance.specialData = spire_card.misc
    # Rampage keeps its per-combat growth in the card's baseDamage (misc stays 0), which the forked
    # mod exposes; the engine reads the growth from specialData (base 8 + specialData).
    if card_id == sts.CardId.RAMPAGE and spire_card.base_damage and spire_card.base_damage > 0:
        instance.specialData = spire_card.base_damage - 8

    # Apply every live upgrade through upgrade(), the engine's canonical path -- it maintains the
    # flag AND the per-card upgrade bookkeeping (Searing Blow counts upgrades in specialData).
    # Constructing with upgraded=True would set only the flag, under-leveling Searing Blow
    # everywhere its multi-upgrade damage schedule reads getUpgradeCount().
    for _ in range(spire_card.upgrades):
        instance.upgrade()

    # Set additional properties (after upgrade(), which adjusts costs itself)
    instance.cost = spire_card.cost
    instance.costForTurn = spire_card.cost

    return instance


# Live monster ids the engine keys under a different string. The engine names the Masked Bandits
# by their in-game display names (Romeo/Bear/Pointy) and drops the "The"/"Body" affixes, so we pin
# the live id -> engine string here. Checked against the engine at import (below).


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
        
        # Lagavulin. Java bytes 4 (STUNNED, set on the damage wake) and 6 (OPEN_NATURAL) both DO
        # NOTHING on their turn and attack the next; parking them on LAGAVULIN_SLEEP with no
        # ASLEEP status (the ASLEEP seeding fires only on the IDLE byte 5) makes the engine's
        # sleep case take its woken branch -- idle this turn, setMove(ATTACK) -- exactly the
        # Java behavior. Mapping them to ATTACK hits one turn early.
        ("Lagavulin", 1): sts.MonsterMoveId.LAGAVULIN_SIPHON_SOUL,  # DEBUFF
        ("Lagavulin", 3): sts.MonsterMoveId.LAGAVULIN_ATTACK,       # STRONG_ATK
        ("Lagavulin", 4): sts.MonsterMoveId.LAGAVULIN_SLEEP,        # OPEN (stun turn, damage wake)
        ("Lagavulin", 5): sts.MonsterMoveId.LAGAVULIN_SLEEP,        # IDLE (sleep)
        ("Lagavulin", 6): sts.MonsterMoveId.LAGAVULIN_SLEEP,        # OPEN_NATURAL (do-nothing open)
        
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


