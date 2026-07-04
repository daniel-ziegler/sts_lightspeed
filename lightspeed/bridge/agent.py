"""STSLightspeedAgent: the live-game driver -- combat MCTS decisions over reconstructed or
persistent-bc state, screen handling, and the shadow fidelity checks."""

import os
import time
import random
import sys
import json

import slaythespire as sts
from spirecomm.spire.game import Game
from spirecomm.spire.character import PlayerClass
from spirecomm.spire.screen import RestOption, ScreenType
from lightspeed import RUNS_DIR
from lightspeed.bridge.mappings import map_card_id, map_relic_id
from lightspeed.bridge.combat import (
    _MISCINFO_DAMAGE_MOVE_INTS, _MISCINFO_HITS_MOVE_INTS, assert_card_damage_matches,
    assert_intent_damage_matches, convert_combat_state,
)
from lightspeed.bridge.overworld import (
    _EVENTS_NOT_FAITHFULLY_RECONSTRUCTED, _inject_nloth_offers, _inject_wemeetagain,
    map_event_to_enum, spirecomm_to_gamecontext,
)
from lightspeed.bridge.actions import (
    _CARD_SELECT_POOL_BY_TASK, _CARD_SELECT_TASK_BY_ACTION, _DISCOVERY_TASKS,
    _MULTI_CARD_SELECT_TASK_BY_ACTION, _PBC_PARK_SELECT_TASKS, map_search_action_to_spirecomm,
)
from lightspeed.bridge.seeds import seed_long_to_string
from spirecomm.communication.action import (
    Action, BossRewardAction, CancelAction, CardRewardAction, CardSelectAction, ChooseAction,
    ChooseMapBossAction, ChooseMapNodeAction, ChooseShopkeeperAction, CombatRewardAction,
    EndTurnAction, OpenChestAction, PlayCardAction, PotionAction, ProceedAction, RestAction,
    RewardType,
    StartGameAction,
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
        self._shadow_reset()
        # Persistent-bc bridge, gated behind STS_PBC_DRIVE (default OFF).
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
        # Live combat turn at the last reconcile: a regression on the same floor marks a NEW fight
        # (Colosseum chains two combats on one floor), so the carried pbc must not survive it.
        self._pbc_live_turn = None
        # Draw-pile cards the player GENUINELY knows the position of -- Headbutt/Warcry put-backs
        # (top, newest first) and Forethought put-unders (bottom, deepest first) -- recorded at
        # their selects as (CardId, upgrades), with the floor they belong to. _mark_known_draw_cards
        # re-validates against the live pile each decision and marks the surviving prefixes known
        # on the freshly converted bc, so the search plans on them instead of re-randomizing.
        self._known_draw_top = []
        self._known_draw_bottom = []
        self._known_draw_floor = None
        # (floor, outcome, via_smoke_bomb) set when _pbc_advance executed to a decided outcome: if
        # the same fight then continues live, the engine mis-simulated the finish and
        # _pbc_reconcile_build crashes -- except a smoke-bomb escape, whose live exit animation
        # handle_combat waits out (bounded by _pbc_escape_waits).
        self._pbc_decided = None
        self._pbc_escape_waits = 0
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
        # The spirecomm action last returned by handle_combat, re-sent verbatim if the position stays
        # frozen past the dedup window (a dropped send), instead of re-deciding on the same position.
        self._last_combat_action_sent = None
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

    def _shadow_reset(self):
        """Invalidate the shadow's one-step prediction (prior bc, action, draw top, floor)."""
        self._shadow_prev_bc = None
        self._shadow_prev_action = None
        self._shadow_prev_draw_top = None
        self._shadow_prev_floor = None

    def _reset_combat_carry(self):
        """Forget every piece of per-fight carried state. Runs on EVERY observed non-combat state
        (room_phase left COMBAT) -- the positive battle-over signal: reward/event/map screens follow
        every fight, including between Colosseum's two same-floor fights, so no floor/turn heuristic
        has to infer the boundary. The floor/turn stale guards in the reconcile and the select path
        remain only as backstops for a missed emit."""
        if self._pbc is not None:
            # A pbc that outlives its fight was left PARKED by a fight-ending play (a clean end
            # reaches a decided outcome and is dropped in _pbc_advance).
            print(f"[pbc] live left combat with the bc still carried "
                  f"(input_state={self._pbc.input_state}); dropping it", file=sys.stderr)
            self._pbc = None
        self._pbc_decided = None
        self._pbc_escape_waits = 0
        self._pbc_floor = None
        self._pbc_live_turn = None
        self._pbc_last_end_turn = None
        self._shadow_reset()
        self._known_draw_top = []
        self._known_draw_bottom = []
        self._known_draw_floor = None
        self._last_acted_combat_sig = None
        self._last_combat_action_sent = None
        self._dedup_stuck_since = None

    def get_next_action_in_game(self, game_state):
        self.choice_count += 1
        self.game = game_state
        self._log_seed_once()
        # Positive battle-over signal: the room left the COMBAT phase, so all per-fight carried
        # state (persistent bc, shadow one-step, draw knowledge, combat dedup) is now stale.
        if not self.game.in_combat:
            self._reset_combat_carry()
        # Persist the raw incoming state BEFORE we touch it, so a silent C++ segfault during
        # processing (no Python traceback) still leaves the triggering state on disk for offline
        # repro. Overwrites each decision; the file is the last state we started to handle.
        try:
            if self.coordinator is not None and self.coordinator.last_raw_communication_state is not None:
                path = os.path.join(RUNS_DIR, "last_instate.json")
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
        live<->offline gap. See REMAINING_DIVERGENCES.md (this directory) for the known classes.

        Phase 1b also covers END_TURN: the reconstruction sets each monster's CURRENT move from the
        visible intent, so executing END_TURN runs the *real* monster moves -- any divergence in the
        post-monster-turn player/monster hp/block is a genuine monster-turn fidelity bug (the boss
        concern). The engine's roll of the NEXT intent during END_TURN diverges by RNG, but we don't
        compare intents so it's harmless here."""
        prev_bc = self._shadow_prev_bc
        prev_action = self._shadow_prev_action
        prev_draw_top = self._shadow_prev_draw_top
        prev_floor = self._shadow_prev_floor
        self._shadow_reset()
        if prev_bc is None or prev_action is None:
            return
        # A combat lives on exactly one floor; the next fight is a higher floor. If the floor changed,
        # prev_bc is the PREVIOUS combat's final state (the bot won/lost and moved on) and comparing it
        # to this fresh fight is a measurement artifact -- e.g. a dead-player end state (php 0) diffed
        # against a full-HP turn 1. The monster-count guard below misses this when both fights happen
        # to have the same number of monsters, so gate on the floor explicitly.
        if prev_floor != self.game.floor:
            return
        # Two combats can share a floor (Colosseum): a bc turn that went backward means prev_bc
        # belongs to the previous fight, so the one-step prediction has no live counterpart.
        if truth_bc.turn < prev_bc.turn:
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
                # Mayhem plays the POST-draw top of the pile at the start of the next turn (the
                # turn-start draw takes the pre-draw top N; Mayhem consumes the card after them),
                # so the single observed pre-draw top can't pin down what it plays -- the shadow's
                # one-step Mayhem replay stays unverifiable below. Forcing the observed top still
                # helps the engine draw one true card; the force status is context only.
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
                elif (len(diffs) == 1 and pred["energy"] != truth["energy"]
                      and (truth_bc.player.hasRelic(sts.RelicId.SNECKO_EYE)
                           or truth_bc.player.hasRelic(sts.RelicId.MUMMIFIED_HAND))):
                    # An energy-ONLY diff under a cost-randomizing relic is a live RNG roll the
                    # one-step can't reproduce: Snecko rerolls the cost of every card drawn inside
                    # the step (end-turn draw, Battle Trance/Pommel draws), Mummified Hand zeroes a
                    # RANDOM hand card on each power play -- either shifts a later play's cost by
                    # the roll delta. Model checks against Java found no real bug in this class
                    # (Art of War, Happy Flower convention, Gremlin Horn, recharge all verified);
                    # an energy diff WITHOUT these relics, or paired with hp/block/hand deltas,
                    # still counts as a divergence.
                    which = ("Snecko Eye" if truth_bc.player.hasRelic(sts.RelicId.SNECKO_EYE)
                             else "Mummified Hand")
                    print(f"[shadow unverifiable] ({tag}) after {desc} ({which} cost roll -- "
                          f"live RNG): " + "; ".join(diffs) + ctx, file=sys.stderr)
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
            path = os.path.join(RUNS_DIR, "spot_weakness_mistarget.jsonl")
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
        live_turn = getattr(self.game, "turn", None) or 0
        if self._pbc is not None and floor != self._pbc_floor:
            self._pbc = None
            self._pbc_decided = None
        # A live turn that went BACKWARD on the same floor means a NEW combat started there (the
        # Colosseum event chains two fights on one floor) -- the carried pbc (and any decided-outcome
        # marker) belongs to the previous fight, so drop it rather than transplanting its hidden
        # monster state onto the new fight's monsters.
        if self._pbc is not None and live_turn < (self._pbc_live_turn or 0):
            print(f"[pbc] new combat on floor {floor} (live turn {self._pbc_live_turn} -> "
                  f"{live_turn}); dropping the previous fight's pbc", file=sys.stderr)
            self._pbc = None
            self._pbc_decided = None
        if self._pbc_decided is not None and self._pbc_decided[0] == floor and live_turn > 1:
            # The engine executed an action to a decided outcome, but the SAME fight is still going
            # live (same floor, no turn reset): a genuine end-of-combat mis-simulation. Crash to
            # surface it -- quietly re-seeding would mask exactly the divergence class the driven
            # pbc exists to expose.
            decided = self._pbc_decided
            self._pbc_decided = None
            raise RuntimeError(f"[pbc] engine predicted combat over ({decided[1]}) at floor {floor} "
                               f"but the live fight continues (turn {live_turn})")
        self._pbc_decided = None
        if self._pbc is None:
            self._pbc = fresh_bc.copy()
            self._pbc_last_end_turn = None
            print(f"[pbc] seeded persistent bc at floor {floor}", file=sys.stderr)
        else:
            self._pbc = self._pbc_reconcile(fresh_bc, fresh_slots)
        self._pbc_slots = dict(fresh_slots)
        self._pbc_floor = floor
        self._pbc_live_turn = live_turn
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
        """Carry the driven persistent bc forward through the action just committed live, to the next
        input point. The action was chosen by searching this exact bc, so it MUST be valid and MUST
        execute -- a failure is a genuine divergence and we crash to surface it. is_valid_action gates
        the execute because execute() asserts(false) on an invalid action (an uncatchable SIGABRT,
        not a Python exception). Drops the pbc (re-seed next decision) when the action ends the
        simulated combat or leaves the engine awaiting a sub-input the drive doesn't handle."""
        if not action.is_valid_action(self._pbc):
            raise RuntimeError("[pbc] chosen action invalid on driven persistent bc: "
                               f"{self._describe_action(action, self._pbc)}")
        pre_shuffles = self._pbc.empty_deck_shuffle_count
        # Pre-advance snapshot for decided-outcome forensics: when this advance ends the simulated
        # combat, the exact input state is the evidence needed to root-cause a phantom outcome
        # (the a20h10k g38 phantom PLAYER_LOSS was unreproducible because only post-crash state
        # survived). Copy is cheap (value-type bc, same op the searcher does per simulation).
        pre_bc = self._pbc.copy()
        action.execute(self._pbc)
        if self._pbc.outcome != sts.BattleOutcome.UNDECIDED:
            print(f"[pbc] advance {self._describe_action(action, pre_bc)} reached "
                  f"{self._pbc.outcome} (full pre/post state in pbc_decided_dumps.jsonl)",
                  file=sys.stderr)
            try:
                with open(os.path.join(RUNS_DIR, "pbc_decided_dumps.jsonl"), "a") as f:
                    raw = self.coordinator.last_raw_communication_state if self.coordinator else None
                    f.write(json.dumps({
                        "floor": self._pbc_floor, "outcome": str(self._pbc.outcome),
                        "action": self._describe_action(action, pre_bc),
                        "pre_bc": str(pre_bc), "post_bc": str(self._pbc), "raw": raw,
                    }) + "\n")
            except Exception:
                pass
            # The engine predicts this combat is OVER. Normally live ends with it and the next combat
            # decision belongs to a new fight; if one arrives for THIS fight instead, the engine
            # mis-simulated the finish -- _pbc_reconcile_build checks the marker and crashes there.
            # A Smoke Bomb escape is flagged: live plays a ~2.5s escape animation during which the
            # room still reports COMBAT, so handle_combat waits that transient out instead.
            #
            # Exception: an advance that crossed an EmptyDeckShuffle reshuffled the discard in an
            # engine-local order the live game rolls differently (an empty-pile Havoc plays a
            # different card), so a fight-ending result downstream of it is UNVERIFIABLE rather
            # than a mis-simulation -- don't arm the crash marker; drop the pbc and let live
            # decide (a re-seed follows if the fight continues). Smoke Bomb escapes stay armed:
            # the escape itself never depends on a reshuffle, and the escape wait needs the flag.
            if (self._pbc.empty_deck_shuffle_count > pre_shuffles
                    and not self._pbc.smoke_bomb_used):
                print(f"[pbc] engine outcome {self._pbc.outcome} rests on an empty-pile reshuffle "
                      f"(order unknowable) -- unverified, deferring to live", file=sys.stderr)
                self._pbc = None
                return
            self._pbc_decided = (self._pbc_floor, str(self._pbc.outcome),
                                 bool(self._pbc.smoke_bomb_used))
            self._pbc = None
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
        # Smoke Bomb ended the fight in the engine (escape = instant PLAYER_VICTORY), but the live
        # game plays a ~2.5s escape animation (player.isEscaping) during which the room still
        # reports COMBAT with the potion already consumed. The fight IS over -- wait for the room
        # to leave combat (the not-in_combat state then resets all carry) instead of treating the
        # transient as a mis-simulated finish. Bounded: an escape that genuinely failed live would
        # sit in combat past the animation, and that IS a divergence worth crashing on.
        if (self._pbc_decided is not None and self._pbc_decided[0] == self.game.floor
                and self._pbc_decided[2]):
            self._pbc_escape_waits += 1
            if self._pbc_escape_waits > 8:
                raise RuntimeError(f"[pbc] smoke-bomb escape never left combat "
                                   f"({self._pbc_escape_waits - 1} waits) -- live escape failed")
            print("[pbc] smoke-bomb escape pending; waiting for the room to leave combat",
                  file=sys.stderr)
            return Action("wait 30")
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
            # Unchanged for far longer than any real resolution transient: the last command didn't
            # take (dropped mid-emit), so waiting will only burn out the 150s watchdog. RE-SEND the
            # exact same command rather than re-deciding: the driven pbc has already been advanced
            # through this action, so a fresh decide would advance it a SECOND time (hidden monster
            # state two steps ahead of live, invisible to the reconcile) and could fire a different
            # command into a still-busy game. A genuinely stuck position re-sends every 8s until the
            # coordinator watchdog kills the run.
            if self._last_combat_action_sent is not None:
                print(f"[combat] position unchanged for {now - self._dedup_stuck_since:.0f}s -- "
                      f"re-sending the last command", file=sys.stderr)
                self._dedup_stuck_since = now
                return self._last_combat_action_sent
            print(f"[combat] position unchanged for {now - self._dedup_stuck_since:.0f}s with no "
                  f"stored command -- re-deciding (dedup released)", file=sys.stderr)
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
        # Card damage check, same contract as the monster intent check above: a hand card whose
        # engine-displayed damage disagrees with the live card is being mis-reconstructed (hidden
        # combat state like strikeCount / specialData / strength), so the search would mis-value
        # playing it. Fail loud rather than plan on wrong numbers.
        assert_card_damage_matches(bc, self.game)
        # Restore the player's genuine draw-order knowledge (Headbutt/Warcry/Forethought) before the
        # search; the reconciled pbc copies this bc, so the marking carries into the drive path.
        self._mark_known_draw_cards(bc)
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
                crash_path = os.path.join(RUNS_DIR, "battle_search_crashes.jsonl")
                with open(crash_path, "a") as f:
                    raw = self.coordinator.last_raw_communication_state if self.coordinator else None
                    f.write(json.dumps({"error": str(e), "raw": raw}) + "\n")
            except Exception:
                pass
            raise

        # Map the search action to a spirecomm action (interpreted against the bc we searched on)
        spirecomm_action = map_search_action_to_spirecomm(first_action, search_bc, self.game, search_slots)

        print(f"Chosen action: {spirecomm_action}", file=sys.stderr)

        # Watch mode: a potion drink shows nothing on screen until it resolves (unlike a card play,
        # which the game animates), so hover the belt potion before committing it. Card plays and
        # end-turn get no hover -- the pre-pause alone paces them.
        if self.watch and isinstance(spirecomm_action, PotionAction):
            verb = "use" if spirecomm_action.use else "discard"
            slot = self.game.potions.index(spirecomm_action.potion)
            self._watch_pause(f"potion {verb} {spirecomm_action.potion.name}", f"potion {slot}")

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
            # advance it here through the chosen action -- with its draw pile forced to the observed
            # live order first, so every top-of-pile read in the resolution replays reality (Havoc
            # chains, mid-resolution draws, potion draws).
            if self._pbc is search_bc:
                self._pbc_force_live_draw_order()
                self._pbc_advance(first_action)
                self._pbc_prev_action_desc = self._describe_action(first_action, bc)
                if first_action.get_action_type() == sts.ActionType.END_TURN:
                    self._pbc_last_end_turn = getattr(self.game, "turn", None)

        # Remembered so a dropped send can be re-issued verbatim (see the dedup release above).
        self._last_combat_action_sent = spirecomm_action
        return spirecomm_action

    def _mark_known_draw_cards(self, bc):
        """Mark on a freshly converted bc the draw-pile positions the player genuinely knows -- the
        Headbutt/Warcry put-backs (top) and Forethought put-unders (bottom) recorded at their
        selects -- so the search plans on them instead of re-randomizing (native play keeps this
        knowledge: the engine's CardPile tracks known tops/bottoms; only the per-decision
        reconstruction forgot it).

        Re-validates the memory against the live pile first: top entries drawn since are dropped
        from the front, and each stack keeps only the prefix that still matches the live order at
        its end of the pile -- a reshuffle or displacement invalidates from the mismatch on.
        Matching is by (CardId, upgrades): a coincidental match with a twin copy marks a position
        that is identical at the level the search sees, so it stays correct. Skipped under Frozen
        Eye, where the whole pile already converts order-observed."""
        if ((self._known_draw_top or self._known_draw_bottom)
                and self._known_draw_floor != self.game.floor):
            self._known_draw_top = []
            self._known_draw_bottom = []
        if not (self._known_draw_top or self._known_draw_bottom) \
                or bc.player.hasRelic(sts.RelicId.FROZEN_EYE):
            return
        live = self.game.draw_pile or []

        def live_key_top(i):       # i cards down from the top (live[-1] is the top)
            return (map_card_id(live[-1 - i].card_id), live[-1 - i].upgrades) if i < len(live) else None

        def live_key_bottom(i):    # i cards up from the bottom (live[0] is the bottom)
            return (map_card_id(live[i].card_id), live[i].upgrades) if i < len(live) else None

        # Top stack: drop consumed entries (a drawn put-back leaves the next remembered card on
        # top; a reshuffle matches nothing and clears), then keep the matching prefix.
        top = self._known_draw_top
        while top and live_key_top(0) != top[0]:
            top.pop(0)
        keep_t = 0
        while keep_t < len(top) and live_key_top(keep_t) == top[keep_t]:
            keep_t += 1
        del top[keep_t:]
        # Bottom stack: anchored at the very bottom; entries above it disappear as they are drawn
        # out the top of a drained pile, so keeping the matching prefix handles both cases.
        bot = self._known_draw_bottom
        if bot and live_key_bottom(0) != bot[0]:
            bot.clear()
        keep_b = 0
        while keep_b < len(bot) and live_key_bottom(keep_b) == bot[keep_b]:
            keep_b += 1
        del bot[keep_b:]

        if not top and not bot:
            return
        base = len(self.game.hand or [])       # conversion uid order: hand cards, then the draw pile
        top_ids = [base + (len(live) - 1 - i) for i in range(keep_t)]
        bot_ids = [base + i for i in range(keep_b)]
        # A nearly-drained pile can put the same card in both views; the top view wins (next draw).
        bot_ids = [u for u in bot_ids if u not in set(top_ids)]
        bc.force_draw_pile_knowledge(top_ids, bot_ids)
        print(f"[known-draw] marked {len(top_ids)} top / {len(bot_ids)} bottom card(s) known "
              f"on the search bc", file=sys.stderr)

    def _pbc_force_live_draw_order(self):
        """Put the driven pbc's draw pile into the exact LIVE order (fully known) before advancing it
        through the committed action, so every top-of-pile read in the resolution replays reality: a
        Havoc pops the true top, a Havoc'd Havoc pops the true next card, and any mid-resolution draw
        (Pommel Strike, Battle Trance, Swift Potion) or the end-turn draw pulls the true cards.

        The live message exposes the whole pile order (draw_pile[-1] is the top -- CommunicationMod
        serializes drawPile.group directly), and the pbc's piles were rebuilt from this same message
        at reconcile with uniqueIds assigned in live list order (convert_combat_state's _add: hand
        first, then the draw pile), so the uid <-> position mapping is exact -- even for twin cards
        that differ only in hidden specialData. Raises on any mismatch: both sides come from the same
        snapshot, so a mismatch is a reconstruction bug, not drift.

        No knowledge cheat: the fully-known order exists ONLY inside this advance, replaying what the
        live game is about to resolve. The next search runs on a fresh reconciliation whose pile is
        rebuilt UNKNOWN-order from the next snapshot (_pbc_reconcile copies the fresh reconstruction),
        so the search never inherits draw-order knowledge the player lacks. Randomness that happens
        DURING the live resolution (a reshuffle, a card shuffled in at a live-rng position) still
        diverges; the next reconcile corrects the observables and the DESYNC oracle reports it."""
        live = self.game.draw_pile or []
        pile = self._pbc.cards.drawPile
        if len(live) != len(pile):
            raise RuntimeError(f"[pbc] draw-order force: pbc pile size {len(pile)} != live {len(live)}")
        if not live:
            return
        base = len(self.game.hand or [])       # conversion uid order: hand cards, then the draw pile
        by_uid = {c.uniqueId: c for c in pile}
        top_first = []
        for i in range(len(live) - 1, -1, -1):
            uid = base + i
            eng = by_uid.get(uid)
            if eng is None or map_card_id(live[i].card_id) != eng.id:
                raise RuntimeError(
                    f"[pbc] draw-order force: uid {uid} mismatch at live index {i} "
                    f"({live[i].card_id} vs {'missing' if eng is None else eng.getName()}) -- "
                    f"conversion uid order no longer matches the live pile")
            top_first.append(uid)
        self._pbc.force_draw_pile_order(top_first)

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
        live_turn = getattr(self.game, "turn", None) or 0
        if self._pbc_floor != self.game.floor or live_turn < (self._pbc_live_turn or 0):
            # Backstop only: a pbc parked by a fight-ending play is normally dropped by
            # _reset_combat_carry on the first non-combat state after the fight. Reaching here
            # means that emit was missed -- a new floor, or a same-floor live-turn regression
            # (Colosseum's second fight), still marks the park as a previous fight's; drop the
            # pbc and resolve on the fresh reconstruction.
            print(f"[pbc] parked select ({self._pbc.card_select_task}) is stale (parked at floor "
                  f"{self._pbc_floor} turn {self._pbc_live_turn}, live at floor {self.game.floor} "
                  f"turn {live_turn}); dropping the pbc", file=sys.stderr)
            self._pbc = None
            self._pbc_decided = None
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
        self._shadow_reset()
        # Invalidate the combat duplicate-emit signature: resolving this select is an action that
        # changes the position, but it's committed here (not through handle_combat), so the sig was
        # never updated for it. Without this reset, a post-select combat position that happens to match
        # the pre-select-card signature (e.g. a Warcry put-back restoring the same hand) reads as a
        # still-resolving duplicate and the next command is never sent -- a hang.
        self._last_acted_combat_sig = None
        # CardRewardScreen (the in-combat Discovery/potion choice) has no num_cards; it always picks 1.
        num_cards = getattr(self.game.screen, "num_cards", None)
        num = num_cards or 1
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
            self._mark_known_draw_cards(bc)
            # Any-number semantics: pickCount == hand size lets the search keep selecting up to the
            # whole hand (openSimpleCardSelectScreen has no canPickAnyNumber), matching the native
            # any-number selects. GamblingChip's combat-start frame omits num_cards entirely -- an
            # absent count must mean "the whole hand", not 1; a reported cap still applies via min().
            mnum = min(num_cards, bc.cards.cardsInHand) if num_cards else bc.cards.cardsInHand
            # When driving, resolve on a bc reconciled from LIVE at the select (hand == the live screen)
            # with the carried hidden monster state, advancing the pbc through the whole select. NO
            # fallback: if it can't resolve, raise (crash) so the divergence is debuggable, not masked.
            if self._pbc_driving_at_select(multi_task):
                sel_bc = self._pbc_reconcile_at_select(bc, slot_to_spire)
                sel_bc.open_card_select(multi_task, mnum)
                chosen = self._run_multi_select(sel_bc, multi_task, action_name, driven=True)
                self._pbc_prev_action_desc = "MULTI_CARD_SELECT"   # DESYNC attribution
                self._watch_select_pause(
                    chosen, f"select {', '.join(c.name for c in chosen) or 'nothing'} ({action_name})")
                return CardSelectAction(chosen)

            bc.open_card_select(multi_task, mnum)
            chosen = self._run_multi_select(bc, multi_task, action_name, driven=False)
            self._watch_select_pause(
                chosen, f"select {', '.join(c.name for c in chosen) or 'nothing'} ({action_name})")
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
        self._mark_known_draw_cards(bc)

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
                return self._resolve_discovery(self._pbc, task, ids, offered, action_name,
                                               driven=True)
            return self._resolve_discovery(bc, task, ids, offered, action_name, driven=False)

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
        # A put-back/put-under select places the chosen card at a position the player KNOWS --
        # Headbutt/Warcry on top, Forethought on the bottom -- knowledge kept until the card moves
        # or the pile reshuffles. Record it so the next decisions' converted bcs mark it known
        # (see _mark_known_draw_cards).
        if task in (sts.CardSelectTask.HEADBUTT, sts.CardSelectTask.WARCRY,
                    sts.CardSelectTask.FORETHOUGHT):
            if self._known_draw_floor != self.game.floor:
                self._known_draw_top = []
                self._known_draw_bottom = []
            self._known_draw_floor = self.game.floor
            key = (map_card_id(live_card.card_id), live_card.upgrades)
            if task == sts.CardSelectTask.FORETHOUGHT:
                self._known_draw_bottom.insert(0, key)   # newest sinks to the very bottom
            else:
                self._known_draw_top.insert(0, key)      # newest lands on top
        print(f"[mcts] card-select ({action_name}) -> {chosen.getName()}"
              f"{'+' if chosen.upgraded else ''} ({pool_name} idx {sel_idx}"
              f"{', pbc-driven' if driven else ''})", file=sys.stderr)
        self._watch_select_pause([live_card], f"select {live_card.name} ({action_name})")
        return live_card, chosen_action

    def _resolve_discovery(self, select_bc, task, ids, offered, action_name, driven):
        """Resolve an in-combat Discovery/Codex select on `select_bc`: inject the live-observed
        offered cards as the candidates (a driven pbc rolled its own from a desynced RNG; reality's
        offered set is observable), search, and commit the pick to the live screen. The chosen index
        lines up with `offered` (ids were built from it in order). When driven, `select_bc` is the
        parked pbc and is also advanced through the pick, which must land it back in a clean player
        turn. Raises (crashes -- no fallback) on an invalid pick or an unclean resolution."""
        if task == sts.CardSelectTask.CODEX:
            select_bc.open_codex_select(ids)
        else:
            select_bc.open_discovery_select(ids, 1, True)
        searcher = sts.BattleSearcher(select_bc)
        searcher.search(self.search_agent.configure_searcher(searcher, select_bc))
        chosen_action = searcher.get_best_action()
        sel_idx = chosen_action.get_select_idx()
        if driven:
            if not chosen_action.is_valid_action(select_bc):
                raise RuntimeError(f"discovery pick invalid on persistent bc ({task})")
            chosen_action.execute(select_bc)      # advance the pbc through the select (CODEX idx==3 = skip)
            if (select_bc.input_state != sts.InputState.PLAYER_NORMAL
                    or select_bc.outcome != sts.BattleOutcome.UNDECIDED):
                raise RuntimeError(f"discovery left pbc unclean (input_state={select_bc.input_state})")
            self._pbc_prev_action_desc = "DISCOVERY"
        suffix = ", pbc-driven" if driven else ""
        # Nilry's Codex is skippable: the engine's CODEX select validates idx in [0,4) and treats
        # idx == 3 (one past the 3 offered cards) as "skip" (Action.cpp / BattleSimulator); the live
        # CARD_REWARD advertises this as skip_available, and the mod routes skip == cancel. Discovery
        # (open_discovery_select) is not skippable, so this only fires for CODEX.
        if task == sts.CardSelectTask.CODEX and sel_idx == len(offered):
            print(f"[mcts] discovery ({action_name}) -> skip (idx {sel_idx}{suffix})",
                  file=sys.stderr)
            self._watch_pause(f"codex skip ({action_name})",
                              -1 if self.game.screen_type == ScreenType.CARD_REWARD else None)
            return CancelAction()
        if not (0 <= sel_idx < len(offered)):
            raise RuntimeError(f"MCTS discovery idx {sel_idx} out of range "
                               f"({len(offered)} offered, {action_name})")
        chosen = offered[sel_idx]
        print(f"[mcts] discovery ({action_name}) -> {chosen.card_id} (idx {sel_idx}{suffix})",
              file=sys.stderr)
        # A Discovery/potion choice is delivered on a CARD_REWARD screen, where the pick is a
        # "choose <index>" command (CardSelectAction only works on HAND_SELECT/GRID). The
        # choice_list / screen.cards order matches sel_idx.
        if self.game.screen_type == ScreenType.CARD_REWARD:
            self._watch_pause(f"discovery {chosen.card_id} ({action_name})", sel_idx)
            return ChooseAction(sel_idx)
        self._watch_select_pause([chosen], f"discovery {chosen.card_id} ({action_name})")
        return CardSelectAction([chosen])

    def _run_multi_select(self, select_bc, multi_task, action_name, driven):
        """Resolve a multi-card select (Gamble / Exhaust-many) on `select_bc` by the engine's own
        sequential protocol: each search picks either a SINGLE_CARD_SELECT (toggle one more hand card
        into the running set; the screen re-opens) or the MULTI confirm (apply the set and exit the
        select). Executes every pick on select_bc and collects the matching live screen cards --
        DISTINCT live instances for duplicate picks (two Strikes must map to two different uuids, or
        the live screen toggles one copy on and back off). Raises (no fallback) on an invalid pick,
        an unmatchable card, non-convergence, or -- for a driven pbc -- an unclean landing state."""
        chosen = []
        used_uuids = set()
        for _ in range(64):                       # bound: hand size is <= ~10; 64 is a safe backstop
            if (select_bc.input_state != sts.InputState.CARD_SELECT
                    or select_bc.card_select_task != multi_task):
                break
            searcher = sts.BattleSearcher(select_bc)
            searcher.search(self.search_agent.configure_searcher(searcher, select_bc))
            best = searcher.get_best_action()
            if best.get_action_type() == sts.ActionType.SINGLE_CARD_SELECT:
                idx = best.get_select_idx()
                hand = select_bc.cards.hand
                if not (0 <= idx < len(hand)):
                    raise RuntimeError(f"multi-select idx {idx} out of hand range "
                                       f"({len(hand)}, {multi_task})")
                live = self._match_live_select_card(hand[idx], exclude_uuids=used_uuids)
                used_uuids.add(live.uuid)
                chosen.append(live)
            if not best.is_valid_action(select_bc):
                raise RuntimeError(f"multi-select action invalid ({multi_task})")
            best.execute(select_bc)               # SINGLE re-opens (loop continues); MULTI confirms (loop exits)
        else:
            raise RuntimeError(f"multi-select did not converge ({multi_task})")
        if driven and (select_bc.input_state != sts.InputState.PLAYER_NORMAL
                       or select_bc.outcome != sts.BattleOutcome.UNDECIDED):
            raise RuntimeError(f"multi-select left pbc unclean (input_state={select_bc.input_state})")
        print(f"[mcts] multi-select ({action_name}, {multi_task}) -> {len(chosen)} card(s)"
              f"{', pbc-driven' if driven else ''}", file=sys.stderr)
        return chosen

    def _match_live_select_card(self, engine_card, exclude_uuids=None):
        """Find the live select-screen candidate matching the engine-chosen card by stable CardId +
        upgrade count (robust to display-name drift), skipping candidates whose uuid is in
        `exclude_uuids` -- a multi-select that picks duplicate cards (two Strikes) needs a distinct
        live instance per pick, since selecting the same uuid twice toggles it back off. Raises if
        the search picked a card not offered (or no longer available) live."""
        want_id, want_upg = engine_card.id, engine_card.upgrade_count
        exclude = exclude_uuids or ()
        for c in self.game.screen.cards:
            if c.uuid not in exclude and map_card_id(c.card_id) == want_id and c.upgrades == want_upg:
                return c
        for c in self.game.screen.cards:  # tolerate an upgrade-count mismatch (e.g. Searing Blow)
            if c.uuid not in exclude and map_card_id(c.card_id) == want_id:
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

    def _watch_select_pause(self, live_cards, desc):
        """Watch mode: hover the pending in-combat select pick before it commits -- the first card
        of a multi-pick (spirecomm clicks the whole set in one command, so one hover is the best
        single signal). The index space matches the mod's choice list for HAND_SELECT (the
        not-yet-selected hand) and GRID (the target group) -- the same screen.cards order
        CardSelectAction resolves against. Pause-only when nothing is picked (an empty Gamble) or
        the card isn't on the live screen."""
        if not self.watch:
            return
        idx = None
        if live_cards:
            try:
                idx = self.game.screen.cards.index(live_cards[0])
            except ValueError:
                idx = None
        self._watch_pause(desc, idx)

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
        from lightspeed.playouts import construct_choice, choose_overworld_action

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
        CombatRewardAction (the relic when the net picks it, else the key -- the net choosing
        skip/key IS the take-key decision) or None when no key is on offer. Raises when the
        reconstruction diverges (wrong screen, out-of-range relic pick): every other net path fails
        loud on divergence, and a silent take-key default would hide the reconstruction bug while
        systematically forfeiting the relic choice."""
        key_item = next((r for r in rewards
                         if r.reward_type in (RewardType.EMERALD_KEY, RewardType.SAPPHIRE_KEY)), None)
        relic_items = [r for r in rewards if r.reward_type == RewardType.RELIC]
        gc = spirecomm_to_gamecontext(self.game)
        if gc.screen_state != sts.ScreenState.REWARDS:
            raise RuntimeError(f"relic-vs-key: reconstructed gc is on {gc.screen_state}, not REWARDS "
                               f"-- reward-screen reconstruction diverged")
        action = self.net_pick_action(gc)
        if action is not None and action.rewards_action_type == sts.RewardsActionType.RELIC:
            idx = action.idx1
            if not (0 <= idx < len(relic_items)):
                raise RuntimeError(f"relic-vs-key: net picked relic idx {idx} but the live screen "
                                   f"offers {len(relic_items)} relic(s) -- reconstruction diverged")
            print(f"[net] reward relic-vs-key -> relic {relic_items[idx].relic.name}",
                  file=sys.stderr)
            return CombatRewardAction(relic_items[idx])
        if key_item is not None:
            print("[net] reward relic-vs-key -> key", file=sys.stderr)
            return CombatRewardAction(key_item)
        return None

