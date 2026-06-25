import sys
import time
import queue
import threading
import json
import collections

from spirecomm.spire.game import Game
from spirecomm.spire.screen import ScreenType
from spirecomm.communication.action import Action, StartGameAction


def read_stdin(input_queue):
    """Read lines from stdin and write them to a queue

    :param input_queue: A queue, to which lines from stdin will be written
    :type input_queue: queue.Queue
    :return: None
    """
    while True:
        stdin_input = ""
        while True:
            input_char = sys.stdin.read(1)
            if input_char == '\n':
                break
            else:
                stdin_input += input_char
        input_queue.put(stdin_input)


def write_stdout(output_queue):
    """Read lines from a queue and write them to stdout

    :param output_queue: A queue, from which this function will receive lines of text
    :type output_queue: queue.Queue
    :return: None
    """
    while True:
        output = output_queue.get()
        print(output, end='\n', flush=True)


class Coordinator:
    """An object to coordinate communication with Slay the Spire"""

    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.input_thread = threading.Thread(target=read_stdin, args=(self.input_queue,))
        self.output_thread = threading.Thread(target=write_stdout, args=(self.output_queue,))
        self.input_thread.daemon = True
        self.input_thread.start()
        self.output_thread.daemon = True
        self.output_thread.start()
        self.action_queue = collections.deque()
        self.state_change_callback = None
        self.out_of_game_callback = None
        self.error_callback = None
        self.game_is_ready = False
        self.stop_after_run = False
        self.in_game = False
        self.last_game_state = None
        self.last_error = None
        # Raw CommunicationMod message dict (game_state + available_commands) for the most
        # recent update, kept so callers can capture the unparsed source of truth for replay.
        self.last_raw_communication_state = None

    def signal_ready(self):
        """Indicate to Communication Mod that setup is complete

        Must be used once, before any other commands can be sent.
        :return: None
        """
        self.send_message("ready")

    def send_message(self, message):
        """Send a command to Communication Mod and start waiting for a response

        :param message: the message to send
        :type message: str
        :return: None
        """
        self.output_queue.put(message)
        self.game_is_ready = False

    def add_action_to_queue(self, action):
        """Queue an action to perform when ready

        :param action: the action to queue
        :type action: Action
        :return: None
        """
        self.action_queue.append(action)

    def clear_actions(self):
        """Remove all actions from the action queue

        :return: None
        """
        self.action_queue.clear()

    def execute_next_action(self):
        """Immediately execute the next action in the action queue

        :return: None
        """
        action = self.action_queue.popleft()
        print(f"Executing action: {action}", file=sys.stderr)
        action.execute(self)

    def execute_next_action_if_ready(self):
        """Immediately execute the next action in the action queue, if ready to do so

        :return: None
        """
        if len(self.action_queue) > 0 and self.action_queue[0].can_be_executed(self):
            self.execute_next_action()

    def register_state_change_callback(self, new_callback):
        """Register a function to be called when a message is received from Communication Mod

        :param new_callback: the function to call
        :type new_callback: function(game_state: Game) -> Action
        :return: None
        """
        self.state_change_callback = new_callback

    def register_command_error_callback(self, new_callback):
        """Register a function to be called when an error is received from Communication Mod

        :param new_callback: the function to call
        :type new_callback: function(error: str) -> Action
        :return: None
        """
        self.error_callback = new_callback

    def register_out_of_game_callback(self, new_callback):
        """Register a function to be called when Communication Mod indicates we are in the main menu

        :param new_callback: the function to call
        :type new_callback: function() -> Action
        :return: None
        """
        self.out_of_game_callback = new_callback

    def get_next_raw_message(self, block=False, timeout=None):
        """Get the next message from Communication Mod as a string

        :param block: set to True to wait for the next message
        :type block: bool
        :param timeout: when blocking, max seconds to wait; raises queue.Empty if it elapses (None
            blocks forever). Lets the caller bound the wait so a hung game can't spin/stall forever.
        :type timeout: float
        :return: the message from Communication Mod
        :rtype: str
        """
        if block:
            return self.input_queue.get(timeout=timeout)   # raises queue.Empty after `timeout` s
        if not self.input_queue.empty():
            return self.input_queue.get()

    def receive_game_state_update(self, block=False, perform_callbacks=True, timeout=None):
        """Using the next message from Communication Mod, update the stored game state

        :param block: set to True to wait for the next message
        :type block: bool
        :param perform_callbacks: set to True to perform callbacks based on the new game state
        :type perform_callbacks: bool
        :param timeout: when blocking, max seconds to wait (queue.Empty propagates on elapse)
        :type timeout: float
        :return: whether a message was received
        """
        message = self.get_next_raw_message(block, timeout)
        if message is not None:
            try:
                communication_state = json.loads(message)
                self.last_raw_communication_state = communication_state
                self.last_error = communication_state.get("error", None)
                self.game_is_ready = communication_state.get("ready_for_command")
                if self.last_error is None:
                    self.in_game = communication_state.get("in_game")
                    if self.in_game:
                        self.last_game_state = Game.from_json(communication_state.get("game_state"), communication_state.get("available_commands"))
                if perform_callbacks:
                    if self.last_error is not None:
                        self.action_queue.clear()
                        new_action = self.error_callback(self.last_error)
                        self.add_action_to_queue(new_action)
                    elif self.in_game:
                        if len(self.action_queue) == 0 and perform_callbacks:
                            new_action = self.state_change_callback(self.last_game_state)
                            # The agent returns None for transient/unactionable states (e.g. the
                            # animation frame right after an in-combat Discovery select, which only
                            # offers key/click/wait). Enqueuing None crashes execute_next_action_if_ready
                            # on `.can_be_executed`; instead skip it and re-poll on the next state. The
                            # agent raises on genuinely-unhandled actionable screens, so a None here is
                            # always a wait, never a swallowed decision.
                            if new_action is not None:
                                self.add_action_to_queue(new_action)
                            else:
                                print("[coordinator] agent returned None (transient state); waiting "
                                      "for next state", file=sys.stderr)
                    elif self.stop_after_run:
                        self.clear_actions()
                    else:
                        new_action = self.out_of_game_callback()
                        self.add_action_to_queue(new_action)
                return True
            except Exception as e:
                print(f"Error handling game state: {e}\nMessage: {message}", file=sys.stderr)
                raise
        return False

    def _state_signature(self):
        """A cheap fingerprint of the current game state that changes whenever the game
        meaningfully advances. Used by the progress watchdog to distinguish a live no-progress
        loop (e.g. an unregistered shop buy) from genuine forward motion.
        """
        raw = self.last_raw_communication_state or {}
        gs = raw.get("game_state") or {}
        cs = gs.get("combat_state") or {}
        return (
            gs.get("floor"), gs.get("act"), gs.get("screen_type"),
            gs.get("gold"), gs.get("current_hp"),
            tuple(gs.get("choice_list") or ()),
            cs.get("turn"),
            len(cs.get("hand", [])),
            tuple((m.get("current_hp"), m.get("block")) for m in cs.get("monsters", [])),
        )

    def run(self):
        """Start executing actions forever

        :return: None
        """
        while True:
            self.execute_next_action_if_ready()
            self.receive_game_state_update(perform_callbacks=True)

    def play_one_game(self, player_class, ascension_level=0, seed=None):
        """

        :param player_class: the class to play
        :type player_class: PlayerClass
        :param ascension_level: the ascension level to use
        :type ascension_level: int
        :param seed: the alphanumeric seed to use
        :type seed: str
        :return: True if the game was a victory, else False
        :rtype: bool
        """
        self.clear_actions()
        while not self.game_is_ready:
            self.receive_game_state_update(block=True, perform_callbacks=False)
        if not self.in_game:
            StartGameAction(player_class, ascension_level, seed).execute(self)
            self.receive_game_state_update(block=True)
        # Drive the game until it ends. Block (with a timeout) on each state instead of busy-polling --
        # a busy poll burns 100% CPU and, worse, hides a hang forever. When the game goes silent,
        # nudge it with `state` (forces a fresh emit past any stuck "ready" flag).
        #
        # Abort on lack of PROGRESS, not lack of messages: a wedged shop buy that never registers is
        # a *live* loop (the game keeps re-emitting the same screen and we keep re-issuing the same
        # action), so silence stays at 0 and a silence timer never fires. Track a fingerprint of the
        # game state and abort if it hasn't changed for far longer than any real animation/load.
        silence = 0.0
        poll = 2.0
        nudged = False
        last_sig = self._state_signature()
        last_progress_t = time.monotonic()
        while self.in_game:
            self.execute_next_action_if_ready()
            try:
                self.receive_game_state_update(block=True, timeout=poll)
                silence = 0.0
                nudged = False
                sig = self._state_signature()
                if sig != last_sig:
                    last_sig = sig
                    last_progress_t = time.monotonic()
            except queue.Empty:
                silence += poll
                if silence >= 30.0 and not nudged:
                    print("[coordinator] no game state for 30s; sending 'state' to nudge the game",
                          file=sys.stderr)
                    self.send_message("state")
                    nudged = True
            # Progress watchdog -- covers both true silence and live no-progress loops.
            if time.monotonic() - last_progress_t >= 150.0:
                screen = getattr(self.last_game_state, "screen_type", "?")
                raise RuntimeError(f"No game-state progress for 150s; game appears hung "
                                   f"(last screen: {screen}).")
        assert self.last_game_state is not None
        if self.last_game_state.screen_type == ScreenType.GAME_OVER:
            return self.last_game_state.screen.victory
        else:
            return False

