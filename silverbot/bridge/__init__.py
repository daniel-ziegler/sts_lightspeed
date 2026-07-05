"""Live-game bridge: spirecomm <-> engine conversion and the MCTS agent driving a real game.

Modules: mappings (string<->enum tables + live-power application), combat (BattleContext
reconstruction), overworld (GameContext/screen conversion), actions (engine action -> live
command), seeds (seed <-> base-35 string), agent (STSLightspeedAgent), cli (entry point).
The repo-root comm.py is the runnable entry point (equivalent to python3 -m
silverbot.bridge.cli) and re-exports this package's public API.
"""
from silverbot.bridge.mappings import *
from silverbot.bridge.combat import *
from silverbot.bridge.overworld import *
from silverbot.bridge.actions import *
from silverbot.bridge.seeds import *
from silverbot.bridge.agent import STSLightspeedAgent
from silverbot.bridge.cli import DEFAULT_CKPT, load_policy_service, run_agent_cli
