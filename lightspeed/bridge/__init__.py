"""Live-game bridge: spirecomm <-> engine conversion and the MCTS agent driving a real game.

Modules: mappings (string<->enum tables + live-power application), combat (BattleContext
reconstruction), overworld (GameContext/screen conversion), actions (engine action -> live
command), seeds (seed <-> base-35 string), agent (STSLightspeedAgent), cli (entry point).
The repo-root comm.py is the runnable entry point (equivalent to python3 -m
lightspeed.bridge.cli) and re-exports this package's public API.
"""
from lightspeed.bridge.mappings import *
from lightspeed.bridge.combat import *
from lightspeed.bridge.overworld import *
from lightspeed.bridge.actions import *
from lightspeed.bridge.seeds import *
from lightspeed.bridge.agent import STSLightspeedAgent
from lightspeed.bridge.cli import DEFAULT_CKPT, load_policy_service, run_agent_cli
