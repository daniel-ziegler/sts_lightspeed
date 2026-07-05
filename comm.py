"""Entry point for the live-game bridge; the implementation lives in silverbot/bridge/.

Kept at the repo root so `python3 comm.py ...` (run_live.sh) and `import comm` (run_grind.sh's
seed generation) keep working. Equivalent to `python3 -m silverbot.bridge.cli`.
"""
import sys

from silverbot.bridge import *
from silverbot.bridge.cli import DEFAULT_CKPT, load_policy_service, run_agent_cli

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run the conversion self-test
        print("Testing spirecomm to GameContext converter...")
        test_basic_conversion()
    else:
        run_agent_cli()
