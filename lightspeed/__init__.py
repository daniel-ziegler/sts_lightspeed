"""sts_lightspeed Python code: training, evaluation, analysis, and the live-game bridge (lightspeed.bridge)."""
import os

# Repo root (the directory containing this package); runs/ holds checkpoints, captures, and logs.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
