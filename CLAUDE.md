# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sts_lightspeed** is a high-performance C++20 implementation of Slay the Spire designed for tree search algorithms and machine learning. It achieves 1M random playouts in 5 seconds with 16 threads and is optimized for ML training and data generation (note: RNG accuracy is no longer maintained).

## Development Environment

- I run things on a cloud GPU machine; it's expected that there's no GPU available locally, but local runs can be useful for small scale testing

## Build Commands

```bash
# Standard build process
cmake .
make -j8

# Python bindings (install development dependencies first)
pyenv shell 3.10.14 && pip install pyarrow tqdm
```

## Testing Commands

```bash
# Local testing command (CPU, small scale)
pyenv shell 3.10.14 && python rl_train.py --algorithm ppo --num-workers 8 --inf-batch-size 4 --inf-batch-size-factor 2 --num-epochs 2 --batch-size 32 --num-games-per-step 32 --torch-compile no --separate-networks false --value-fork-layer 1 --num-value-layers 1 --model-dim 64
```