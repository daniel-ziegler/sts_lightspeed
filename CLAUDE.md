# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sts_lightspeed** is a high-performance C++20 implementation of Slay the Spire designed for tree search algorithms and machine learning. It achieves 1M random playouts in 5 seconds with 16 threads and is optimized for ML training and data generation (note: RNG accuracy is no longer maintained).

## Build Commands

```bash
# Standard build process
cmake .
make -j8

# Python bindings (install development dependencies first)
pyenv shell 3.10.14 && pip install pyarrow tqdm
```

This creates four executables:
- `slaythespire` - Python module for ML workflows
- `main` - Interactive console simulator  
- `test` - Testing and benchmarking suite
- `small-test` - Lightweight test executable

## Testing Commands

The `test` executable provides various testing and simulation modes:

```bash
# Multi-threaded agent testing
./test agent_mt [threads] [depth] [ascension] [seed] [playouts] [print_level]

# Simple agent testing  
./test simple_agent_mt [threads] [seed] [playouts] [print]

# MCTS from save file
./test mcts_save [save_file] [simulations]

# Convert save file to JSON
./test json [save_file_path] [json_output_path]

# Convert JSON back to save file
./test json_to_save [json_input_path] [save_file_output_path]
```

## Console Simulation

```bash
# Interactive play
./main
# Input format: seed character(I/S/D/W) ascensionLevel
# Example: 12345 I 0
```

## Architecture Overview

### Core Components

- **Game Logic** (`src/game/`, `include/game/`): GameContext, Card/Deck management, Map progression, SaveFile handling
- **Combat System** (`src/combat/`, `include/combat/`): BattleContext, Player/Monster, Actions, turn-based sequencing  
- **Constants** (`include/constants/`): All game data definitions (Cards, Relics, Monsters, Events, StatusEffects)
- **Simulation** (`src/sim/`, `include/sim/`): ConsoleSimulator, BattleSimulator, debug utilities
- **AI Agents** (`src/sim/search/`): ScumSearchAgent2 (MCTS), SimpleAgent, ExpertKnowledge heuristics
- **Python Bindings** (`bindings/`): PyBind11 integration for ML workflows

### Key Implementation Details

- **Performance**: Fixed-size containers (`fixed_list.h`), compile flags `-O1 -Wno-shift-count-overflow`
- **Dependencies**: nlohmann/json and PyBind11 as git submodules
- **Data Generation**: Focused on generating training data for neural networks rather than RNG accuracy
- **Multi-threading**: Built-in parallel simulation support across agents

### Python Integration

The codebase includes Python files for ML training and data generation:

- **`network.py`** - Complete neural network architecture using transformer layers to predict win probabilities for card/relic choices and fixed actions. Includes custom embeddings, attention mechanisms, and data processing utilities
- **`train.py`** - Training pipeline with hyperparameter sweeping, validation splits, and comprehensive evaluation including ROC curves and card/relic statistics. Supports command-line arguments for flexible training configuration
- **`randomplayouts.py`** - High-performance data generation script that runs thousands of games using neural network guidance with Boltzmann sampling. Features multi-threaded batched inference, choice statistics, and parallel game execution
- **`inputs.py`** - Generic input space framework with embedding builders for sequences, enums, fixed vectors, and composite types. Provides abstraction layer for neural network input processing
- **`run.py`** - Simple game runner that plays a single game with neural network agent, useful for testing and debugging
- Various `.parquet` files contain training data rollouts from different experiments

To use the right Python environment, prefix all python commands with `pyenv shell 3.10.14 &&`

## Development Notes

- C++20 standard required
- Uses CMake build system with git submodules for dependencies
- All Ironclad cards and colorless cards implemented
- Complete enemy roster and relic system
- Console playable with full overworld/map system

# Important instructions

Do not make code changes backward-compatible! Just refactor things to use the new way of doing things. I want to keep the code clean without backward compatibility shims.

