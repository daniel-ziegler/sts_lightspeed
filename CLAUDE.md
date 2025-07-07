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
    - Main definitions: `bindings/slaythespire.cpp`

### Key Implementation Details

- **Performance**: Fixed-size containers (`fixed_list.h`), compile flags `-O1 -Wno-shift-count-overflow`
- **Dependencies**: nlohmann/json and PyBind11 as git submodules
- **Data Generation**: Focused on generating training data for neural networks rather than RNG accuracy
- **Multi-threading**: Built-in parallel simulation support across agents

### Python Integration

The codebase includes Python files for ML training and data generation:

- **`network.py`** - Complete neural network architecture using transformer layers to predict win probabilities for card/relic choices and fixed actions. Includes custom embeddings, attention mechanisms, and data processing utilities
- **`train.py`** - Training pipeline with hyperparameter sweeping, validation splits, and comprehensive evaluation including ROC curves and card/relic statistics. Supports command-line arguments for flexible training configuration
- **`playouts.py`** - High-performance data generation script that runs thousands of games using neural network guidance with Boltzmann sampling. Features multi-threaded batched inference, choice statistics, and parallel game execution
- **`ppo_train.py`** - Proximal Policy Optimization (PPO) reinforcement learning training system. Collects experience trajectories from games and trains policy/value networks using GAE advantages
- **`inputs.py`** - Generic input space framework with embedding builders for sequences, enums, fixed vectors, and composite types. Provides abstraction layer for neural network input processing
- **`run.py`** - Simple game runner that plays a single game with neural network agent, useful for testing and debugging
- Various `.parquet` files contain training data rollouts from different experiments

To use the right Python environment, prefix all python commands with `pyenv shell 3.10.14 &&`

### PPO Training Details

The **`ppo_train.py`** implements Proximal Policy Optimization reinforcement learning with the following key characteristics:

- **Experience Collection**: Runs parallel game episodes using `ThreadPoolExecutor` and `as_completed()` to collect trajectories
- **Trajectory Bias**: Since `as_completed()` returns finished games in completion order, shorter (typically worse-performing) games complete first and appear earlier in the batch. This creates a systematic bias where the first trajectories are often poor performers
- **Debug Output**: Uses random trajectory selection instead of first trajectory to avoid the completion order bias when displaying training progress
- **Network Architecture**: Supports both single network with value head or separate policy/value networks
- **Reward Functions**: Multiple reward function options including sparse victory rewards, dense floor progress, and perfected strike counting
- **Checkpointing**: Automatic model saving with resume functionality using `--resume-from-step` and checkpoint paths based on `--save-path`
- **GAE Advantages**: Computes Generalized Advantage Estimation for stable policy gradient training

### Neural Network Action Support

The ML pipeline supports neural network decision-making for specific screen states through a structured choice system. Here's what's required to add support for a new action type:

#### Adding New FixedAction Types
When adding actions that map to existing choice categories (like rest site actions → fixed actions):

1. **`network.py`**: Add new `FixedAction` enum values
2. **`playouts.py`**: 
   - Add new screen state case in `construct_choice()` function
   - Map C++ action indices to appropriate `FixedAction` types
   - Add screen state to neural network condition in `run_game()`
3. **`ppo_train.py`**: Add screen state to neural network condition in `run_ppo_episode()`

#### Adding New Choice Categories
If you needed to add a completely new choice type (like `events_offered` field to `Choice`):

1. **`playouts.py`**: 
   - Add new field to `Choice` dataclass constructor and `as_dict()` method
   - Add handling in `construct_choice()` for the new choice type
   - Add path handling in `pick_card_with_net()` function
   - Add choice type mapping in action path processing
2. **`network.py`**: 
   - Add new `ActionType` enum value
   - Add new field to `choice_space` DictSpace definition
   - Update network architecture if needed for new input dimensions
3. **`ppo_train.py`**: Add path handling for new choice type in experience collection
4. **`train.py`**: Update validation and statistics collection for new choice type

#### Key Patterns
- **C++ Integration**: Game actions use `idx1`, `idx2` fields and `rewards_action_type` for structured actions
- **Choice Mapping**: `construct_choice()` maps C++ actions to typed Python choice objects
- **Path System**: `choice_space.ix_to_path()` converts flat neural network indices back to semantic choices
- **Consistency**: Both `playouts.py` and `ppo_train.py` must handle the same screen states identically

The system is designed to be extensible - most new action types can be added by following these established patterns without changing the core neural network architecture.

## Development Notes

- C++20 standard required
- Uses CMake build system with git submodules for dependencies
- All Ironclad cards and colorless cards implemented
- Complete enemy roster and relic system
- Console playable with full overworld/map system

# Important instructions

- Do not make code changes backward-compatible! Just refactor things to use the new way of doing things. I want to keep the code clean without backward compatibility shims.
- Always make everything *uniform*! Never handle multiple possible input data types or formats. Instead refactor at least one of the sources of input so that they are the same.
- DO NOT assume warnings are not a problem. They are a problem! Figure out why they are being produced.
- Fail fast and hard! DO NOT swallow errors. Add in asserts and throw exceptions liberally. If anything surprising could be due to a bug or a bad input, *throw an exception*!
- Never leave comments like "# Changed to new thing". Only leave comments that describe the current state of the code. And do so sparingly - prefer not to comment, unless the code would be unclear without.
- NEVER swallow errors just to make tests pass.