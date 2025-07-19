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

## Reinforcement Learning Training

The unified RL trainer supports both PPO and PPG algorithms:

```bash
# PPO training (single-phase)
pyenv shell 3.10.14 && python rl_train.py --algorithm ppo --separate-networks true

# PPG training with Reloaded enhancements (recommended)
pyenv shell 3.10.14 && python rl_train.py --algorithm ppg --separate-networks true

# PPG with custom settings
pyenv shell 3.10.14 && python rl_train.py --algorithm ppg \
  --separate-networks true \
  --adaptive-kl-reg true \
  --policy-reg-coef 1.0 \
  --n-policy-iterations 4 \
  --n-aux-epochs 2

# Training with specific parameters
pyenv shell 3.10.14 && python rl_train.py --algorithm ppg \
  --num-iterations 1000 \
  --num-games-per-step 256 \
  --num-workers 40 \
  --reward-function victory \
  --save-path my_model
```

**Key Parameters:**
- `--algorithm`: Choose `ppo` or `ppg`
- `--separate-networks`: Use separate policy/value networks (recommended for PPG)
- `--adaptive-kl-reg`: Enable adaptive KL regularization (PPG Reloaded)
- `--n-policy-iterations`: Auxiliary phase frequency (higher = less frequent, more efficient)
- `--reward-function`: Choose from `victory`, `smooth`, `perfected_strike`, `no_pstrikes`

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
- **`rl_train.py`** - Unified PPO and PPG (Phasic Policy Gradient) reinforcement learning training system with full PPG Reloaded algorithm implementation. Supports both single and separate network architectures with adaptive regularization
- **`inputs.py`** - Generic input space framework with embedding builders for sequences, enums, fixed vectors, and composite types. Provides abstraction layer for neural network input processing
- **`run.py`** - Simple game runner that plays a single game with neural network agent, useful for testing and debugging
- Various `.parquet` files contain training data rollouts from different experiments

To use the right Python environment, prefix all python commands with `pyenv shell 3.10.14 &&`

### Reinforcement Learning Training Details

The **`rl_train.py`** implements both PPO (Proximal Policy Optimization) and PPG (Phasic Policy Gradient) with PPG Reloaded enhancements:

#### PPO Mode (`--algorithm ppo`)
- **Standard PPO**: Single-phase training with policy and value updates
- **Network Options**: Single network with value head or separate policy/value networks
- **Experience Collection**: Parallel game episodes using trajectory collection
- **GAE Advantages**: Generalized Advantage Estimation for stable policy gradients

#### PPG Mode (`--algorithm ppg`)
- **Two-Phase Training**: Alternates between policy phase (PPO) and auxiliary phase (feature distillation)
- **Auxiliary Value Heads**: Policy network includes auxiliary value head for joint training during auxiliary phase
- **Behavioral Cloning**: Preserves original policy distribution during auxiliary phase updates
- **Trajectory Sampling**: Uses complete trajectory sampling for auxiliary phase data diversity

#### PPG Reloaded Enhancements
- **Adaptive KL Regularization**: Dynamically adjusts regularization strength based on measured policy drift
- **Enhanced Policy Regularization**: Stronger default regularization (1.0 vs 0.5) for better stability
- **Computational Efficiency**: Configurable auxiliary phase frequency via `n_policy_iterations` parameter
- **Data Diversity**: Trajectory-level buffer management for better experience diversity

#### Common Features
- **Reward Functions**: Multiple options including sparse victory, dense floor progress, and perfected strike counting
- **Checkpointing**: Automatic model saving with resume functionality using `--resume-from-step`
- **Multi-threading**: Parallel experience collection with configurable worker count
- **TensorBoard Logging**: Comprehensive metrics tracking including adaptive coefficients

### Neural Network Action Support

The ML pipeline supports neural network decision-making for specific screen states through a structured choice system. Here's what's required to add support for a new action type:

#### Adding New FixedAction Types
When adding actions that map to existing choice categories (like rest site actions → fixed actions):

1. **`network.py`**: Add new `FixedAction` enum values
2. **`playouts.py`**: 
   - Add new screen state case in `construct_choice()` function
   - Map C++ action indices to appropriate `FixedAction` types
   - Add screen state to neural network condition in `run_game()`
3. **`rl_train.py`**: Add screen state to neural network condition in training episodes

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
3. **`rl_train.py`**: Add path handling for new choice type in experience collection
4. **`train.py`**: Update validation and statistics collection for new choice type

#### Key Patterns
- **C++ Integration**: Game actions use `idx1`, `idx2` fields and `rewards_action_type` for structured actions
- **Choice Mapping**: `construct_choice()` maps C++ actions to typed Python choice objects
- **Path System**: `choice_space.ix_to_path()` converts flat neural network indices back to semantic choices
- **Consistency**: Both `playouts.py` and `rl_train.py` must handle the same screen states identically

The system is designed to be extensible - most new action types can be added by following these established patterns without changing the core neural network architecture.

## Development Notes

- C++20 standard required
- Uses CMake build system with git submodules for dependencies
- All Ironclad cards and colorless cards implemented
- Complete enemy roster and relic system
- Console playable with full overworld/map system

## Common Pitfalls and Solutions

### Python/C++ Integration Issues

1. **Enum Bounds Checking**: All `IntEnum` classes used with `EnumSpace` must start at 0, not 1
   - **Wrong**: `class MyEnum(IntEnum): NONE = auto()` (starts at 1)
   - **Right**: `class MyEnum(IntEnum): NONE = 0; OTHER = auto()` (starts at 0)
   - **Why**: `EnumSpace` checks `0 <= x < len(enum_class)` but `auto()` starts at 1

2. **Object Attribute Access**: Check object types before accessing attributes
   - **Wrong**: `gc.deck.cards[idx]`, `gc.relics.relics[idx]` (deck and relics are lists, not objects)
   - **Right**: `gc.deck[idx]`, `gc.relics[idx]` (they are the lists themselves)
   - **Check**: Use `type(obj)` and `dir(obj)` to verify structure

3. **Card/Relic ID Access**: Use correct attribute names and extract IDs from objects
   - **Wrong**: `card.getId()`, `relic.getId()` (methods don't exist)
   - **Right**: `card.id`, `relic.id` (direct attribute access)
   - **Important**: `gc.relics[idx]` returns a `Relic` object, use `gc.relics[idx].id` for the ID
   - **Check**: Test with `hasattr(obj, 'method_name')` first

### C++ Bindings

1. **Missing Bindings**: Add all needed fields to `bindings/slaythespire.cpp`
   - **Pattern**: `.def_readwrite("pythonName", &CppClass::cppFieldName)`
   - **Required**: Rebuild with `make` after adding bindings
   - **Check**: Use `dir(obj)` to verify fields are exposed

2. **Naming Conventions**: C++ uses camelCase, Python expectations vary
   - **C++**: `skillCardDeckIdx`, `relicIdx0`, `hpAmount1`
   - **Python**: Access exactly as defined in C++ (don't convert to snake_case)

3. **Nested Namespaces**: PyBind11 doesn't handle nested namespaces well
   - **Wrong**: `pybind11::enum_<Neow::Bonus>(m, "Neow.Bonus")`
   - **Right**: `pybind11::enum_<Neow::Bonus>(m, "NeowBonus")`
   - **Access**: Use `sts.NeowBonus.THREE_CARDS` not `sts.Neow.Bonus.THREE_CARDS`

### Neural Network Pipeline

1. **Choice Space Structure**: Match tensor structure to space definitions
   - **DictAddSpace**: Requires nested dictionary in batch tensors
   - **Collation**: Manually populate each field, can't use `torch.tensor()` on individual elements
   - **Defaults**: Use `.get()` with sensible defaults for optional fields

2. **Batch Tensor Initialization**: Structure must match space definition
   - **Wrong**: Single tensor for `DictAddSpace`
   - **Right**: Nested dict with tensor for each field
   - **Example**: `'fixed': {'value': {'action': tensor, 'gold': tensor}, 'mask': tensor}`

3. **Data Type Consistency**: Ensure all tensor assignments use correct types
   - **Wrong**: `batch['field'][i,j] = torch.tensor(value)`
   - **Right**: `batch['field'][i,j] = int(value)` (for int32 tensors)

### Choice System Updates

1. **Backward Compatibility**: Update all related functions when changing data structures
   - **Required**: Update `as_dict()`, `construct_choice()`, `path_to_action_and_desc()`
   - **Pattern**: Use `isinstance(obj, dict)` to handle both old and new formats temporarily

2. **Serialization**: Use `flatten_dict()` for nested structures before collation
   - **Pattern**: `flattened = flatten_dict(choice.as_dict())`
   - **Required**: Add dummy fields for training: `choice_type`, `chosen_idx`, `outcome`

3. **Optional Fields**: Design for sparse data to avoid boilerplate
   - **Pattern**: Only include non-default values in dictionaries
   - **Collation**: Use `.get(key, default)` to handle missing fields

### Testing and Debugging

1. **Incremental Testing**: Test each component separately
   - **Order**: Choice creation → Serialization → Collation → Neural network
   - **Tools**: Use `dir()`, `type()`, `hasattr()` to inspect objects

2. **Event Action Mapping**: Check GameContext.cpp for correct action structure
   - **Example**: NLOTH event has 3 cases (0, 1, 2)
   - **Pattern**: Look at `GameContext::chooseEventOption()` switch statements
   - **Verify**: Action indices match the C++ implementation exactly

3. **Build After Changes**: Always rebuild after C++ binding changes
   - **Command**: `pyenv shell 3.10.14 && make`
   - **Check**: Import and test immediately after build

# Important instructions

- Do not make code changes backward-compatible! Just refactor things to use the new way of doing things. I want to keep the code clean without backward compatibility shims.
- Always make everything *uniform*! Never handle multiple possible input data types or formats. Instead refactor at least one of the sources of input so that they are the same.
- DO NOT assume warnings are not a problem. They are a problem! Figure out why they are being produced.
- NO DEFENSIVE PROGRAMMING! DO NOT swallow errors or handle unexpected states. Add in asserts and throw exceptions liberally. If anything surprising could be due to a bug or a bad input, *throw an exception*!
- Either figure things out or give up. Do not revert to simpler approaches or hack around issues.
- Never leave comments like "# Changed to new thing". Only leave comments that describe the current state of the code. And do so sparingly - prefer not to comment, unless the code would be unclear without.
- NEVER swallow errors just to make tests pass.