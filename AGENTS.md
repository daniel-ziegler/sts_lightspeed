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
pip install pyarrow tqdm
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

### C++ Bindings

1. **Missing Bindings**: Add all needed fields to `bindings/slaythespire.cpp`
   - **Pattern**: `.def_readwrite("pythonName", &CppClass::cppFieldName)`
   - **Required**: Rebuild with `make -j8` after adding bindings
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
   - **Command**: `make -j8`
   - **Check**: Import and test immediately after build

## Implementing New Relics

When implementing new relics that affect game mechanics, follow these patterns established in the codebase:

### Relic Data Storage and Initialization

Relics can store persistent state using the `data` field in `RelicInstance`. Initialize this in `GameContext::obtainRelic()`:

```cpp
// In GameContext.cpp obtainRelic() switch statement
case RelicId::YOUR_RELIC: {
    relicData = 3;  // Set initial uses/counter
    break;
}
```

**Common patterns:**
- **Limited uses**: `NEOWS_LAMENT` (3), `WING_BOOTS` (3), `OMAMORI` (2), `MATRYOSHKA` (2)
- **Counters**: `MAW_BANK` (1 for tracking state)
- **No data needed**: Most passive relics don't need data (leave `relicData = 0`)

### Game Logic Integration

Different relic types require different integration points:

#### 1. **UI/Action Availability** (like WING_BOOTS)
- **Validation**: Update `isValidXXXAction()` functions in `GameAction.cpp`
- **Action Generation**: Update `getAllXXXActions()` functions to include new options
- **Console Display**: Update `printXXXActions()` in `ConsoleSimulator.cpp` to show new options
- **Usage Consumption**: Add logic in `GameAction::execute()` to consume relic uses

```cpp
// Example: In isValidMapAction()
if (gc.relics.has(RelicId::YOUR_RELIC) && gc.relics.getRelicValue(RelicId::YOUR_RELIC) > 0) {
    // Allow special action
}

// Example: In execute() for MAP_SCREEN
if (usedSpecialRelic) {
    gc.relics.getRelicValueRef(RelicId::YOUR_RELIC)--;
}
```

#### 2. **Battle Effects**
- Modify `BattleContext` methods for combat-related relics
- Add checks in damage calculation, card play, turn start/end

#### 3. **Event/Room Effects**
- Add cases in `GameContext::chooseEventOption()` for event-specific relics
- Modify room entry/exit logic in `transitionToMapNode()` and related methods

### UI Integration Checklist

When adding relics that provide new player choices:

1. **Core Logic** (`GameAction.cpp`):
   - [ ] Add validation in `isValidXXXAction()`
   - [ ] Add action generation in `getAllXXXActions()`
   - [ ] Add usage consumption in `execute()`

2. **Console Interface** (`ConsoleSimulator.cpp`):
   - [ ] Update `printXXXActions()` to show new options
   - [ ] Add visual indicators (like "(WING_BOOTS)")
   - [ ] Show remaining uses when relevant

3. **Initialization** (`GameContext.cpp`):
   - [ ] Add case in `obtainRelic()` switch statement
   - [ ] Set appropriate `relicData` value

4. **Testing**:
   - [ ] Verify relic initializes with correct data
   - [ ] Test that new actions appear in console
   - [ ] Confirm usage consumption works correctly
   - [ ] Build and test Python bindings

### Common Gotchas

- **ConsoleSimulator Sync**: UI display functions must match the logic in `getAllXXXActions()`, or players won't see new options
- **Screen State Logic**: Different screen states have different validation/action patterns
- **Relic Access**: Use `gc.relics.has()` and `gc.relics.getRelicValue()` for checks
- **Python Bindings**: Relics appear as `gc.relics` list with `.id` and `.data` fields
- **Save Compatibility**: RelicId enum values are mapped in `SaveFileMappings.h`

### Examples in Codebase

- **WING_BOOTS**: Map navigation bypass (this implementation)
- **NEOWS_LAMENT**: 3-use enemy kill counter
- **GIRYA**: Campfire action enabler with 3 upgrade limit
- **MAW_BANK**: State tracking relic
- **OMAMORI**: Curse negation with limited uses

## Action Systems

The codebase has two distinct Action systems: **combat Actions** and **search Actions**.

### Combat Actions (`combat/Actions.h`)

Combat Actions represent atomic game effects that modify battle state. They are defined using a macro-based system for code generation.

#### Action Definition

Actions are defined using the `FOREACH_ACTIONTYPE` macro, which generates:
- Struct definitions with typed fields (e.g., `_BuffPlayer` with fields `PlayerStatus s` and `int amount`)
- An enum `ActionType` for each action variant
- Static factory methods in the `Actions` namespace
- Operator overloads for copying, moving, and comparison

Each action struct implements a `call operator` that modifies the `BattleContext`:

```cpp
void _BuffPlayer::operator()(BattleContext &bc) const {
    if (s == PlayerStatus::CORRUPTION && !bc.player.hasStatus<PS::CORRUPTION>()) {
        bc.cards.onBuffCorruption();
    }
    bc.player.buff(s, amount);
}
```

#### Action Queue

The `BattleContext` maintains an `ActionQueue<50>` (a fixed-capacity deque) that holds pending actions:

```cpp
ActionQueue<50> actionQueue;
CardQueue cardQueue;
```

**Queue Operations:**
- `addToTop(Action)` / `actionQueue.pushFront()` - Adds action to front (executes next)
- `addToBot(Action)` / `actionQueue.pushBack()` - Adds action to back (executes later)
- `actionQueue.popFront()` - Removes and returns next action

**Execution Loop:**

The `BattleContext::executeActions()` method processes both queues:

1. Check for loop/turn limits (prevents infinite loops)
2. If `actionQueue` not empty: pop and execute an `Action`
3. If `cardQueue` not empty: pop and play a `CardQueueItem`
4. Continue until `inputState != EXECUTING_ACTIONS` or combat ends

Actions can add more actions to either queue during execution, enabling complex card effect chains. The queue architecture allows fine control over execution order - cards typically add effects to the bottom (after current effects), while some mechanics like "reactive" debuffs add to top (before pending effects).

### Search Actions (`sim/search/Action.h`)

Search Actions represent high-level player decisions for AI agents and tree search. These are compact 32-bit encoded actions.

#### Bit-Packed Structure

Actions pack into a single `uint32_t`:
- Bits 29-31 (3 bits): `ActionType` enum
- Bits 0-15 (16 bits): `idx1` (source index or select index)
- Bits 16-28 (13 bits): `idx2` (target index)

**Action Types:**
- `CARD` - Play card from hand at target
- `POTION` - Use/discard potion
- `SINGLE_CARD_SELECT` - Pick one card (Armaments, Exhume, etc.)
- `MULTI_CARD_SELECT` - Pick multiple cards (bit flags for indices)
- `END_TURN` - End player turn

#### Validation and Execution

Search Actions validate against `BattleContext` state:

```cpp
bool Action::isValidAction(const BattleContext &bc) const {
    // Check outcome, input state, and action-specific constraints
}
```

When executed, search Actions translate to combat Actions:

```cpp
void Action::execute(BattleContext &bc) const {
    switch (getActionType()) {
        case ActionType::CARD:
            const CardQueueItem item(bc.cards.hand[getSourceIdx()], getTargetIdx(), bc.player.energy);
            bc.addToBotCard(item);
            break;
        // ... handle other types
    }
    bc.inputState = InputState::EXECUTING_ACTIONS;
    bc.executeActions();  // Process the combat action queue
}
```

Search Actions serve as the interface between AI agents and the game engine, while combat Actions implement the actual game mechanics.

### RNG System

`BattleContext` maintains a **single unified RNG stream**, seeded from `GameContext`:

```cpp
Random rng;  // All combat randomness
```

Each `Random` object uses an XORShift algorithm with a 64-bit state and maintains a `counter` for deterministic replay. The RNG is initialized at battle start:

```cpp
rng = Random(gc.seed + gc.floorNum);
```

After battle, the RNG state syncs back to `GameContext.rng` to persist across floors within the same act.

**Key RNG Methods:**
- `random(int range)` - Returns `[0, range]` inclusive, increments counter
- `random(int start, int end)` - Returns `[start, end]` inclusive
- `randomBoolean()` - Returns true/false
- `randomFloat()` / `randomDouble()` - Returns normalized floating point

The counter enables RNG state synchronization for replay/debugging. All combat-related randomness (monster AI, card generation, HP variance, potion drops, deck shuffling) uses this single stream.

`GameContext` maintains separate RNG streams for non-combat systems: `cardRng` (card rewards), `eventRng` (events), `merchantRng` (shop), `monsterRng` (encounter generation), `neowRng` (Neow blessings), `relicRng` (relic generation), `treasureRng` (treasure rooms), and `rng` (general/combat-related).

**Important:** RNG accuracy is no longer maintained relative to the base game, since we've unified the battle RNG streams into a single one.

# Important instructions

- Do not make code changes backward-compatible! Just refactor things to use the new way of doing things. I want to keep the code clean without backward compatibility shims.
- Always make everything *uniform*! Never handle multiple possible input data types or formats. Instead refactor at least one of the sources of input so that they are the same.
- DO NOT assume warnings are not a problem. They are a problem! Figure out why they are being produced.
- NO DEFENSIVE PROGRAMMING! DO NOT swallow errors or handle unexpected states. Add in asserts and throw exceptions liberally. If anything surprising could be due to a bug or a bad input, *throw an exception*!
- If you cannot figure out how to fully satisfy request, say so and give up! Do not revert to simpler approaches or hack around issues.
- Never leave comments like "# Changed to new thing". Only leave comments that describe the current state of the code. And do so sparingly - prefer not to comment, unless the code would be unclear without.
- NEVER swallow errors just to make tests pass.