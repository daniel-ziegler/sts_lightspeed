# STS Lightspeed AI Agent for CommunicationMod

This module provides a command-line interface that enables our high-performance C++ STS implementation to control the real Slay the Spire game via [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod).

## Features

- **Real-time Game Control**: Interfaces with CommunicationMod to play the actual game
- **High-Performance AI**: Uses our optimized C++ battle simulation for decision making  
- **Full State Conversion**: Converts spirecomm game state to our internal BattleContext format
- **Multiple Characters**: Supports Ironclad, Silent, and Defect characters
- **Extensible Design**: Easy to plug in different AI strategies

## Usage

### Command Line Interface

```bash
# Run the AI agent for CommunicationMod
python comm.py --character ironclad --games 1

# Available options
python comm.py --help

# Run conversion tests
python comm.py --test
```

### Shell Script Wrapper

```bash
# Make executable (one time setup)
chmod +x run_agent.sh

# Run the agent
./run_agent.sh ironclad 1
./run_agent.sh silent 5
./run_agent.sh defect 0   # Play infinitely
```

### Integration with CommunicationMod

1. Install CommunicationMod in Slay the Spire
2. Set up the mod to call our agent:
   ```bash
   # In CommunicationMod settings, set the AI command to:
   python /path/to/sts_lightspeed/comm.py --character ironclad --games 1
   ```
3. Start Slay the Spire and begin a game
4. The AI will take control and play automatically

## How It Works

1. **Communication Protocol**: Uses stdin/stdout to communicate with CommunicationMod
2. **State Conversion**: Converts spirecomm's JSON game state to our C++ BattleContext
3. **AI Decision Making**: Uses our battle simulation to evaluate actions
4. **Action Translation**: Converts our actions back to spirecomm command format
5. **Real-time Execution**: Sends commands to control the real game

## Key Components

### STSLightspeedAgent Class
- Main AI agent that handles game state and makes decisions
- Converts between spirecomm format and our internal representation
- Implements fallback logic for robust operation

### State Conversion Functions
- `spirecomm_to_gamecontext()`: Convert basic game state
- `spirecomm_to_battlecontext()`: Convert full combat state with card piles and powers  
- `convert_combat_state()`: Handle battle-specific conversion
- `gamecontext_to_spirecomm_action()`: Convert AI actions to game commands
- `map_move_id()`: Maps monster move IDs from spirecomm to C++ MonsterMoveId enum
- `map_power_id()`: Maps player status effects from spirecomm to C++ PlayerStatus enum
- `map_monster_power_id()`: Maps monster status effects to C++ MonsterStatus enum

### Battle Context Integration
- Full player state: HP, energy, block, powers/buffs/debuffs
- Monster states: HP, block, powers, targeting information
- Card piles: Hand, draw pile, discard pile, exhaust pile
- Combat tracking: Turn state, energy management
- Monster move history tracking for AI decision making

### Enum Mapping System
- **Monster Move Mapping**: 200+ mappings from spirecomm monster strings + move IDs to MonsterMoveId enum
- **Status Effect Mapping**: Player and monster status effects with proper naming alignment
- **Monster ID Mapping**: Monster identification strings aligned between spirecomm and C++ systems

## Supported Features

- ✅ Combat decision making with card play and targeting
- ✅ Non-combat screens (card rewards, map navigation, rest sites)
- ✅ Power/buff/debuff conversion and management
- ✅ Full card pile state synchronization
- ✅ Monster state tracking and targeting
- ✅ Error handling and recovery
- ✅ Multiple character classes

## Extending the AI

To improve the AI decision making, modify the following methods in `STSLightspeedAgent`:

- `handle_combat()`: Improve combat strategy using BattleContext
- `handle_choice_screen()`: Add smarter non-combat decisions
- Add integration with existing ML models or search algorithms

## Mapping System Details

### Monster Move ID Mapping

The system maps spirecomm monster move data to our C++ `MonsterMoveId` enum using a comprehensive lookup table in `map_move_id()`:

```python
def map_move_id(monster_string: str, move_id: int) -> sts.MonsterMoveId:
    key = (monster_string, move_id)
    move_mapping = {
        ("Cultist", 1): sts.MonsterMoveId.CULTIST_DARK_STRIKE,
        ("Cultist", 3): sts.MonsterMoveId.CULTIST_INCANTATION,
        # ... 200+ mappings for all monsters
    }
```

**Key Implementation Details:**
- **Java Source Analysis**: Move IDs extracted from decompiled Java monster classes
- **Gap Handling**: Many monsters have non-sequential move numbering (e.g., SlaverBlue uses moves 1,4 not 1,2)
- **Escape Moves**: Gremlin types use move_id=99 for escape behavior
- **Special Numbering**: Some monsters start at move_id=0, others at 1
- **State-Based Moves**: Complex monsters like Lagavulin have moves for different internal states

**Comprehensive Coverage:**
- **Exordium**: 16 monsters (Cultist, JawWorm, Louses, Gremlins, Slimes, etc.)
- **City**: 11 monsters (Chosen, Byrds, Slavers, Taskmaster, etc.)  
- **Beyond**: 11 monsters (Darkling, Orb Walker, Repulsors, etc.)
- **Ending**: 4 elite monsters (Spire Growth/Shield/Spear, corrupt bosses)

### Status Effect Mapping Challenges

**Player Status Effects:**
- **Naming Mismatches**: spirecomm sends "Strength Down" but C++ expected "Lose Strength"
- **Solution**: Updated `playerStatusStrings[]` array to match spirecomm naming
- **Similar Issues**: Handled "Weakened" vs "Weak", focus vs strength debuff variations

**Monster Status Effects:**
- **Duplicate Names**: "Lock-On" vs "Lockon", "Regenerate" vs "Regeneration" 
- **Solution**: Added multiple mappings in `map_monster_power_id()` for common variants

### Monster ID Mapping Issues

**String Format Mismatches:**
- **Space vs No-Space**: spirecomm "GremlinNob" vs C++ "Gremlin Nob" 
- **Solution**: Updated `monsterIdStrings[]` to match spirecomm format exactly
- **Pattern**: Prefer spirecomm naming since it comes from the actual game data

### Lessons Learned

1. **Systematic Verification Required**: Runtime errors revealed gaps not found by static analysis
2. **Java Source Truth**: Decompiled Java code provides exact move ID constants and patterns
3. **Non-Sequential Numbering**: Many monsters skip move IDs or use special patterns
4. **Naming Consistency**: Always align C++ string constants with spirecomm format
5. **Comprehensive Testing**: Need actual game runs to catch all mapping edge cases
6. **Error Handling**: INVALID fallbacks with warnings prevent crashes while highlighting issues
7. **Documentation**: Track all special cases and numbering patterns for future reference

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure spirecomm is installed: `pip install spirecomm`
2. **Build Errors**: Make sure the C++ module is built: `make -j8`
3. **Communication Timeouts**: Check CommunicationMod is properly configured
4. **Unknown Mapping Warnings**: Check runtime output for missing monster/status mappings

### Mapping Errors

```bash
# Common error patterns and solutions:
Unknown monster move mapping for 'MonsterName' move_id=X
→ Check Java source for exact move ID constants
→ Add mapping to move_mapping dict in map_move_id()

Unknown status name: StatusName
→ Check playerStatusStrings[] in PlayerStatusEffects.h
→ Update string to match spirecomm format

Unknown monster id 'MonsterName' 
→ Check monsterIdStrings[] in MonsterIds.h
→ Update string to match spirecomm format
```

### Testing

```bash
# Test basic conversion
python comm.py --test

# Test agent creation
python -c "import comm; agent = comm.STSLightspeedAgent(); print('OK')"

# Test CLI without game
timeout 5 python comm.py --character ironclad --games 1
# Should output "ready" and wait for input

# Check for mapping warnings during actual play
python comm.py --character ironclad --games 1 2>&1 | grep -i "warning\|unknown"
```

## Architecture

```
Real Slay the Spire Game
         ↓ (JSON game state)
    CommunicationMod
         ↓ (stdin/stdout)
    STSLightspeedAgent  
         ↓ (state conversion + mapping)
    C++ BattleContext
         ↓ (AI decision)
    GameAction
         ↓ (command translation)
    Spirecomm Action
         ↓ (stdout)
    CommunicationMod
         ↓ (game command)
Real Slay the Spire Game
```

## Technical Implementation Notes

### Monster Move ID Extraction Process

The monster move mappings were systematically extracted using the following process:

1. **Java Source Analysis**: Examined decompiled Java files in `/mnt/c/Users/zieDa/Downloads/sts_java/com/megacrit/cardcrawl/monsters/`
2. **Constant Extraction**: Located `static final byte` move ID constants in each monster class
3. **Pattern Recognition**: Identified gaps, special numbering, and escape move patterns
4. **Runtime Verification**: Used actual game runs to verify mappings and catch edge cases
5. **Systematic Correction**: Fixed each runtime warning individually with precise mappings

### String Alignment Strategy

When spirecomm and C++ string formats differed, we consistently updated C++ to match spirecomm:
- **Rationale**: spirecomm data comes directly from the game, making it authoritative
- **Examples**: `"Lose Strength"` → `"Strength Down"`, `"GremlinWarrior"` → `"GremlinNob"`
- **Files Modified**: `include/constants/PlayerStatusEffects.h`, `include/constants/MonsterIds.h`

### Error Handling Design

For now, we try to continue instead of immediately crashing, but we will change that later.
- **INVALID Fallback**: Return enum INVALID values instead of crashing  
- **Warning Messages**: Print to stderr for debugging without breaking game flow

### C++ Bindings Completeness

Updated Python bindings to expose all 210+ MonsterMoveId enum values:
- **File**: `bindings/slaythespire.cpp` lines 1440-1650+

This creates a complete pipeline from the real game state to our high-performance AI decision making and back to game control, with comprehensive enum mapping ensuring data fidelity throughout the conversion process.