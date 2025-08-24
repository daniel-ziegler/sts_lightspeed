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

### Battle Context Integration
- Full player state: HP, energy, block, powers/buffs/debuffs
- Monster states: HP, block, powers, targeting information
- Card piles: Hand, draw pile, discard pile, exhaust pile
- Combat tracking: Turn state, energy management

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

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure spirecomm is installed: `pip install spirecomm`
2. **Build Errors**: Make sure the C++ module is built: `make -j8`
3. **Communication Timeouts**: Check CommunicationMod is properly configured

### Testing

```bash
# Test basic conversion
python comm.py --test

# Test agent creation
python -c "import comm; agent = comm.STSLightspeedAgent(); print('OK')"

# Test CLI without game
timeout 5 python comm.py --character ironclad --games 1
# Should output "ready" and wait for input
```

## Architecture

```
Real Slay the Spire Game
         ↓ (JSON game state)
    CommunicationMod
         ↓ (stdin/stdout)
    STSLightspeedAgent  
         ↓ (state conversion)
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

This creates a complete pipeline from the real game state to our high-performance AI decision making and back to game control.