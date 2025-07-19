G# sts_lightspeed

High-performance C++20 implementation of Slay the Spire designed for tree search algorithms and machine learning training.

**Features**
* **Performance**: 1M random playouts in 5 seconds with 16 threads
* **ML Optimized**: Focused on training data generation and neural network integration
* **Complete Implementation**: All Ironclad cards, colorless cards, enemies, relics, and game mechanics
* **Python Integration**: PyBind11 bindings for seamless ML workflows
* **Advanced RL**: PPO and PPG (Phasic Policy Gradient) with PPG Reloaded enhancements
* **Playable Console**: Interactive game simulation
* **Save File Support**: Load and convert game states to/from JSON
* **Multi-threaded**: Parallel simulation and training support

## Quick Start

### Building
```bash
# Standard build
cmake .
make -j8

# Python bindings (install dependencies first)
pyenv shell 3.10.14 && pip install pyarrow tqdm
```

### Training Neural Networks
```bash
# PPO training
pyenv shell 3.10.14 && python rl_train.py --algorithm ppo --separate-networks true

# PPG training (recommended)
pyenv shell 3.10.14 && python rl_train.py --algorithm ppg --separate-networks true
```

### Interactive Play
```bash
./main
# Input: seed character(I/S/D/W) ascensionLevel
# Example: 12345 I 0
```

### Testing & Simulation
```bash
# Multi-threaded agent testing
./test agent_mt [threads] [depth] [ascension] [seed] [playouts] [print_level]

# Convert save files to JSON
./test json [save_file_path] [json_output_path]
```

## Machine Learning Components

* **`rl_train.py`**: Unified PPO/PPG trainer with PPG Reloaded algorithm
* **`network.py`**: Transformer-based neural network architecture
* **`playouts.py`**: High-performance data generation with neural network guidance  
* **`train.py`**: Supervised learning pipeline for choice prediction

## Architecture

* **C++20 Core**: High-performance game simulation engine
* **Python Bindings**: PyBind11 integration for ML workflows
* **Fixed-size Containers**: Optimized for speed and memory efficiency
* **Multi-threading**: Parallel game execution and neural network inference

## PPG Reloaded Features

* **Adaptive KL Regularization**: Dynamic policy regularization based on drift measurement
* **Trajectory Sampling**: Complete rollout sampling for auxiliary phase data diversity
* **Behavioral Cloning**: Policy preservation during feature distillation
* **Computational Efficiency**: Configurable auxiliary phase frequency for optimal performance

See `CLAUDE.md` for detailed development guidelines and implementation details.
