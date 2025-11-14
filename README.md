# SIMA-Like Agent

A modular AI agent that learns to play video games through self-improvement, inspired by Google's SIMA 2. Now with Minecraft support!

## Overview

This project implements the skeleton of a research application where an AI agent learns to play video games through a self-improvement cycle similar to Google's SIMA 2. The system includes:

- **Task Setter**: Proposes tasks with estimated rewards
- **Agent**: Executes tasks in game environments  
- **Reward Model**: Evaluates episode performance
- **Self-Generated Experience**: Stores and manages episode data
- **Self-Improvement Loop**: Orchestrates the learning cycle

This virtuous cycle of iterative improvement paves the way for a future where agents can learn and grow with minimal human intervention, becoming open-ended learners in embodied AI.

**NEW**: The system now supports **Minecraft** as a game environment, enabling research into embodied AI in complex, open-world scenarios.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Task Setter │────│    Agent     │────│ Game Env    │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │
       │            ┌──────▼──────┐
       │            │  Episodes   │
       │            └──────┬──────┘
       │                   │
       ▼            ┌──────▼──────┐
┌─────────────┐     │ Self-Gen.   │     ┌─────────────┐
│Reward Model │◄────│ Experience  │────►│   Storage   │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Installation

### Basic Installation
```bash
pip install -e .
```

### For Development
```bash
pip install -e ".[dev]"
```

### For Minecraft Support (Optional)
For MineRL environment support:
```bash
pip install minerl
```

Note: MineRL has additional dependencies including Java JDK 8. See the [MineRL installation guide](https://minerl.readthedocs.io/en/latest/tutorials/index.html) for detailed setup instructions.

For raw Minecraft control (experimental):
```bash
pip install pyautogui pynput mss pillow
```

## Quick Start

### Train the agent (dummy environment)
```bash
python -m src.main --mode train --generations 10 --episodes-per-gen 5
```

### Train the agent in Minecraft (MineRL)
```bash
python -m src.main --mode train --env minecraft --generations 5 --episodes-per-gen 3
```

### Run a single episode (dummy environment)
```bash
python -m src.main --mode play-once --task-id "reach_goal"
```

### Run a single Minecraft episode
```bash
python -m src.main --mode play-once --env minecraft --task-id "collect_wood"
```

### Test Minecraft environment only
```bash
python -m src.main --mode test-env --env minecraft
```

### Inspect stored experience
```bash
python -m src.main --mode inspect-buffer
```

## Project Structure

```
sima_like_agent/
├── pyproject.toml
├── README.md
├── src/
│   ├── main.py                    # Entry point and CLI
│   ├── config/
│   │   └── config.py             # Configuration classes
│   ├── env/
│   │   ├── base_env.py           # Abstract game environment
│   │   ├── dummy_env.py          # Simple test environment
│   │   ├── minecraft_env.py      # Minecraft environment wrapper ⭐
│   │   ├── io_controller.py      # Game I/O interfaces
│   │   └── vision.py             # Observation encoding
│   ├── agent/
│   │   ├── policy.py             # Policy implementations
│   │   └── agent.py              # Main agent orchestrator
│   ├── tasks/
│   │   ├── task_schema.py        # Task data structures
│   │   └── task_setter.py        # Task generation logic ⭐
│   ├── reward/
│   │   └── reward_model.py       # Episode scoring ⭐
│   ├── experience/
│   │   ├── types.py              # Core data types
│   │   ├── buffer.py             # In-memory experience store
│   │   └── storage.py            # Persistent experience store
│   ├── training/
│   │   └── self_improvement_loop.py  # Main training loop
│   └── utils/
│       ├── logging_utils.py      # Structured logging
│       └── seed.py               # Random seed management
└── tests/
    ├── test_experience.py        # Experience system tests
    └── test_agent_interfaces.py  # Agent integration tests
```

## Key Components

### Environment Interface
The `GameEnv` abstract base class provides a clean interface for any video game. Includes:
- `DummyGameEnv` for testing and development
- `MinecraftEnv` for Minecraft gameplay via MineRL or raw control

### Minecraft Support
The system now supports Minecraft through two modes:
- **MineRL**: Uses the MineRL research platform for structured Minecraft gameplay
- **Raw Control**: Direct keyboard/mouse control of Minecraft client (experimental)

Minecraft-specific features:
- Specialized action space for movement, building, mining, crafting
- Minecraft vision encoder for processing pixel observations
- Task templates for common Minecraft objectives (collect wood, build structures, etc.)
- Reward model understanding Minecraft progress indicators

### Experience System
Episodes are stored both in-memory (`ReplayBuffer`) and persistently (`storage.py`) as the "Self-Generated Experience" that drives learning.

### Task Generation
The `TaskSetter` proposes new tasks based on previous performance, with hooks for LLM-based task generation.

### Reward Learning
The `RewardModel` scores episodes, with clear interfaces for plugging in learned reward functions or LLM-based evaluation. Now includes Minecraft-specific scoring based on inventory changes, health preservation, and exploration.

## Minecraft Configuration

The Minecraft environment can be configured via CLI arguments or config files:

```python
# Example configuration
minecraft_config = MinecraftConfig(
    backend="minerl",  # "minerl" or "raw" 
    environment_name="MineRLNavigateDense-v0",  # MineRL environment
    max_steps=1000,
    frame_skip=1,
    render=True,
    action_space="discrete"  # "discrete" or "continuous"
)
```

### MineRL Environments
Supported MineRL environments include:
- `MineRLNavigateDense-v0`: Navigation with dense rewards
- `MineRLTreechop-v0`: Tree chopping tasks
- `MineRLObtainDiamond-v0`: Complex diamond obtaining task

### Raw Control Mode
For direct Minecraft control:
1. Start Minecraft client
2. Use `--env minecraft --minecraft-backend raw` 
3. Ensure Minecraft window is focused and accessible

**Note**: Raw control mode is experimental and requires additional setup.

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
black src/ tests/
ruff src/ tests/
mypy src/
```

## TODOs for Production Use

### Core System
1. **RL Algorithms**: Implement proper policy gradient/Q-learning in `policy.py`
2. **Scalability**: Add distributed training and experience storage
3. **Vision Models**: Plug in CNN/transformer backbones in `vision.py`

### Environment Integration
1. **More Games**: Add support for other games beyond Minecraft
2. **Real Input Control**: Complete the raw Minecraft controller implementation
3. **MineRL Integration**: Test and optimize MineRL environment performance

### Intelligence Components  
1. **Reward Learning**: Train neural reward models or integrate LLM evaluation
2. **Task Generation**: Add LLM-based creative task generation
3. **Hierarchical Tasks**: Implement task decomposition and sub-goals

### Minecraft-Specific Enhancements
1. **Advanced Actions**: Implement crafting, building, combat mechanics
2. **State Detection**: Add inventory tracking, health monitoring, environment analysis
3. **Long-Term Goals**: Add support for complex, multi-step Minecraft objectives

## License

MIT License - see LICENSE file for details.

