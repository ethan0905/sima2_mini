# SIMA-like Agent Example Configurations

This directory contains example configuration files for different use cases.

## Available Configurations

### `advanced_config.json`
A comprehensive configuration for serious learning experiments:
- Epsilon-greedy policy with learning
- Larger grid environment (7x7)
- 50 generations with 8 episodes each
- Learning every 2 generations
- Enhanced reward parameters

Usage:
```bash
python -m src.main --mode train --config examples/advanced_config.json
```

## Creating Custom Configurations

You can create your own configuration files by copying and modifying these examples. The configuration format supports:

- **Environment settings**: Grid size, max steps, reward values
- **Agent configuration**: Policy type, learning parameters, encoder settings
- **Training parameters**: Generation count, learning frequency, evaluation settings
- **Experience management**: Buffer size, storage format, sampling parameters
- **Experiment metadata**: Name, logging, checkpoints

## Configuration Templates

### Quick Testing
```json
{
  "training": {
    "num_generations": 5,
    "episodes_per_generation": 3
  },
  "experiment_name": "quick_test"
}
```

### Learning Experiment
```json
{
  "agent": {
    "policy_type": "epsilon_greedy",
    "epsilon": 0.2,
    "learning_rate": 0.01
  },
  "training": {
    "num_generations": 30,
    "episodes_per_generation": 6
  }
}
```

### Large Scale Training
```json
{
  "environment": {
    "grid_size": 10,
    "max_steps": 150
  },
  "training": {
    "num_generations": 100,
    "episodes_per_generation": 12
  },
  "experience": {
    "max_buffer_size": 1000,
    "learning_sample_size": 200
  }
}
```
