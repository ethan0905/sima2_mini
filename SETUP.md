# SIMA Minecraft Agent - Setup & Testing Guide

## 🚀 Quick Status Check
✅ **Agent Architecture**: Ready and working  
✅ **Dummy Environment**: Functional for testing  
✅ **Minecraft Environment**: Implemented with fallback  
❗ **MineRL**: Not installed (optional)  
❗ **OpenAI API**: Not required for basic functionality  

## 📋 Prerequisites

### Required Dependencies (Already Installed)
- Python 3.11+
- NumPy, PyTorch, Loguru
- Core SIMA architecture

### Optional Dependencies

#### For MineRL Support (Recommended)
```bash
# Install MineRL for proper Minecraft integration
pip install minerl

# Note: MineRL requires Java JDK 8
# macOS: brew install openjdk@8
# Ubuntu: sudo apt-get install openjdk-8-jdk
```

#### For Raw Minecraft Control (Experimental)
```bash
# For screen capture and keyboard/mouse automation
pip install pyautogui pynput mss pillow
```

## 🎮 Testing the Agent

### 1. Basic Functionality Test
```bash
# Test with dummy environment (always works)
python -m src.main --mode play-once --env dummy

# Expected output: Agent completes episode with basic game mechanics
```

### 2. Minecraft Environment Test (Without MineRL)
```bash
# Test Minecraft environment with fallback mode
python -m src.main --mode play-once --env minecraft

# Expected: Runs with warnings about raw control being placeholder
```

### 3. Training Test
```bash
# Train for a few generations with dummy environment
python -m src.main --mode train --env dummy --generations 2 --episodes-per-gen 3

# Monitor: src/experiments/sima_experiment/ for results
```

## 🎯 Running Different Modes

### Single Episode Testing
```bash
# Dummy environment - always works
python -m src.main --mode play-once --env dummy --task-id "reach_goal"

# Minecraft environment - needs MineRL for full functionality
python -m src.main --mode play-once --env minecraft --task-id "collect_wood"
```

### Training Mode
```bash
# Quick training session
python -m src.main --mode train --env dummy --generations 5 --episodes-per-gen 3

# Minecraft training (requires MineRL for proper functionality)
python -m src.main --mode train --env minecraft --generations 3 --episodes-per-gen 2
```

### Inspect Experience Data
```bash
# View stored episodes and metrics
python -m src.main --mode inspect-buffer

# View training metrics
python -m src.main --mode view-metrics
```

## ⚙️ Configuration

### No OpenAI API Key Required
The agent uses built-in components:
- **Random Policy**: For basic exploration
- **Simple Reward Model**: For episode evaluation  
- **Dummy Vision Encoder**: For observation processing

### Custom Configuration
Create a custom config file:
```json
{
  "environment": {
    "env_type": "minecraft",
    "max_steps": 1000
  },
  "minecraft": {
    "use_minerl": true,
    "env_id": "MineRLTreechop-v0",
    "max_episode_steps": 500
  },
  "agent": {
    "policy_type": "random",
    "encoder_type": "dummy"
  }
}
```

Use with:
```bash
python -m src.main --config custom_config.json --mode train
```

## 🔧 Current Implementation Status

### ✅ Working Components
- **Agent Architecture**: Complete self-improvement loop
- **Dummy Environment**: Fully functional 5x5 grid world
- **Task Generation**: Automatic task proposal and execution
- **Reward System**: Simple but effective reward modeling
- **Experience Storage**: Episode recording and replay
- **CLI Interface**: Full command-line interface

### ⚠️ Limited Components  
- **Minecraft Raw Control**: Placeholder implementation
- **Vision Processing**: Basic dummy encoder
- **Action Space**: Currently using discrete actions from dummy env

### 🔄 With MineRL Installation
- **Full Minecraft Integration**: Proper environment simulation
- **Complex Action Space**: Continuous camera, keyboard, mouse
- **Rich Observations**: RGB video frames, inventory, health
- **Realistic Tasks**: Tree chopping, mining, navigation

## 🐛 Common Issues & Solutions

### MineRL Installation Issues
```bash
# If MineRL fails to install
pip install --upgrade pip setuptools wheel
pip install minerl

# For Java issues on macOS
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
```

### Action Space Warnings
The warnings about "Unknown action type: discrete" are expected when using Minecraft environment without MineRL. The agent falls back to a placeholder implementation.

### Performance
- **Dummy Environment**: Very fast, good for testing architecture
- **Minecraft (Raw)**: Currently just simulation, no real interaction
- **Minecraft (MineRL)**: Slower but realistic, requires proper installation

## 🎯 Next Steps

1. **Install MineRL** for full Minecraft functionality:
   ```bash
   pip install minerl
   ```

2. **Test with MineRL**:
   ```bash
   python -m src.main --mode play-once --env minecraft
   ```

3. **Train the agent**:
   ```bash
   python -m src.main --mode train --env minecraft --generations 10
   ```

4. **Monitor progress**:
   ```bash
   python -m src.main --mode view-metrics
   ```

## 🎉 Ready to Use!

The agent is **ready for testing and development** right now with the dummy environment. For full Minecraft integration, install MineRL as described above.

**No OpenAI API keys or external services required** - everything runs locally!
