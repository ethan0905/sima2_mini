# SIMA Agent Architecture - Complete Implementation

This document describes the complete SIMA (Scalable Instructable Multiworld Agent) architecture implementation for intelligent Minecraft control.

## 🏗️ Architecture Overview

The SIMA agent follows the architecture shown in your diagram with these key components:

### 1. **Agent** (Central Coordinator)
- `ConversationalMinecraftAgent` class
- Manages conversation flow and high-level decision making
- Coordinates between all other components
- Handles user interaction and command processing

### 2. **Vision System** (Computer Vision)
- `MinecraftVision` class in `src/vision/minecraft_vision.py`
- Real-time screen capture and analysis
- Health/hunger bar detection
- Block and entity recognition
- Situational awareness and state tracking
- Natural language situation descriptions

### 3. **Task Setter** (Intelligent Planning)
- `IntelligentTaskPlanner` class
- Plans action sequences based on user requests + visual analysis
- Adapts behavior to current game state
- Prioritizes urgent needs (health, hunger, safety)
- Context-aware task decomposition

### 4. **Reward Model** (Learning)
- Experience database storage
- Action outcome tracking
- Success/failure pattern recognition
- Adaptive behavior improvement

### 5. **Self-Generated Experience** (Memory)
- Experience storage with timestamps
- Game state snapshots
- Action sequence records
- Learning from past interactions

## 🔄 Complete Control Flow

```
User Request → Vision Analysis → Task Planning → Agent Response → Action Execution → Experience Storage
     ↑                                                                                        ↓
     └─────────────────── Learning & Adaptation ←──────────────────────────────────────────┘
```

### Step-by-Step Process:

1. **User Input**: Natural language request (e.g., "find food")

2. **Vision Analysis**: 
   - Capture current screen
   - Analyze health, hunger, blocks, entities
   - Generate situation description

3. **Intelligent Planning**:
   - Combine user request with visual state
   - Plan optimal action sequence
   - Consider priorities and safety

4. **Agent Response**:
   - Generate conversational response
   - Explain planned actions
   - Provide context and reasoning

5. **Action Execution**:
   - Execute planned action sequence
   - Monitor progress and adapt
   - Handle unexpected situations

6. **Experience Storage**:
   - Record complete interaction
   - Store outcomes and results
   - Enable learning and improvement

## 🎮 Enhanced Capabilities

### Vision-Guided Intelligence
- **Health Monitoring**: "Your health is low - I'll find food first"
- **Threat Detection**: "I see mobs nearby - let's stay safe"
- **Resource Recognition**: "I can see wood blocks - perfect for building"
- **Time Awareness**: "It's getting dark - we should find shelter"

### Adaptive Task Planning
- **Context-Aware**: Plans change based on what agent sees
- **Priority-Based**: Urgent needs (health/hunger) take precedence
- **Multi-Step**: Complex tasks broken into intelligent sequences
- **Adaptive**: Plans adjust if situation changes

### Natural Conversation
- **Situational**: "I can see you're hungry and there are animals nearby"
- **Explanatory**: "Let me approach them carefully to get food"
- **Adaptive**: "Your health dropped - should I pause mining to find food?"

## 🚀 Usage Examples

### Basic Usage
```bash
python minecraft_chat.py --auto-focus
```

### Enhanced Commands
```
👤 "Find food" 
🤖 Analyzes: Health 85%, Hunger 25%, Animals nearby
    Plans: Approach animals, careful movement, food acquisition
    Executes: Look around → Move toward animals → Interact

👤 "What do you see?"
🤖 "I can see: health is moderate, very hungry, looking at grass, animals nearby, it's daytime"

👤 "Build something useful"
🤖 Analyzes current resources and situation
    Plans: Assess materials, plan structure, execute construction
    Adapts: "I see you have wood blocks - let me build a shelter platform"
```

## 💡 Key Features

### 1. **Robust Window Focus** (Already Implemented)
- Automatic Minecraft window targeting
- macOS AppleScript integration  
- Verification and fallback handling
- Manual focus commands

### 2. **Computer Vision** (New)
- Real-time screen analysis
- Health/hunger monitoring
- Block/entity detection
- Situational awareness

### 3. **Intelligent Planning** (New)
- Context-aware action sequences
- Priority-based decision making
- Multi-step task decomposition
- Adaptive execution

### 4. **Experience Learning** (New)
- Interaction history storage
- Pattern recognition
- Adaptive behavior improvement
- Success/failure tracking

## 🛠️ Technical Implementation

### Files Modified/Created:
- ✅ `src/agent/conversational_agent.py` - Enhanced with vision integration
- ✅ `src/vision/minecraft_vision.py` - New computer vision system
- ✅ `requirements.txt` - Updated with vision dependencies
- ✅ `minecraft_chat.py` - Enhanced startup and control

### Dependencies Added:
- `opencv-python` - Computer vision processing
- `numpy` - Array operations and image analysis
- `mss` - Fast screen capture
- `Pillow` - Image processing utilities

### Architecture Benefits:
1. **Intelligent**: Makes decisions based on visual analysis
2. **Adaptive**: Behavior changes with game state
3. **Conversational**: Natural language interaction
4. **Learning**: Improves through experience
5. **Robust**: Reliable window focus and control

## 🎯 Next Steps

### Immediate Enhancements:
- [ ] Enhanced object detection models
- [ ] More sophisticated task planning
- [ ] Better learning algorithms
- [ ] Inventory analysis
- [ ] Advanced building patterns

### Advanced Features:
- [ ] Multi-modal learning
- [ ] Collaborative gameplay
- [ ] Complex goal planning
- [ ] Performance optimization

## 🔧 Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Minecraft**:
   - Set "Pause on Lost Focus: OFF" (Options → Controls)
   - Ensure game is running and visible

3. **Set OpenAI API Key** (optional for AI chat):
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

4. **Run Enhanced Agent**:
   ```bash
   python minecraft_chat.py --auto-focus
   ```

The agent now fully implements the SIMA architecture with intelligent, adaptive control of Minecraft through computer vision, natural language processing, and autonomous action selection!
