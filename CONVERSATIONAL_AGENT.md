# 🎮 Conversational Minecraft Age### 2. Test the Agent
```bash
# Start the conversational agent (uses gpt-4o-mini by default)
python minecraft_chat.py

# Or with API key directly  
python minecraft_chat.py --openai-key "your-key-here"

# Or specify a different model (o1-mini, gpt-4o, etc.)
python minecraft_chat.py --model o1-mini
```

## 🎯 What This Agent Does

This is a **conversational AI assistant** that can:
- 💬 **Chat with you** in natural language
- 🎮 **Control Minecraft** via keyboard and mouse  
- 🤖 **Execute instructions** like "go mine that tree" or "build a house"
- 🧠 **Understand complex tasks** and break them into steps

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Essential for real Minecraft control
pip install pyautogui pynput mss pillow

# Optional but highly recommended for smart conversations  
pip install openai

# Check what's installed
python minecraft_chat.py --check-deps
```

### 2. Set up OpenAI API Key (Optional but Recommended)
```bash
# Get your API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-api-key-here"

# Or add to your shell profile (~/.bashrc, ~/.zshrc)
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
```

### 3. Test the Agent
```bash
# Start the conversational agent
python minecraft_chat.py

# Or with API key directly
python minecraft_chat.py --openai-key "your-key-here"
```

## 🎮 How to Use

### 1. Prepare Minecraft
- Open Minecraft
- Load any world (Creative mode recommended for testing)
- Make sure Minecraft window is **focused and visible**

### 2. Start the Agent
```bash
python minecraft_chat.py
```

### 3. Chat with Your Assistant!
```
👤 You: Go forward and mine some wood
🤖 Agent: I'll help you mine that tree! Let me move forward and start breaking the wood blocks.

👤 You: Build a small platform here  
🤖 Agent: I'd be happy to build a platform! I'll start by placing blocks in a 3x3 pattern.

👤 You: Look around for animals
🤖 Agent: I'll turn the camera to scan the area for any nearby animals!
```

## 💬 Example Conversations

### Basic Movement and Actions
```
"go forward for 3 seconds"
"turn left and look up"  
"jump over that gap"
"break the block I'm looking at"
"place a block here"
```

### Complex Tasks
```
"mine all the trees in this area"
"build a 5x5 house with a door"
"find and collect some food"
"dig down until you find stone"
"make a path from here to that mountain"
```

### Interactive Tasks
```
"what do you see around us?"
"check my inventory"
"find the best spot to build a base"
"help me farm this area"
"protect me while I build"
```

## ⚙️ Configuration Options

### With OpenAI API Key (Smart Mode)
- **Natural language understanding**: Understands complex instructions
- **Conversational**: Asks clarifying questions, provides updates
- **Task planning**: Breaks down complex tasks into steps
- **Context awareness**: Remembers conversation history
- **Model options**: gpt-4o-mini (default), gpt-4o, o1-mini, etc.

### Without OpenAI API Key (Basic Mode) 
- **Keyword recognition**: Understands simple commands like "go forward", "mine"
- **Direct actions**: Executes basic movements and interactions
- **Fast response**: No API calls, immediate action
- **Privacy**: Everything runs locally
- **Automatic fallback**: If OpenAI fails, switches to basic mode

## 🛠️ Available Commands

### Movement
- `move(direction, duration)`: forward, backward, left, right
- `look(yaw, pitch)`: turn camera/player view
- `jump()`: make player jump

### Interaction  
- `dig()`: break blocks (left-click and hold)
- `place()`: place blocks (right-click)
- `use()`: interact with blocks/items (right-click)
- `attack()`: attack entities (left-click)

### Utility
- `select_slot(1-9)`: choose hotbar slot
- `open_inventory()`: open inventory screen
- `type_chat(message)`: type in game chat

### Meta Commands (in chat)
- `help`: show available commands
- `status`: show current task and agent status
- `quit`: exit the agent

## 🔧 Troubleshooting

### "Control libraries not available"
```bash
pip install pyautogui pynput mss pillow
```

### "Mouse/keyboard not working"
- Ensure Minecraft is the **focused window**
- Try running as administrator (if on Windows)
- Check PyAutoGUI failsafe: move mouse to screen corner to stop

### "OpenAI API error"  
- Check your API key is correct
- Ensure you have OpenAI credits
- Agent will fall back to basic mode if OpenAI fails

### "Actions too fast/slow"
- The agent has built-in delays for realistic gameplay
- You can modify timing in the `conversational_agent.py` file

## 🎯 Advanced Usage

### Custom Actions
Edit `conversational_agent.py` to add new actions:
```python
def _custom_action(self, params):
    # Your custom Minecraft action here
    pass
```

### Screen Analysis
The agent can capture screenshots for visual context:
```python
screenshot = self._capture_screen()
# Process image to understand game state
```

### Task Memory
The agent maintains conversation history and can reference previous tasks:
```
👤 You: Do the same thing you did before
🤖 Agent: I'll repeat the tree mining task from earlier!
```

## 🚀 Example Session

```bash
$ python minecraft_chat.py

🎮 Minecraft Conversational Agent
==================================================
I'm your Minecraft assistant! I can help you play the game.
Just tell me what you'd like me to do in Minecraft!
Type 'quit' to exit, 'help' for commands, 'status' for current task info.
==================================================

💡 Ready! Make sure Minecraft is open and focused.
   Example: 'Go forward and mine some wood'
   Example: 'Look around and find animals'  
   Example: 'Build a 3x3 platform here'

👤 You: go forward and mine the tree in front of me

🤖 Agent: I'll help you mine that tree! Let me move forward and start breaking the wood blocks.

👤 You: now build a small house

🤖 Agent: I'd be happy to help build a house! Could you tell me what materials you'd like me to use? I can see what's in your inventory and start with a simple structure.

👤 You: use the wood we just collected

🤖 Agent: Perfect! I'll use the wood blocks we collected to build a cozy wooden house. Let me start with a 5x5 foundation and build up the walls.

👤 You: quit

🤖 Agent: Goodbye! Happy crafting! 👋
```

## 🎉 You're Ready!

Your conversational Minecraft agent is ready to use! The agent can work in two modes:

1. **Basic Mode** (no API key): Simple command recognition, immediate actions
2. **Smart Mode** (with OpenAI): Natural conversations, complex task understanding

Start with basic commands to test the controls, then try more complex instructions as you get comfortable with the system.

**Have fun building, mining, and exploring with your AI assistant!** 🎮✨
