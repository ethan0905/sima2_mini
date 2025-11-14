"""
Conversational Minecraft Agent

A chat-based agent that can understand natural language instructions
and execute them in a real Minecraft game using keyboard and mouse control.
Integrates computer vision and intelligent task planning following the SIMA architecture.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple
import threading
import queue

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Try to import required dependencies for chat functionality
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import pyautogui
    import pynput
    import mss
    from PIL import Image
    HAS_CONTROL = True
    # Configure pyautogui
    pyautogui.PAUSE = 0.1
    pyautogui.FAILSAFE = True  # Move mouse to corner to stop
except ImportError:
    HAS_CONTROL = False

# Import vision system and task planner
try:
    from vision.minecraft_vision import MinecraftVision, IntelligentTaskPlanner, GameState
    HAS_VISION = True
except ImportError as e:
    try:
        # Try alternative import path
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from vision.minecraft_vision import MinecraftVision, IntelligentTaskPlanner, GameState
        HAS_VISION = True
    except ImportError:
        HAS_VISION = False
        print(f"⚠️  Vision system not available - {e}")
        # Define dummy classes to prevent import errors
        class GameState:
            def __init__(self):
                self.health = 100.0
                self.hunger = 100.0
                self.current_block = "unknown"
                self.nearby_entities = []
                self.time_of_day = "day"
        
        class MinecraftVision:
            def __init__(self):
                self.current_state = GameState()
            def analyze_current_situation(self):
                return self.current_state
            def get_situation_description(self):
                return "Vision system not available"
        
        class IntelligentTaskPlanner:
            def __init__(self, vision_system=None):
                self.vision = vision_system
            def plan_action_sequence(self, user_request, game_state):
                return []


class ConversationalMinecraftAgent:
    """
    A chat-based Minecraft agent following SIMA architecture that can:
    1. Have conversations with the user
    2. Use computer vision to understand game state
    3. Intelligently plan and execute tasks
    4. Control Minecraft via keyboard/mouse
    5. Learn from experience and adapt behavior
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini", auto_focus: bool = True):
        """
        Initialize the conversational agent.
        
        Args:
            openai_api_key: OpenAI API key for chat functionality
            model: OpenAI model to use (gpt-4o-mini, gpt-4o, o1-mini, etc.)
            auto_focus: Whether to automatically focus Minecraft before each command
        """
        self.openai_api_key = openai_api_key
        self.model = model
        self.auto_focus = auto_focus
        self.chat_history: List[Dict[str, str]] = []
        self.current_task: Optional[str] = None
        self.task_status = "idle"  # idle, planning, executing, completed, failed
        
        # Control components
        if HAS_CONTROL:
            self.screen_monitor = mss.mss()
            self.mouse = pynput.mouse.Controller()
            self.keyboard = pynput.keyboard.Controller()
        
        # Vision and intelligence components (SIMA architecture)
        if HAS_VISION:
            try:
                self.vision_system = MinecraftVision()
                self.task_planner = IntelligentTaskPlanner(self.vision_system)
                self.enable_vision = True
                logger.info("Vision system and intelligent task planner initialized")
            except Exception as e:
                logger.error(f"Failed to initialize vision system: {e}")
                self.vision_system = None
                self.task_planner = None
                self.enable_vision = False
                print(f"⚠️  Vision system failed to initialize: {e}")
        else:
            self.vision_system = None
            self.task_planner = None
            self.enable_vision = False
            logger.warning("Vision system not available - using basic mode")
        
        # Experience storage for learning (simplified for now)
        self.experience_database = []
        self.reward_history = []
        
        # Set up OpenAI client if available
        if HAS_OPENAI and openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            self.can_chat = True
            logger.info("Conversational agent initialized with OpenAI support")
        else:
            self.openai_client = None
            self.can_chat = False
            logger.warning("OpenAI not available - using basic command parsing")
        
        # Initialize system prompt
        self.system_prompt = self._create_system_prompt()
        
        # Basic command patterns for fallback mode
        self.basic_commands = {
            "move": ["walk", "go", "move", "run"],
            "look": ["look", "turn", "face"],
            "dig": ["mine", "dig", "break", "destroy"],
            "place": ["place", "put", "build"],
            "jump": ["jump", "hop"],
            "sneak": ["sneak", "crouch"],
            "attack": ["attack", "hit", "fight"],
            "use": ["use", "right click", "interact"],
            "inventory": ["inventory", "inv", "open inventory"],
            "chat": ["say", "chat", "type"]
        }
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the AI assistant with vision awareness."""
        base_prompt = """You are an advanced Minecraft assistant with computer vision capabilities. You can:

1. See and understand the current game state through visual analysis
2. Intelligently plan actions based on what you observe
3. Adapt your behavior to the current situation
4. Control the game through keyboard and mouse

VISION CAPABILITIES:
- Health and hunger bar analysis
- Block and entity detection
- Time of day awareness
- Situational assessment

INTELLIGENT PLANNING:
- You can analyze the current situation visually
- Plan sequences of actions to achieve goals
- Adapt actions based on visual feedback
- Prioritize urgent needs (health, hunger)

Available actions:
- move(direction, duration): Move forward/backward/left/right for a duration
- look(yaw, pitch): Turn the camera/player view
- dig(): Break the block you're looking at
- place(): Place a block from your hotbar
- jump(): Jump once
- sneak(enable): Enable/disable sneaking
- attack(): Attack or break blocks
- use(): Right-click/interact with blocks/items
- select_slot(number): Select hotbar slot 1-9
- open_inventory(): Open inventory
- type_chat(message): Type in chat

ADAPTIVE BEHAVIOR:
When you receive an instruction, I will:
1. Analyze the current visual situation
2. Plan the optimal sequence of actions
3. Execute actions with visual feedback
4. Adapt if the situation changes

Action Keywords (use these in your responses to trigger actions):
- "move forward", "go forward" → moves forward
- "move left", "go left" → moves left  
- "move right", "go right" → moves right
- "move back", "go back" → moves backward
- "look left", "turn left" → turns camera left
- "look right", "turn right" → turns camera right
- "look up" → looks up
- "look down" → looks down
- "mine", "dig", "break" → breaks blocks
- "place", "build" → places blocks
- "jump" → jumps
- "attack", "hit" → attacks
- "use", "interact" → right-clicks

INTELLIGENT RESPONSES:
Based on visual analysis, I will:
- Prioritize urgent needs (low health/hunger)
- Suggest optimal actions for the situation
- Warn about potential dangers
- Adapt plans based on what I observe

Example intelligent responses:
User: "Go get some food"
Analysis: [Health: 85%, Hunger: 25%, Animals nearby]
You: "I can see your hunger is low and there are animals nearby! Let me approach them to get food."

User: "Mine some blocks"
Analysis: [Looking at stone, Health: 30%]  
You: "I notice your health is low - should I find food/shelter first, or proceed with mining the stone I can see?"

Be conversational, intelligent, and adaptive based on visual analysis!"""

        if self.enable_vision:
            return base_prompt
        else:
            # Fallback to basic prompt without vision capabilities
            return """You are a helpful Minecraft assistant that can control the game through keyboard and mouse.

You can understand natural language instructions and translate them into specific Minecraft actions.

Available actions:
- move(direction, duration): Move forward/backward/left/right for a duration
- look(yaw, pitch): Turn the camera/player view
- dig(): Break the block you're looking at
- place(): Place a block from your hotbar
- jump(): Jump once
- sneak(enable): Enable/disable sneaking
- attack(): Attack or break blocks
- use(): Right-click/interact with blocks/items
- select_slot(number): Select hotbar slot 1-9
- open_inventory(): Open inventory
- type_chat(message): Type in chat

Be conversational but include action keywords so the system knows what to do!"""

    def start_conversation(self) -> None:
        """Start the conversational interface."""
        print("\n🎮 SIMA Minecraft Agent - Intelligent & Adaptive")
        print("=" * 60)
        print("I'm your advanced Minecraft assistant with computer vision!")
        print("I can see the game, understand situations, and adapt my actions.")
        print("Just tell me what you'd like me to do in Minecraft!")
        print("Type 'quit' to exit, 'help' for commands, 'status' for current task info.")
        print("=" * 60)
        
        if self.can_chat:
            print(f"🤖 AI Mode: Using {self.model} for enhanced conversations")
        else:
            print("⚠️  Note: OpenAI API not configured - using basic command recognition")
            print("   For full chat: set OPENAI_API_KEY environment variable")
        
        # Vision system status
        if self.enable_vision:
            print("👁️  Computer Vision: ENABLED")
            print("🧠 Intelligent Planning: ENABLED")
            print("📊 Experience Learning: ENABLED")
        else:
            print("⚠️  Computer Vision: DISABLED (install cv2, numpy, mss)")
            print("   For full vision: pip install opencv-python numpy mss")
        
        if not HAS_CONTROL:
            print("⚠️  Note: Control libraries not installed")
            print("   Install with: pip install pyautogui pynput mss pillow")
            return
        
        # Initial Minecraft focus attempt
        print("\n🎯 Attempting to focus Minecraft window...")
        initial_focus_success = self._focus_minecraft_window()
        if initial_focus_success:
            time.sleep(0.5)
            is_focused = self._verify_minecraft_focus()
            if is_focused:
                print("✅ Minecraft window focused and verified!")
            else:
                print("⚠️  Focus attempted but verification unclear")
                print("💡 If commands don't work, manually click Minecraft window")
        else:
            print("⚠️  Could not automatically focus Minecraft")
            print("💡 Please click on the Minecraft window manually before giving commands")
            
        print("\n💡 Ready! Enhanced Examples:")
        print("   'Go find food' (I'll analyze hunger and search intelligently)")
        print("   'Mine whatever looks good' (I'll assess what's visible)")
        print("   'Keep me safe at night' (I'll monitor threats and health)")
        print("   'Build something useful' (I'll plan based on current state)")
        print("   'What do you see?' (I'll describe the current situation)")
        print("   'focus minecraft' (to manually focus)")
        print("   'check focus' (to verify focus)")
        print("\n⚠️  CRITICAL: Set Minecraft 'Pause on Lost Focus: OFF' (Options → Controls)")
        
        # Main conversation loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("🤖 Agent: Goodbye! Happy crafting! 👋")
                    break
                
                if user_input.lower() == "help":
                    self._show_help()
                    continue
                
                if user_input.lower() == "status":
                    self._show_status()
                    continue
                
                # Process the instruction
                response = self._process_instruction(user_input)
                print(f"🤖 Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n🤖 Agent: Interrupted! Goodbye!")
                break
            except Exception as e:
                print(f"🤖 Agent: Sorry, I encountered an error: {e}")
                logger.error(f"Error in conversation: {e}")
    
    def _process_instruction(self, instruction: str) -> str:
        """
        Process a user instruction with vision analysis and intelligent planning.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            Response message
        """
        # Automatically focus Minecraft before processing any instruction
        if (self.auto_focus and HAS_CONTROL and 
            not any(cmd in instruction.lower() for cmd in ["help", "status", "quit", "exit", "focus", "check"])):
            
            # Check if Minecraft is already focused
            is_already_focused = self._verify_minecraft_focus()
            if not is_already_focused:
                print("⚠️  Minecraft is not focused. Attempting to focus it...")
                focus_success = self._focus_minecraft_window()
                
                if focus_success:
                    time.sleep(0.5)
                    is_focused = self._verify_minecraft_focus()
                    if not is_focused:
                        print("❌ Focus attempt failed. Please manually click the Minecraft window now.")
                        print("   Press Enter when Minecraft is focused and ready...")
                        input("   👆 Click Minecraft window, then press Enter: ")
                else:
                    print("❌ Could not auto-focus. Please manually click the Minecraft window now.")
                    print("   Press Enter when Minecraft is focused and ready...")
                    input("   👆 Click Minecraft window, then press Enter: ")
            # If already focused, continue silently
        
        # SIMA ARCHITECTURE: Vision Analysis + Task Setting + Intelligent Planning
        game_state = None
        situation_desc = ""
        planned_actions = []
        
        # Step 1: Vision System - Analyze current game state
        if self.enable_vision and self.vision_system:
            print("👁️  Analyzing visual state...")
            game_state = self.vision_system.analyze_current_situation()
            situation_desc = self.vision_system.get_situation_description()
            
            # Store experience for learning
            experience_entry = {
                "timestamp": time.time(),
                "user_request": instruction,
                "game_state": game_state,
                "situation": situation_desc
            }
            
        # Step 2: Task Setter - Plan intelligent actions based on request + vision
        if self.enable_vision and self.task_planner and game_state:
            print("🧠 Planning intelligent actions...")
            planned_actions = self.task_planner.plan_action_sequence(instruction, game_state)
            
            # Add planned actions to experience
            if planned_actions:
                experience_entry["planned_actions"] = planned_actions
                print(f"   📋 Planned {len(planned_actions)} actions")
        
        # Add to chat history with context
        context_instruction = instruction
        if situation_desc:
            context_instruction = f"{instruction}\n\nCurrent situation: {situation_desc}"
        
        self.chat_history.append({"role": "user", "content": context_instruction})
        
        # Step 3: Agent Response - Generate response with awareness
        if self.can_chat:
            # Use OpenAI with vision context
            try:
                response = self._chat_with_ai(context_instruction)
            except Exception as e:
                logger.warning(f"OpenAI failed, falling back to basic mode: {e}")
                response = f"🔄 OpenAI is having issues, using basic mode instead.\n{self._parse_basic_command(instruction)}"
        else:
            # Use basic pattern matching with vision awareness
            if situation_desc:
                response = f"{situation_desc}\n{self._parse_basic_command(instruction)}"
            else:
                response = self._parse_basic_command(instruction)
        
        # Step 4: Execute planned actions (if any)
        if planned_actions:
            print("⚙️  Executing intelligent action plan...")
            self._execute_planned_actions(planned_actions)
            
        # Step 5: Store experience for learning
        if self.enable_vision and game_state:
            experience_entry["agent_response"] = response
            experience_entry["execution_time"] = time.time()
            self.experience_database.append(experience_entry)
            
            # Keep only recent experiences (simple memory management)
            if len(self.experience_database) > 100:
                self.experience_database = self.experience_database[-50:]
        
        # Add response to chat history
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _chat_with_ai(self, instruction: str) -> str:
        """Use OpenAI to understand instruction and generate response."""
        try:
            # Create messages for OpenAI API
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add recent chat history (last 10 messages)
            recent_history = self.chat_history[-10:]
            messages.extend(recent_history)
            
            # Add current instruction
            messages.append({"role": "user", "content": instruction})
            
            # Get AI response using new OpenAI API format
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract and execute any actions mentioned in the response
            self._execute_actions_from_response(ai_response, instruction)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error with OpenAI chat: {e}")
            # Fall back to basic command parsing on any error
            fallback_response = self._parse_basic_command(instruction)
            return f"OpenAI error, using basic mode: {fallback_response}"
    
    def _parse_basic_command(self, instruction: str) -> str:
        """Parse basic commands without AI."""
        instruction_lower = instruction.lower()
        
        # Check for focus commands (check this FIRST before other commands)
        if "focus" in instruction_lower and "minecraft" in instruction_lower:
            print("🎯 Manually focusing Minecraft...")
            success = self._focus_minecraft_window()
            if success:
                time.sleep(0.5)
                is_focused = self._verify_minecraft_focus()
                if is_focused:
                    return "✅ Successfully focused Minecraft window and verified!"
                else:
                    return "⚠️  Attempted to focus Minecraft, but verification unclear. Please check manually."
            else:
                return "❌ Could not automatically focus Minecraft. Please click on the Minecraft window manually."
        
        # Check for focus verification
        if "check focus" in instruction_lower or "verify focus" in instruction_lower:
            is_focused = self._verify_minecraft_focus()
            if is_focused:
                return "✅ Minecraft is currently focused and ready for commands!"
            else:
                return "❌ Minecraft is not focused. Use 'focus minecraft' or click on the Minecraft window."
        
        # Check for pause menu handling
        if "unpause" in instruction_lower or ("close" in instruction_lower and "menu" in instruction_lower):
            success = self._handle_minecraft_pause()
            if success:
                return "Handled Minecraft pause menu!"
            else:
                return "Could not handle pause menu automatically."
        
        # Check for movement commands
        if any(word in instruction_lower for word in self.basic_commands["move"]):
            if "forward" in instruction_lower:
                self._execute_minecraft_action("move", {"direction": "forward", "duration": 2.0})
                return "Moving forward!"
            elif "back" in instruction_lower:
                self._execute_minecraft_action("move", {"direction": "backward", "duration": 2.0})
                return "Moving backward!"
            elif "left" in instruction_lower:
                self._execute_minecraft_action("move", {"direction": "left", "duration": 2.0})
                return "Moving left!"
            elif "right" in instruction_lower:
                self._execute_minecraft_action("move", {"direction": "right", "duration": 2.0})
                return "Moving right!"
            else:
                # Default forward movement for general "move" commands
                self._execute_minecraft_action("move", {"direction": "forward", "duration": 2.0})
                return "Moving forward!"
        
        # Check for mining/digging
        if any(word in instruction_lower for word in self.basic_commands["dig"]):
            self._execute_minecraft_action("dig", {})
            return "Mining/breaking the block in front of me!"
        
        # Check for jumping
        if any(word in instruction_lower for word in self.basic_commands["jump"]):
            self._execute_minecraft_action("jump", {})
            return "Jumping!"
        
        # Check for looking/turning
        if any(word in instruction_lower for word in self.basic_commands["look"]):
            if "left" in instruction_lower:
                self._execute_minecraft_action("look", {"yaw": -45, "pitch": 0})
                return "Looking left!"
            elif "right" in instruction_lower:
                self._execute_minecraft_action("look", {"yaw": 45, "pitch": 0})
                return "Looking right!"
            elif "up" in instruction_lower:
                self._execute_minecraft_action("look", {"yaw": 0, "pitch": -30})
                return "Looking up!"
            elif "down" in instruction_lower:
                self._execute_minecraft_action("look", {"yaw": 0, "pitch": 30})
                return "Looking down!"
        
        # Check for placing/building
        if any(word in instruction_lower for word in self.basic_commands["place"]):
            self._execute_minecraft_action("place", {})
            return "Placing a block!"
        
        # Check for using/interacting
        if any(word in instruction_lower for word in self.basic_commands["use"]):
            self._execute_minecraft_action("use", {})
            return "Using/interacting with the item in front of me!"
        
        # Check for attacking
        if any(word in instruction_lower for word in self.basic_commands["attack"]):
            self._execute_minecraft_action("attack", {})
            return "Attacking!"
        
        return f"I understand you want me to: {instruction}. I'm not sure how to do that specific action yet. Try commands like 'go forward', 'mine', 'jump', etc."
    
    def _execute_actions_from_response(self, ai_response: str, original_instruction: str) -> None:
        """Extract and execute actions from AI response and original instruction."""
        # Parse both the AI response and original instruction for action keywords
        response_lower = ai_response.lower()
        instruction_lower = original_instruction.lower()
        combined_text = f"{response_lower} {instruction_lower}"
        
        logger.info(f"Parsing actions from: '{original_instruction}'")
        
        # Movement actions
        if any(word in combined_text for word in ["move forward", "go forward", "walk forward"]):
            self._execute_minecraft_action("move", {"direction": "forward", "duration": 2.0})
        elif any(word in combined_text for word in ["move back", "go back", "walk back"]):
            self._execute_minecraft_action("move", {"direction": "backward", "duration": 2.0})
        elif any(word in combined_text for word in ["move left", "go left", "walk left"]):
            self._execute_minecraft_action("move", {"direction": "left", "duration": 2.0})
        elif any(word in combined_text for word in ["move right", "go right", "walk right"]):
            self._execute_minecraft_action("move", {"direction": "right", "duration": 2.0})
        
        # Looking actions
        if any(word in combined_text for word in ["look left", "turn left"]):
            self._execute_minecraft_action("look", {"yaw": -45, "pitch": 0})
        elif any(word in combined_text for word in ["look right", "turn right"]):
            self._execute_minecraft_action("look", {"yaw": 45, "pitch": 0})
        elif any(word in combined_text for word in ["look up"]):
            self._execute_minecraft_action("look", {"yaw": 0, "pitch": -30})
        elif any(word in combined_text for word in ["look down"]):
            self._execute_minecraft_action("look", {"yaw": 0, "pitch": 30})
        
        # Mining/digging actions
        if any(word in combined_text for word in ["mine", "dig", "break", "destroy"]):
            self._execute_minecraft_action("dig", {})
        
        # Jumping
        if "jump" in combined_text:
            self._execute_minecraft_action("jump", {})
        
        # Placing blocks
        if any(word in combined_text for word in ["place", "build", "put"]):
            self._execute_minecraft_action("place", {})
        
        # Using/interacting
        if any(word in combined_text for word in ["use", "interact", "right click"]):
            self._execute_minecraft_action("use", {})
        
        # Attacking
        if any(word in combined_text for word in ["attack", "hit", "fight"]):
            self._execute_minecraft_action("attack", {})
    
    def _execute_minecraft_action(self, action_type: str, params: Dict) -> None:
        """Execute a Minecraft action via keyboard/mouse control."""
        if not HAS_CONTROL:
            print(f"⚠️  Cannot execute {action_type} - control libraries not available")
            logger.warning("Cannot execute action - control libraries not available")
            return
        
        print(f"🎮 Executing {action_type} action with params: {params}")
        logger.info(f"Executing {action_type} action: {params}")
        
        try:
            # Ensure Minecraft is focused
            self._ensure_minecraft_focus()
            
            if action_type == "move":
                self._move(params["direction"], params.get("duration", 1.0))
            elif action_type == "look":
                self._look(params.get("yaw", 0), params.get("pitch", 0))
            elif action_type == "dig":
                self._dig()
            elif action_type == "jump":
                self._jump()
            elif action_type == "place":
                self._place()
            elif action_type == "use":
                self._use()
            elif action_type == "attack":
                self._attack()
            else:
                logger.warning(f"Unknown action type: {action_type}")
                print(f"⚠️  Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            print(f"❌ Error executing action {action_type}: {e}")
    
    def _move(self, direction: str, duration: float) -> None:
        """Move the player in the specified direction."""
        key_map = {
            "forward": "w",
            "backward": "s", 
            "left": "a",
            "right": "d"
        }
        
        key = key_map.get(direction)
        if key:
            # Use pyautogui for key presses for better reliability
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            print(f"   ✅ Moved {direction} for {duration} seconds")
            logger.info(f"Moved {direction} for {duration} seconds")
    
    def _look(self, yaw: float, pitch: float) -> None:
        """Turn the camera/player view."""
        # Convert angles to mouse movement
        # This is approximate and may need calibration
        mouse_sensitivity = 2.0
        dx = int(yaw * mouse_sensitivity)
        dy = int(pitch * mouse_sensitivity)
        
        # Use move instead of moveRel for better compatibility
        current_pos = pyautogui.position()
        new_x = current_pos.x + dx
        new_y = current_pos.y + dy
        pyautogui.moveTo(new_x, new_y)
        print(f"   ✅ Looked yaw={yaw}, pitch={pitch} (moved mouse {dx}, {dy})")
        logger.info(f"Looked yaw={yaw}, pitch={pitch}")
    
    def _dig(self) -> None:
        """Break the block being looked at."""
        pyautogui.mouseDown(button='left')
        time.sleep(0.5)  # Hold to break
        pyautogui.mouseUp(button='left')
        print(f"   ✅ Mining/digging block (held left click for 0.5s)")
        logger.info("Executed dig action")
    
    def _jump(self) -> None:
        """Make the player jump."""
        pyautogui.press('space')
        print(f"   ✅ Jumped!")
        logger.info("Executed jump")
    
    def _place(self) -> None:
        """Place a block from hotbar."""
        pyautogui.click(button='right')
        print(f"   ✅ Placed block (right click)")
        logger.info("Executed place action")
    
    def _use(self) -> None:
        """Right-click/interact."""
        pyautogui.click(button='right')
        print(f"   ✅ Used/interacted (right click)")
        logger.info("Executed use action")
    
    def _attack(self) -> None:
        """Attack/left-click."""
        pyautogui.click(button='left')
        print(f"   ✅ Attacked (left click)")
        logger.info("Executed attack action")
    
    def _capture_screen(self) -> Optional[Image.Image]:
        """Capture the current screen."""
        if not HAS_CONTROL:
            return None
        
        try:
            with self.screen_monitor as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            return None
    
    def _show_help(self) -> None:
        """Show help information."""
        print("\n🤖 Agent: Here's what I can help you with:")
        print("\n🎯 INTELLIGENT ACTIONS (with vision analysis):")
        print("  • Smart Planning: 'find food', 'stay safe', 'gather resources'")
        print("  • Adaptive Tasks: 'mine whatever looks good', 'build something useful'")
        print("  • Situational: 'what do you see?', 'analyze the area', 'what should I do?'")
        
        print("\n⚙️ BASIC MOVEMENT & ACTIONS:")
        print("  • Movement: 'go forward', 'move left', 'walk backward'")
        print("  • Mining: 'dig this block', 'mine the tree', 'break the stone'")
        print("  • Building: 'place a block', 'build a wall', 'make a house'")
        print("  • Looking: 'look left', 'turn right', 'look up', 'look around'")
        print("  • Actions: 'jump', 'attack', 'use this'")
        
        print("\n🎮 SYSTEM COMMANDS:")
        print("  • Focus: 'focus minecraft', 'check focus', 'unpause', 'close menu'")
        print("  • Info: 'help', 'status', 'quit'")
        
        if self.enable_vision:
            print("\n👁️ VISION CAPABILITIES:")
            print("  • Health/Hunger monitoring")
            print("  • Block and entity detection")
            print("  • Situational awareness")
            print("  • Intelligent action planning")
            print("  • Experience-based learning")
        
        print("\n💡 Focus Tips:")
        print("     • If actions don't work, try: 'focus minecraft'")
        print("     • Check if focused with: 'check focus'") 
        print("     • Set Minecraft 'Pause on Lost Focus: OFF' in Options → Controls")
    
    def _show_status(self) -> None:
        """Show current status."""
        print(f"\n🤖 Agent: Current status: {self.task_status}")
        if self.current_task:
            print(f"  Current task: {self.current_task}")
        else:
            print("  No active task")
        print(f"  Chat history: {len(self.chat_history)} messages")
        print(f"  Control available: {HAS_CONTROL}")
        print(f"  AI chat available: {self.can_chat}")
        
        # Vision system status
        if self.enable_vision:
            print(f"  👁️  Vision system: ENABLED")
            print(f"  🧠 Task planner: ENABLED")
            print(f"  📊 Experience entries: {len(self.experience_database)}")
            
            # Show current game state if available
            if self.vision_system and hasattr(self.vision_system, 'current_state'):
                state = self.vision_system.current_state
                print(f"  🎮 Last game analysis:")
                print(f"     Health: {state.health:.1f}%, Hunger: {state.hunger:.1f}%")
                print(f"     Looking at: {state.current_block}")
                print(f"     Entities: {', '.join(state.nearby_entities) if state.nearby_entities else 'None'}")
                print(f"     Time: {state.time_of_day}")
        else:
            print(f"  👁️  Vision system: DISABLED")
    
    def _focus_minecraft_window(self) -> bool:
        """
        Try to focus the Minecraft window using multiple methods.
        
        Returns:
            True if successful, False otherwise
        """
        if not HAS_CONTROL:
            return False
        
        import platform
        system = platform.system()
        
        try:
            # Method 1: Platform-specific application switching
            if system == "Darwin":  # macOS
                try:
                    print("   🍎 Trying macOS Minecraft focus (Java process)...")
                    import subprocess
                    
                    # Use the working AppleScript that targets the Java process
                    apple_script = '''
                    tell application "System Events"
                        if exists process "java" then
                            tell process "java"
                                set frontmost to true
                                try
                                    tell (first window whose name contains "Minecraft") to perform action "AXRaise"
                                end try
                            end tell
                            return "success"
                        else
                            return "java_not_found"
                        end if
                    end tell
                    '''
                    
                    result = subprocess.run(['osascript', '-e', apple_script], 
                                          capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        output = result.stdout.strip()
                        if "success" in output:
                            time.sleep(0.7)  # Give time for app switch
                            print("   ✅ Successfully focused Minecraft (Java process)")
                            return True
                        elif "java_not_found" in output:
                            print("   ⚠️  Java process not found - is Minecraft running?")
                        else:
                            print(f"   ⚠️  Unexpected result: {output}")
                    else:
                        print(f"   ⚠️  AppleScript error: {result.stderr}")
                        
                except Exception as e:
                    print(f"   ⚠️  AppleScript method failed: {e}")
            
            elif system == "Windows":  # Windows
                try:
                    print("   🪟 Trying Windows application switching...")
                    import pygetwindow as gw
                    
                    # Find Minecraft window
                    minecraft_windows = []
                    for title in ["Minecraft", "minecraft"]:
                        minecraft_windows.extend(gw.getWindowsWithTitle(title))
                    
                    if minecraft_windows:
                        window = minecraft_windows[0]
                        window.restore()  # Restore if minimized
                        window.activate()  # Bring to front
                        time.sleep(0.5)
                        print(f"   ✅ Focused Minecraft window: '{window.title[:30]}...'")
                        return True
                    else:
                        print("   ⚠️  No Minecraft window found")
                        
                except Exception as e:
                    print(f"   ⚠️  Windows focus method failed: {e}")
            
            else:  # Linux
                try:
                    print("   🐧 Trying Linux window focus...")
                    import subprocess
                    
                    # Use wmctrl to focus Minecraft window
                    result = subprocess.run(['wmctrl', '-a', 'Minecraft'], 
                                          capture_output=True, timeout=5)
                    
                    if result.returncode == 0:
                        time.sleep(0.5)
                        print("   ✅ Focused Minecraft window using wmctrl")
                        return True
                    else:
                        print("   ⚠️  wmctrl failed or Minecraft not found")
                        
                except Exception as e:
                    print(f"   ⚠️  Linux focus method failed: {e}")
            
            # Method 2: Try pygetwindow as fallback
            try:
                print("   �️  Trying pygetwindow fallback...")
                import pygetwindow as gw
                
                # Get all windows and search for Minecraft
                all_windows = gw.getAllTitles()
                minecraft_window = None
                
                for title in all_windows:
                    if 'minecraft' in title.lower() and len(title) > 1:
                        minecraft_windows = gw.getWindowsWithTitle(title)
                        if minecraft_windows:
                            minecraft_window = minecraft_windows[0]
                            break
                
                if minecraft_window:
                    minecraft_window.restore()
                    minecraft_window.activate()
                    time.sleep(0.7)  # Longer wait for focus
                    print(f"   ✅ Focused window: '{minecraft_window.title[:30]}...'")
                    return True
                else:
                    print("   ⚠️  No Minecraft window found via pygetwindow")
                    
            except Exception as e:
                print(f"   ⚠️  pygetwindow fallback failed: {e}")
            
            # Method 3: Alt+Tab/Cmd+Tab as last resort
            try:
                print("   ⌨️  Trying keyboard shortcut as last resort...")
                if system == "Darwin":
                    # On macOS, use Cmd+Tab
                    pyautogui.hotkey('cmd', 'tab')
                    time.sleep(0.5)
                    # Try once more to cycle if needed
                    pyautogui.hotkey('cmd', 'tab')
                else:
                    # On Windows/Linux, use Alt+Tab
                    pyautogui.hotkey('alt', 'tab')
                    time.sleep(0.5)
                    pyautogui.hotkey('alt', 'tab')
                    
                time.sleep(0.7)
                print("   ⚠️  Used keyboard shortcuts (may need manual adjustment)")
                return True  # Assume it worked since we tried
                
            except Exception as e:
                print(f"   ⚠️  Keyboard shortcut failed: {e}")
            
            print("   ❌ All focus methods failed")
            return False
            
        except Exception as e:
            print(f"   ❌ Focus error: {e}")
            logger.error(f"Error focusing Minecraft window: {e}")
            return False
    
    def _ensure_minecraft_focus(self) -> None:
        """Ensure Minecraft is focused before executing actions (silent version)."""
        # This is a silent version used during action execution
        # The verbose focusing is done at the instruction level
        try:
            # Quick focus attempt without verbose output
            if HAS_CONTROL:
                # Just try the quickest method (screen center click)
                import mss
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    center_x = monitor['width'] // 2
                    center_y = monitor['height'] // 2
                    pyautogui.click(center_x, center_y)
                    time.sleep(0.1)  # Brief pause
        except Exception as e:
            logger.debug(f"Silent focus attempt failed: {e}")
            # Don't print errors here to avoid spam
    
    def _handle_minecraft_pause(self) -> bool:
        """
        Handle Minecraft pause menu if it's open.
        
        Returns:
            True if handled/no pause detected, False if couldn't handle
        """
        if not HAS_CONTROL:
            return False
        
        try:
            # Method 1: Try to detect pause menu and close it
            # Press Escape twice to ensure we're not in pause menu
            # First Escape: If in pause menu, closes it. If not, opens it.
            # Second Escape: If first one opened it, this closes it. If first closed it, does nothing.
            
            logger.debug("Checking for Minecraft pause menu...")
            
            # Give Minecraft focus first
            self._focus_minecraft_window()
            time.sleep(0.1)
            
            # Double-tap escape to ensure we're not in pause menu
            pyautogui.press('escape')
            time.sleep(0.1)
            pyautogui.press('escape')  
            time.sleep(0.1)
            
            logger.debug("Handled potential Minecraft pause menu")
            return True
            
        except Exception as e:
            logger.error(f"Error handling pause menu: {e}")
            return False
    
    def _detect_minecraft_pause_menu(self) -> bool:
        """
        Try to detect if Minecraft pause menu is open using screen capture.
        
        Returns:
            True if pause menu is detected, False otherwise
        """
        if not HAS_CONTROL:
            return False
            
        try:
            # Capture screen and look for pause menu indicators
            screenshot = self._capture_screen()
            if screenshot is None:
                return False
            
            # Convert to grayscale for easier text detection
            import numpy as np
            screenshot_gray = screenshot.convert('L')
            screenshot_array = np.array(screenshot_gray)
            
            # Look for common pause menu text patterns
            # This is a simple approach - could be enhanced with OCR
            
            # For now, return False and rely on the escape key method
            return False
            
        except Exception as e:
            logger.debug(f"Could not detect pause menu: {e}")
            return False
    
    def process_instruction(self, instruction: str) -> str:
        """
        Public method to process an instruction (for testing/API use).
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            Response message
        """
        return self._process_instruction(instruction)
    
    def _verify_minecraft_focus(self) -> bool:
        """
        Verify if Minecraft is currently the focused application.
        
        Returns:
            True if Minecraft appears to be focused, False otherwise
        """
        if not HAS_CONTROL:
            return False
        
        import platform
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                import subprocess
                
                # Check if Java process (Minecraft) is frontmost
                apple_script = '''
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    if frontApp is "java" then
                        return "minecraft_focused"
                    else
                        return frontApp
                    end if
                end tell
                '''
                
                result = subprocess.run(['osascript', '-e', apple_script], 
                                      capture_output=True, text=True, timeout=3)
                
                if result.returncode == 0:
                    active_app = result.stdout.strip().lower()
                    return 'minecraft_focused' in active_app or 'java' in active_app
                    
            elif system == "Windows":
                try:
                    import pygetwindow as gw
                    active_window = gw.getActiveWindow()
                    if active_window:
                        return 'minecraft' in active_window.title.lower()
                except:
                    pass
                    
            return False  # Default to False if we can't determine
            
        except Exception as e:
            logger.debug(f"Could not verify Minecraft focus: {e}")
            return False
    
    def _ensure_minecraft_focus(self) -> None:
        """Ensure Minecraft is focused before executing actions (silent version)."""
        # This is a silent version used during action execution
        # The verbose focusing is done at the instruction level
        try:
            # Quick focus attempt without verbose output
            if HAS_CONTROL:
                # Just try the quickest method (screen center click)
                import mss
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    center_x = monitor['width'] // 2
                    center_y = monitor['height'] // 2
                    pyautogui.click(center_x, center_y)
                    time.sleep(0.1)  # Brief pause
        except Exception as e:
            logger.debug(f"Silent focus attempt failed: {e}")
            # Don't print errors here to avoid spam
    
    def _handle_minecraft_pause(self) -> bool:
        """
        Handle Minecraft pause menu if it's open.
        
        Returns:
            True if handled/no pause detected, False if couldn't handle
        """
        if not HAS_CONTROL:
            return False
        
        try:
            # Method 1: Try to detect pause menu and close it
            # Press Escape twice to ensure we're not in pause menu
            # First Escape: If in pause menu, closes it. If not, opens it.
            # Second Escape: If first one opened it, this closes it. If first closed it, does nothing.
            
            logger.debug("Checking for Minecraft pause menu...")
            
            # Give Minecraft focus first
            self._focus_minecraft_window()
            time.sleep(0.1)
            
            # Double-tap escape to ensure we're not in pause menu
            pyautogui.press('escape')
            time.sleep(0.1)
            pyautogui.press('escape')  
            time.sleep(0.1)
            
            logger.debug("Handled potential Minecraft pause menu")
            return True
            
        except Exception as e:
            logger.error(f"Error handling pause menu: {e}")
            return False
    
    def _detect_minecraft_pause_menu(self) -> bool:
        """
        Try to detect if Minecraft pause menu is open using screen capture.
        
        Returns:
            True if pause menu is detected, False otherwise
        """
        if not HAS_CONTROL:
            return False
            
        try:
            # Capture screen and look for pause menu indicators
            screenshot = self._capture_screen()
            if screenshot is None:
                return False
            
            # Convert to grayscale for easier text detection
            import numpy as np
            screenshot_gray = screenshot.convert('L')
            screenshot_array = np.array(screenshot_gray)
            
            # Look for common pause menu text patterns
            # This is a simple approach - could be enhanced with OCR
            
            # For now, return False and rely on the escape key method
            return False
            
        except Exception as e:
            logger.debug(f"Could not detect pause menu: {e}")
            return False
    
    def process_instruction(self, instruction: str) -> str:
        """
        Public method to process an instruction (for testing/API use).
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            Response message
        """
        return self._process_instruction(instruction)
    
    def _execute_planned_actions(self, planned_actions: List[Dict]) -> None:
        """
        Execute a sequence of planned actions from the intelligent task planner.
        
        Args:
            planned_actions: List of action dictionaries from task planner
        """
        if not planned_actions:
            return
            
        try:
            for i, action in enumerate(planned_actions):
                action_type = action.get("type", "")
                
                print(f"   {i+1}/{len(planned_actions)}: {action_type}")
                
                if action_type == "message":
                    # Display informational message
                    print(f"   💬 {action.get('text', '')}")
                
                elif action_type == "move":
                    direction = action.get("direction", "forward")
                    if direction == "toward_animal":
                        # Look for animals and move toward them
                        self._execute_minecraft_action("look", {"direction": "around"})
                        time.sleep(0.5)
                        self._execute_minecraft_action("move", {"direction": "forward", "duration": 1.5})
                    elif direction == "explore":
                        # Exploration pattern
                        self._execute_minecraft_action("look", {"yaw": 90, "pitch": 0})
                        time.sleep(0.3)
                        self._execute_minecraft_action("move", {"direction": "forward", "duration": 2.0})
                    else:
                        # Standard movement
                        self._execute_minecraft_action("move", {"direction": direction, "duration": 1.5})
                
                elif action_type == "look":
                    direction = action.get("direction", "around")
                    if direction == "around":
                        # Look around pattern
                        for angle in [90, 180, 270, 360]:
                            self._execute_minecraft_action("look", {"yaw": angle, "pitch": 0})
                            time.sleep(0.3)
                    else:
                        self._execute_minecraft_action("look", {"direction": direction})
                
                elif action_type == "mine":
                    target = action.get("target", "any")
                    if target == "any":
                        self._execute_minecraft_action("dig", {})
                    else:
                        print(f"   ⛏️ Mining {target}...")
                        self._execute_minecraft_action("dig", {})
                
                elif action_type == "place":
                    self._execute_minecraft_action("place", {})
                
                elif action_type == "build":
                    structure = action.get("structure", "block")
                    if structure == "house":
                        print("   🏠 Building simple house foundation...")
                        # Simple house foundation - 4x4 platform
                        for x in range(4):
                            for z in range(4):
                                if x == 0 or x == 3 or z == 0 or z == 3:  # Only edges
                                    self._execute_minecraft_action("place", {})
                                    time.sleep(0.2)
                                    self._execute_minecraft_action("move", {"direction": "right", "duration": 0.5})
                            self._execute_minecraft_action("move", {"direction": "forward", "duration": 0.5})
                    elif structure == "wall":
                        print("   🧱 Building wall...")
                        # Simple wall - 5 blocks in a line
                        for _ in range(5):
                            self._execute_minecraft_action("place", {})
                            time.sleep(0.2)
                            self._execute_minecraft_action("move", {"direction": "right", "duration": 0.5})
                
                elif action_type == "search":
                    target = action.get("target", "")
                    print(f"   🔍 Searching for {target}...")
                    # Exploration movement pattern
                    self._execute_minecraft_action("look", {"direction": "around"})
                    time.sleep(0.5)
                    self._execute_minecraft_action("move", {"direction": "forward", "duration": 2.0})
                
                else:
                    print(f"   ⚠️ Unknown action type: {action_type}")
                
                # Small delay between actions
                time.sleep(0.5)
                
        except Exception as e:
            print(f"   ❌ Error executing planned action: {e}")
            logger.error(f"Error in planned action execution: {e}")
    
def start_conversational_agent(openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini", auto_focus: bool = True) -> None:
    """
    Start the conversational Minecraft agent.
    
    Args:
        openai_api_key: Optional OpenAI API key for enhanced chat
        model: OpenAI model to use
        auto_focus: Whether to automatically focus Minecraft before each command
    """
    agent = ConversationalMinecraftAgent(openai_api_key, model, auto_focus)
    agent.start_conversation()


if __name__ == "__main__":
    # You can run this directly for testing
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    start_conversational_agent(api_key)
