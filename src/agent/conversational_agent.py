"""
Conversational Minecraft Agent

A chat-based agent that can understand natural language instructions
and execute them in a real Minecraft game using keyboard and mouse control.
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


class ConversationalMinecraftAgent:
    """
    A chat-based Minecraft agent that can:
    1. Have conversations with the user
    2. Understand natural language instructions
    3. Control Minecraft via keyboard/mouse
    4. Execute tasks autonomously
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the conversational agent.
        
        Args:
            openai_api_key: OpenAI API key for chat functionality
            model: OpenAI model to use (gpt-4o-mini, gpt-4o, o1-mini, etc.)
        """
        self.openai_api_key = openai_api_key
        self.model = model
        self.chat_history: List[Dict[str, str]] = []
        self.current_task: Optional[str] = None
        self.task_status = "idle"  # idle, planning, executing, completed, failed
        
        # Control components
        if HAS_CONTROL:
            self.screen_monitor = mss.mss()
            self.mouse = pynput.mouse.Controller()
            self.keyboard = pynput.keyboard.Controller()
        
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
        """Create the system prompt for the AI assistant."""
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

IMPORTANT: When you receive an instruction, I will automatically execute the corresponding Minecraft actions based on keywords in your response and the user's request. You should:

1. Acknowledge what the user wants to do
2. Briefly explain what action you're taking
3. Use action keywords in your response so the system can execute them

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

Example responses:
User: "Go forward and mine the tree"
You: "I'll help you get that tree! Let me move forward to reach it and then mine the wood blocks."

User: "Turn around and build a wall"
You: "I'll turn around by looking right and then place blocks to build a wall for you."

Be conversational but include the action keywords so the system knows what to do!"""

    def start_conversation(self) -> None:
        """Start the conversational interface."""
        print("\n🎮 Minecraft Conversational Agent")
        print("=" * 50)
        print("I'm your Minecraft assistant! I can help you play the game.")
        print("Just tell me what you'd like me to do in Minecraft!")
        print("Type 'quit' to exit, 'help' for commands, 'status' for current task info.")
        print("=" * 50)
        
        if self.can_chat:
            print(f"🤖 AI Mode: Using {self.model} for enhanced conversations")
        else:
            print("⚠️  Note: OpenAI API not configured - using basic command recognition")
            print("   For full chat: set OPENAI_API_KEY environment variable")
        
        if not HAS_CONTROL:
            print("⚠️  Note: Control libraries not installed")
            print("   Install with: pip install pyautogui pynput mss pillow")
            return
        
        print("\n💡 Ready! Make sure Minecraft is open and focused.")
        print("   Example: 'Go forward and mine some wood'")
        print("   Example: 'Look around and find animals'")
        print("   Example: 'Build a 3x3 platform here'")
        print("\n⚠️  FOCUS TIP: If Minecraft pauses when you click terminal:")
        print("   1. Set Minecraft to Windowed mode (not Fullscreen)")
        print("   2. In Options → Controls → Set 'Pause on Lost Focus: OFF'")
        print("   3. Or say 'focus minecraft' to refocus the game window")
        
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
        Process a user instruction and execute it.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            Response message
        """
        # Add to chat history
        self.chat_history.append({"role": "user", "content": instruction})
        
        if self.can_chat:
            # Use OpenAI to understand and plan
            try:
                response = self._chat_with_ai(instruction)
            except Exception as e:
                logger.warning(f"OpenAI failed, falling back to basic mode: {e}")
                response = f"🔄 OpenAI is having issues, using basic mode instead.\n{self._parse_basic_command(instruction)}"
        else:
            # Use basic pattern matching
            response = self._parse_basic_command(instruction)
        
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
            success = self._focus_minecraft_window()
            if success:
                return "Successfully focused Minecraft window!"
            else:
                return "Could not automatically focus Minecraft. Please click on the Minecraft window manually."
        
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
        
        pyautogui.move(dx, dy, relative=True)
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
        print("  • Movement: 'go forward', 'move left', 'walk backward'")
        print("  • Mining: 'dig this block', 'mine the tree', 'break the stone'")
        print("  • Building: 'place a block', 'build a wall'")
        print("  • Looking: 'look left', 'turn right', 'look up'")
        print("  • Actions: 'jump', 'attack', 'use this'")
        print("  • Complex: 'build a house', 'find diamonds', 'make a farm'")
        print("  • Focus: 'focus minecraft', 'unpause', 'close menu'")
        print("\n  Commands: 'help', 'status', 'quit'")
        print("\n💡 Tip: Set Minecraft 'Pause on Lost Focus: OFF' in Options → Controls")
    
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
    
    def _focus_minecraft_window(self) -> bool:
        """
        Try to focus the Minecraft window.
        
        Returns:
            True if successful, False otherwise
        """
        if not HAS_CONTROL:
            return False
        
        try:
            # Method 1: Try using pygetwindow with correct API
            try:
                import pygetwindow as gw
                
                # Get all windows and filter for Minecraft
                all_windows = gw.getAllWindows()
                minecraft_window = None
                
                for window in all_windows:
                    window_title = window.title.lower()
                    if ('minecraft' in window_title and 
                        len(window_title) > 1 and  # Avoid empty titles
                        window.visible):
                        minecraft_window = window
                        break
                
                if minecraft_window:
                    minecraft_window.activate()
                    time.sleep(0.3)  # Give window time to focus
                    logger.info(f"Successfully focused Minecraft window: {minecraft_window.title}")
                    return True
                    
            except Exception as e:
                logger.debug(f"pygetwindow method failed: {e}")
            
            # Method 2: Fallback - Click on center of screen (assumes Minecraft is visible)
            try:
                # Get screen size
                import mss
                with mss.mss() as sct:
                    monitor = sct.monitors[1]  # Primary monitor
                    screen_width = monitor['width']
                    screen_height = monitor['height']
                
                # Click on center of screen (where Minecraft likely is)
                center_x = screen_width // 2
                center_y = screen_height // 2
                
                pyautogui.click(center_x, center_y)
                time.sleep(0.2)
                logger.info("Clicked center of screen to focus Minecraft")
                return True
                
            except Exception as e:
                logger.debug(f"Screen click method failed: {e}")
            
            # Method 3: Use Alt+Tab to cycle to Minecraft (macOS: Cmd+Tab)
            try:
                import platform
                if platform.system() == "Darwin":  # macOS
                    pyautogui.hotkey('cmd', 'tab')
                else:  # Windows/Linux
                    pyautogui.hotkey('alt', 'tab')
                    
                time.sleep(0.5)
                logger.info("Used keyboard shortcut to switch windows")
                return True
                
            except Exception as e:
                logger.debug(f"Keyboard shortcut method failed: {e}")
                
            logger.warning("All focus methods failed - manual focus required")
            return False
            
        except Exception as e:
            logger.error(f"Error focusing Minecraft window: {e}")
            return False
    
    def _ensure_minecraft_focus(self) -> None:
        """Ensure Minecraft is focused before executing actions."""
        # Try to focus Minecraft window
        focus_success = self._focus_minecraft_window()
        
        if not focus_success:
            # If we can't focus automatically, provide helpful instructions
            print("⚠️  Could not automatically focus Minecraft window.")
            print("   Manual steps:")
            print("   1. Click on the Minecraft window to focus it")
            print("   2. Or set Minecraft: Options → Controls → 'Pause on Lost Focus: OFF'")
            print("   3. Or use windowed mode for easier window management")
            print("   Continuing with action...")
            time.sleep(1)  # Give user time to manually focus if needed
    
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


def start_conversational_agent(openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
    """
    Start the conversational Minecraft agent.
    
    Args:
        openai_api_key: Optional OpenAI API key for enhanced chat
        model: OpenAI model to use
    """
    agent = ConversationalMinecraftAgent(openai_api_key, model)
    agent.start_conversation()


if __name__ == "__main__":
    # You can run this directly for testing
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    start_conversational_agent(api_key)
