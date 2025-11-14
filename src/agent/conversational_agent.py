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

from ..utils.logging_utils import get_logger

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

When the user gives you an instruction:
1. Acknowledge what they want to do
2. Break it down into steps if complex
3. Execute the actions needed
4. Report back on what you accomplished

Be conversational and helpful. Ask for clarification if instructions are unclear.

Example responses:
User: "Go forward and mine the tree in front of me"
You: "I'll help you mine that tree! Let me move forward and start breaking the wood blocks."

User: "Build a small house"
You: "I'd be happy to help build a house! Could you tell me what materials you'd like me to use? I can see what's in your inventory and start with a simple structure."
"""

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
        
        # Check for mining/digging
        if any(word in instruction_lower for word in self.basic_commands["dig"]):
            self._execute_minecraft_action("dig", {})
            return "Breaking the block in front of me!"
        
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
        
        return f"I understand you want me to: {instruction}. I'll try my best to help! (Note: install OpenAI for better understanding)"
    
    def _execute_actions_from_response(self, ai_response: str, original_instruction: str) -> None:
        """Extract and execute actions from AI response."""
        # This is a simplified version - in a full implementation,
        # the AI would return structured action commands
        
        # For now, use basic keyword detection
        response_lower = ai_response.lower()
        
        # Look for action keywords in the AI response
        if "move forward" in response_lower or "go forward" in response_lower:
            self._execute_minecraft_action("move", {"direction": "forward", "duration": 2.0})
        
        if "mine" in response_lower or "dig" in response_lower:
            self._execute_minecraft_action("dig", {})
        
        if "jump" in response_lower:
            self._execute_minecraft_action("jump", {})
        
        # More sophisticated action parsing would be implemented here
    
    def _execute_minecraft_action(self, action_type: str, params: Dict) -> None:
        """Execute a Minecraft action via keyboard/mouse control."""
        if not HAS_CONTROL:
            logger.warning("Cannot execute action - control libraries not available")
            return
        
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
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
    
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
            self.keyboard.press(key)
            time.sleep(duration)
            self.keyboard.release(key)
            logger.info(f"Moved {direction} for {duration} seconds")
    
    def _look(self, yaw: float, pitch: float) -> None:
        """Turn the camera/player view."""
        # Convert angles to mouse movement
        # This is approximate and may need calibration
        mouse_sensitivity = 2.0
        dx = int(yaw * mouse_sensitivity)
        dy = int(pitch * mouse_sensitivity)
        
        pyautogui.move(dx, dy, relative=True)
        logger.info(f"Looked yaw={yaw}, pitch={pitch}")
    
    def _dig(self) -> None:
        """Break the block being looked at."""
        pyautogui.mouseDown(button='left')
        time.sleep(0.5)  # Hold to break
        pyautogui.mouseUp(button='left')
        logger.info("Executed dig action")
    
    def _jump(self) -> None:
        """Make the player jump."""
        pyautogui.press('space')
        logger.info("Executed jump")
    
    def _place(self) -> None:
        """Place a block from hotbar."""
        pyautogui.click(button='right')
        logger.info("Executed place action")
    
    def _use(self) -> None:
        """Right-click/interact."""
        pyautogui.click(button='right')
        logger.info("Executed use action")
    
    def _attack(self) -> None:
        """Attack/left-click."""
        pyautogui.click(button='left')
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
        print("\n  Commands: 'help', 'status', 'quit'")
    
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
    
    def _handle_minecraft_pause(self) -> None:
        """Handle Minecraft pause menu if it's open."""
        if not HAS_CONTROL:
            return
        
        try:
            # Press Escape to close pause menu if it's open
            # This is safe - if not paused, it will pause then immediately unpause
            pyautogui.press('escape')
            time.sleep(0.1)
            pyautogui.press('escape') 
            time.sleep(0.1)
            logger.debug("Handled potential Minecraft pause menu")
        except Exception as e:
            logger.error(f"Error handling pause menu: {e}")


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
