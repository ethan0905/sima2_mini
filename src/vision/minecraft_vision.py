"""
Enhanced Minecraft Vision System

Provides computer vision capabilities to understand what's happening
in the Minecraft game and make intelligent decisions.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

try:
    import mss
    HAS_SCREEN_CAPTURE = True
except ImportError:
    HAS_SCREEN_CAPTURE = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

@dataclass
class GameState:
    """Represents the current state of the Minecraft game."""
    health: float = 100.0
    hunger: float = 100.0
    inventory_items: List[str] = None
    current_block: Optional[str] = None
    nearby_entities: List[str] = None
    time_of_day: str = "day"
    biome: str = "unknown"
    position_estimate: Tuple[int, int] = (0, 0)
    
    def __post_init__(self):
        if self.inventory_items is None:
            self.inventory_items = []
        if self.nearby_entities is None:
            self.nearby_entities = []

class MinecraftVision:
    """Computer vision system for analyzing Minecraft gameplay."""
    
    def __init__(self):
        self.screen_monitor = None
        if HAS_SCREEN_CAPTURE:
            self.screen_monitor = mss.mss()
        
        # Vision analysis cache
        self.last_screenshot = None
        self.last_analysis = None
        self.last_analysis_time = 0
        
        # Game state tracking
        self.current_state = GameState()
        
    def capture_minecraft_screen(self) -> Optional[np.ndarray]:
        """
        Capture the current Minecraft screen.
        
        Returns:
            Screenshot as numpy array or None if failed
        """
        if not HAS_SCREEN_CAPTURE or not self.screen_monitor:
            return None
            
        try:
            # Capture primary monitor
            with self.screen_monitor as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                
                # Convert to numpy array
                img = np.array(screenshot)
                
                # Convert BGRA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                
                self.last_screenshot = img
                return img
                
        except Exception as e:
            print(f"❌ Error capturing screen: {e}")
            return None
    
    def analyze_health_hunger(self, img: np.ndarray) -> Tuple[float, float]:
        """
        Analyze health and hunger bars from the screenshot.
        
        Args:
            img: Screenshot as numpy array
            
        Returns:
            Tuple of (health_percentage, hunger_percentage)
        """
        try:
            height, width = img.shape[:2]
            
            # Health/hunger bars are typically at the bottom
            bottom_region = img[int(height * 0.85):height, :]
            
            # Look for red (health) and brown/orange (hunger) colors
            # This is a simplified approach - could be enhanced with template matching
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2HSV)
            
            # Red color range for health (accounting for different shades)
            red_lower = np.array([0, 50, 50])
            red_upper = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, red_lower, red_upper)
            
            # Orange/brown range for hunger
            orange_lower = np.array([10, 50, 50])
            orange_upper = np.array([25, 255, 255])
            orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Estimate percentages based on colored pixels
            health_pixels = np.sum(red_mask > 0)
            hunger_pixels = np.sum(orange_mask > 0)
            
            # Rough estimation (would need calibration for accuracy)
            health_percentage = min(100.0, max(0.0, health_pixels / 200.0))
            hunger_percentage = min(100.0, max(0.0, hunger_pixels / 200.0))
            
            return health_percentage, hunger_percentage
            
        except Exception as e:
            print(f"⚠️  Error analyzing health/hunger: {e}")
            return 80.0, 80.0  # Default values
    
    def detect_blocks_and_entities(self, img: np.ndarray) -> Tuple[List[str], str]:
        """
        Detect nearby blocks and entities in the field of view.
        
        Args:
            img: Screenshot as numpy array
            
        Returns:
            Tuple of (entities_list, current_block_type)
        """
        try:
            # Center region for current block detection
            height, width = img.shape[:2]
            center_region = img[int(height*0.4):int(height*0.6), 
                              int(width*0.4):int(width*0.6)]
            
            # Convert to HSV for color analysis
            hsv_center = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
            
            # Simple block type detection based on dominant colors
            current_block = self._analyze_block_type(hsv_center)
            
            # Entity detection in wider field of view
            entities = self._detect_entities(img)
            
            return entities, current_block
            
        except Exception as e:
            print(f"⚠️  Error detecting blocks/entities: {e}")
            return [], "unknown"
    
    def _analyze_block_type(self, hsv_region: np.ndarray) -> str:
        """Analyze the type of block being looked at."""
        # Get dominant color
        mean_color = np.mean(hsv_region.reshape(-1, 3), axis=0)
        hue, sat, val = mean_color
        
        # Simple heuristic block detection
        if val < 30:  # Very dark
            return "air/sky"
        elif hue < 15 or hue > 165:  # Red-ish
            return "stone/dirt"
        elif 15 <= hue < 35:  # Orange/brown
            return "wood/dirt"
        elif 35 <= hue < 85:  # Green
            return "grass/leaves"
        elif 85 <= hue < 135:  # Blue
            return "water/sky"
        else:
            return "unknown"
    
    def _detect_entities(self, img: np.ndarray) -> List[str]:
        """Detect entities like animals, mobs, or players."""
        entities = []
        
        try:
            # Simple motion-based entity detection
            # (In a real implementation, you might use object detection models)
            
            # Look for common Minecraft entity colors/patterns
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # Brown colors (cows, horses)
            brown_mask = cv2.inRange(hsv, np.array([8, 50, 50]), np.array([20, 255, 255]))
            if np.sum(brown_mask) > 500:  # Threshold for detection
                entities.append("animal")
            
            # White colors (sheep, polar bears)
            white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
            if np.sum(white_mask) > 500:
                entities.append("animal")
            
            # Dark colors (potential mobs)
            dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
            if np.sum(dark_mask) > 1000:
                entities.append("mob")
                
        except Exception as e:
            print(f"⚠️  Error in entity detection: {e}")
        
        return entities
    
    def analyze_current_situation(self) -> GameState:
        """
        Perform a complete analysis of the current game situation.
        
        Returns:
            Updated GameState object
        """
        # Avoid too frequent analysis
        current_time = time.time()
        if current_time - self.last_analysis_time < 1.0:  # Max once per second
            return self.current_state
        
        # Capture screenshot
        img = self.capture_minecraft_screen()
        if img is None:
            return self.current_state
        
        print("🔍 Analyzing current game situation...")
        
        # Analyze health and hunger
        health, hunger = self.analyze_health_hunger(img)
        
        # Detect blocks and entities
        entities, current_block = self.detect_blocks_and_entities(img)
        
        # Update game state
        self.current_state.health = health
        self.current_state.hunger = hunger
        self.current_state.nearby_entities = entities
        self.current_state.current_block = current_block
        
        # Simple time detection based on overall brightness
        brightness = np.mean(img)
        self.current_state.time_of_day = "day" if brightness > 100 else "night"
        
        self.last_analysis_time = current_time
        
        print(f"   📊 Health: {health:.1f}%, Hunger: {hunger:.1f}%")
        print(f"   🎯 Looking at: {current_block}")
        print(f"   👁️  Entities nearby: {', '.join(entities) if entities else 'None'}")
        print(f"   🌅 Time: {self.current_state.time_of_day}")
        
        return self.current_state
    
    def get_situation_description(self) -> str:
        """
        Get a natural language description of the current situation.
        
        Returns:
            Human-readable situation description
        """
        state = self.current_state
        
        # Build description
        parts = []
        
        # Health status
        if state.health < 30:
            parts.append("health is critically low")
        elif state.health < 60:
            parts.append("health is moderate")
        
        # Hunger status  
        if state.hunger < 30:
            parts.append("very hungry")
        elif state.hunger < 60:
            parts.append("getting hungry")
        
        # Current focus
        if state.current_block and state.current_block != "air/sky":
            parts.append(f"looking at {state.current_block}")
        
        # Nearby entities
        if "animal" in state.nearby_entities:
            parts.append("animals nearby")
        if "mob" in state.nearby_entities:
            parts.append("potential threats nearby")
        
        # Time of day
        if state.time_of_day == "night":
            parts.append("it's nighttime")
        
        if parts:
            return "I can see: " + ", ".join(parts)
        else:
            return "Everything looks normal"

class IntelligentTaskPlanner:
    """Plans tasks based on visual analysis and user requests."""
    
    def __init__(self, vision_system: MinecraftVision):
        self.vision = vision_system
        
    def plan_action_sequence(self, user_request: str, game_state: GameState) -> List[Dict]:
        """
        Plan a sequence of actions based on user request and current game state.
        
        Args:
            user_request: Natural language request from user
            game_state: Current game state from vision analysis
            
        Returns:
            List of action dictionaries
        """
        actions = []
        request_lower = user_request.lower()
        
        # Analyze user intent and current situation
        if "food" in request_lower or "eat" in request_lower or game_state.hunger < 30:
            actions.extend(self._plan_food_actions(game_state))
            
        elif "heal" in request_lower or "health" in request_lower or game_state.health < 30:
            actions.extend(self._plan_healing_actions(game_state))
            
        elif "mine" in request_lower or "dig" in request_lower:
            actions.extend(self._plan_mining_actions(user_request, game_state))
            
        elif "build" in request_lower or "place" in request_lower:
            actions.extend(self._plan_building_actions(user_request, game_state))
            
        elif "find" in request_lower or "look for" in request_lower:
            actions.extend(self._plan_search_actions(user_request, game_state))
            
        else:
            # Default movement actions
            actions.extend(self._plan_movement_actions(user_request, game_state))
        
        return actions
    
    def _plan_food_actions(self, game_state: GameState) -> List[Dict]:
        """Plan actions to find/consume food."""
        actions = []
        
        if "animal" in game_state.nearby_entities:
            actions.append({"type": "message", "text": "I see animals nearby for food!"})
            actions.append({"type": "look", "direction": "around"})
            actions.append({"type": "move", "direction": "toward_animal"})
        else:
            actions.append({"type": "message", "text": "Looking for food sources..."})
            actions.append({"type": "look", "direction": "around"})
            actions.append({"type": "search", "target": "food"})
        
        return actions
    
    def _plan_mining_actions(self, request: str, game_state: GameState) -> List[Dict]:
        """Plan mining actions based on what's visible."""
        actions = []
        
        # Check if already looking at mineable block
        if game_state.current_block in ["stone", "wood", "dirt"]:
            actions.append({"type": "message", "text": f"Mining {game_state.current_block}..."})
            actions.append({"type": "mine", "target": game_state.current_block})
        else:
            actions.append({"type": "message", "text": "Looking for blocks to mine..."})
            actions.append({"type": "look", "direction": "down"})
            actions.append({"type": "mine", "target": "any"})
        
        return actions
    
    def _plan_building_actions(self, request: str, game_state: GameState) -> List[Dict]:
        """Plan building actions."""
        actions = []
        
        # Simple building logic
        actions.append({"type": "message", "text": "Starting to build..."})
        
        if "house" in request.lower():
            actions.append({"type": "build", "structure": "house"})
        elif "wall" in request.lower():
            actions.append({"type": "build", "structure": "wall"})
        else:
            actions.append({"type": "place", "target": "block"})
        
        return actions
    
    def _plan_search_actions(self, request: str, game_state: GameState) -> List[Dict]:
        """Plan search actions."""
        actions = []
        
        if "animal" in request.lower():
            if "animal" in game_state.nearby_entities:
                actions.append({"type": "message", "text": "Found animals nearby!"})
            else:
                actions.append({"type": "message", "text": "Searching for animals..."})
                actions.append({"type": "look", "direction": "around"})
                actions.append({"type": "move", "direction": "explore"})
        
        return actions
    
    def _plan_movement_actions(self, request: str, game_state: GameState) -> List[Dict]:
        """Plan basic movement actions."""
        actions = []
        
        # Extract movement direction from request
        if "forward" in request.lower():
            actions.append({"type": "move", "direction": "forward"})
        elif "back" in request.lower():
            actions.append({"type": "move", "direction": "backward"})
        elif "left" in request.lower():
            actions.append({"type": "move", "direction": "left"})
        elif "right" in request.lower():
            actions.append({"type": "move", "direction": "right"})
        
        return actions
