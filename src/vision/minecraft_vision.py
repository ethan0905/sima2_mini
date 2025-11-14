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
import os
from datetime import datetime

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
    item_in_hand: str = "unknown"
    hotbar_items: List[str] = None
    
    def __post_init__(self):
        if self.inventory_items is None:
            self.inventory_items = []
        if self.nearby_entities is None:
            self.nearby_entities = []
        if self.hotbar_items is None:
            self.hotbar_items = ["empty"] * 9

class MinecraftVision:
    """Computer vision system for analyzing Minecraft gameplay."""
    
    def __init__(self, save_screenshots: bool = True, screenshots_folder: str = "screenshots"):
        self.screen_monitor = None
        if HAS_SCREEN_CAPTURE:
            self.screen_monitor = mss.mss()
        
        # Screenshot saving configuration
        self.save_screenshots = save_screenshots
        self.screenshots_folder = screenshots_folder
        
        # Create screenshots folder if it doesn't exist
        if self.save_screenshots:
            os.makedirs(self.screenshots_folder, exist_ok=True)
            print(f"📁 Screenshots will be saved to: {os.path.abspath(self.screenshots_folder)}")
        
        # Vision analysis cache
        self.last_screenshot = None
        self.last_analysis = None
        self.last_analysis_time = 0
        
        # Game state tracking
        self.current_state = GameState()
        
    def capture_minecraft_screen(self, save_with_timestamp: bool = True, force_focus: bool = True) -> Optional[np.ndarray]:
        """
        Capture the current Minecraft screen with precise window targeting.
        
        Args:
            save_with_timestamp: Whether to save screenshot with timestamp in filename
            force_focus: Whether to focus Minecraft window before capture
        
        Returns:
            Screenshot as numpy array or None if failed
        """
        if not HAS_SCREEN_CAPTURE or not self.screen_monitor:
            print("❌ Screen capture not available")
            return None
            
        try:
            # Step 1: Focus Minecraft window first if requested
            if force_focus:
                print("🎯 Focusing Minecraft window before capture...")
                focus_success = self._focus_minecraft_window()
                if not focus_success:
                    print("⚠️  Could not focus Minecraft - capturing anyway")
                else:
                    # Give the window focus time to settle
                    import time
                    time.sleep(0.5)
            
            # Step 2: Try to get Minecraft window bounds for precise capture
            minecraft_bounds = self._get_minecraft_window_bounds()
            
            if minecraft_bounds:
                print(f"📊 Minecraft window found: {minecraft_bounds['width']}x{minecraft_bounds['height']}")
                # Capture only the Minecraft window
                with self.screen_monitor as sct:
                    screenshot = sct.grab(minecraft_bounds)
            else:
                print("⚠️  Could not find Minecraft window bounds - capturing primary monitor")
                # Fallback: capture primary monitor
                with self.screen_monitor as sct:
                    monitor = sct.monitors[1]
                    screenshot = sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert BGRA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            # Step 3: Save screenshot if enabled
            if self.save_screenshots:
                self._save_screenshot(img, save_with_timestamp)
            
            self.last_screenshot = img
            return img
                
        except Exception as e:
            print(f"❌ Error capturing Minecraft screen: {e}")
            return None
    
    def _save_screenshot(self, img: np.ndarray, with_timestamp: bool = True) -> str:
        """
        Save a screenshot to the screenshots folder.
        
        Args:
            img: Screenshot as numpy array
            with_timestamp: Whether to include timestamp in filename
            
        Returns:
            Path to saved screenshot
        """
        try:
            # Generate filename
            if with_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
                filename = f"minecraft_screenshot_{timestamp}.png"
            else:
                filename = "minecraft_latest.png"
            
            filepath = os.path.join(self.screenshots_folder, filename)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Save image
            success = cv2.imwrite(filepath, img_bgr)
            
            if success:
                print(f"📸 Screenshot saved: {filename}")
                return filepath
            else:
                print("⚠️ Failed to save screenshot")
                return ""
                
        except Exception as e:
            print(f"⚠️ Error saving screenshot: {e}")
            return ""
    
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
    
    def analyze_current_situation(self, save_annotated: bool = False, user_request: str = "") -> GameState:
        """
        Perform a complete analysis of the current game situation.
        
        Args:
            save_annotated: Whether to save an annotated screenshot
            user_request: Optional user request for annotation context
        
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
        
        # Analyze hotbar and what player is holding
        item_in_hand, hotbar_items = self.analyze_hotbar_and_hand(img)
        
        # Update game state
        self.current_state.health = health
        self.current_state.hunger = hunger
        self.current_state.nearby_entities = entities
        self.current_state.current_block = current_block
        self.current_state.item_in_hand = item_in_hand
        self.current_state.hotbar_items = hotbar_items
        
        # Simple time detection based on overall brightness
        brightness = np.mean(img)
        self.current_state.time_of_day = "day" if brightness > 100 else "night"
        
        # Save annotated screenshot if requested
        if save_annotated:
            self.save_annotated_screenshot(img, self.current_state, user_request)
        
        self.last_analysis_time = current_time
        
        print(f"   📊 Health: {health:.1f}%, Hunger: {hunger:.1f}%")
        print(f"   🎯 Looking at: {current_block}")
        print(f"   🤲 Holding: {item_in_hand}")
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
        
        # What's in hand
        if state.item_in_hand and state.item_in_hand != "empty hand" and state.item_in_hand != "unknown":
            parts.append(f"holding {state.item_in_hand}")
        elif state.item_in_hand == "empty hand":
            parts.append("hands are empty")
        
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

    def analyze_hotbar_and_hand(self, img: np.ndarray) -> Tuple[str, List[str]]:
        """
        Analyze the hotbar and what the player is holding.
        
        Args:
            img: Screenshot as numpy array
            
        Returns:
            Tuple of (item_in_hand, hotbar_items)
        """
        try:
            height, width = img.shape[:2]
            
            # Hotbar is typically at the bottom center of the screen
            hotbar_region = img[int(height * 0.85):int(height * 0.95), 
                              int(width * 0.25):int(width * 0.75)]
            
            # Hand/tool area is typically bottom right
            hand_region = img[int(height * 0.75):int(height * 0.95), 
                             int(width * 0.85):width]
            
            # Analyze what's in hand based on colors and patterns
            item_in_hand = self._analyze_hand_item(hand_region)
            
            # Analyze hotbar items (simplified)
            hotbar_items = self._analyze_hotbar_items(hotbar_region)
            
            return item_in_hand, hotbar_items
            
        except Exception as e:
            print(f"⚠️  Error analyzing hotbar/hand: {e}")
            return "unknown", []
    
    def _analyze_hand_item(self, hand_region: np.ndarray) -> str:
        """Analyze what item the player is holding."""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(hand_region, cv2.COLOR_RGB2HSV)
            mean_color = np.mean(hsv.reshape(-1, 3), axis=0)
            hue, sat, val = mean_color
            
            # Simple heuristic detection based on dominant colors
            if val < 20:  # Very dark - likely empty hand or air
                return "empty hand"
            elif sat < 30:  # Low saturation - likely stone/metal tools
                if val > 100:
                    return "stone/iron tool"
                else:
                    return "stone block"
            elif 8 <= hue <= 25 and sat > 50:  # Brown/orange - wood items
                return "wood item/tool"
            elif 35 <= hue <= 85:  # Green - plant items
                return "plant/food item"
            elif hue < 8 or hue > 170:  # Red - could be redstone, brick, etc.
                return "red item"
            else:
                return "unknown item"
                
        except Exception as e:
            print(f"⚠️  Error analyzing hand item: {e}")
            return "unknown"
    
    def _analyze_hotbar_items(self, hotbar_region: np.ndarray) -> List[str]:
        """Analyze items in the hotbar (simplified)."""
        try:
            # This is a simplified implementation
            # In a real system, you'd use more sophisticated image recognition
            
            # Divide hotbar into 9 slots
            slot_width = hotbar_region.shape[1] // 9
            items = []
            
            for i in range(9):
                start_x = i * slot_width
                end_x = (i + 1) * slot_width
                slot_region = hotbar_region[:, start_x:end_x]
                
                # Check if slot has an item (based on color variance)
                if slot_region.size > 0:
                    color_variance = np.var(slot_region)
                    if color_variance > 100:  # Threshold for "has item"
                        items.append("item")
                    else:
                        items.append("empty")
                else:
                    items.append("empty")
            
            return items
            
        except Exception as e:
            print(f"⚠️  Error analyzing hotbar: {e}")
            return ["unknown"] * 9

    def save_annotated_screenshot(self, img: np.ndarray, game_state: GameState, user_request: str = "") -> str:
        """
        Save a screenshot with analysis annotations overlay.
        
        Args:
            img: Screenshot as numpy array
            game_state: Current game state analysis
            user_request: Optional user request context
            
        Returns:
            Path to saved annotated screenshot
        """
        try:
            # Create a copy for annotation
            annotated_img = img.copy()
            
            # Add text annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (255, 255, 255)  # White text
            thickness = 2
            
            # Background rectangles for better text visibility
            overlay = annotated_img.copy()
            
            # Health and hunger info (top left)
            health_text = f"Health: {game_state.health:.1f}%"
            hunger_text = f"Hunger: {game_state.hunger:.1f}%"
            time_text = f"Time: {game_state.time_of_day}"
            
            # Text positions
            y_offset = 30
            x_offset = 10
            
            # Draw background rectangles
            cv2.rectangle(overlay, (5, 5), (300, 120), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(overlay, health_text, (x_offset, y_offset), font, font_scale, color, thickness)
            cv2.putText(overlay, hunger_text, (x_offset, y_offset + 25), font, font_scale, color, thickness)
            cv2.putText(overlay, time_text, (x_offset, y_offset + 50), font, font_scale, color, thickness)
            
            # Current focus info (top right)
            if game_state.current_block:
                block_text = f"Looking at: {game_state.current_block}"
                cv2.rectangle(overlay, (annotated_img.shape[1] - 320, 5), (annotated_img.shape[1] - 5, 60), (0, 0, 0), -1)
                cv2.putText(overlay, block_text, (annotated_img.shape[1] - 310, 30), font, font_scale, color, thickness)
            
            # Hand item info (bottom right)
            if game_state.item_in_hand:
                hand_text = f"Holding: {game_state.item_in_hand}"
                cv2.rectangle(overlay, (annotated_img.shape[1] - 320, annotated_img.shape[0] - 60), 
                             (annotated_img.shape[1] - 5, annotated_img.shape[0] - 5), (0, 0, 0), -1)
                cv2.putText(overlay, hand_text, (annotated_img.shape[1] - 310, annotated_img.shape[0] - 30), 
                           font, font_scale, color, thickness)
            
            # User request context (bottom left)
            if user_request:
                request_text = f"Request: {user_request[:40]}..."
                cv2.rectangle(overlay, (5, annotated_img.shape[0] - 60), 
                             (min(400, len(request_text) * 8), annotated_img.shape[0] - 5), (0, 0, 0), -1)
                cv2.putText(overlay, request_text, (x_offset, annotated_img.shape[0] - 30), 
                           font, font_scale, color, thickness)
            
            # Blend overlay with original image
            annotated_img = cv2.addWeighted(annotated_img, 0.7, overlay, 0.3, 0)
            
            # Save annotated screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"annotated_{timestamp}.png"
            filepath = os.path.join(self.screenshots_folder, filename)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(filepath, img_bgr)
            
            if success:
                print(f"📊 Annotated screenshot saved: {filename}")
                return filepath
            else:
                print("⚠️ Failed to save annotated screenshot")
                return ""
                
        except Exception as e:
            print(f"⚠️ Error saving annotated screenshot: {e}")
            return ""

    def _focus_minecraft_window(self) -> bool:
        """
        Focus the Minecraft window using platform-specific methods.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                return self._focus_minecraft_macos()
            elif system == "Windows":
                return self._focus_minecraft_windows()
            elif system == "Linux":
                return self._focus_minecraft_linux()
            else:
                print(f"⚠️  Unsupported platform: {system}")
                return False
                
        except Exception as e:
            print(f"❌ Error focusing Minecraft window: {e}")
            return False
    
    def _focus_minecraft_macos(self) -> bool:
        """Focus Minecraft on macOS using AppleScript."""
        try:
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
            
            if result.returncode == 0 and "success" in result.stdout:
                print("   ✅ Minecraft focused via Java process")
                return True
            else:
                print("   ❌ Could not focus Minecraft Java process")
                return False
                
        except Exception as e:
            print(f"   ❌ macOS focus error: {e}")
            return False
    
    def _focus_minecraft_windows(self) -> bool:
        """Focus Minecraft on Windows."""
        try:
            # Try using pygetwindow if available
            import pygetwindow as gw
            
            # Look for Minecraft windows
            minecraft_windows = []
            for window in gw.getAllWindows():
                if "minecraft" in window.title.lower() or "java" in window.title.lower():
                    minecraft_windows.append(window)
            
            if minecraft_windows:
                # Focus the first Minecraft window found
                minecraft_windows[0].activate()
                print("   ✅ Minecraft window focused")
                return True
            else:
                print("   ❌ No Minecraft window found")
                return False
                
        except ImportError:
            print("   ⚠️  pygetwindow not available - install with: pip install pygetwindow")
            return False
        except Exception as e:
            print(f"   ❌ Windows focus error: {e}")
            return False
    
    def _focus_minecraft_linux(self) -> bool:
        """Focus Minecraft on Linux."""
        try:
            import subprocess
            
            # Try using wmctrl to focus Minecraft window
            result = subprocess.run(['wmctrl', '-a', 'Minecraft'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ Minecraft focused via wmctrl")
                return True
            else:
                # Try focusing Java process
                result = subprocess.run(['wmctrl', '-a', 'java'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("   ✅ Java process focused")
                    return True
                else:
                    print("   ❌ Could not focus Minecraft")
                    return False
                    
        except FileNotFoundError:
            print("   ⚠️  wmctrl not available - install with: sudo apt install wmctrl")
            return False
        except Exception as e:
            print(f"   ❌ Linux focus error: {e}")
            return False
    
    def _get_minecraft_window_bounds(self) -> Optional[Dict]:
        """
        Get the exact bounds of the Minecraft window for precise capture.
        
        Returns:
            Dictionary with window bounds or None if not found
        """
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                return self._get_minecraft_bounds_macos()
            elif system == "Windows":
                return self._get_minecraft_bounds_windows()
            elif system == "Linux":
                return self._get_minecraft_bounds_linux()
            else:
                print(f"⚠️  Window bounds detection not supported on {system}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting window bounds: {e}")
            return None
    
    def _get_minecraft_bounds_macos(self) -> Optional[Dict]:
        """Get Minecraft window bounds on macOS."""
        try:
            import subprocess
            import re
            
            # Simplified AppleScript to get window bounds
            apple_script = '''
            tell application "System Events"
                if exists process "java" then
                    tell process "java"
                        try
                            set minecraft_window to (first window whose name contains "Minecraft")
                            set {x, y} to position of minecraft_window
                            set {w, h} to size of minecraft_window
                            return x & " " & y & " " & w & " " & h
                        on error
                            return "error"
                        end try
                    end tell
                else
                    return "java_not_found"
                end if
            end tell
            '''
            
            result = subprocess.run(['osascript', '-e', apple_script], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                coords_str = result.stdout.strip()
                print(f"   📊 Raw coordinates: '{coords_str}'")
                
                if coords_str and coords_str not in ["error", "java_not_found"]:
                    # Use regex to extract numbers from the string
                    numbers = re.findall(r'\d+', coords_str)
                    print(f"   📊 Extracted numbers: {numbers}")
                    
                    if len(numbers) >= 4:
                        try:
                            x, y, w, h = map(int, numbers[:4])
                            
                            bounds = {
                                "left": x,
                                "top": y, 
                                "width": w,
                                "height": h
                            }
                            print(f"   ✅ Window bounds: {bounds}")
                            return bounds
                            
                        except ValueError as ve:
                            print(f"   ⚠️  Could not parse numbers: {numbers} - {ve}")
                
                print(f"   ⚠️  Could not extract valid coordinates from: '{coords_str}'")
                return None
            else:
                print(f"   ⚠️  AppleScript failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"   ⚠️  macOS bounds detection error: {e}")
            return None
    
    def _get_minecraft_bounds_windows(self) -> Optional[Dict]:
        """Get Minecraft window bounds on Windows."""
        try:
            import pygetwindow as gw
            
            # Look for Minecraft windows
            for window in gw.getAllWindows():
                if "minecraft" in window.title.lower() or "java" in window.title.lower():
                    return {
                        "left": window.left,
                        "top": window.top,
                        "width": window.width,
                        "height": window.height
                    }
            
            return None
            
        except ImportError:
            print("   ⚠️  pygetwindow not available for bounds detection")
            return None
        except Exception as e:
            print(f"   ⚠️  Windows bounds detection error: {e}")
            return None
    
    def _get_minecraft_bounds_linux(self) -> Optional[Dict]:
        """Get Minecraft window bounds on Linux."""
        try:
            import subprocess
            
            # Use xwininfo to get window bounds
            result = subprocess.run(['xwininfo', '-name', 'Minecraft'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                bounds = {}
                
                for line in lines:
                    if 'Absolute upper-left X:' in line:
                        bounds['left'] = int(line.split(':')[1].strip())
                    elif 'Absolute upper-left Y:' in line:
                        bounds['top'] = int(line.split(':')[1].strip())
                    elif 'Width:' in line:
                        bounds['width'] = int(line.split(':')[1].strip())
                    elif 'Height:' in line:
                        bounds['height'] = int(line.split(':')[1].strip())
                
                if len(bounds) == 4:
                    return bounds
            
            return None
            
        except FileNotFoundError:
            print("   ⚠️  xwininfo not available - install X11 utilities")
            return None
        except Exception as e:
            print(f"   ⚠️  Linux bounds detection error: {e}")
