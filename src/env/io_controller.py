from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np

__all__ = [
    "ScreenCapture", "InputController", "GameIOController", 
    "DummyScreenCapture", "DummyInputController"
]


class ScreenCapture(Protocol):
    """Protocol for capturing game screen content."""
    
    def capture(self) -> np.ndarray:
        """
        Capture the current screen as a numpy array.
        
        Returns:
            RGB image array of shape (height, width, 3)
        """
        ...
        
    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Capture a specific screen region.
        
        Args:
            x, y: Top-left corner coordinates
            width, height: Region dimensions
            
        Returns:
            RGB image array of the specified region
        """
        ...


class InputController(Protocol):
    """Protocol for sending input to games."""
    
    def key_press(self, key: str) -> None:
        """
        Send a key press event.
        
        Args:
            key: Key identifier (e.g., 'w', 'space', 'ctrl+c')
        """
        ...
        
    def key_release(self, key: str) -> None:
        """
        Send a key release event.
        
        Args:
            key: Key identifier
        """
        ...
        
    def mouse_move(self, x: int, y: int) -> None:
        """
        Move mouse to absolute coordinates.
        
        Args:
            x, y: Screen coordinates
        """
        ...
        
    def mouse_click(self, x: int, y: int, button: str = "left") -> None:
        """
        Click mouse at specified coordinates.
        
        Args:
            x, y: Screen coordinates
            button: Mouse button ("left", "right", "middle")
        """
        ...


class GameIOController:
    """
    High-level controller that combines screen capture and input control
    for interacting with real games.
    
    This represents the bridge between the abstract GameEnv interface
    and actual game control. In production, this would use libraries like:
    - pyautogui, pynput for input control
    - mss, pillow for screen capture
    - opencv for image processing
    """
    
    def __init__(self, screen_capture: ScreenCapture, input_controller: InputController):
        self.screen_capture = screen_capture
        self.input_controller = input_controller
        
    def get_observation(self) -> Dict[str, Any]:
        """
        Capture current game state as an observation.
        
        Returns:
            Dictionary with 'pixels' key containing screen capture
        """
        # TODO: Add preprocessing like resizing, normalization
        pixels = self.screen_capture.capture()
        return {"pixels": pixels, "info": {}}
        
    def execute_action(self, action: Dict[str, Any]) -> None:
        """
        Execute a game action through input control.
        
        Args:
            action: Action dictionary containing input commands
        """
        # TODO: Parse action and convert to appropriate input events
        # Example action formats:
        # {"type": "key", "key": "w", "duration": 0.1}
        # {"type": "mouse", "x": 100, "y": 200, "button": "left"}
        # {"type": "combo", "keys": ["w", "a"], "duration": 0.5}
        
        action_type = action.get("type", "noop")
        
        if action_type == "key":
            key = action["key"]
            self.input_controller.key_press(key)
            # TODO: Handle key release timing
            
        elif action_type == "mouse":
            x, y = action["x"], action["y"]
            button = action.get("button", "left")
            self.input_controller.mouse_click(x, y, button)
            
        # TODO: Add more sophisticated action types
        

# Dummy implementations for testing

class DummyScreenCapture:
    """Dummy screen capture that returns random noise."""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        
    def capture(self) -> np.ndarray:
        """Return random RGB image."""
        return np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        
    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Return random RGB image of specified size."""
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class DummyInputController:
    """Dummy input controller that logs actions."""
    
    def __init__(self):
        self.actions_log: List[Dict[str, Any]] = []
        
    def key_press(self, key: str) -> None:
        """Log key press."""
        self.actions_log.append({"type": "key_press", "key": key})
        print(f"[DummyInput] Key press: {key}")
        
    def key_release(self, key: str) -> None:
        """Log key release.""" 
        self.actions_log.append({"type": "key_release", "key": key})
        print(f"[DummyInput] Key release: {key}")
        
    def mouse_move(self, x: int, y: int) -> None:
        """Log mouse movement."""
        self.actions_log.append({"type": "mouse_move", "x": x, "y": y})
        print(f"[DummyInput] Mouse move: ({x}, {y})")
        
    def mouse_click(self, x: int, y: int, button: str = "left") -> None:
        """Log mouse click."""
        self.actions_log.append({"type": "mouse_click", "x": x, "y": y, "button": button})
        print(f"[DummyInput] Mouse click: ({x}, {y}) {button}")


# TODO: Real implementations would use:
# 
# class RealScreenCapture:
#     def __init__(self):
#         import mss
#         self.sct = mss.mss()
#         
#     def capture(self) -> np.ndarray:
#         screenshot = self.sct.grab(self.sct.monitors[1])  # Primary monitor
#         return np.array(screenshot)[:, :, :3]  # Remove alpha channel
#         
# class RealInputController:
#     def key_press(self, key: str) -> None:
#         import pyautogui
#         pyautogui.keyDown(key)
#         
#     def mouse_click(self, x: int, y: int, button: str = "left") -> None:
#         import pyautogui
#         pyautogui.click(x, y, button=button)
