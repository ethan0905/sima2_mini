"""
Minecraft environment wrapper for the SIMA agent.

This module provides a MinecraftEnv that conforms to the GameEnv interface,
allowing the agent to interact with Minecraft through either MineRL or
direct keyboard/mouse control.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np

from .base_env import Action, GameEnv, Observation
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Try to import MineRL if available
try:
    import minerl
    HAS_MINERL = True
    logger.info("MineRL is available for Minecraft integration")
except ImportError:
    minerl = None
    HAS_MINERL = False
    logger.warning("MineRL not found - will use fallback control methods")


class MinecraftEnv(GameEnv):
    """
    Wrapper around a running Minecraft instance.

    This class abstracts away how we talk to Minecraft:
    - Either via an existing RL env like MineRL (preferred if installed),
    - Or via keyboard/mouse automation + screen capture.
    
    The environment provides standard GameEnv interface while handling
    the complexity of Minecraft's action space and observation format.
    """

    def __init__(self, config: "MinecraftConfig"):
        """
        Initialize the Minecraft environment.
        
        Args:
            config: MinecraftConfig with settings for the environment
        """
        self.config = config
        self._step_count = 0
        self._max_steps = getattr(config, 'max_episode_steps', 1000)
        self._last_obs = None
        self._done = False
        
        # Initialize the underlying environment
        if config.use_minerl and HAS_MINERL:
            logger.info(f"Initializing MineRL environment: {config.env_id}")
            self._init_minerl()
        else:
            if config.use_minerl and not HAS_MINERL:
                raise ImportError(
                    "MineRL requested but not installed. "
                    "Install with: pip install minerl"
                )
            logger.info("Using raw Minecraft control (not fully implemented)")
            self._init_raw_control()
    
    def _init_minerl(self) -> None:
        """Initialize MineRL environment."""
        try:
            self._env = minerl.make(self.config.env_id)
            logger.info(f"Created MineRL environment: {self.config.env_id}")
        except Exception as e:
            logger.error(f"Failed to create MineRL environment: {e}")
            raise
    
    def _init_raw_control(self) -> None:
        """Initialize raw Minecraft control (placeholder)."""
        from .io_controller import RawMinecraftController
        self._controller = RawMinecraftController()
        logger.warning("Raw Minecraft control is not fully implemented")
    
    def reset(self) -> Observation:
        """
        Reset the Minecraft environment.
        
        Returns:
            Initial observation with 'pixels' key containing RGB frame
        """
        self._step_count = 0
        self._done = False
        
        if hasattr(self, '_env'):  # MineRL
            try:
                obs = self._env.reset()
                # MineRL typically returns obs["pov"] as the main camera view
                if isinstance(obs, dict) and "pov" in obs:
                    pixels = obs["pov"]
                else:
                    pixels = obs
                    
                # Ensure proper format and size
                pixels = self._process_frame(pixels)
                
                self._last_obs = {
                    "pixels": pixels,
                    "info": {
                        "position": getattr(obs, 'player_pos', [0, 64, 0]),
                        "inventory": getattr(obs, 'inventory', {}),
                        "step_count": self._step_count
                    }
                }
                
                logger.info("Reset Minecraft environment successfully")
                return self._last_obs
                
            except Exception as e:
                logger.error(f"Failed to reset MineRL environment: {e}")
                # Return a dummy observation
                return self._create_dummy_obs()
        
        else:  # Raw control
            # TODO: implement reset for raw control
            # This would involve:
            # 1. Teleporting player to spawn
            # 2. Clearing inventory 
            # 3. Resetting world state if needed
            self._last_obs = self._controller.capture_screen()
            logger.info("Reset raw Minecraft control (placeholder)")
            return self._last_obs
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step in the Minecraft environment.
        
        Args:
            action: Dictionary with Minecraft action commands
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Environment is done, call reset() first")
        
        self._step_count += 1
        
        if hasattr(self, '_env'):  # MineRL
            try:
                # Convert high-level action to MineRL format
                minerl_action = self._convert_to_minerl_action(action)
                
                obs, reward, done, info = self._env.step(minerl_action)
                
                # Process observation
                if isinstance(obs, dict) and "pov" in obs:
                    pixels = obs["pov"]
                else:
                    pixels = obs
                    
                pixels = self._process_frame(pixels)
                
                observation = {
                    "pixels": pixels,
                    "info": {
                        **info,
                        "step_count": self._step_count,
                        "action_taken": action
                    }
                }
                
                # Check if episode should end
                done = done or self._step_count >= self._max_steps
                self._done = done
                
                self._last_obs = observation
                
                return observation, float(reward), done, info
                
            except Exception as e:
                logger.error(f"Error during MineRL step: {e}")
                # Return safe defaults
                return (
                    self._last_obs or self._create_dummy_obs(),
                    0.0,
                    True,
                    {"error": str(e)}
                )
        
        else:  # Raw control
            # TODO: implement step for raw control
            try:
                self._controller.send_action(action)
                
                # Wait for action to complete
                time.sleep(1.0 / 20.0)  # ~20 FPS
                
                # Capture new observation
                observation = self._controller.capture_screen()
                
                # Compute basic reward (placeholder)
                reward = self._compute_raw_reward(action, observation)
                
                # Check termination
                done = self._step_count >= self._max_steps
                self._done = done
                
                info = {
                    "step_count": self._step_count,
                    "action_taken": action,
                    "control_method": "raw"
                }
                
                self._last_obs = observation
                
                return observation, reward, done, info
                
            except Exception as e:
                logger.error(f"Error during raw control step: {e}")
                return (
                    self._last_obs or self._create_dummy_obs(),
                    0.0,
                    True,
                    {"error": str(e)}
                )
    
    def _convert_to_minerl_action(self, action: Action) -> Dict[str, Any]:
        """
        Convert high-level action dict to MineRL-compatible format.
        
        Args:
            action: High-level action dictionary
            
        Returns:
            MineRL-compatible action dictionary
        """
        # Default MineRL action
        minerl_action = {
            "camera": [0.0, 0.0],  # [pitch, yaw] in degrees
            "forward": 0,
            "back": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "sneak": 0,
            "sprint": 0,
            "attack": 0,
            "use": 0,
        }
        
        # Map from high-level action
        if "move_forward" in action:
            if action["move_forward"] > 0:
                minerl_action["forward"] = 1
            elif action["move_forward"] < 0:
                minerl_action["back"] = 1
        
        if "strafe" in action:
            if action["strafe"] > 0:
                minerl_action["right"] = 1
            elif action["strafe"] < 0:
                minerl_action["left"] = 1
        
        if action.get("jump", False):
            minerl_action["jump"] = 1
            
        if action.get("attack", False):
            minerl_action["attack"] = 1
        
        if "camera_pitch" in action or "camera_yaw" in action:
            minerl_action["camera"] = [
                action.get("camera_pitch", 0.0),
                action.get("camera_yaw", 0.0)
            ]
        
        return minerl_action
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process raw frame from Minecraft to standard format.
        
        Args:
            frame: Raw frame from Minecraft
            
        Returns:
            Processed frame as float32 array
        """
        if frame is None:
            # Return black frame as fallback
            return np.zeros((self.config.frame_height, self.config.frame_width, 3), dtype=np.float32)
        
        # Ensure it's a numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Handle different input formats
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Resize if needed
        if frame.shape[:2] != (self.config.frame_height, self.config.frame_width):
            try:
                import cv2
                frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))
            except ImportError:
                # Fallback: simple interpolation (not ideal but works)
                from scipy.ndimage import zoom
                h_ratio = self.config.frame_height / frame.shape[0]
                w_ratio = self.config.frame_width / frame.shape[1]
                frame = zoom(frame, (h_ratio, w_ratio, 1), order=1).astype(np.uint8)
        
        # Ensure 3 channels (RGB)
        if len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 4:  # RGBA -> RGB
            frame = frame[:, :, :3]
        
        # Convert to float32 and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def _compute_raw_reward(self, action: Action, observation: Observation) -> float:
        """
        Compute reward for raw control mode (placeholder).
        
        Args:
            action: Action taken
            observation: Resulting observation
            
        Returns:
            Reward value
        """
        # TODO: Implement proper reward computation
        # This could look at:
        # - Movement (encourage exploration)
        # - Block breaking/placing
        # - Inventory changes
        # - Achievement unlocks
        
        base_reward = -0.01  # Small penalty for each step
        
        # Encourage some actions
        if action.get("move_forward", 0) > 0:
            base_reward += 0.005  # Small reward for moving forward
        
        if action.get("attack", False):
            base_reward += 0.01   # Reward for interaction
        
        return base_reward
    
    def _create_dummy_obs(self) -> Observation:
        """Create a dummy observation for fallback cases."""
        dummy_frame = np.zeros((self.config.frame_height, self.config.frame_width, 3), dtype=np.float32)
        return {
            "pixels": dummy_frame,
            "info": {
                "step_count": self._step_count,
                "dummy": True
            }
        }
    
    def render(self) -> None:
        """
        Render the environment.
        
        For MineRL this may be a no-op since the game window is already visible.
        For raw control, this could show the captured frame.
        """
        if hasattr(self, '_env'):
            try:
                self._env.render()
            except Exception as e:
                logger.debug(f"Render failed: {e}")
        else:
            # For raw control, rendering is handled by the game window
            pass
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        logger.info("Closing Minecraft environment")
        
        if hasattr(self, '_env'):
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing MineRL environment: {e}")
        
        if hasattr(self, '_controller'):
            try:
                self._controller.close()
            except Exception as e:
                logger.warning(f"Error closing raw controller: {e}")


# TODO: More sophisticated Minecraft environments could include:
#
# class MinecraftMultiTaskEnv(MinecraftEnv):
#     """Environment that can switch between different Minecraft tasks."""
#     pass
#     
# class MinecraftTeamEnv(MinecraftEnv):
#     """Environment for multi-agent Minecraft collaboration."""
#     pass
#     
# class MinecraftBuildingEnv(MinecraftEnv):
#     """Specialized environment for building tasks with structure rewards."""
#     pass
