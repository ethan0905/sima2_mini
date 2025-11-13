from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import numpy as np

from .base_env import Action, GameEnv, Observation

__all__ = ["DummyGameEnv"]


class DummyGameEnv(GameEnv):
    """
    A simple dummy game environment for testing and development.
    
    Implements a basic gridworld where the agent starts at (0,0) and tries to
    reach a goal position. This allows the entire SIMA-like system to run
    end-to-end without requiring actual game integration.
    
    State space:
    - 5x5 grid
    - Agent position (x, y)
    - Goal position (fixed at (4, 4))
    
    Action space:
    - "move": "up", "down", "left", "right"
    - "action": "noop" (placeholder for game-specific actions)
    """

    def __init__(self, grid_size: int = 5, max_steps: int = 50, goal_reward: float = 10.0):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        
        # Goal is always at bottom-right
        self.goal_pos = (grid_size - 1, grid_size - 1)
        
        # Current state
        self.agent_pos = (0, 0)
        self.step_count = 0
        self._done = False
        
    def reset(self) -> Observation:
        """Reset to starting position."""
        self.agent_pos = (0, 0)
        self.step_count = 0
        self._done = False
        return self._get_observation()
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step in the dummy environment.
        
        Args:
            action: Dictionary with "move" key containing direction
            
        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            return self._get_observation(), 0.0, True, {"error": "Episode already done"}
            
        self.step_count += 1
        
        # Parse action
        move = action.get("move", "noop")
        
        # Update position based on movement
        x, y = self.agent_pos
        if move == "up" and y > 0:
            y -= 1
        elif move == "down" and y < self.grid_size - 1:
            y += 1
        elif move == "left" and x > 0:
            x -= 1
        elif move == "right" and x < self.grid_size - 1:
            x += 1
            
        self.agent_pos = (x, y)
        
        # Calculate reward
        reward = 0.0
        info: Dict[str, Any] = {"step": self.step_count}
        
        # Check if reached goal
        if self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            self._done = True
            info["goal_reached"] = True
        
        # Check if max steps exceeded
        elif self.step_count >= self.max_steps:
            reward = -1.0  # Small penalty for timeout
            self._done = True
            info["timeout"] = True
        
        # Small step penalty to encourage efficiency
        else:
            reward = -0.1
            
        info.update({
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "distance_to_goal": abs(self.agent_pos[0] - self.goal_pos[0]) + 
                               abs(self.agent_pos[1] - self.goal_pos[1])
        })
        
        return self._get_observation(), reward, self._done, info
    
    def render(self) -> None:
        """Print a simple ASCII representation of the grid."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place goal
        gx, gy = self.goal_pos
        grid[gy][gx] = 'G'
        
        # Place agent
        ax, ay = self.agent_pos
        grid[ay][ax] = 'A' if (ax, ay) != self.goal_pos else '@'  # @ if agent on goal
        
        print(f"\nStep {self.step_count}/{self.max_steps}")
        for row in grid:
            print(' '.join(row))
        print(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}")
        
    def close(self) -> None:
        """No cleanup needed for dummy environment."""
        pass
        
    def seed(self, seed: int | None = None) -> None:
        """Set random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _get_observation(self) -> Observation:
        """
        Create an observation from current state.
        
        Returns a dummy "pixel" representation as a flattened array
        and structured info about the game state.
        """
        # Create a simple "image" representation
        # Each cell is represented by a value: 0=empty, 1=agent, 2=goal
        pixels = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Mark goal position
        gx, gy = self.goal_pos
        pixels[gy, gx] = 2.0
        
        # Mark agent position
        ax, ay = self.agent_pos
        pixels[ay, ax] = 1.0
        
        # Flatten to simulate a camera image
        pixels_flat = pixels.flatten()
        
        return Observation(
            pixels=pixels_flat,
            info={
                "agent_pos": self.agent_pos,
                "goal_pos": self.goal_pos,
                "step_count": self.step_count,
                "done": self._done,
                "grid_size": self.grid_size
            }
        )
