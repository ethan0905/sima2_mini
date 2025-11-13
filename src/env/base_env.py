from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, TypedDict

__all__ = ["Observation", "Action", "GameEnv"]


class Observation(TypedDict):
    """
    Generic observation structure for game environments.
    
    Contains raw pixel data and any additional metadata that might be useful
    for the agent's decision making or reward computation.
    """
    pixels: Any  # numpy array or other image representation
    info: Dict[str, Any]  # Additional state information


Action = Dict[str, Any]  # e.g. keys/mouse movements, or discrete actions


class GameEnv(ABC):
    """
    Abstract interface for a video-game environment.
    
    This represents the "Game Environment" component in the SIMA architecture,
    providing a standardized way to interact with any video game through
    observations and actions.
    
    The interface is inspired by OpenAI Gym but simplified for gaming contexts
    where observations are primarily visual and actions can be complex input
    combinations.
    """

    @abstractmethod
    def reset(self) -> Observation:
        """
        Reset the game environment to an initial state.
        
        Returns:
            Initial observation after reset
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.
        
        Args:
            action: Action to execute (e.g., keyboard/mouse inputs)
            
        Returns:
            Tuple of (observation, reward, done, info)
            - observation: New state after action
            - reward: Sparse reward from environment (main learning comes from RewardModel)
            - done: Whether episode has terminated
            - info: Additional metadata about the step
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        Render the current game state (for debugging/visualization).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up the environment and release any resources.
        """
        pass

    def seed(self, seed: int | None = None) -> None:
        """
        Set random seed for reproducible behavior.
        
        Args:
            seed: Random seed value, None for random initialization
        """
        # Default implementation - environments can override if needed
        pass
