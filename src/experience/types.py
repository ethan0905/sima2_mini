from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

from ..env.base_env import Observation, Action

__all__ = ["Transition", "Episode"]


@dataclass
class Transition:
    """
    A single state-action-reward transition within an episode.
    
    This represents one step of agent interaction with the environment,
    containing all information needed for learning and analysis.
    """
    obs: Observation
    action: Action
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate transition data after initialization."""
        if not isinstance(self.obs, dict) or "pixels" not in self.obs:
            raise ValueError("obs must be an Observation with 'pixels' key")
        if not isinstance(self.action, dict):
            raise ValueError("action must be a dictionary")
        if not isinstance(self.reward, (int, float)):
            raise ValueError("reward must be a number")
        if not isinstance(self.done, bool):
            raise ValueError("done must be a boolean")


@dataclass  
class Episode:
    """
    A complete episode of agent interaction with the environment.
    
    This represents the core unit of "Self-Generated Experience" in the SIMA
    architecture. Each episode contains:
    - The task the agent was trying to solve
    - All transitions (state-action-reward sequences)
    - Final evaluation from the RewardModel
    - Success/failure outcome
    
    Episodes are stored in the experience buffer and used for:
    - Training the agent's policy
    - Updating the reward model
    - Informing the task setter about difficulty
    """
    task_id: str
    task_description: str
    estimated_reward: float
    transitions: List[Transition]
    final_reward: float
    success: bool
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate episode data after initialization."""
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if not self.task_description.strip():
            raise ValueError("task_description cannot be empty")
        if not isinstance(self.transitions, list):
            raise ValueError("transitions must be a list")
        if not all(isinstance(t, Transition) for t in self.transitions):
            raise ValueError("All transitions must be Transition objects")
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")

    @property
    def length(self) -> int:
        """Get the number of transitions in this episode."""
        return len(self.transitions)

    @property
    def total_environment_reward(self) -> float:
        """Get the sum of all environment rewards during the episode."""
        return sum(t.reward for t in self.transitions)

    @property
    def was_truncated(self) -> bool:
        """Check if episode was truncated (ended without natural termination)."""
        if not self.transitions:
            return False
        last_transition = self.transitions[-1]
        return last_transition.done and last_transition.info.get("timeout", False)

    @property
    def reached_goal(self) -> bool:
        """Check if episode successfully reached its goal."""
        if not self.transitions:
            return False
        last_transition = self.transitions[-1]
        return last_transition.info.get("goal_reached", False)

    def add_transition(self, transition: Transition) -> None:
        """
        Add a transition to the episode.
        
        Args:
            transition: Transition to add
        """
        self.transitions.append(transition)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert episode to dictionary for serialization.
        
        Returns:
            Dictionary representation of the episode
        """
        import numpy as np
        
        def convert_numpy(obj):
            """Convert numpy arrays to lists for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        return {
            "task_id": self.task_id,
            "task_description": self.task_description, 
            "estimated_reward": self.estimated_reward,
            "transitions": [
                {
                    "obs": convert_numpy(t.obs),
                    "action": convert_numpy(t.action),
                    "reward": t.reward,
                    "done": t.done,
                    "info": convert_numpy(t.info)
                } for t in self.transitions
            ],
            "final_reward": self.final_reward,
            "success": self.success,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """
        Create episode from dictionary representation.
        
        Args:
            data: Dictionary containing episode data
            
        Returns:
            Episode object
        """
        import numpy as np
        
        def convert_to_numpy(obj):
            """Convert lists back to numpy arrays if they look like pixel data."""
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    if k == "pixels" and isinstance(v, list):
                        # Convert pixel data back to numpy array
                        new_dict[k] = np.array(v, dtype=np.float32)
                    else:
                        new_dict[k] = convert_to_numpy(v)
                return new_dict
            elif isinstance(obj, list):
                return [convert_to_numpy(item) for item in obj]
            else:
                return obj
        
        transitions = [
            Transition(
                obs=convert_to_numpy(t["obs"]),
                action=t["action"],
                reward=t["reward"],
                done=t["done"],
                info=t["info"]
            ) for t in data["transitions"]
        ]
        
        return cls(
            task_id=data["task_id"],
            task_description=data["task_description"],
            estimated_reward=data["estimated_reward"],
            transitions=transitions,
            final_reward=data["final_reward"],
            success=data["success"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {})
        )
