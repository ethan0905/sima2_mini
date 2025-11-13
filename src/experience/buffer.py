from __future__ import annotations

import random
from typing import List, Optional

from .types import Episode

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """
    In-memory storage for agent episodes that represents the "Self-Generated Experience" 
    component in the SIMA architecture.
    
    This buffer stores episodes from agent interactions and provides sampling
    functionality for training. It supports both random sampling for general
    learning and specific queries for analysis.
    
    The buffer maintains episodes in memory for fast access during training,
    while persistent storage is handled by the storage module.
    """

    def __init__(self, max_size: Optional[int] = None, auto_evict: bool = True):
        """
        Initialize the replay buffer.
        
        Args:
            max_size: Maximum number of episodes to store. If None, unlimited.
            auto_evict: If True, automatically remove oldest episodes when max_size exceeded
        """
        self.max_size = max_size
        self.auto_evict = auto_evict
        self._episodes: List[Episode] = []

    def add_episode(self, episode: Episode) -> None:
        """
        Add a new episode to the buffer.
        
        Args:
            episode: Episode to add to the buffer
        """
        self._episodes.append(episode)
        
        # Handle size limit
        if self.max_size is not None and len(self._episodes) > self.max_size:
            if self.auto_evict:
                # Remove oldest episode
                self._episodes.pop(0)
            else:
                raise RuntimeError(f"Buffer size limit {self.max_size} exceeded")

    def sample_episodes(self, n: int, replace: bool = False) -> List[Episode]:
        """
        Randomly sample episodes from the buffer.
        
        Args:
            n: Number of episodes to sample
            replace: Whether to sample with replacement
            
        Returns:
            List of sampled episodes
            
        Raises:
            ValueError: If n > buffer size and replace=False
        """
        if n <= 0:
            return []
            
        if not replace and n > len(self._episodes):
            raise ValueError(f"Cannot sample {n} episodes without replacement from buffer of size {len(self._episodes)}")
            
        if replace:
            return [random.choice(self._episodes) for _ in range(n)]
        else:
            return random.sample(self._episodes, n)

    def sample_by_task(self, task_id: str, n: Optional[int] = None) -> List[Episode]:
        """
        Sample episodes for a specific task.
        
        Args:
            task_id: ID of the task to sample episodes for
            n: Maximum number of episodes to return. If None, return all.
            
        Returns:
            List of episodes for the specified task
        """
        task_episodes = [ep for ep in self._episodes if ep.task_id == task_id]
        
        if n is None or n >= len(task_episodes):
            return task_episodes
        else:
            return random.sample(task_episodes, n)

    def sample_successful_episodes(self, n: int) -> List[Episode]:
        """
        Sample only successful episodes.
        
        Args:
            n: Number of successful episodes to sample
            
        Returns:
            List of successful episodes
        """
        successful_episodes = [ep for ep in self._episodes if ep.success]
        
        if n > len(successful_episodes):
            # Return all successful episodes if not enough available
            return successful_episodes
        else:
            return random.sample(successful_episodes, n)

    def sample_failed_episodes(self, n: int) -> List[Episode]:
        """
        Sample only failed episodes.
        
        Args:
            n: Number of failed episodes to sample
            
        Returns:
            List of failed episodes
        """
        failed_episodes = [ep for ep in self._episodes if not ep.success]
        
        if n > len(failed_episodes):
            # Return all failed episodes if not enough available
            return failed_episodes
        else:
            return random.sample(failed_episodes, n)

    def get_recent_episodes(self, n: int) -> List[Episode]:
        """
        Get the n most recent episodes.
        
        Args:
            n: Number of recent episodes to return
            
        Returns:
            List of most recent episodes
        """
        return self._episodes[-n:] if n <= len(self._episodes) else self._episodes

    def clear(self) -> None:
        """Remove all episodes from the buffer."""
        self._episodes.clear()

    def get_task_statistics(self) -> dict:
        """
        Get statistics about tasks in the buffer.
        
        Returns:
            Dictionary with task statistics including success rates
        """
        if not self._episodes:
            return {}
            
        task_stats = {}
        
        for episode in self._episodes:
            task_id = episode.task_id
            if task_id not in task_stats:
                task_stats[task_id] = {
                    "total_episodes": 0,
                    "successful_episodes": 0,
                    "total_reward": 0.0,
                    "avg_length": 0.0
                }
            
            stats = task_stats[task_id]
            stats["total_episodes"] += 1
            if episode.success:
                stats["successful_episodes"] += 1
            stats["total_reward"] += episode.final_reward
            stats["avg_length"] = (stats["avg_length"] * (stats["total_episodes"] - 1) + 
                                  episode.length) / stats["total_episodes"]
        
        # Calculate success rates
        for task_id, stats in task_stats.items():
            stats["success_rate"] = stats["successful_episodes"] / stats["total_episodes"]
            stats["avg_reward"] = stats["total_reward"] / stats["total_episodes"]
            
        return task_stats

    def __len__(self) -> int:
        """Get the number of episodes in the buffer."""
        return len(self._episodes)

    def __iter__(self):
        """Allow iteration over episodes."""
        return iter(self._episodes)

    def __getitem__(self, index: int) -> Episode:
        """Allow indexing into the buffer."""
        return self._episodes[index]

    def __repr__(self) -> str:
        """String representation of the buffer."""
        return f"ReplayBuffer(size={len(self)}, max_size={self.max_size})"
