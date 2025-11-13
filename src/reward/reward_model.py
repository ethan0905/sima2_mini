from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..experience.types import Episode
from ..tasks.task_schema import Task

__all__ = ["RewardModel", "SimpleRewardModel"]


class RewardModel(ABC):
    """
    Abstract base class for evaluating episode performance in the SIMA architecture.
    
    The RewardModel serves as an external evaluator that scores complete episodes
    based on task completion criteria. This decoupling allows for:
    - Learning reward functions from human feedback
    - Using LLMs to evaluate complex objectives
    - Incorporating domain-specific knowledge
    
    The model provides both initial task reward estimates (for the TaskSetter)
    and final episode scoring (for agent training).
    """

    @abstractmethod
    def estimate_task_reward(self, task: Task) -> float:
        """
        Provide an initial reward estimate for a task.
        
        This estimate is used by the TaskSetter when proposing tasks
        to calibrate task difficulty and selection.
        
        Args:
            task: Task to estimate reward for
            
        Returns:
            Estimated reward value
        """
        pass

    @abstractmethod 
    def score_episode(self, task: Task, episode: Episode) -> float:
        """
        Compute the final reward for a completed episode.
        
        This is the primary signal used to train the agent's policy,
        replacing or augmenting the sparse environment rewards.
        
        Args:
            task: Task the agent was attempting
            episode: Completed episode data
            
        Returns:
            Final reward score for the episode
        """
        pass

    @abstractmethod
    def update_from_experience(self, episodes: List[Episode]) -> None:
        """
        Update the reward model based on observed episodes.
        
        This allows the reward model to adapt and improve its scoring
        based on accumulated experience, potentially learning from:
        - Success/failure patterns
        - Human feedback on episode quality
        - Comparative preferences between episodes
        
        Args:
            episodes: List of episodes to learn from
        """
        pass


class SimpleRewardModel(RewardModel):
    """
    A simple heuristic-based reward model for baseline functionality.
    
    This implementation uses basic rules and patterns to score episodes:
    - Rewards goal completion highly
    - Penalizes inefficiency (too many steps)
    - Gives partial credit for progress toward objectives
    
    This serves as a foundation that can be replaced with learned models.
    """

    def __init__(self, 
                 goal_completion_reward: float = 10.0,
                 step_penalty: float = 0.1,
                 progress_reward_scale: float = 2.0):
        """
        Initialize the simple reward model.
        
        Args:
            goal_completion_reward: Reward for successfully completing the task
            step_penalty: Penalty per step taken (encourages efficiency)
            progress_reward_scale: Scale factor for progress-based rewards
        """
        self.goal_completion_reward = goal_completion_reward
        self.step_penalty = step_penalty
        self.progress_reward_scale = progress_reward_scale
        
        # Track performance for adaptive estimation
        self._task_completion_rates: Dict[str, List[bool]] = {}
        self._task_avg_steps: Dict[str, List[int]] = {}

    def estimate_task_reward(self, task: Task) -> float:
        """
        Estimate task reward based on historical performance and task metadata.
        
        Args:
            task: Task to estimate reward for
            
        Returns:
            Estimated reward value
        """
        # Use task's own estimate as baseline
        base_estimate = task.estimated_reward
        
        # Adjust based on historical performance if available
        task_type = task.metadata.get("template", task.id)
        
        if task_type in self._task_completion_rates:
            completion_rates = self._task_completion_rates[task_type]
            avg_completion_rate = sum(completion_rates) / len(completion_rates)
            
            # Adjust estimate based on historical difficulty
            if avg_completion_rate > 0.8:  # Too easy
                adjusted_estimate = base_estimate * 0.8
            elif avg_completion_rate < 0.2:  # Too hard
                adjusted_estimate = base_estimate * 1.2
            else:
                adjusted_estimate = base_estimate
        else:
            adjusted_estimate = base_estimate
            
        return max(0.0, min(10.0, adjusted_estimate))

    def score_episode(self, task: Task, episode: Episode) -> float:
        """
        Score an episode based on completion, efficiency, and progress.
        
        Args:
            task: Task that was attempted
            episode: Episode data to score
            
        Returns:
            Final reward score
        """
        total_reward = 0.0
        
        # Check if goal was reached
        if episode.reached_goal:
            total_reward += self.goal_completion_reward
        
        # Efficiency penalty (encourage shorter successful episodes)
        step_penalty = len(episode.transitions) * self.step_penalty
        total_reward -= step_penalty
        
        # Progress reward (partial credit for getting closer to goal)
        progress_reward = self._calculate_progress_reward(episode)
        total_reward += progress_reward
        
        # Bonus for not timing out
        if episode.success and not episode.was_truncated:
            total_reward += 1.0
            
        # Small bonus for environment rewards (if any)
        env_reward_bonus = episode.total_environment_reward * 0.1
        total_reward += env_reward_bonus
        
        # Ensure reasonable bounds
        total_reward = max(0.0, min(15.0, total_reward))
        
        return total_reward

    def update_from_experience(self, episodes: List[Episode]) -> None:
        """
        Update internal statistics based on observed episodes.
        
        Args:
            episodes: Episodes to learn from
        """
        for episode in episodes:
            task_type = episode.metadata.get("template", episode.task_id)
            
            # Track completion rates
            if task_type not in self._task_completion_rates:
                self._task_completion_rates[task_type] = []
            self._task_completion_rates[task_type].append(episode.success)
            
            # Track average steps
            if task_type not in self._task_avg_steps:
                self._task_avg_steps[task_type] = []
            self._task_avg_steps[task_type].append(episode.length)
            
            # Keep only recent history
            max_history = 50
            if len(self._task_completion_rates[task_type]) > max_history:
                self._task_completion_rates[task_type] = self._task_completion_rates[task_type][-max_history:]
                self._task_avg_steps[task_type] = self._task_avg_steps[task_type][-max_history:]

    def get_model_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about the reward model's performance.
        
        Returns:
            Dictionary with model statistics by task type
        """
        stats = {}
        
        for task_type in self._task_completion_rates:
            completion_rates = self._task_completion_rates[task_type]
            avg_steps = self._task_avg_steps.get(task_type, [])
            
            if completion_rates:
                stats[task_type] = {
                    "completion_rate": sum(completion_rates) / len(completion_rates),
                    "num_episodes": len(completion_rates),
                    "avg_steps": sum(avg_steps) / len(avg_steps) if avg_steps else 0.0,
                }
                
        return stats

    def _calculate_progress_reward(self, episode: Episode) -> float:
        """
        Calculate reward based on progress toward the goal.
        
        Args:
            episode: Episode to calculate progress for
            
        Returns:
            Progress-based reward value
        """
        if not episode.transitions:
            return 0.0
            
        # Look for progress indicators in the episode info
        initial_info = episode.transitions[0].info
        final_info = episode.transitions[-1].info
        
        # Check for distance-based progress (for gridworld)
        if "distance_to_goal" in initial_info and "distance_to_goal" in final_info:
            initial_distance = initial_info["distance_to_goal"]
            final_distance = final_info["distance_to_goal"]
            
            # Reward for reducing distance to goal
            distance_improvement = initial_distance - final_distance
            progress_reward = distance_improvement * self.progress_reward_scale
            
            return max(0.0, progress_reward)
        
        # TODO: Add other progress indicators:
        # - Items collected
        # - Areas explored
        # - Objectives completed
        # - Health remaining
        
        return 0.0


# TODO: Learned reward model would look like:
# 
# class LearnedRewardModel(RewardModel):
#     def __init__(self):
#         import torch
#         import torch.nn as nn
#         
#         # Simple MLP for scoring episodes
#         self.network = nn.Sequential(
#             nn.Linear(episode_feature_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(), 
#             nn.Linear(64, 1)
#         )
#         self.optimizer = torch.optim.Adam(self.network.parameters())
#         
#     def score_episode(self, task: Task, episode: Episode) -> float:
#         features = self._extract_episode_features(task, episode)
#         with torch.no_grad():
#             score = self.network(torch.tensor(features))
#         return score.item()
#         
#     def update_from_experience(self, episodes: List[Episode]) -> None:
#         # Train on episode features with supervision from:
#         # - Human preferences 
#         # - Success/failure labels
#         # - Comparative rankings
#         pass
#
# class LLMRewardModel(RewardModel):
#     def __init__(self, llm_client):
#         self.llm_client = llm_client
#         
#     def score_episode(self, task: Task, episode: Episode) -> float:
#         # Summarize episode as text
#         summary = self._summarize_episode(episode)
#         
#         prompt = f"""
#         Task: {task.description}
#         Agent behavior: {summary}
#         
#         Score this episode from 0-10 based on:
#         - Task completion
#         - Efficiency 
#         - Learning demonstration
#         
#         Provide only the numeric score.
#         """
#         
#         response = self.llm_client.generate(prompt)
#         return float(response.strip())
