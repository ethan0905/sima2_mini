from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from ..env.base_env import Action

__all__ = ["Policy", "RandomPolicy", "MLPPolicy", "EpsilonGreedyPolicy"]


class Policy(ABC):
    """
    Abstract base class for agent policies in the SIMA architecture.
    
    The Policy represents the agent's decision-making component that maps
    observations to actions. This can range from simple heuristics to
    complex neural networks trained with RL algorithms.
    """

    @abstractmethod
    def act(self, obs_vector: np.ndarray) -> Action:
        """
        Select an action given an observation.
        
        Args:
            obs_vector: Encoded observation from the environment
            
        Returns:
            Action dictionary to execute in the environment
        """
        pass

    @abstractmethod
    def update(self, episodes_data: List[Dict]) -> None:
        """
        Update the policy based on experience.
        
        Args:
            episodes_data: List of episode data for learning
        """
        pass

    def reset(self) -> None:
        """Reset any internal state (for stateful policies)."""
        pass


class RandomPolicy(Policy):
    """
    Random policy for baseline and testing.
    
    This policy selects actions uniformly at random from the action space.
    Useful for establishing baselines and testing the environment interface.
    """

    def __init__(self, action_space: List[str] = None):
        """
        Initialize random policy.
        
        Args:
            action_space: List of possible actions. If None, uses default gridworld actions.
        """
        if action_space is None:
            action_space = ["up", "down", "left", "right", "noop"]
        self.action_space = action_space

    def act(self, obs_vector: np.ndarray) -> Action:
        """Select a random action."""
        move = random.choice(self.action_space)
        return {"move": move, "type": "discrete"}

    def update(self, episodes_data: List[Dict]) -> None:
        """Random policy doesn't learn, so this is a no-op."""
        pass


class EpsilonGreedyPolicy(Policy):
    """
    Simple epsilon-greedy policy for Q-learning style algorithms.
    
    This policy maintains a Q-table (or Q-function approximator) and uses
    epsilon-greedy exploration to balance exploitation vs exploration.
    """

    def __init__(self, 
                 action_space: List[str] = None,
                 epsilon: float = 0.1,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.95):
        """
        Initialize epsilon-greedy policy.
        
        Args:
            action_space: Available actions
            epsilon: Exploration probability
            learning_rate: Learning rate for Q-updates
            discount_factor: Discount factor for future rewards
        """
        if action_space is None:
            action_space = ["up", "down", "left", "right", "noop"]
            
        self.action_space = action_space
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Simple Q-table (state -> action -> value)
        # In practice, would use function approximation for large state spaces
        self.q_table: Dict[str, Dict[str, float]] = {}

    def act(self, obs_vector: np.ndarray) -> Action:
        """Select action using epsilon-greedy strategy."""
        state_key = self._obs_to_state_key(obs_vector)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.action_space}
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: random action
            move = random.choice(self.action_space)
        else:
            # Exploit: best known action
            q_values = self.q_table[state_key]
            move = max(q_values.keys(), key=lambda a: q_values[a])
        
        return {"move": move, "type": "discrete"}

    def update(self, episodes_data: List[Dict]) -> None:
        """
        Update Q-values using episode data.
        
        Args:
            episodes_data: List containing episode transitions
        """
        for episode_data in episodes_data:
            transitions = episode_data.get("transitions", [])
            
            for i, transition in enumerate(transitions):
                obs_vector = transition["obs_vector"]
                action = transition["action"] 
                reward = transition["reward"]
                next_obs_vector = transition.get("next_obs_vector")
                done = transition["done"]
                
                state_key = self._obs_to_state_key(obs_vector)
                action_key = action.get("move", "noop")
                
                # Initialize Q-values if needed
                if state_key not in self.q_table:
                    self.q_table[state_key] = {a: 0.0 for a in self.action_space}
                
                # Q-learning update
                current_q = self.q_table[state_key][action_key]
                
                if done or next_obs_vector is None:
                    # Terminal state
                    target_q = reward
                else:
                    # Bootstrap from next state
                    next_state_key = self._obs_to_state_key(next_obs_vector)
                    if next_state_key not in self.q_table:
                        self.q_table[next_state_key] = {a: 0.0 for a in self.action_space}
                    
                    max_next_q = max(self.q_table[next_state_key].values())
                    target_q = reward + self.discount_factor * max_next_q
                
                # Update Q-value
                self.q_table[state_key][action_key] += self.learning_rate * (target_q - current_q)

    def _obs_to_state_key(self, obs_vector: np.ndarray) -> str:
        """
        Convert observation vector to a state key for the Q-table.
        
        Args:
            obs_vector: Observation vector
            
        Returns:
            String key representing the state
        """
        # For gridworld, extract position info if available
        if len(obs_vector) >= 5:
            # Assume first few elements are normalized position features
            x = int(obs_vector[0] * 10)  # Denormalize
            y = int(obs_vector[1] * 10)
            return f"pos_{x}_{y}"
        else:
            # Fallback: hash the observation
            return str(hash(obs_vector.tobytes()))


class MLPPolicy(Policy):
    """
    Multi-layer perceptron policy using PyTorch.
    
    This implements a neural network policy that can be trained using
    policy gradient methods like REINFORCE or actor-critic algorithms.
    
    Note: This is a minimal implementation. Production use would require
    proper RL frameworks like Stable Baselines3 or Ray RLLib.
    """

    def __init__(self, 
                 obs_dim: int,
                 hidden_dims: List[int] = None,
                 action_space: List[str] = None,
                 learning_rate: float = 0.001):
        """
        Initialize MLP policy.
        
        Args:
            obs_dim: Dimension of observation vector
            hidden_dims: Hidden layer dimensions
            action_space: Available actions
            learning_rate: Learning rate for policy updates
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch is required for MLPPolicy")
        
        if action_space is None:
            action_space = ["up", "down", "left", "right", "noop"]
        if hidden_dims is None:
            hidden_dims = [128, 64]
            
        self.action_space = action_space
        self.action_dim = len(action_space)
        self.learning_rate = learning_rate
        
        # Build network
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, self.action_dim))
        
        self.network = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Store for gradient computation
        self.saved_log_probs: List[torch.Tensor] = []
        self.saved_rewards: List[float] = []

    def act(self, obs_vector: np.ndarray) -> Action:
        """Select action using policy network."""
        import torch
        import torch.nn.functional as F
        
        obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0)
        
        # Forward pass
        logits = self.network(obs_tensor)
        action_probs = F.softmax(logits, dim=-1)
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample()
        
        # Store for learning
        self.saved_log_probs.append(action_dist.log_prob(action_idx))
        
        # Convert to action dictionary
        action_name = self.action_space[action_idx.item()]
        return {"move": action_name, "type": "discrete"}

    def update(self, episodes_data: List[Dict]) -> None:
        """
        Update policy using REINFORCE algorithm.
        
        Args:
            episodes_data: Episode data containing rewards
        """
        if not self.saved_log_probs:
            return
            
        import torch
        
        # Extract rewards from episode data
        all_rewards = []
        for episode_data in episodes_data:
            episode_rewards = episode_data.get("rewards", [])
            all_rewards.extend(episode_rewards)
        
        # Ensure we have matching rewards and log_probs
        min_len = min(len(self.saved_log_probs), len(all_rewards))
        if min_len == 0:
            return
            
        log_probs = self.saved_log_probs[:min_len]
        rewards = all_rewards[:min_len]
        
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R  # Discount factor
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns (for stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy gradient loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear stored data
        self.saved_log_probs.clear()
        self.saved_rewards.clear()

    def reset(self) -> None:
        """Reset stored gradients."""
        self.saved_log_probs.clear()
        self.saved_rewards.clear()
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save the policy model to disk.
        
        Args:
            filepath: Path to save the model checkpoint
        """
        import torch
        
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_space': self.action_space,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'architecture': {
                'obs_dim': self.network[0].in_features,
                'hidden_dims': [layer.out_features for layer in self.network if hasattr(layer, 'out_features')][:-1]
            }
        }
        
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load the policy model from disk.
        
        Args:
            filepath: Path to load the model checkpoint from
        """
        import torch
        
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore other attributes
        self.action_space = checkpoint['action_space']
        self.action_dim = checkpoint['action_dim']
        self.learning_rate = checkpoint['learning_rate']


# TODO: More sophisticated policies would include:
# 
# class PPOPolicy(Policy):
#     """Proximal Policy Optimization implementation."""
#     pass
#     
# class A3CPolicy(Policy): 
#     """Asynchronous Actor-Critic implementation."""
#     pass
#     
# class TransformerPolicy(Policy):
#     """Transformer-based policy for sequence modeling.""" 
#     pass
