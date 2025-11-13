from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from .base_env import Observation

__all__ = ["ObservationEncoder", "FlattenEncoder", "DummyVisionEncoder"]


class ObservationEncoder(ABC):
    """
    Abstract base class for encoding game observations into feature vectors.
    
    This component processes raw pixel data and metadata from the game environment
    into a numerical representation suitable for the agent's policy network.
    
    In production, this would typically involve:
    - CNN or Vision Transformer for processing pixel data
    - Embedding layers for categorical information
    - Normalization and feature fusion
    """

    @abstractmethod
    def encode(self, observation: Observation) -> np.ndarray:
        """
        Encode a game observation into a feature vector.
        
        Args:
            observation: Raw observation from the game environment
            
        Returns:
            Encoded feature vector as numpy array
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Get the dimensionality of the encoded feature vector.
        
        Returns:
            Number of features in the encoded representation
        """
        pass


class FlattenEncoder(ObservationEncoder):
    """
    Simple encoder that flattens pixel data and concatenates with info features.
    
    This is a naive baseline that treats pixels as a flat feature vector.
    Suitable for simple environments but not for complex visual games.
    """

    def __init__(self, normalize_pixels: bool = True):
        self.normalize_pixels = normalize_pixels
        self._feature_dim: int | None = None

    def encode(self, observation: Observation) -> np.ndarray:
        """Flatten and optionally normalize pixel data."""
        pixels = observation["pixels"]
        
        # Handle different pixel formats
        if isinstance(pixels, np.ndarray):
            pixel_features = pixels.flatten()
        else:
            # Convert to numpy if needed
            pixel_features = np.array(pixels).flatten()
            
        # Normalize to [0, 1] range
        if self.normalize_pixels:
            pixel_features = pixel_features.astype(np.float32)
            if pixel_features.max() > 1.0:  # Assume 8-bit images
                pixel_features = pixel_features / 255.0
                
        # TODO: Add info features (position, health, etc.)
        # For now, just return pixel features
        info_features = np.array([], dtype=np.float32)
        
        # Cache feature dimension
        if self._feature_dim is None:
            self._feature_dim = len(pixel_features) + len(info_features)
            
        return np.concatenate([pixel_features, info_features])

    def get_feature_dim(self) -> int:
        """Get feature dimension (must call encode once first)."""
        if self._feature_dim is None:
            raise ValueError("Must call encode() at least once to determine feature dimension")
        return self._feature_dim


class DummyVisionEncoder(ObservationEncoder):
    """
    Dummy encoder for testing that returns fixed-size random features.
    
    Useful for testing the agent pipeline without requiring actual vision processing.
    """

    def __init__(self, feature_dim: int = 128, add_noise: bool = False):
        self.feature_dim = feature_dim
        self.add_noise = add_noise

    def encode(self, observation: Observation) -> np.ndarray:
        """Return dummy features based on observation info."""
        # Create deterministic features based on info
        info = observation.get("info", {})
        
        # Use position info if available (for dummy gridworld)
        if "agent_pos" in info and "goal_pos" in info:
            agent_pos = info["agent_pos"]
            goal_pos = info["goal_pos"]
            
            # Create simple positional features
            features = np.zeros(self.feature_dim, dtype=np.float32)
            features[0] = agent_pos[0] / 10.0  # Normalized x
            features[1] = agent_pos[1] / 10.0  # Normalized y
            features[2] = goal_pos[0] / 10.0   # Goal x
            features[3] = goal_pos[1] / 10.0   # Goal y
            
            # Distance to goal
            distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
            features[4] = distance / 10.0
            
        else:
            # Fallback to random features
            features = np.random.normal(0, 0.1, self.feature_dim).astype(np.float32)
            
        # Add noise if requested
        if self.add_noise:
            noise = np.random.normal(0, 0.01, self.feature_dim).astype(np.float32)
            features += noise
            
        return features

    def get_feature_dim(self) -> int:
        """Return the configured feature dimension."""
        return self.feature_dim


# TODO: Production vision encoder would look like:
# 
# class CNNVisionEncoder(ObservationEncoder):
#     def __init__(self):
#         import torch
#         import torch.nn as nn
#         
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2), 
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((7, 7)),
#             nn.Flatten(),
#             nn.Linear(128 * 7 * 7, 512)
#         )
#         
#     def encode(self, observation: Observation) -> np.ndarray:
#         pixels = torch.tensor(observation["pixels"]).permute(2, 0, 1).unsqueeze(0)
#         with torch.no_grad():
#             features = self.cnn(pixels)
#         return features.squeeze(0).numpy()
#         
#     def get_feature_dim(self) -> int:
#         return 512
