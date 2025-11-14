from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional

import numpy as np

from .base_env import Observation

__all__ = ["ObservationEncoder", "FlattenEncoder", "DummyVisionEncoder", "MinecraftVisionEncoder"]


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


class MinecraftVisionEncoder(ObservationEncoder):
    """
    Vision encoder specialized for Minecraft observations.
    
    Processes RGB frames from Minecraft, extracting relevant visual features
    for agent decision making. Assumes frames are normalized to [0, 1].
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (84, 84),
                 use_cnn: bool = False,
                 feature_dim: Optional[int] = None):
        """
        Initialize Minecraft vision encoder.
        
        Args:
            target_size: Target (height, width) for frame processing
            use_cnn: Whether to use a small CNN for feature extraction
            feature_dim: Output feature dimension (auto-calculated if None)
        """
        self.target_size = target_size
        self.use_cnn = use_cnn
        self._feature_dim = feature_dim
        self._cnn = None
        
        if self.use_cnn:
            self._init_cnn()
    
    def _init_cnn(self) -> None:
        """Initialize a simple CNN for feature extraction."""
        try:
            import torch
            import torch.nn as nn
            
            # Simple CNN for Minecraft visual features
            self._cnn = nn.Sequential(
                # Conv layer 1: 3 -> 32 channels
                nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
                nn.ReLU(),
                
                # Conv layer 2: 32 -> 64 channels  
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                
                # Conv layer 3: 64 -> 64 channels
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                
                # Global average pooling
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            
            # Calculate output dimension
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, *self.target_size)
                output = self._cnn(dummy_input)
                self._cnn_output_dim = output.shape[1]
                
        except ImportError:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("PyTorch not available, CNN features disabled")
            self.use_cnn = False
    
    def encode(self, observation: Observation) -> np.ndarray:
        """
        Encode a Minecraft observation into a feature vector.
        
        Args:
            observation: Minecraft observation with 'pixels' field
            
        Returns:
            Feature vector for the observation
        """
        pixels = observation["pixels"]
        
        # Handle different input types
        if not isinstance(pixels, np.ndarray):
            pixels = np.array(pixels)
        
        # Ensure float32 format
        if pixels.dtype != np.float32:
            if pixels.dtype == np.uint8:
                pixels = pixels.astype(np.float32) / 255.0
            else:
                pixels = pixels.astype(np.float32)
        
        # Resize to target size if needed
        if pixels.shape[:2] != self.target_size:
            pixels = self._resize_frame(pixels)
        
        # Extract features
        if self.use_cnn and self._cnn is not None:
            features = self._extract_cnn_features(pixels)
        else:
            features = self._extract_basic_features(pixels)
        
        # Add metadata features if available
        info = observation.get("info", {})
        info_features = self._extract_info_features(info)
        
        # Combine all features
        all_features = np.concatenate([features, info_features])
        
        # Cache feature dimension
        if self._feature_dim is None:
            self._feature_dim = len(all_features)
            
        return all_features
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target size."""
        try:
            import cv2
            return cv2.resize(frame, self.target_size[::-1])  # cv2 uses (width, height)
        except ImportError:
            # Fallback: simple interpolation
            from scipy.ndimage import zoom
            h_ratio = self.target_size[0] / frame.shape[0]
            w_ratio = self.target_size[1] / frame.shape[1]
            return zoom(frame, (h_ratio, w_ratio, 1), order=1).astype(np.float32)
    
    def _extract_cnn_features(self, pixels: np.ndarray) -> np.ndarray:
        """Extract features using CNN."""
        import torch
        
        # Convert to torch tensor (H, W, C) -> (C, H, W)
        pixels_tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)
        
        with torch.no_grad():
            features = self._cnn(pixels_tensor)
        
        return features.squeeze(0).numpy()
    
    def _extract_basic_features(self, pixels: np.ndarray) -> np.ndarray:
        """Extract basic hand-crafted features from Minecraft frames."""
        # Color histograms for each channel
        red_hist = np.histogram(pixels[:, :, 0], bins=8, range=(0, 1))[0]
        green_hist = np.histogram(pixels[:, :, 1], bins=8, range=(0, 1))[0]
        blue_hist = np.histogram(pixels[:, :, 2], bins=8, range=(0, 1))[0]
        
        # Basic statistics
        mean_color = np.mean(pixels, axis=(0, 1))
        std_color = np.std(pixels, axis=(0, 1))
        
        # Edge detection (simple gradient)
        gray = np.mean(pixels, axis=2)
        edges_x = np.abs(np.gradient(gray, axis=1))
        edges_y = np.abs(np.gradient(gray, axis=0))
        edge_density = np.mean(edges_x + edges_y)
        
        # Sky detection (top third of image, typically blue)
        top_third = pixels[:pixels.shape[0]//3, :, :]
        sky_blue = np.mean(top_third[:, :, 2]) - np.mean(top_third[:, :, :2])
        
        # Combine all features
        features = np.concatenate([
            red_hist, green_hist, blue_hist,  # 24 features
            mean_color, std_color,             # 6 features
            [edge_density, sky_blue]           # 2 features
        ]).astype(np.float32)
        
        return features
    
    def _extract_info_features(self, info: Dict[str, Any]) -> np.ndarray:
        """Extract features from info metadata."""
        features = []
        
        # Player position (if available)
        if "position" in info:
            pos = info["position"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                features.extend([float(pos[0]), float(pos[1]), float(pos[2])])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Inventory information (if available)
        if "inventory" in info:
            inv = info["inventory"]
            # Count total items
            total_items = sum(inv.values()) if isinstance(inv, dict) else 0
            features.append(float(total_items))
        else:
            features.append(0.0)
        
        # Step count
        features.append(float(info.get("step_count", 0)))
        
        return np.array(features, dtype=np.float32)
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        if self._feature_dim is None:
            if self.use_cnn and self._cnn is not None:
                base_dim = self._cnn_output_dim
            else:
                base_dim = 32  # 24 (color hist) + 6 (stats) + 2 (edge/sky)
            
            info_dim = 5  # 3 (position) + 1 (inventory) + 1 (step)
            self._feature_dim = base_dim + info_dim
            
        return self._feature_dim


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
