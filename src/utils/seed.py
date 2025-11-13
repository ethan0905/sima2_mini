from __future__ import annotations

import random
from typing import Optional

import numpy as np

__all__ = ["set_random_seed", "get_random_state", "create_reproducible_random"]


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducible experiments across all libraries.
    
    This function seeds:
    - Python's built-in random module
    - NumPy's random number generator  
    - PyTorch (if available)
    - Any other libraries that support seeding
    
    Args:
        seed: Random seed value to use
    """
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        
        # CUDA determinism (if using GPU)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # TensorFlow (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
        
    print(f"Set random seed to {seed} for reproducible experiments")


def get_random_state() -> dict:
    """
    Get the current random state from all seeded libraries.
    
    Returns:
        Dictionary containing random states for restoration
    """
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state()
    }
    
    # PyTorch state
    try:
        import torch
        state["torch_random"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_random"] = torch.cuda.get_rng_state()
    except ImportError:
        pass
    
    return state


def restore_random_state(state: dict) -> None:
    """
    Restore random state from a previously saved state dictionary.
    
    Args:
        state: Dictionary containing random states (from get_random_state)
    """
    if "python_random" in state:
        random.setstate(state["python_random"])
        
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
        
    if "torch_random" in state:
        try:
            import torch
            torch.set_rng_state(state["torch_random"])
            if "torch_cuda_random" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state(state["torch_cuda_random"])
        except ImportError:
            pass


class ReproducibleRandom:
    """
    Context manager for reproducible random number generation.
    
    Usage:
        with ReproducibleRandom(42):
            # All random operations here use seed 42
            result = some_stochastic_function()
        # Random state is restored after the block
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self.saved_state: Optional[dict] = None
        
    def __enter__(self):
        # Save current state
        self.saved_state = get_random_state()
        
        # Set new seed
        set_random_seed(self.seed)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous state
        if self.saved_state is not None:
            restore_random_state(self.saved_state)


def create_reproducible_random(seed: int) -> random.Random:
    """
    Create a separate Random instance with its own seed.
    
    This is useful when you want reproducible random behavior that doesn't
    affect the global random state.
    
    Args:
        seed: Seed for the Random instance
        
    Returns:
        Random instance with the specified seed
    """
    rng = random.Random(seed)
    return rng


def generate_experiment_seed(experiment_name: str) -> int:
    """
    Generate a consistent seed based on experiment name.
    
    This allows reproducible experiments with memorable names instead
    of arbitrary numbers.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Consistent integer seed based on the name
    """
    # Use hash of experiment name to generate seed
    seed = hash(experiment_name) % (2**31)  # Keep within int32 range
    return abs(seed)  # Ensure positive


# Utilities for seeding specific components

def seed_environment(env, seed: int) -> None:
    """
    Seed an environment if it supports seeding.
    
    Args:
        env: Environment instance
        seed: Seed value
    """
    if hasattr(env, 'seed'):
        env.seed(seed)
    elif hasattr(env, 'reset') and hasattr(env.reset, '__code__'):
        # Check if reset accepts a seed parameter
        import inspect
        sig = inspect.signature(env.reset)
        if 'seed' in sig.parameters:
            env.reset(seed=seed)


def seed_policy(policy, seed: int) -> None:
    """
    Seed a policy if it supports seeding.
    
    Args:
        policy: Policy instance
        seed: Seed value
    """
    if hasattr(policy, 'seed'):
        policy.seed(seed)


# Example usage:
# 
# # Set global seed for experiment
# set_random_seed(42)
# 
# # Use context manager for temporary seeding
# with ReproducibleRandom(123):
#     result = run_stochastic_test()
# 
# # Create separate random generator
# rng = create_reproducible_random(456)
# value = rng.random()
# 
# # Generate seed from experiment name
# seed = generate_experiment_seed("baseline_gridworld_v1")
# set_random_seed(seed)
