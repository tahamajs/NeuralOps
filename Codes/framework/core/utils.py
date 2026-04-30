# framework/core/utils.py
"""
Utility functions for the framework.
"""

import torch
import random
import numpy as np
from typing import Union, Optional


def get_device(device_name: str = "auto") -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device_name: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device
        
    Example:
        >>> device = get_device()
        >>> device = get_device("cuda")
    """
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    
    # Auto selection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to set deterministic algorithms (slower)
        
    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device of a model.
    
    Args:
        model: The model
        
    Returns:
        torch.device
    """
    return next(model.parameters()).device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: The model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """
    Format a number with commas and suffix.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string (e.g., "1.2M", "45K")
        
    Example:
        >>> format_number(1234567)
        '1.2M'
    """
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num/1_000:.0f}K"
    return str(num)


def clear_memory():
    """
    Clear GPU/MPS memory cache.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_memory_usage() -> Optional[float]:
    """
    Get current GPU/MPS memory usage in MB.
    
    Returns:
        Memory usage in MB, or None if not available
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return None


def human_readable_time(seconds: float) -> str:
    """
    Convert seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"