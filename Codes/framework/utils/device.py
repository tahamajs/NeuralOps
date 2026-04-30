# framework/utils/device.py
"""
Device management utilities.
"""

import torch
import gc
from typing import Optional


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device_str: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device
        
    Example:
        >>> device = get_device()
        >>> device = get_device("cuda")
    """
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    
    return torch.device("cpu")


def clear_memory():
    """Clear GPU/MPS memory cache."""
    gc.collect()
    
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
    
    if torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, use psutil
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    return None


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device of a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        torch.device
    """
    return next(model.parameters()).device


def to_device(data, device: torch.device):
    """
    Recursively move data to device.
    
    Args:
        data: Input data (tensor, dict, list, tuple)
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    return data