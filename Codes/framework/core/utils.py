"""General utilities (device, seeding, formatting)."""

import random

import numpy as np
import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Return the device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    if device.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but MPS is not available")
    if device.type == "cuda" and device.index is not None:
        count = torch.cuda.device_count()
        if device.index >= count:
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; {count} device(s) detected"
            )
    return device


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the model device."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """Format number."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num/1_000:.0f}K"
    return str(num)


def clear_memory() -> None:
    """Clear memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_memory_usage() -> float | None:
    """Return the memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return None


def human_readable_time(seconds: float) -> str:
    """Human readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def ste_round(x: torch.Tensor) -> torch.Tensor:
    r"""Straight-Through Estimator (STE) for rounding in quantization-aware training."""
    return x + (torch.round(x) - x).detach()


def ste_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    r"""Straight-Through Estimator (STE) for clamping."""
    return x + (torch.clamp(x, min_val, max_val) - x).detach()
