"""Memory footprint measurement tools for GPU VRAM and system CPU RAM."""

from __future__ import annotations
import gc
import torch
try:
    import psutil
except ImportError:
    psutil = None


def measure_gpu_memory() -> float:
    """
    Purge cached memory and return peak allocated GPU VRAM in megabytes (MB).
    Falls back to 0.0 on non-CUDA systems.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
    return 0.0


def get_cpu_memory_mb() -> float:
    """Return resident CPU memory footprint in MB."""
    if psutil is not None:
        process = psutil.Process()
        return process.memory_info().rss / (1024.0 * 1024.0)
    return 0.0
