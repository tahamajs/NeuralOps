"""
Strict 3-Level Fallback Dispatcher for NeuralOps Hot-Paths with Telemetry & Backend Forcing.

Level 1: CUDA C++ (SM >= 7.0, nvcc present)
Level 2: Triton JIT Kernel (if CUDA available and triton enabled)
Level 3: Pure PyTorch Reference (guaranteed fallback)
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Generator

import torch

logger = logging.getLogger(__name__)

# Active forced backend: "auto", "cuda", "triton", "pytorch"
_FORCED_BACKEND: str = "auto"

# Dispatch telemetry stats
_DISPATCH_STATS: dict[str, int] = {
    "cuda": 0,
    "triton": 0,
    "pytorch": 0,
    "cuda_failures": 0,
    "triton_failures": 0,
}


@contextlib.contextmanager
def force_kernel_backend(backend: str) -> Generator[None, None, None]:
    """
    Context manager to force kernel execution backend.
    Supported backends: 'auto', 'cuda', 'triton', 'pytorch'.
    """
    global _FORCED_BACKEND
    prev = _FORCED_BACKEND
    assert backend in ("auto", "cuda", "triton", "pytorch"), f"Unknown backend: {backend}"
    _FORCED_BACKEND = backend
    try:
        yield
    finally:
        _FORCED_BACKEND = prev


def get_dispatch_stats() -> dict[str, int]:
    """Returns copy of dispatch telemetry counters."""
    return dict(_DISPATCH_STATS)


def reset_dispatch_stats() -> None:
    """Resets dispatch telemetry counters."""
    global _DISPATCH_STATS
    _DISPATCH_STATS = {
        "cuda": 0,
        "triton": 0,
        "pytorch": 0,
        "cuda_failures": 0,
        "triton_failures": 0,
    }


def _is_first_tensor_cpu(*args: Any, **kwargs: Any) -> bool:
    """Check if the first tensor argument is on CPU."""
    for arg in args:
        if isinstance(arg, torch.Tensor):
            return not arg.is_cuda
    for val in kwargs.values():
        if isinstance(val, torch.Tensor):
            return not val.is_cuda
    return False


def dispatch(
    cuda_fn: Callable[..., Any] | None,
    triton_fn: Callable[..., Any] | None,
    pytorch_fn: Callable[..., Any],
    *args: Any,
    kernel_name: str = "unknown",
    allow_none: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Executes through the 3-level fallback chain:
    1. Check CPU: if CPU tensor, route immediately to PyTorch reference without logging warnings.
    2. If forced backend:
       - 'cuda': try cuda_fn only (or fail if None/error)
       - 'triton': try triton_fn only (or fail if None/error)
       - 'pytorch': call pytorch_fn directly
    3. If 'auto':
       Level 1 -> Level 2 -> Level 3.
    """
    global _DISPATCH_STATS

    # CPU fast-path: Pure PyTorch Reference immediately
    if _is_first_tensor_cpu(*args, **kwargs):
        _DISPATCH_STATS["pytorch"] += 1
        return pytorch_fn(*args, **kwargs)

    forced = _FORCED_BACKEND

    # Forced Level 3
    if forced == "pytorch":
        _DISPATCH_STATS["pytorch"] += 1
        return pytorch_fn(*args, **kwargs)

    # Forced Level 1
    if forced == "cuda":
        if cuda_fn is None:
            _DISPATCH_STATS["cuda_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'cuda', but no CUDA C++ implementation is available.")
        try:
            res = cuda_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["cuda"] += 1
                return res
        except Exception as e:
            _DISPATCH_STATS["cuda_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'cuda', but execution failed: {e}") from e

    # Forced Level 2
    if forced == "triton":
        if triton_fn is None:
            _DISPATCH_STATS["triton_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'triton', but no Triton implementation is available.")
        try:
            res = triton_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["triton"] += 1
                return res
        except Exception as e:
            _DISPATCH_STATS["triton_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'triton', but execution failed: {e}") from e

    # AUTO: Level 1 -> Level 2 -> Level 3
    if cuda_fn is not None:
        try:
            res = cuda_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["cuda"] += 1
                return res
        except Exception as e:
            _DISPATCH_STATS["cuda_failures"] += 1
            logger.debug(f"[{kernel_name}] CUDA C++ path failed: {e}. Falling back to Triton / PyTorch.")

    if triton_fn is not None:
        try:
            res = triton_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["triton"] += 1
                return res
        except Exception as e:
            _DISPATCH_STATS["triton_failures"] += 1
            logger.debug(f"[{kernel_name}] Triton path failed: {e}. Falling back to PyTorch.")

    _DISPATCH_STATS["pytorch"] += 1
    return pytorch_fn(*args, **kwargs)
