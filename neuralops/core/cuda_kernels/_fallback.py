"""
Strict 3-Level Fallback Dispatcher for NeuralOps Hot-Paths.

Level 1: CUDA C++ (SM >= 7.0, nvcc present)
Level 2: Triton JIT Kernel (if CUDA available and triton enabled)
Level 3: Pure PyTorch Reference (guaranteed fallback)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def dispatch(
    cuda_fn: Callable[..., Any] | None,
    triton_fn: Callable[..., Any] | None,
    pytorch_fn: Callable[..., Any],
    *args: Any,
    kernel_name: str = "unknown",
    **kwargs: Any,
) -> Any:
    """
    Executes through the 3-level fallback chain:
    1. Try CUDA C++ kernel
    2. Try Triton kernel
    3. Fall back to PyTorch reference
    """
    if cuda_fn is not None:
        try:
            return cuda_fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[{kernel_name}] CUDA C++ path failed: {e}. Falling back to Triton / PyTorch.")

    if triton_fn is not None:
        try:
            return triton_fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[{kernel_name}] Triton path failed: {e}. Falling back to PyTorch.")

    return pytorch_fn(*args, **kwargs)
