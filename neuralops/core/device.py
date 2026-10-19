#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Universal Hardware Device Abstraction

Device detection, multi-GPU topology allocation, and safe context management for CUDA, MPS, and CPU.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Callable, Generator, Optional, Sequence, TypeVar, Union

import torch
import torch.nn as nn

from neuralops.core.hardware import (
    HardwareBackend,
    HardwareTopology,
    HardwareVendor,
    probe_hardware,
)

logger = logging.getLogger("neuralops.core.device")

T = TypeVar("T")


def get_current_device(preferred: Optional[str | torch.device] = None) -> torch.device:
    """
    Get the optimal available torch.device.
    Respects NEURALOPS_DEVICE environment variable if set.
    """
    if preferred is not None:
        if isinstance(preferred, str) and preferred.lower() != "auto":
            return torch.device(preferred)
        elif isinstance(preferred, torch.device):
            return preferred

    env_device = os.environ.get("NEURALOPS_DEVICE", "").strip().lower()
    if env_device and env_device != "auto":
        return torch.device(env_device)

    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def is_mps_available() -> bool:
    """Returns True if Apple Silicon Metal (MPS) is available."""
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def is_cuda_available() -> bool:
    """Returns True if NVIDIA CUDA or AMD ROCm is available."""
    return bool(torch.cuda.is_available())


def is_rocm_available() -> bool:
    """Returns True if AMD ROCm HIP is available."""
    return bool(hasattr(torch.version, "hip") and torch.version.hip is not None)


def is_xpu_available() -> bool:
    """Returns True if Intel XPU is available."""
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def harmonize_dtype(tensor_or_dtype: Union[torch.Tensor, torch.dtype], device: torch.device) -> Any:
    """
    Harmonize tensor dtype to guarantee safety on target device.
    For instance:
    - MPS does not support float64 in many ops -> cast to float32.
    - MPS older runtimes do not support bfloat16 -> cast to float32 or float16.
    """
    is_tensor = isinstance(tensor_or_dtype, torch.Tensor)
    dtype = tensor_or_dtype.dtype if is_tensor else tensor_or_dtype
    target_dev = torch.device(device)

    safe_dtype = dtype
    if target_dev.type == "mps":
        if dtype == torch.float64:
            safe_dtype = torch.float32
        elif dtype == torch.bfloat16:
            topo = probe_hardware(target_dev)
            if not topo.supports_bf16:
                safe_dtype = torch.float32

    if is_tensor:
        if tensor_or_dtype.dtype != safe_dtype:
            return tensor_or_dtype.to(dtype=safe_dtype)
        return tensor_or_dtype
    return safe_dtype


def safe_to_device(
    obj: Any,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = False,
) -> Any:
    """
    Safely move tensor, module, or collection to device with automatic dtype harmonization.
    Guarantees no crash on Apple Silicon MPS or CPU.
    """
    target_dev = get_current_device(device)

    if isinstance(obj, torch.Tensor):
        harmonized_obj = obj
        if target_dev.type == "mps":
            if obj.dtype == torch.float64:
                harmonized_obj = obj.to(dtype=torch.float32)
            elif obj.dtype == torch.bfloat16:
                topo = probe_hardware(target_dev)
                if not topo.supports_bf16:
                    harmonized_obj = obj.to(dtype=torch.float32)

        t = harmonized_obj.to(device=target_dev, non_blocking=non_blocking)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return harmonize_dtype(t, target_dev)

    if isinstance(obj, nn.Module):
        if target_dev.type == "mps":
            for param in obj.parameters():
                if param.dtype == torch.float64:
                    param.data = param.data.to(torch.float32)
            for buf in obj.buffers():
                if buf.dtype == torch.float64:
                    buf.data = buf.data.to(torch.float32)

        m = obj.to(device=target_dev)
        if dtype is not None:
            m = m.to(dtype=dtype)
        return m

    if isinstance(obj, (list, tuple)):
        converted = [safe_to_device(item, target_dev, dtype, non_blocking) for item in obj]
        return type(obj)(converted)

    if isinstance(obj, dict):
        return {k: safe_to_device(v, target_dev, dtype, non_blocking) for k, v in obj.items()}

    return obj


@contextlib.contextmanager
def device_scope(device: Union[str, torch.device]) -> Generator[torch.device, None, None]:
    """
    Context manager to execute a block within a specific device scope.
    Handles CUDA stream/device setting and restores previous active device.
    """
    target_device = torch.device(device)
    prev_cuda_device = None

    if target_device.type == "cuda" and torch.cuda.is_available():
        prev_cuda_device = torch.cuda.current_device()
        idx = target_device.index if target_device.index is not None else 0
        torch.cuda.set_device(idx)

    try:
        yield target_device
    finally:
        if prev_cuda_device is not None and torch.cuda.is_available():
            torch.cuda.set_device(prev_cuda_device)


class UniversalDispatcher:
    """
    3-Tier Multi-Backend Dispatcher for NeuralOps Kernels & Operations.

    Tier 1: Hardware-Native Acceleration (CUDA C++ / ROCm HIP kernels)
    Tier 2: OpenAI Triton JIT Kernels (SM >= 8.0, ROCm)
    Tier 3: Pure Vectorized Universal PyTorch Fallback (CPU, MPS, Generic GPU)
    """

    @staticmethod
    def dispatch(
        op_name: str,
        *args: Any,
        native_fn: Optional[Callable[..., Any]] = None,
        triton_fn: Optional[Callable[..., Any]] = None,
        pytorch_fallback_fn: Optional[Callable[..., Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> Any:
        first_tensor = None
        for a in args:
            if isinstance(a, torch.Tensor):
                first_tensor = a
                break
        if first_tensor is None:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor):
                    first_tensor = v
                    break

        exec_device = torch.device(device) if device else (first_tensor.device if first_tensor is not None else get_current_device())
        topo = probe_hardware(exec_device)

        if topo.is_cuda_or_rocm and native_fn is not None:
            try:
                return native_fn(*args, **kwargs)
            except Exception as e:
                logger.debug("Tier 1 Native kernel for '%s' unavailable/failed (%s), trying Tier 2...", op_name, e)

        if topo.supports_triton_jit and triton_fn is not None:
            try:
                import triton  # noqa: F401
                return triton_fn(*args, **kwargs)
            except Exception as e:
                logger.debug("Tier 2 Triton JIT for '%s' failed (%s), falling back to Tier 3 Universal PyTorch...", op_name, e)

        if pytorch_fallback_fn is not None:
            return pytorch_fallback_fn(*args, **kwargs)

        raise RuntimeError(
            f"UniversalDispatcher: Op '{op_name}' failed across all tiers on device '{exec_device}' (topology: {topo.device_name})."
        )
