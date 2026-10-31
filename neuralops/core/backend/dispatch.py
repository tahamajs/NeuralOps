#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Backend Dispatch Engine

Multi-tier execution dispatcher routing kernel requests across Native CUDA, Triton JIT, and PyTorch fallbacks.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch

from neuralops.core.hardware import (
    HardwareBackend,
    HardwareTopology,
    probe_hardware,
)

logger = logging.getLogger("neuralops.core.backend.dispatch")


class BackendDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    XPU = "xpu"
    TRITON = "triton"


def available_backends() -> list[str]:
    """
    Return all compute backends available in the current environment.
    """
    backends = ["cpu"]
    if torch.cuda.is_available():
        is_rocm = bool(hasattr(torch.version, "hip") and torch.version.hip is not None)
        backends.insert(0, "rocm" if is_rocm else "cuda")
        try:
            import triton  # noqa: F401
            backends.insert(0, "triton")
        except ImportError:
            pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends.insert(0, "mps")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        backends.insert(0, "xpu")

    return backends


def get_available_backend(device: Optional[torch.device | str] = None) -> BackendDevice:
    """
    Return highest tier backend device for the given or current target device.
    """
    topo = probe_hardware(device)
    if topo.backend == HardwareBackend.CUDA:
        if topo.supports_triton_jit:
            try:
                import triton  # noqa: F401
                return BackendDevice.TRITON
            except ImportError:
                pass
        return BackendDevice.CUDA
    elif topo.backend == HardwareBackend.ROCM:
        if topo.supports_triton_jit:
            try:
                import triton  # noqa: F401
                return BackendDevice.TRITON
            except ImportError:
                pass
        return BackendDevice.ROCM
    elif topo.backend == HardwareBackend.MPS:
        return BackendDevice.MPS
    elif topo.backend == HardwareBackend.XPU:
        return BackendDevice.XPU
    return BackendDevice.CPU


def get_backend(tensor: Optional[torch.Tensor] = None) -> str:
    """
    Determine the optimal execution backend string for a given tensor.
    """
    if tensor is not None:
        dev_type = tensor.device.type
        if dev_type == "cuda":
            topo = probe_hardware(tensor.device)
            if topo.supports_triton_jit:
                try:
                    import triton  # noqa: F401
                    return "triton"
                except ImportError:
                    pass
            return "rocm" if topo.backend == HardwareBackend.ROCM else "cuda"
        elif dev_type == "mps":
            return "mps"
        elif dev_type == "xpu":
            return "xpu"
        return "cpu"

    backends = available_backends()
    return backends[0] if backends else "cpu"


def chunk_gla_cpu(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: Optional[torch.Tensor] = None, scale: Optional[float] = None) -> torch.Tensor:
    """Universal pure PyTorch CPU/MPS implementation of chunked/recurrent GLA."""
    if scale is not None:
        q = q * scale

    B, H, N, D = q.shape
    state = torch.zeros(B, H, D, v.shape[-1], device=q.device, dtype=torch.float32)
    outputs = []

    for t in range(N):
        alpha = g[:, :, t].exp() if g is not None else 1.0
        if isinstance(alpha, torch.Tensor):
            state = state * alpha[:, :, None, None]
        k_t = k[:, :, t].float()
        v_t = v[:, :, t].float()
        state = state + torch.einsum("bhd,bhe->bhde", k_t, v_t)
        o_t = torch.einsum("bhd,bhde->bhe", q[:, :, t].float(), state)
        outputs.append(o_t)

    return torch.stack(outputs, dim=2).to(q.dtype)


def dispatch(op_name: str, *args: Any, backend: Optional[str] = None, **kwargs: Any) -> Any:
    """
    Universal 3-Tier Hardware Dispatch:
    Level 1: Hardware-Native Acceleration (CUDA C++ / ROCm HIP)
    Level 2: OpenAI Triton JIT Kernel (NVIDIA SM >= 8.0 / ROCm)
    Level 3: Pure Vectorized Universal PyTorch (CPU / Apple Silicon MPS / Generic GPU)
    """
    if backend is None:
        tensor = args[0] if args and isinstance(args[0], torch.Tensor) else kwargs.get("tensor")
        backend = get_backend(tensor)

    if op_name == "chunk_gla":
        if backend in ("triton", "cuda", "rocm"):
            try:
                from neuralops.attentions.linear.gla_attention import _chunked_gla_forward
                return _chunked_gla_forward(*args, **kwargs)
            except Exception as e:
                logger.warning("[chunk_gla] Triton/CUDA accelerated path failed (%s). Falling back to Universal CPU/MPS.", e)
        return chunk_gla_cpu(*args, **kwargs)

    elif op_name == "fused_linear_cross_entropy":
        try:
            from neuralops.core.fused import fused_linear_cross_entropy
            return fused_linear_cross_entropy(*args, **kwargs)
        except Exception as e:
            logger.warning("[fused_linear_cross_entropy] Accelerated path failed (%s). Falling back to PyTorch reference.", e)
            logits = torch.matmul(args[0], args[1].t()) if len(args) >= 2 else kwargs.get("logits")
            target = args[2] if len(args) >= 3 else kwargs.get("target")
            return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

    elif op_name == "fused_linear_kl_divergence":
        try:
            from neuralops.core.fused import fused_linear_kl_divergence
            return fused_linear_kl_divergence(*args, **kwargs)
        except Exception as e:
            logger.warning("[fused_linear_kl_divergence] Accelerated path failed (%s). Falling back to PyTorch reference.", e)
            p = args[0] if len(args) >= 1 else kwargs.get("p")
            q = args[1] if len(args) >= 2 else kwargs.get("q")
            return torch.nn.functional.kl_div(p.log(), q, reduction="batchmean")

    else:
        raise NotImplementedError(
            f"Operation '{op_name}' not implemented in any available backend. "
            f"Available backends: {available_backends()}"
        )
