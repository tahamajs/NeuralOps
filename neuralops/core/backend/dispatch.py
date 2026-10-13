"""
Backend Dispatch Registry and Dispatcher for Hardware-Specific Ops.
"""
from __future__ import annotations

import torch
import logging
from enum import Enum
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BackendDevice(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    TRITON = "triton"


def available_backends() -> list[str]:
    """
    Return the list of available compute backends.
    """
    backends = ["cpu"]
    if torch.cuda.is_available():
        backends.insert(0, "cuda")
        try:
            import triton
            backends.insert(0, "triton")
        except ImportError:
            pass
    return backends


def get_available_backend() -> BackendDevice:
    if torch.cuda.is_available():
        try:
            import triton
            return BackendDevice.TRITON
        except ImportError:
            return BackendDevice.CUDA
    return BackendDevice.CPU


def get_backend(tensor: Optional[torch.Tensor] = None) -> str:
    """
    Determine which backend to use for a given tensor.
    """
    if tensor is not None:
        if tensor.is_cuda:
            try:
                import triton
                return "triton"
            except ImportError:
                return "cuda"
        return "cpu"
    backends = available_backends()
    return backends[0]


def chunk_gla_cpu(q, k, v, g=None, scale=None):
    """CPU implementation of chunked/recurrent GLA."""
    if scale is not None:
        q = q * scale

    B, H, N, D = q.shape
    state = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
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


def dispatch(op_name: str, *args, backend: Optional[str] = None, **kwargs):
    """
    Dispatch an operation to the appropriate backend.
    """
    if backend is None:
        tensor = args[0] if args and isinstance(args[0], torch.Tensor) else kwargs.get("tensor")
        backend = get_backend(tensor)

    if op_name == "chunk_gla":
        if backend == "triton":
            try:
                from neuralops.attentions.linear.gla_attention import _chunked_gla_forward
                return _chunked_gla_forward(*args, **kwargs)
            except Exception:
                pass
        return chunk_gla_cpu(*args, **kwargs)
    elif op_name == "fused_linear_cross_entropy":
        from neuralops.core.fused import fused_linear_cross_entropy
        return fused_linear_cross_entropy(*args, **kwargs)
    elif op_name == "fused_linear_kl_divergence":
        from neuralops.core.fused import fused_linear_kl_divergence
        return fused_linear_kl_divergence(*args, **kwargs)
    else:
        raise NotImplementedError(
            f"Operation '{op_name}' not implemented in any available backend. "
            f"Available backends: {available_backends()}"
        )
