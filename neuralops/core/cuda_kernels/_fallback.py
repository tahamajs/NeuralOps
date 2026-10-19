# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Three-Level Kernel Fallback Dispatcher

Runtime dispatcher managing three-level fallback across native CUDA, Triton JIT, and PyTorch reference implementations.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Generator

import torch

logger = logging.getLogger(__name__)

_FORCED_BACKEND: str = "auto"
FORCE_PYTORCH_FALLBACK: bool = False

_LAST_DISPATCH_BACKEND: dict[str, str] = {}
_LAST_DISPATCH_ERROR: dict[str, str] = {}

def get_last_dispatch_backend(kernel_name: str) -> str:
    return _LAST_DISPATCH_BACKEND.get(kernel_name, 'unknown')

def get_last_dispatch_error(kernel_name: str) -> str:
    return _LAST_DISPATCH_ERROR.get(kernel_name, '')

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


class _AutogradBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fwd_fn, bwd_ref_fn, reconstruct, *tensors):
        with torch.no_grad():
            args, kwargs = reconstruct(tensors)
            out = fwd_fn(*args, **kwargs)
        ctx.bwd_ref_fn = bwd_ref_fn
        ctx.reconstruct = reconstruct
        ctx.save_for_backward(*tensors)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        with torch.enable_grad():
            detached = [t.detach().requires_grad_(t.requires_grad) for t in saved]
            args, kwargs = ctx.reconstruct(detached)
            ref_out = ctx.bwd_ref_fn(*args, **kwargs)
            req_tensors = [t for t in detached if t.requires_grad]
            grads = torch.autograd.grad(ref_out, req_tensors, grad_output, allow_unused=True)

        grad_iter = iter(grads)
        tensor_grads = tuple(next(grad_iter) if t.requires_grad else None for t in detached)
        return (None, None, None, *tensor_grads)


def _pack_for_autograd(args: tuple, kwargs: dict):
    tensor_list = []
    arg_is_tensor = []
    for a in args:
        if isinstance(a, torch.Tensor):
            arg_is_tensor.append(True)
            tensor_list.append(a)
        else:
            arg_is_tensor.append(False)

    kw_is_tensor = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            kw_is_tensor[k] = True
            tensor_list.append(v)
        else:
            kw_is_tensor[k] = False

    def reconstruct(t_list):
        t_iter = iter(t_list)
        new_args = tuple(next(t_iter) if is_t else a for is_t, a in zip(arg_is_tensor, args))
        new_kwargs = {k: next(t_iter) if kw_is_tensor[k] else v for k, v in kwargs.items()}
        return new_args, new_kwargs

    return tensor_list, reconstruct


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
    if kernel_name != "unknown" and _is_first_tensor_cpu(*args, **kwargs):
        _DISPATCH_STATS["pytorch"] += 1
        _LAST_DISPATCH_BACKEND[kernel_name] = "pytorch"
        return pytorch_fn(*args, **kwargs)

    forced = _FORCED_BACKEND

    if FORCE_PYTORCH_FALLBACK or forced == "pytorch":
        _DISPATCH_STATS["pytorch"] += 1
        _LAST_DISPATCH_BACKEND[kernel_name] = "pytorch"
        return pytorch_fn(*args, **kwargs)

    has_grad = any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args) or any(
        isinstance(val, torch.Tensor) and val.requires_grad for val in kwargs.values()
    )

    if has_grad:
        tensor_list, reconstruct = _pack_for_autograd(args, kwargs)

        if forced == "cuda":
            if cuda_fn is None:
                _DISPATCH_STATS["cuda_failures"] += 1
                raise RuntimeError(f"[{kernel_name}] Backend forced to 'cuda', but no CUDA C++ implementation is available.")
            try:
                res = _AutogradBridge.apply(cuda_fn, pytorch_fn, reconstruct, *tensor_list)
                if res is not None or allow_none:
                    _DISPATCH_STATS["cuda"] += 1
                    _LAST_DISPATCH_BACKEND[kernel_name] = "cuda"
                    return res
            except Exception as e:
                _DISPATCH_STATS["cuda_failures"] += 1
                raise RuntimeError(f"[{kernel_name}] Backend forced to 'cuda', but execution failed: {e}") from e

        if forced == "triton":
            if triton_fn is None:
                _DISPATCH_STATS["triton_failures"] += 1
                raise RuntimeError(f"[{kernel_name}] Backend forced to 'triton', but no Triton implementation is available.")
            try:
                res = _AutogradBridge.apply(triton_fn, pytorch_fn, reconstruct, *tensor_list)
                if res is not None or allow_none:
                    _DISPATCH_STATS["triton"] += 1
                    _LAST_DISPATCH_BACKEND[kernel_name] = "triton"
                    return res
            except Exception as e:
                _DISPATCH_STATS["triton_failures"] += 1
                raise RuntimeError(f"[{kernel_name}] Backend forced to 'triton', but execution failed: {e}") from e

        if cuda_fn is not None:
            try:
                res = _AutogradBridge.apply(cuda_fn, pytorch_fn, reconstruct, *tensor_list)
                if res is not None or allow_none:
                    _DISPATCH_STATS["cuda"] += 1
                    _LAST_DISPATCH_BACKEND[kernel_name] = "cuda"
                    return res
            except Exception as e:
                _DISPATCH_STATS["cuda_failures"] += 1
                logger.debug(f"[{kernel_name}] CUDA C++ autograd path failed: {e}. Falling back.")

        if triton_fn is not None:
            try:
                res = _AutogradBridge.apply(triton_fn, pytorch_fn, reconstruct, *tensor_list)
                if res is not None or allow_none:
                    _DISPATCH_STATS["triton"] += 1
                    _LAST_DISPATCH_BACKEND[kernel_name] = "triton"
                    return res
            except Exception as e:
                _DISPATCH_STATS["triton_failures"] += 1
                logger.debug(f"[{kernel_name}] Triton autograd path failed: {e}. Falling back.")

        _DISPATCH_STATS["pytorch"] += 1
        _LAST_DISPATCH_BACKEND[kernel_name] = "pytorch"
        return pytorch_fn(*args, **kwargs)

    if forced == "cuda":
        if cuda_fn is None:
            _DISPATCH_STATS["cuda_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'cuda', but no CUDA C++ implementation is available.")
        try:
            res = cuda_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["cuda"] += 1
                _LAST_DISPATCH_BACKEND[kernel_name] = "cuda"
                return res
        except Exception as e:
            _DISPATCH_STATS["cuda_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'cuda', but execution failed: {e}") from e

    if forced == "triton":
        if triton_fn is None:
            _DISPATCH_STATS["triton_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'triton', but no Triton implementation is available.")
        try:
            res = triton_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["triton"] += 1
                _LAST_DISPATCH_BACKEND[kernel_name] = "triton"
                return res
        except Exception as e:
            _DISPATCH_STATS["triton_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'triton', but execution failed: {e}") from e

    import os
    import sys
    debug_dispatch = os.environ.get("NEURALOPS_DEBUG_DISPATCH") == "1"
    if debug_dispatch:
        print(f"[dispatch] {kernel_name}: cuda={'yes' if cuda_fn else 'no'} triton={'yes' if triton_fn else 'no'} pytorch=yes", file=sys.stderr)

    if cuda_fn is not None:
        try:
            res = cuda_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["cuda"] += 1
                _LAST_DISPATCH_BACKEND[kernel_name] = "cuda"
                if debug_dispatch:
                    print(f"[dispatch] {kernel_name}: executed branch = CUDA", file=sys.stderr)
                return res
        except Exception as e:
            _DISPATCH_STATS["cuda_failures"] += 1
            if debug_dispatch:
                print(f"[dispatch] {kernel_name}: CUDA failed with {type(e).__name__}: {e}", file=sys.stderr)
            logger.debug(f"[{kernel_name}] CUDA C++ path failed: {e}. Falling back to Triton / PyTorch.")
            _LAST_DISPATCH_ERROR[kernel_name] = str(e)
            if os.environ.get("NEURALOPS_STRICT_CUDA") == "1":
                raise

    if triton_fn is not None:
        try:
            res = triton_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["triton"] += 1
                _LAST_DISPATCH_BACKEND[kernel_name] = "triton"
                if debug_dispatch:
                    print(f"[dispatch] {kernel_name}: executed branch = TRITON", file=sys.stderr)
                return res
        except Exception as e:
            _DISPATCH_STATS["triton_failures"] += 1
            if debug_dispatch:
                print(f"[dispatch] {kernel_name}: TRITON failed with {type(e).__name__}: {e}", file=sys.stderr)
            logger.debug(f"[{kernel_name}] Triton path failed: {e}. Falling back to PyTorch.")

    _DISPATCH_STATS["pytorch"] += 1
    _LAST_DISPATCH_BACKEND[kernel_name] = "pytorch"
    if debug_dispatch:
        print(f"[dispatch] {kernel_name}: executed branch = PYTORCH_FALLBACK", file=sys.stderr)
    return pytorch_fn(*args, **kwargs)


__all__ = [
    "FORCE_PYTORCH_FALLBACK",
    "dispatch",
    "force_kernel_backend",
    "get_dispatch_stats",
    "reset_dispatch_stats",
    "get_last_dispatch_backend",
    "get_last_dispatch_error",
]
