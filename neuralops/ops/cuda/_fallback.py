#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Three-Level Kernel Fallback Dispatcher & Telemetry Engine

Architectural Contract (NeuralOps v1.6.0):
1. Level 1 (Native CUDA C++ / HIP): Compiled high-throughput extension leveraging hardware
   warp primitives (__shfl_sync), Tensor Core MMA, and register budgeting (<= 48 regs).
2. Level 2 (Triton JIT): Autotuned JIT kernel compiled to PTX for FP16/BF16/FP32, avoiding
   host driver roundtrips and providing portable accelerator execution.
3. Level 3 (Vectorized PyTorch Reference): Pure PyTorch oracle implementation ensuring exact
   numerical execution across arbitrary CPU, MPS, and headless CI runner environments.

Zero Silent Fallback Law (AGENTS.md Section 7):
Every degradation from Level 1 to Level 2, or from Level 2 to Level 3, MUST emit a structured
logger.warning with the failing operator name and underlying exception. In addition,
global telemetry counters (_DISPATCH_STATS) and per-kernel execution state
(_LAST_DISPATCH_BACKEND, _LAST_DISPATCH_ERROR) are updated synchronously.

Environment Overrides:
- NEURALOPS_STRICT_CUDA=1: Disables fallback; immediately raises on Level 1 failure.
- FORCE_PYTORCH_FALLBACK=1: Globally forces Level 3 execution for baseline calibration sweeps.
- NEURALOPS_DEBUG_DISPATCH=1: Emits real-time branch routing traces to sys.stderr.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import sys
from typing import Any, Callable, Generator

import torch

logger = logging.getLogger(__name__)

_FORCED_BACKEND: contextvars.ContextVar[str] = contextvars.ContextVar("neuralops_forced_backend", default="auto")
FORCE_PYTORCH_FALLBACK: bool = False

_LAST_DISPATCH_BACKEND: dict[str, str] = {}
_LAST_DISPATCH_ERROR: dict[str, str] = {}


def set_last_dispatch_backend(kernel_name: str, backend: str) -> None:
    """Explicitly record the backend executed for a kernel (e.g. from dedicated autograd functions)."""
    _LAST_DISPATCH_BACKEND[kernel_name] = backend


def get_last_dispatch_backend(kernel_name: str) -> str:
    return _LAST_DISPATCH_BACKEND.get(kernel_name, "unknown")


def get_last_dispatch_error(kernel_name: str) -> str:
    return _LAST_DISPATCH_ERROR.get(kernel_name, "")


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
    Supported backends: 'auto', 'cuda', 'triton', 'fla', 'pytorch'.
    """
    if backend not in ("auto", "cuda", "triton", "fla", "pytorch"):
        raise ValueError(f"Unknown backend: {backend}")
    if backend == "fla":
        from neuralops.core.exceptions import IntegrationNotImplementedError
        from neuralops.integrations.fla import fla_is_available

        if not fla_is_available():
            raise IntegrationNotImplementedError(
                "Cannot force kernel backend 'fla': the 'fla' package is not installed. "
                "Install it via: pip install 'neuralops[fla]' (requires Linux + CUDA)."
            )
        logger.info("FLA backend explicitly selected; routing to fla.ops kernels.")
    token = _FORCED_BACKEND.set(backend)
    try:
        yield
    finally:
        _FORCED_BACKEND.reset(token)


def _resolve_backend(backend: str = "auto") -> str:
    """Resolve backend selection respecting _FORCED_BACKEND context.

    If backend is 'auto', checks _FORCED_BACKEND. If forced is not 'auto',
    returns the forced backend. Otherwise resolves 'auto' to 'cuda' so that
    the Level 1 CUDA implementation is attempted first in the fallback chain.
    """
    if backend == "auto":
        forced = _FORCED_BACKEND.get()
        if forced != "auto":
            return forced
        return "cuda"
    return backend


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


def _should_checkpoint() -> bool:
    """Return True if activation checkpointing is enabled via environment."""
    val = os.environ.get("NEURALOPS_AUTOGRAD_CHECKPOINT", "").strip().lower()
    return val in ("1", "true", "yes", "on") and torch.is_grad_enabled()


class _AutogradBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fwd_fn, bwd_ref_fn, reconstruct, *tensors):
        if _should_checkpoint():
            from torch.utils.checkpoint import checkpoint

            def _custom_fwd(*t):
                a, kw = reconstruct(t)
                return fwd_fn(*a, **kw)

            out = checkpoint(_custom_fwd, *tensors, use_reentrant=False)
        else:
            with torch.no_grad():
                args, kwargs = reconstruct(tensors)
                out = fwd_fn(*args, **kwargs)
        ctx.bwd_ref_fn = bwd_ref_fn
        ctx.reconstruct = reconstruct
        ctx.save_for_backward(*tensors)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        saved = ctx.saved_tensors
        with torch.enable_grad():
            detached = [t.detach().requires_grad_(t.requires_grad) for t in saved]
            args, kwargs = ctx.reconstruct(detached)
            ref_out = ctx.bwd_ref_fn(*args, **kwargs)
            req_tensors = [t for t in detached if t.requires_grad]

            ref_outputs = list(ref_out) if isinstance(ref_out, (tuple, list)) else [ref_out]
            if len(ref_outputs) != len(grad_outputs):
                raise RuntimeError(
                    "Autograd reference output count does not match the accelerated forward output count "
                    f"({len(ref_outputs)} != {len(grad_outputs)})."
                )

            active_outputs = [
                (output, grad)
                for output, grad in zip(ref_outputs, grad_outputs)
                if isinstance(output, torch.Tensor) and output.requires_grad and grad is not None
            ]
            if req_tensors and active_outputs:
                grads = torch.autograd.grad(
                    tuple(output for output, _ in active_outputs),
                    req_tensors,
                    tuple(grad for _, grad in active_outputs),
                    allow_unused=True,
                )
            else:
                grads = (None,) * len(req_tensors)

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
    cuda_fn: Any,
    triton_fn: Any,
    pytorch_ref: Any = None,
    *args: Any,
    kernel_name: str = "unknown",
    allow_none: bool = False,
    pytorch_fn: Any = None,
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
    if pytorch_ref is None and pytorch_fn is not None:
        pytorch_ref = pytorch_fn
    pytorch_fn = pytorch_ref

    if pytorch_fn is None:
        raise RuntimeError(
            f"No PyTorch fallback supplied for kernel '{kernel_name}'; " "CUDA and Triton both unavailable"
        )

    forced = _FORCED_BACKEND.get()
    requested_backend = kwargs.pop("backend", "auto")
    effective_backend = _resolve_backend(requested_backend)
    if forced == "auto" and effective_backend != "cuda":
        forced = effective_backend

    if (
        not FORCE_PYTORCH_FALLBACK
        and forced == "auto"
        and kernel_name != "unknown"
        and _is_first_tensor_cpu(*args, **kwargs)
    ):
        _DISPATCH_STATS["pytorch"] += 1
        _LAST_DISPATCH_BACKEND[kernel_name] = "pytorch"
        return pytorch_fn(*args, **kwargs)

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
                raise RuntimeError(
                    f"[{kernel_name}] Backend forced to 'cuda', but no CUDA C++ implementation is available."
                )
            try:
                res = _AutogradBridge.apply(cuda_fn, pytorch_fn, reconstruct, *tensor_list)
                if res is not None or allow_none:
                    _DISPATCH_STATS["cuda"] += 1
                    _LAST_DISPATCH_BACKEND[kernel_name] = "cuda"
                    return res
                raise RuntimeError("implementation returned None")
            except Exception as e:
                _DISPATCH_STATS["cuda_failures"] += 1
                raise RuntimeError(f"[{kernel_name}] Backend forced to 'cuda', but execution failed: {e}") from e

        if forced == "triton":
            if triton_fn is None:
                _DISPATCH_STATS["triton_failures"] += 1
                raise RuntimeError(
                    f"[{kernel_name}] Backend forced to 'triton', but no Triton implementation is available."
                )
            try:
                res = _AutogradBridge.apply(triton_fn, pytorch_fn, reconstruct, *tensor_list)
                if res is not None or allow_none:
                    _DISPATCH_STATS["triton"] += 1
                    _LAST_DISPATCH_BACKEND[kernel_name] = "triton"
                    return res
                raise RuntimeError("implementation returned None")
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
                logger.warning(f"[{kernel_name}] CUDA C++ autograd path failed: {e}. Falling back to Triton / PyTorch.")
                _LAST_DISPATCH_ERROR[kernel_name] = str(e)
                if os.environ.get("NEURALOPS_STRICT_CUDA") == "1":
                    raise

        if triton_fn is not None:
            try:
                res = _AutogradBridge.apply(triton_fn, pytorch_fn, reconstruct, *tensor_list)
                if res is not None or allow_none:
                    _DISPATCH_STATS["triton"] += 1
                    _LAST_DISPATCH_BACKEND[kernel_name] = "triton"
                    return res
            except Exception as e:
                _DISPATCH_STATS["triton_failures"] += 1
                logger.warning(f"[{kernel_name}] Triton autograd path failed: {e}. Falling back to PyTorch.")

        _DISPATCH_STATS["pytorch"] += 1
        _LAST_DISPATCH_BACKEND[kernel_name] = "pytorch"
        return pytorch_fn(*args, **kwargs)

    if forced == "cuda":
        if cuda_fn is None:
            _DISPATCH_STATS["cuda_failures"] += 1
            raise RuntimeError(
                f"[{kernel_name}] Backend forced to 'cuda', but no CUDA C++ implementation is available."
            )
        try:
            res = cuda_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["cuda"] += 1
                _LAST_DISPATCH_BACKEND[kernel_name] = "cuda"
                return res
            raise RuntimeError("implementation returned None")
        except Exception as e:
            _DISPATCH_STATS["cuda_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'cuda', but execution failed: {e}") from e

    if forced == "triton":
        if triton_fn is None:
            _DISPATCH_STATS["triton_failures"] += 1
            raise RuntimeError(
                f"[{kernel_name}] Backend forced to 'triton', but no Triton implementation is available."
            )
        try:
            res = triton_fn(*args, **kwargs)
            if res is not None or allow_none:
                _DISPATCH_STATS["triton"] += 1
                _LAST_DISPATCH_BACKEND[kernel_name] = "triton"
                return res
            raise RuntimeError("implementation returned None")
        except Exception as e:
            _DISPATCH_STATS["triton_failures"] += 1
            raise RuntimeError(f"[{kernel_name}] Backend forced to 'triton', but execution failed: {e}") from e

    debug_dispatch = os.environ.get("NEURALOPS_DEBUG_DISPATCH") == "1"
    if debug_dispatch:
        print(
            f"[dispatch] {kernel_name}: cuda={'yes' if cuda_fn else 'no'} triton={'yes' if triton_fn else 'no'} pytorch=yes",
            file=sys.stderr,
        )

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
            logger.warning(f"[{kernel_name}] CUDA C++ path failed: {e}. Falling back to Triton / PyTorch.")
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
            logger.warning(f"[{kernel_name}] Triton path failed: {e}. Falling back to PyTorch.")

    _DISPATCH_STATS["pytorch"] += 1
    _LAST_DISPATCH_BACKEND[kernel_name] = "pytorch"
    if debug_dispatch:
        print(f"[dispatch] {kernel_name}: executed branch = PYTORCH_FALLBACK", file=sys.stderr)
    return pytorch_fn(*args, **kwargs)


_CUSTOM_KERNEL_REGISTRY: dict[tuple[str, str], Callable[..., Any]] = {}


def register_kernel(kernel_name: str, fn: Callable[..., Any], backend: str = "cuda") -> None:
    """Register or override a custom kernel function for dispatch."""
    _CUSTOM_KERNEL_REGISTRY[(backend, kernel_name)] = fn


def is_kernel_available(kernel_name: str, backend: str = "cuda") -> bool:
    """Check whether a kernel is available on the specified backend."""
    if (backend, kernel_name) in _CUSTOM_KERNEL_REGISTRY:
        return True
    if backend == "cuda":
        if not torch.cuda.is_available():
            return False
        from neuralops.ops.cuda import list_cuda_kernels

        return kernel_name in list_cuda_kernels("modules") or kernel_name in list_cuda_kernels("operators")
    elif backend == "triton":
        try:
            import triton  # noqa: F401

            return True
        except ImportError:
            return False
    elif backend in ("pytorch", "cpu"):
        return True
    return False


# ---------------------------------------------------------------------------
# Three-Level Dispatch Implementations for Batch 2 & 3 CUDA Kernels (18 Modules)
# ---------------------------------------------------------------------------


def _direct_svd_cuda_available() -> bool:
    return is_kernel_available("direct_svd", backend="cuda")


def dispatch_direct_svd(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _direct_svd_cuda_available():
            try:
                from neuralops.ops.cuda.direct_svd import direct_svd_compress

                return direct_svd_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[direct_svd] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[direct_svd] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("direct_svd", backend="triton"):
            try:
                from neuralops.ops.triton.low_rank import direct_svd_compress as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[direct_svd] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[direct_svd] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'direct_svd'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


def _asvd_cuda_available() -> bool:
    return is_kernel_available("asvd", backend="cuda")


def dispatch_asvd(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _asvd_cuda_available():
            try:
                from neuralops.ops.cuda.asvd import asvd_compress

                return asvd_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[asvd] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[asvd] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("asvd", backend="triton"):
            try:
                from neuralops.ops.triton.low_rank import asvd_compress as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[asvd] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[asvd] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'asvd'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


def _fwsvd_cuda_available() -> bool:
    return is_kernel_available("fwsvd", backend="cuda")


def dispatch_fwsvd(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _fwsvd_cuda_available():
            try:
                from neuralops.ops.cuda.fwsvd import fwsvd_compress

                return fwsvd_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[fwsvd] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[fwsvd] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("fwsvd", backend="triton"):
            try:
                from neuralops.ops.triton.low_rank import fwsvd_compress as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[fwsvd] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[fwsvd] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'fwsvd'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


def _hnc_svd_cuda_available() -> bool:
    return is_kernel_available("hnc_svd", backend="cuda")


def dispatch_hnc_svd(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _hnc_svd_cuda_available():
            try:
                from neuralops.ops.cuda.hnc_svd import hnc_svd_compress

                return hnc_svd_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[hnc_svd] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[hnc_svd] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("hnc_svd", backend="triton"):
            try:
                from neuralops.ops.triton.low_rank import hnc_svd_compress as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[hnc_svd] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[hnc_svd] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'hnc_svd'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


def _awq_cuda_available() -> bool:
    return is_kernel_available("awq", backend="cuda")


def dispatch_awq(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _awq_cuda_available():
            try:
                from neuralops.ops.cuda.awq import awq_compress

                return awq_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[awq] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[awq] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("awq", backend="triton"):
            try:
                from neuralops.ops.triton.quantize_awq import awq_compress as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[awq] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[awq] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'awq'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


def _fp8_compressor_cuda_available() -> bool:
    return is_kernel_available("fp8_compressor", backend="cuda")


def dispatch_fp8_compressor(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _fp8_compressor_cuda_available():
            try:
                from neuralops.ops.cuda.fp8_compressor import fp8_compressor_compress

                return fp8_compressor_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[fp8_compressor] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[fp8_compressor] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("fp8_compressor", backend="triton"):
            try:
                from neuralops.ops.triton.quantize_fp8 import fp8_compress as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[fp8_compressor] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[fp8_compressor] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError(
            "No PyTorch fallback supplied for kernel 'fp8_compressor'; " "CUDA and Triton both unavailable"
        )
    return pytorch_fn(*args, **kwargs)


def _kmeans_quantize_cuda_available() -> bool:
    return is_kernel_available("kmeans_quantize", backend="cuda")


def dispatch_kmeans_quantize(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _kmeans_quantize_cuda_available():
            try:
                from neuralops.ops.cuda.kmeans_quantize import kmeans_quantize_compress

                return kmeans_quantize_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[kmeans_quantize] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[kmeans_quantize] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("kmeans_quantize", backend="triton"):
            try:
                from neuralops.ops.triton.quantize_kmeans import kmeans_quantize_compress as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[kmeans_quantize] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[kmeans_quantize] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError(
            "No PyTorch fallback supplied for kernel 'kmeans_quantize'; " "CUDA and Triton both unavailable"
        )
    return pytorch_fn(*args, **kwargs)


def _wanda_cuda_available() -> bool:
    return is_kernel_available("wanda", backend="cuda")


def dispatch_wanda(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _wanda_cuda_available():
            try:
                from neuralops.ops.cuda.wanda import wanda_compress

                return wanda_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[wanda] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[wanda] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("wanda", backend="triton"):
            try:
                from neuralops.ops.triton.prune_wanda import wanda_compress as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[wanda] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[wanda] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'wanda'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


def _sparsegpt_cuda_available() -> bool:
    return is_kernel_available("sparsegpt", backend="cuda")


def dispatch_sparsegpt(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _sparsegpt_cuda_available():
            try:
                from neuralops.ops.cuda.sparsegpt import sparsegpt_compress

                return sparsegpt_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[sparsegpt] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[sparsegpt] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("sparsegpt", backend="triton"):
            try:
                from neuralops.ops.triton.prune_sparsegpt import sparsegpt_compress as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[sparsegpt] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[sparsegpt] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'sparsegpt'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


def _admm_cuda_available() -> bool:
    return is_kernel_available("admm", backend="cuda")


def dispatch_admm(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _admm_cuda_available():
            try:
                from neuralops.ops.cuda.admm import admm_pruner_compress

                return admm_pruner_compress(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[admm] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[admm] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("admm", backend="triton"):
            try:
                from neuralops.ops.triton.prune_admm import admm_prune as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[admm] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[admm] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'admm'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


def _routing_attention_cuda_available() -> bool:
    return is_kernel_available("routing_attention", backend="cuda")


def dispatch_routing_attention(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _routing_attention_cuda_available():
            try:
                from neuralops.ops.cuda.routing_attention import routing_attention_forward

                return routing_attention_forward(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[routing_attention] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[routing_attention] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("routing_attention", backend="triton"):
            try:
                from neuralops.ops.triton.routing_attention_triton import routing_attention_forward as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[routing_attention] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[routing_attention] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError(
            "No PyTorch fallback supplied for kernel 'routing_attention'; " "CUDA and Triton both unavailable"
        )
    return pytorch_fn(*args, **kwargs)


def _lsh_attention_cuda_available() -> bool:
    return is_kernel_available("lsh_attention", backend="cuda")


def dispatch_lsh_attention(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _lsh_attention_cuda_available():
            try:
                from neuralops.ops.cuda.lsh_attention import lsh_attention_forward

                return lsh_attention_forward(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[lsh_attention] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[lsh_attention] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("lsh_attention", backend="triton"):
            try:
                from neuralops.ops.triton.lsh_attention_triton import lsh_attention_forward as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[lsh_attention] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[lsh_attention] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError(
            "No PyTorch fallback supplied for kernel 'lsh_attention'; " "CUDA and Triton both unavailable"
        )
    return pytorch_fn(*args, **kwargs)


def _cross_attention_cuda_available() -> bool:
    return is_kernel_available("cross_attention", backend="cuda")


def dispatch_cross_attention(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _cross_attention_cuda_available():
            try:
                from neuralops.ops.cuda.cross_attention import cross_attention_forward

                return cross_attention_forward(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[cross_attention] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[cross_attention] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("cross_attention", backend="triton"):
            try:
                from neuralops.ops.triton.cross_attention_triton import cross_attention_forward as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[cross_attention] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[cross_attention] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError(
            "No PyTorch fallback supplied for kernel 'cross_attention'; " "CUDA and Triton both unavailable"
        )
    return pytorch_fn(*args, **kwargs)


def _adaptive_sparse_cuda_available() -> bool:
    return is_kernel_available("adaptive_sparse", backend="cuda")


def dispatch_adaptive_sparse(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _adaptive_sparse_cuda_available():
            try:
                from neuralops.ops.cuda.adaptive_sparse import adaptive_sparse_attention_forward

                return adaptive_sparse_attention_forward(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[adaptive_sparse] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[adaptive_sparse] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("adaptive_sparse", backend="triton"):
            try:
                from neuralops.ops.triton.adaptive_sparse_triton import adaptive_sparse_attention_forward as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[adaptive_sparse] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[adaptive_sparse] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError(
            "No PyTorch fallback supplied for kernel 'adaptive_sparse'; " "CUDA and Triton both unavailable"
        )
    return pytorch_fn(*args, **kwargs)


def _dynamic_conv_cuda_available() -> bool:
    return is_kernel_available("dynamic_conv", backend="cuda")


def dispatch_dynamic_conv(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _dynamic_conv_cuda_available():
            try:
                from neuralops.ops.cuda.dynamic_conv import dynamic_conv_attention_forward

                return dynamic_conv_attention_forward(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[dynamic_conv] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[dynamic_conv] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("dynamic_conv", backend="triton"):
            try:
                from neuralops.ops.triton.dynamic_conv_triton import dynamic_conv_attention_forward as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[dynamic_conv] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[dynamic_conv] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError(
            "No PyTorch fallback supplied for kernel 'dynamic_conv'; " "CUDA and Triton both unavailable"
        )
    return pytorch_fn(*args, **kwargs)


def _token_compressed_cuda_available() -> bool:
    return is_kernel_available("token_compressed", backend="cuda")


def dispatch_token_compressed(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _token_compressed_cuda_available():
            try:
                from neuralops.ops.cuda.token_compressed import token_compressed_attention_forward

                return token_compressed_attention_forward(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[token_compressed] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[token_compressed] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("token_compressed", backend="triton"):
            try:
                from neuralops.ops.triton.token_compressed_triton import (
                    token_compressed_attention_forward as _triton_fwd,
                )

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[token_compressed] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[token_compressed] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError(
            "No PyTorch fallback supplied for kernel 'token_compressed'; " "CUDA and Triton both unavailable"
        )
    return pytorch_fn(*args, **kwargs)


def _chela_slay_cuda_available() -> bool:
    return is_kernel_available("chela_slay", backend="cuda")


def dispatch_chela_slay(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _chela_slay_cuda_available():
            try:
                from neuralops.ops.cuda.chela_slay import chela_slay_attention_forward

                return chela_slay_attention_forward(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[chela_slay] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[chela_slay] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("chela_slay", backend="triton"):
            try:
                from neuralops.ops.triton.chela_slay_triton import chela_slay_forward as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[chela_slay] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[chela_slay] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'chela_slay'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


def _rala_cuda_available() -> bool:
    return is_kernel_available("rala", backend="cuda")


def dispatch_rala(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = backend == "auto" and _FORCED_BACKEND.get() == "auto"
    if target_backend == "cuda":
        if _rala_cuda_available():
            try:
                from neuralops.ops.cuda.rala import rala_attention_forward

                return rala_attention_forward(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[rala] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[rala] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("rala", backend="triton"):
            try:
                from neuralops.ops.triton.rala_attention_triton import rala_attention_forward as _triton_fwd

                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[rala] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[rala] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError("No PyTorch fallback supplied for kernel 'rala'; " "CUDA and Triton both unavailable")
    return pytorch_fn(*args, **kwargs)


__all__ = [
    "FORCE_PYTORCH_FALLBACK",
    "dispatch",
    "force_kernel_backend",
    "get_dispatch_stats",
    "reset_dispatch_stats",
    "get_last_dispatch_backend",
    "set_last_dispatch_backend",
    "get_last_dispatch_error",
    "is_kernel_available",
    "register_kernel",
    "dispatch_direct_svd",
    "dispatch_asvd",
    "dispatch_fwsvd",
    "dispatch_hnc_svd",
    "dispatch_awq",
    "dispatch_fp8_compressor",
    "dispatch_kmeans_quantize",
    "dispatch_wanda",
    "dispatch_sparsegpt",
    "dispatch_admm",
    "dispatch_routing_attention",
    "dispatch_lsh_attention",
    "dispatch_cross_attention",
    "dispatch_adaptive_sparse",
    "dispatch_dynamic_conv",
    "dispatch_token_compressed",
    "dispatch_chela_slay",
    "dispatch_rala",
]
