# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
PyTorch compatibility patches and shims (Rule 2.1 compliant - standard torch.amp).
"""

from __future__ import annotations

import contextlib
import logging
import math
import sys
from typing import Any

_logger = logging.getLogger("neuralops.core._torch_compat")


def patch_torch_compat() -> None:
    """Install PyTorch 1.12+ backward compatibility polyfills (SDPA, torch.compile, torch.amp)."""
    if "torch" not in sys.modules:
        return
    try:
        import torch
        import torch.nn.functional as F

        # 1. Polyfill F.scaled_dot_product_attention if missing (PyTorch < 2.0)
        if not hasattr(F, "scaled_dot_product_attention"):

            def _scaled_dot_product_attention(
                query: Any,
                key: Any,
                value: Any,
                attn_mask: Any = None,
                dropout_p: float = 0.0,
                is_causal: bool = False,
                scale: float | None = None,
            ) -> Any:
                scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
                scores = (query @ key.transpose(-2, -1)) * scale_factor
                if is_causal:
                    q_len, k_len = query.size(-2), key.size(-2)
                    causal_mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril(diagonal=0)
                    scores = scores.masked_fill(causal_mask.logical_not(), float("-inf"))
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        scores = scores.masked_fill(attn_mask.logical_not(), float("-inf"))
                    else:
                        scores = scores + attn_mask
                probs = torch.softmax(scores, dim=-1)
                if dropout_p > 0.0:
                    probs = torch.dropout(probs, dropout_p, train=True)
                return probs @ value

            F.scaled_dot_product_attention = _scaled_dot_product_attention

        # 2. Polyfill torch.compile if missing (PyTorch < 2.0)
        if not hasattr(torch, "compile"):

            def _dummy_compile(fn: Any = None, *args: Any, **kwargs: Any) -> Any:
                if fn is not None:
                    return fn

                def decorator(func: Any) -> Any:
                    return func

                return decorator

            torch.compile = _dummy_compile

        # 3. Polyfill torch.compiler if missing (PyTorch < 2.0)
        if not hasattr(torch, "compiler"):

            class _CompilerCompat:
                @staticmethod
                def is_compiling() -> bool:
                    return False

                @staticmethod
                def cudagraph_mark_step_begin() -> None:
                    pass

            torch.compiler = _CompilerCompat

        # 4. Polyfill torch.amp if missing (PyTorch < 2.0) - strictly avoiding deprecated amp per Rule 2.1
        if not hasattr(torch, "amp"):

            class _AmpCompat:
                @staticmethod
                def autocast(
                    device_type: str = "cuda",
                    dtype: Any = None,
                    enabled: bool = True,
                    cache_enabled: Any = None,
                ) -> Any:
                    if device_type == "cuda" and torch.cuda.is_available():
                        kwargs: dict[str, Any] = {"enabled": enabled}
                        if dtype is not None:
                            kwargs["dtype"] = dtype
                        if cache_enabled is not None:
                            kwargs["cache_enabled"] = cache_enabled
                        cuda_amp = getattr(torch, "cuda", None)
                        if cuda_amp and hasattr(cuda_amp, "amp"):
                            return cuda_amp.amp.autocast(**kwargs)
                        return contextlib.nullcontext()
                    elif hasattr(torch, "cpu") and hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast"):
                        kwargs = {"enabled": enabled}
                        if dtype is not None:
                            kwargs["dtype"] = dtype
                        return torch.cpu.amp.autocast(**kwargs)
                    return contextlib.nullcontext()

                cuda_mod = getattr(torch, "cuda", None)
                GradScaler = getattr(getattr(cuda_mod, "amp", None), "GradScaler", None) if cuda_mod else None

            torch.amp = _AmpCompat

        # 5. Fix torch.is_autocast_enabled to safely accept optional device_type
        if hasattr(torch, "is_autocast_enabled"):
            _orig_is_autocast_enabled = torch.is_autocast_enabled

            def _safe_is_autocast_enabled(device_type: str | None = None) -> bool:
                try:
                    if device_type is not None:
                        return _orig_is_autocast_enabled(device_type)
                    return _orig_is_autocast_enabled()
                except TypeError:
                    if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
                        return torch.is_autocast_cpu_enabled()
                    return _orig_is_autocast_enabled()

            torch.is_autocast_enabled = _safe_is_autocast_enabled

        # 6. Polyfill torch.nn.RMSNorm for PyTorch < 2.4
        if not hasattr(torch.nn, "RMSNorm"):

            class RMSNorm(torch.nn.Module):
                def __init__(self, normalized_shape: int | tuple[int, ...] | list[int], eps: float = 1e-6) -> None:
                    super().__init__()
                    if isinstance(normalized_shape, int):
                        normalized_shape = (normalized_shape,)
                    self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
                    self.eps = eps

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    variance = x.pow(2).mean(-1, keepdim=True)
                    x_normed = x * torch.rsqrt(variance + self.eps)
                    return self.weight * x_normed

            torch.nn.RMSNorm = RMSNorm

        # 7. Polyfill torch.cuda.OutOfMemoryError for PyTorch < 2.0
        if hasattr(torch, "cuda") and not hasattr(torch.cuda, "OutOfMemoryError"):

            class _OutOfMemoryError(RuntimeError):
                pass

            torch.cuda.OutOfMemoryError = _OutOfMemoryError

    except Exception as exc:
        _logger.debug("Failed to patch torch compatibility: %s", exc)


def patch_pascal_svd_if_needed() -> None:
    """Install a lazy SVD compatibility shim without probing CUDA at import time."""
    if "torch" not in sys.modules:
        return
    try:
        import torch

        if getattr(torch.linalg, "_pascal_patched", False):
            return

        original_svd = torch.linalg.svd
        capability_cache: dict[int, tuple[int, int]] = {}

        def _call_orig_svd(mat: Any, full: bool, drv: str | None, ot: Any) -> Any:
            if drv is not None:
                try:
                    if ot is not None:
                        return original_svd(mat, full_matrices=full, driver=drv, out=ot)
                    return original_svd(mat, full_matrices=full, driver=drv)
                except TypeError:
                    _logger.debug("Driver argument not supported by torch.linalg.svd; falling back")
            if ot is not None:
                return original_svd(mat, full_matrices=full, out=ot)
            return original_svd(mat, full_matrices=full)

        def _safe_pascal_linalg_svd(
            matrix: Any,
            full_matrices: bool = True,
            *,
            driver: str | None = None,
            out: Any = None,
        ) -> Any:
            """Use CPU SVD only for CUDA devices whose capability requires it."""
            if isinstance(matrix, torch.Tensor) and matrix.is_cuda:
                device_index = matrix.device.index or 0
                capability = capability_cache.get(device_index)
                if capability is None:
                    try:
                        capability = torch.cuda.get_device_capability(matrix.device)
                    except Exception as exc:
                        _logger.debug("Failed to query device capability for %s: %s", matrix.device, exc)
                        return _call_orig_svd(matrix, full_matrices, driver, out)
                    capability_cache[device_index] = capability

                if capability < (7, 0):
                    device = matrix.device
                    cpu_matrix = matrix.detach().cpu().float()
                    u_cpu, s_cpu, vh_cpu = _call_orig_svd(cpu_matrix, full_matrices, driver, None)
                    result_dtype = matrix.dtype if matrix.dtype != torch.float16 else torch.float32
                    u = u_cpu.to(device=device, dtype=result_dtype)
                    s = s_cpu.to(device=device, dtype=result_dtype)
                    vh = vh_cpu.to(device=device, dtype=result_dtype)
                    if out is not None:
                        out[0].copy_(u)
                        out[1].copy_(s)
                        out[2].copy_(vh)
                        return out
                    return u, s, vh

            return _call_orig_svd(matrix, full_matrices, driver, out)

        torch.linalg.svd = _safe_pascal_linalg_svd
        torch.linalg._pascal_patched = True
    except Exception as exc:
        _logger.debug("Failed to patch torch.linalg.svd for Pascal architecture: %s", exc)
