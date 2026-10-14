"""
Automatic Mixed Precision (AMP) and FP16 Numerical Safety Utilities.

Provides context managers and numerical guards to prevent underflows, overflows,
and NaNs in FP16/BF16 linear attention, cumulative summation, normalization,
and linear algebra operations (SVD, Cholesky, QR).
"""

from __future__ import annotations

import contextlib
import math
from typing import Generator, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def fp32_accumulator_context(device: Optional[torch.device] = None) -> Generator[None, None, None]:
    """
    Context manager that temporarily disables torch.amp.autocast to ensure
    sensitive mathematical operations (cumulative summations, normalizations,
    SVD decompositions) execute entirely in FP32 with 32-bit registers.
    """
    device_type = "cuda" if (torch.cuda.is_available() and (device is None or device.type == "cuda")) else "cpu"
    
    # Check if autocast is currently active
    was_autocast_enabled = torch.is_autocast_enabled(device_type) if hasattr(torch, "is_autocast_enabled") else False
    
    if was_autocast_enabled:
        with torch.amp.autocast(device_type=device_type, enabled=False):
            yield
    else:
        yield


def safe_linear_attn_denominator(
    denominator: torch.Tensor,
    eps: float = 1e-6,
    floor_val: float = 1e-4,
) -> torch.Tensor:
    """
    Stabilizes denominator tensor for linear attention:
        Z = sum(K, dim=-2)
        out = (Q @ KV) / (Q @ Z + eps)
    Clamps positive values away from zero and replaces non-finite values (NaN / Inf) with floor_val.
    """
    denom_clean = torch.nan_to_num(denominator, nan=floor_val, posinf=1e4, neginf=-1e4)
    sign = torch.sign(denom_clean)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return torch.where(
        denom_clean.abs() < eps,
        sign * floor_val,
        denom_clean,
    )


def safe_l2_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Safe L2 normalization in FP32 preventing divide-by-zero or zero-gradient freeze.
    """
    orig_dtype = x.dtype
    x_fp32 = x.float()
    norm = torch.norm(x_fp32, p=2, dim=dim, keepdim=True)
    norm_safe = torch.clamp(norm, min=eps)
    out = x_fp32 / norm_safe
    return out.to(orig_dtype)


def safe_svd(
    matrix: torch.Tensor,
    full_matrices: bool = False,
    driver: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes SVD in float32 / float64 with AMP autocast disabled,
    catching convergence failures with jitter fallback.
    """
    orig_dtype = matrix.dtype
    orig_device = matrix.device
    
    # Cast to float32 minimum (float64 if double precision requested)
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float64
    
    with torch.amp.autocast(device_type="cuda" if orig_device.type == "cuda" else "cpu", enabled=False):
        M = matrix.to(dtype=work_dtype)
        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=full_matrices, driver=driver)
        except RuntimeError:
            # Fallback with small diagonal jitter if convergence fails
            jitter = 1e-6 * torch.eye(M.shape[-2], M.shape[-1], device=M.device, dtype=work_dtype)
            U, S, Vh = torch.linalg.svd(M + jitter, full_matrices=full_matrices)

    return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)


def safe_cholesky(
    matrix: torch.Tensor,
    upper: bool = False,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Executes Cholesky decomposition in float32/float64 with AMP disabled and progressive diagonal jitter.
    """
    orig_dtype = matrix.dtype
    orig_device = matrix.device
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float64

    with torch.amp.autocast(device_type="cuda" if orig_device.type == "cuda" else "cpu", enabled=False):
        A = matrix.to(dtype=work_dtype)
        try:
            L = torch.linalg.cholesky(A, upper=upper)
        except RuntimeError:
            diag_jitter = jitter * torch.eye(A.shape[-1], device=A.device, dtype=work_dtype)
            try:
                L = torch.linalg.cholesky(A + diag_jitter, upper=upper)
            except RuntimeError:
                diag_jitter_heavy = (jitter * 100.0) * torch.eye(A.shape[-1], device=A.device, dtype=work_dtype)
                L = torch.linalg.cholesky(A + diag_jitter_heavy, upper=upper)

    return L.to(orig_dtype)


__all__ = [
    "fp32_accumulator_context",
    "safe_linear_attn_denominator",
    "safe_l2_normalize",
    "safe_svd",
    "safe_cholesky",
]
