"""Production Triton kernels for elementwise SwiGLU, Fast GELU, and RMSNorm.

Falls back transparently to PyTorch when Triton or CUDA is unavailable.
Public API:
    fused_swiglu(x, gate)              -> SiLU(gate) * x
    fast_gelu(x)                       -> tanh-approximated GELU
    fused_rmsnorm(x, weight, eps=1e-6) -> x / sqrt(mean(x^2) + eps) * weight

Backward-compatible aliases are kept at the bottom of this module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl

__all__ = [
    "fused_swiglu",
    "fast_gelu",
    "fused_rmsnorm",
    # backward-compat aliases
    "fused_fast_gelu",
]


# ═══════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_SWIGLU_BLOCK = 1024
_RMSNORM_BLOCK = 1024


def _cuda_ok(*tensors: torch.Tensor) -> bool:
    """True if Triton is available and every tensor is a CUDA float tensor."""
    if not (HAS_TRITON and tl is not None and triton is not None):
        return False
    return all(
        t.is_cuda and t.dtype in _FLOAT_DTYPES for t in tensors
    )


# ═══════════════════════════════════════════════════════════════════
#  SwiGLU
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON and tl is not None:

    @triton.jit
    def _swiglu_kernel(
        X, GATE, OUT,
        N_ELEMENTS,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_ELEMENTS

        x = tl.load(X + offs, mask=mask, other=0.0)
        g = tl.load(GATE + offs, mask=mask, other=0.0)

        # Numerically stable SiLU: sigmoid via tl.sigmoid.
        silu = g * tl.sigmoid(g)
        out = silu * x

        tl.store(OUT + offs, out, mask=mask)


def fused_swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute SiLU(gate) * x elementwise.

    Uses a Triton kernel when both tensors are CUDA floats; otherwise
    falls back to ``F.silu(gate) * x`` in pure PyTorch.
    """
    if x.shape != gate.shape:
        raise ValueError(
            f"fused_swiglu: shape mismatch x={tuple(x.shape)} gate={tuple(gate.shape)}"
        )

    if _cuda_ok(x, gate):
        x_c = x.contiguous()
        g_c = gate.contiguous()
        out = torch.empty_like(x_c)
        n = x_c.numel()
        grid = (triton.cdiv(n, _SWIGLU_BLOCK),)
        _swiglu_kernel[grid](x_c, g_c, out, n, BLOCK_SIZE=_SWIGLU_BLOCK)
        return out

    return F.silu(gate) * x


# ═══════════════════════════════════════════════════════════════════
#  Fast GELU
# ═══════════════════════════════════════════════════════════════════

def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Tanh-approximated GELU.

    Matches ``F.gelu(x, approximate='tanh')``. Kept as a thin wrapper
    rather than a Triton kernel because the operation is trivially
    memory-bound and PyTorch already emits a single fused kernel for it.
    """
    return F.gelu(x, approximate="tanh")


# ═══════════════════════════════════════════════════════════════════
#  RMSNorm
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON and tl is not None:

    @triton.jit
    def _rmsnorm_kernel(
        X, W, OUT,
        N,
        eps,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X + row * N + cols, mask=mask, other=0.0)

        # Single-pass reduction: sum of squares.
        sq_sum = tl.sum(x * x, axis=0)
        inv_rms = 1.0 / tl.sqrt(sq_sum / N + eps)

        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=mask, other=0.0)
            out = x * inv_rms * w
        else:
            out = x * inv_rms

        tl.store(OUT + row * N + cols, out, mask=mask)


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight.

    Normalizes over the last dimension. ``weight`` is optional.

    Uses a Triton kernel when x is a 2-D CUDA float tensor with a
    power-of-two-friendly last dim and Triton is available. Otherwise
    falls back to a pure PyTorch implementation.
    """

    orig_shape = x.shape
    if x.dim() < 2:
        x2 = x.unsqueeze(0)
    else:
        x2 = x
    N = x2.shape[-1]
    M = x2.numel() // N
    x2d = x2.reshape(M, N)

    has_weight = weight is not None

    can_use_triton = (
        _cuda_ok(x2d)
        and (weight is None or _cuda_ok(weight))
        and N <= _RMSNORM_BLOCK * 8   # avoid silly tile sizes
    )

    if can_use_triton:
        x_c = x2d.contiguous()
        w_c = weight.contiguous() if has_weight else x_c  # unused when HAS_WEIGHT=False
        out = torch.empty_like(x_c)

        BLOCK_N = triton.next_power_of_2(N)
        grid = (M,)
        _rmsnorm_kernel[grid](
            x_c, w_c, out,
            N, eps,
            HAS_WEIGHT=has_weight,
            BLOCK_N=BLOCK_N,
        )
        return out.reshape(orig_shape)

    # PyTorch fallback.
    variance = x2d.pow(2).mean(-1, keepdim=True)
    y = x2d * torch.rsqrt(variance + eps)
    if has_weight:
        y = y * weight
    return y.reshape(orig_shape)


# ═══════════════════════════════════════════════════════════════════
#  Backward-compatibility aliases
# ═══════════════════════════════════════════════════════════════════

# Old name → new name. Kept so existing imports do not break.
fused_fast_gelu = fast_gelu