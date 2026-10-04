#!/usr/bin/env python3
"""
Common utilities, validation, and PyTorch fallback implementations for shared Triton kernels.

Provides the unified `triton_dispatch` router with automatic capability checking:
  1. Triton package presence (HAS_TRITON)
  2. CUDA device availability and tensor placement
  3. Floating-point precision (fp16 / bf16 for Triton, fp32 -> PyTorch fallback)
  4. Supported head dimensions ({32, 64, 128, 256})
  5. Maximum sequence length threshold (<= 4096 tokens)
  6. Graceful, single-log fallback to mathematically equivalent PyTorch operations.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger("TritonKernels")

# ---------------------------------------------------------------------------
# Triton Environment Detection
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_HEAD_DIMS: set[int] = {32, 64, 128, 256}
MAX_TRITON_SEQ_LEN: int = 4096

_WARNED_REASONS: set[str] = set()


def log_fallback_reason(kernel_name: str, reason: str) -> None:
    """Log the fallback reason once per unique (kernel, reason) pair."""
    key = f"{kernel_name}:{reason}"
    if key not in _WARNED_REASONS:
        _WARNED_REASONS.add(key)
        logger.info("Triton kernel '%s' using PyTorch fallback: %s", kernel_name, reason)


def check_triton_suitability(
    tensor: torch.Tensor,
    kernel_name: str,
    head_dim: int | None = None,
    seq_len: int | None = None,
) -> tuple[bool, str | None]:
    """
    Validate whether the input tensor satisfies all requirements for Triton execution.
    Returns (is_suitable, failure_reason).
    """
    if not HAS_TRITON:
        return False, "triton package is not installed"

    if not torch.cuda.is_available():
        return False, "CUDA is not available on this system"

    if not tensor.is_cuda:
        return False, f"tensor is on {tensor.device}, expected CUDA"

    if tensor.dtype not in (torch.float16, torch.bfloat16):
        return (
            False,
            f"dtype {tensor.dtype} not in (float16, bfloat16); Triton kernels require fp16/bf16",
        )

    if head_dim is not None and head_dim not in SUPPORTED_HEAD_DIMS:
        return (
            False,
            f"head dimension {head_dim} not in supported set {SUPPORTED_HEAD_DIMS}",
        )

    if seq_len is not None and seq_len > MAX_TRITON_SEQ_LEN:
        return (
            False,
            f"sequence length {seq_len} exceeds max supported length {MAX_TRITON_SEQ_LEN}",
        )

    return True, None


# ===========================================================================
# PYTORCH FALLBACK IMPLEMENTATIONS
# ===========================================================================


def linear_attn_pytorch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    phi: str = "elu+1",
    eps: float = 1e-6,
    cos_weight: torch.Tensor | None = None,
    sin_weight: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Pure PyTorch fallback for linear attention with O(N) memory efficiency.
    Supports causal chunked prefix recurrence and bidirectional parallel reduction,
    with non-negative activation map phi and optional cos/sin positional modulation.
    """
    orig_dtype = q.dtype
    orig_dim = q.dim()

    if orig_dim == 3:
        # [B, N, D] -> [B, 1, N, D]
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    # 1. Feature Map phi
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    if phi == "elu+1":
        phi_q = F.elu(q_f) + 1.0
        phi_k = F.elu(k_f) + 1.0
    elif phi == "relu":
        phi_q = F.relu(q_f)
        phi_k = F.relu(k_f)
    elif phi == "identity":
        phi_q = q_f
        phi_k = k_f
    elif phi == "softmax_free":
        phi_q = F.elu(q_f) + 1.0
        phi_k = F.elu(k_f) + 1.0
    else:
        phi_q = F.elu(q_f) + 1.0
        phi_k = F.elu(k_f) + 1.0

    B, H, N, D_k = phi_q.shape
    D_v = v_f.shape[-1]

    # 2. Positional modulation (e.g. cosformer)
    has_modulation = cos_weight is not None and sin_weight is not None
    if has_modulation:
        cos_w = cos_weight.float()
        sin_w = sin_weight.float()
        # Broadcast weights to [B, H, N, 1] or [1, 1, N, 1]
        if cos_w.dim() == 1:
            cos_w = cos_w.view(1, 1, -1, 1)
            sin_w = sin_w.view(1, 1, -1, 1)
        elif cos_w.dim() == 2:
            cos_w = cos_w.unsqueeze(1).unsqueeze(-1)
            sin_w = sin_w.unsqueeze(1).unsqueeze(-1)
        elif cos_w.dim() == 3:
            cos_w = cos_w.unsqueeze(-1)
            sin_w = sin_w.unsqueeze(-1)

        q_c = phi_q * cos_w
        q_s = phi_q * sin_w
        k_c = phi_k * cos_w
        k_s = phi_k * sin_w

    eff_eps = max(eps, 1e-4) if orig_dtype == torch.float16 else eps

    if is_causal:
        # Memory-efficient streaming chunked prefix recurrence
        chunk_size = min(64, max(1, N))
        num_chunks = (N + chunk_size - 1) // chunk_size
        out = torch.empty((B, H, N, D_v), dtype=torch.float32, device=q.device)

        if has_modulation:
            S_cos = torch.zeros((B, H, D_k, D_v), dtype=torch.float32, device=q.device)
            S_sin = torch.zeros((B, H, D_k, D_v), dtype=torch.float32, device=q.device)
            Z_cos = torch.zeros((B, H, D_k), dtype=torch.float32, device=q.device)
            Z_sin = torch.zeros((B, H, D_k), dtype=torch.float32, device=q.device)

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)

                qc = q_c[:, :, start:end]
                qs = q_s[:, :, start:end]
                kc = k_c[:, :, start:end]
                ks = k_s[:, :, start:end]
                vc = v_f[:, :, start:end]

                # Inter-chunk projection
                num_inter = torch.matmul(qc, S_cos) + torch.matmul(qs, S_sin)
                denom_inter = torch.matmul(qc, Z_cos.unsqueeze(-1)).squeeze(-1) + torch.matmul(
                    qs, Z_sin.unsqueeze(-1)
                ).squeeze(-1)

                # Intra-chunk causal scores
                A_intra = torch.matmul(qc, kc.transpose(-1, -2)) + torch.matmul(
                    qs, ks.transpose(-1, -2)
                )
                causal_mask = torch.tril(
                    torch.ones(end - start, end - start, device=q.device, dtype=torch.bool)
                )
                A_intra = torch.where(causal_mask, A_intra, torch.zeros_like(A_intra))
                num_intra = torch.matmul(A_intra, vc)
                denom_intra = A_intra.sum(dim=-1)

                denom_val = denom_inter.unsqueeze(-1) + denom_intra.unsqueeze(-1)
                if phi in ("elu+1", "relu", "softmax_free"):
                    denom_safe = torch.clamp(denom_val, min=eff_eps)
                else:
                    denom_sign = torch.sign(denom_val)
                    denom_sign = torch.where(
                        denom_sign == 0, torch.ones_like(denom_sign), denom_sign
                    )
                    denom_safe = torch.where(
                        denom_val.abs() < eff_eps, eff_eps * denom_sign, denom_val
                    )

                chunk_out = (num_inter + num_intra) / denom_safe
                out[:, :, start:end] = chunk_out

                # State update
                S_cos = S_cos + torch.matmul(kc.transpose(-1, -2), vc)
                S_sin = S_sin + torch.matmul(ks.transpose(-1, -2), vc)
                Z_cos = Z_cos + kc.sum(dim=-2)
                Z_sin = Z_sin + ks.sum(dim=-2)
        else:
            S = torch.zeros((B, H, D_k, D_v), dtype=torch.float32, device=q.device)
            Z = torch.zeros((B, H, D_k), dtype=torch.float32, device=q.device)

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)

                qc = phi_q[:, :, start:end]
                kc = phi_k[:, :, start:end]
                vc = v_f[:, :, start:end]

                num_inter = torch.matmul(qc, S)
                denom_inter = torch.matmul(qc, Z.unsqueeze(-1)).squeeze(-1)

                A_intra = torch.matmul(qc, kc.transpose(-1, -2))
                causal_mask = torch.tril(
                    torch.ones(end - start, end - start, device=q.device, dtype=torch.bool)
                )
                A_intra = torch.where(causal_mask, A_intra, torch.zeros_like(A_intra))
                num_intra = torch.matmul(A_intra, vc)
                denom_intra = A_intra.sum(dim=-1)

                denom_val = denom_inter.unsqueeze(-1) + denom_intra.unsqueeze(-1)
                if phi in ("elu+1", "relu", "softmax_free"):
                    denom_safe = torch.clamp(denom_val, min=eff_eps)
                else:
                    denom_sign = torch.sign(denom_val)
                    denom_sign = torch.where(
                        denom_sign == 0, torch.ones_like(denom_sign), denom_sign
                    )
                    denom_safe = torch.where(
                        denom_val.abs() < eff_eps, eff_eps * denom_sign, denom_val
                    )

                chunk_out = (num_inter + num_intra) / denom_safe
                out[:, :, start:end] = chunk_out

                S = S + torch.matmul(kc.transpose(-1, -2), vc)
                Z = Z + kc.sum(dim=-2)

        res = out.to(orig_dtype)
    else:
        # Bidirectional global parallel reduction
        if has_modulation:
            S_cos = torch.matmul(k_c.transpose(-1, -2), v_f)
            S_sin = torch.matmul(k_s.transpose(-1, -2), v_f)
            Z_cos = k_c.sum(dim=-2, keepdim=True)
            Z_sin = k_s.sum(dim=-2, keepdim=True)

            num = torch.matmul(q_c, S_cos) + torch.matmul(q_s, S_sin)
            denom = torch.matmul(q_c, Z_cos.transpose(-1, -2)) + torch.matmul(
                q_s, Z_sin.transpose(-1, -2)
            )
            if phi in ("elu+1", "relu", "softmax_free"):
                denom_safe = torch.clamp(denom, min=eff_eps)
            else:
                denom_sign = torch.sign(denom)
                denom_sign = torch.where(denom_sign == 0, torch.ones_like(denom_sign), denom_sign)
                denom_safe = torch.where(denom.abs() < eff_eps, eff_eps * denom_sign, denom)
            res = (num / denom_safe).to(orig_dtype)
        else:
            KV = torch.matmul(phi_k.transpose(-1, -2), v_f)
            num = torch.matmul(phi_q, KV)
            Z = phi_k.sum(dim=-2, keepdim=True)
            denom = torch.matmul(phi_q, Z.transpose(-1, -2))
            if phi in ("elu+1", "relu", "softmax_free"):
                denom_safe = torch.clamp(denom, min=eff_eps)
            else:
                denom_sign = torch.sign(denom)
                denom_sign = torch.where(denom_sign == 0, torch.ones_like(denom_sign), denom_sign)
                denom_safe = torch.where(denom.abs() < eff_eps, eff_eps * denom_sign, denom)
            res = (num / denom_safe).to(orig_dtype)

    if orig_dim == 3:
        res = res.squeeze(1)
    return res


def sparse_attn_pytorch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor | None = None,
    block_size: int = 64,
    is_causal: bool = False,
    eps: float = 1e-6,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Pure PyTorch fallback for block-sparse attention.
    Expands block_mask to full attention matrix or falls back to causal/dense attention.
    """
    orig_dtype = q.dtype
    orig_dim = q.dim()

    if orig_dim == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    N_q, N_k = scores.shape[-2], scores.shape[-1]

    if block_mask is not None:
        # Expand block_mask to element-wise boolean mask
        bm = block_mask.to(torch.bool)
        if bm.dim() == 2:
            expanded_mask = bm.repeat_interleave(block_size, dim=0).repeat_interleave(
                block_size, dim=1
            )
            expanded_mask = expanded_mask[:N_q, :N_k]
            scores = scores.masked_fill(~expanded_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        elif bm.dim() == 4:
            expanded_mask = bm.repeat_interleave(block_size, dim=2).repeat_interleave(
                block_size, dim=3
            )
            expanded_mask = expanded_mask[..., :N_q, :N_k]
            scores = scores.masked_fill(~expanded_mask, float("-inf"))

    if is_causal:
        causal_mask = torch.tril(torch.ones(N_q, N_k, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    attn_probs = F.softmax(scores, dim=-1)
    attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
    out = torch.matmul(attn_probs, v.float()).to(orig_dtype)

    if orig_dim == 3:
        out = out.squeeze(1)
    return out


def low_rank_pytorch_fallback(
    x: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Pure PyTorch fallback for low-rank projection: y = (x @ V.T) @ U.T + bias.
    Handles flexible shape conventions for U and V.
    """
    # Contract with V
    if V.shape[-1] == x.shape[-1]:
        z = torch.matmul(x, V.transpose(-1, -2))
    else:
        z = torch.matmul(x, V)

    # Contract with U
    if U.shape[-1] == z.shape[-1]:
        y = torch.matmul(z, U.transpose(-1, -2))
    else:
        y = torch.matmul(z, U)

    if bias is not None:
        y = y + bias
    return y


def state_update_pytorch_fallback(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    eps: float = 1e-6,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch fallback for SSM state updates (Mamba/S4/RWKV recurrence).
    Computes discrete recurrent state scan: h_t = A_bar * h_{t-1} + B_bar * u_t, y_t = C * h_t + D * u_t.
    """
    # Shapes: u: [B, L, D], delta: [B, L, D] or [B, L, 1]
    B_sz, L_sz, D_sz = u.shape
    N_sz = A.shape[-1] if A.dim() >= 2 else 16

    if state is None:
        state = torch.zeros((B_sz, D_sz, N_sz), dtype=torch.float32, device=u.device)
    else:
        state = state.float().clone()

    y_list = []
    # Vectorized / sequential scan
    for t in range(L_sz):
        u_t = u[:, t, :].float()  # [B, D]
        delta_t = delta[:, t, :].float() if delta.dim() == 3 else delta.float()
        if delta_t.shape[-1] == 1:
            delta_t = delta_t.expand(-1, D_sz)

        # Discretization
        dA = torch.exp(delta_t.unsqueeze(-1) * A.float())  # [B, D, N]
        if B.dim() == 3:
            B_t = B[:, t, :].float()
            if B_t.dim() == 2 and B_t.shape[-1] == N_sz:
                dB = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, D, N]
            else:
                dB = delta_t.unsqueeze(-1) * B_t
        else:
            dB = delta_t.unsqueeze(-1) * B.float()

        state = state * dA + u_t.unsqueeze(-1) * dB

        if C.dim() == 3:
            C_t = C[:, t, :].float()
            if C_t.dim() == 2 and C_t.shape[-1] == N_sz:
                y_t = (state * C_t.unsqueeze(1)).sum(dim=-1)
            else:
                y_t = (state * C_t).sum(dim=-1)
        else:
            y_t = (state * C.float()).sum(dim=-1)

        if D is not None:
            y_t = y_t + u_t * D.float()
        y_list.append(y_t)

    y = torch.stack(y_list, dim=1).to(u.dtype)
    return y, state.to(u.dtype)


def gqa_pytorch_fallback(
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch fallback for GQA repeat and reshape along the head dimension.
    """
    if num_kv_heads == num_heads:
        return k, v

    n_rep = num_heads // num_kv_heads
    # Expected shape: [B, H_kv, N, D]
    B, H_kv, N, D = k.shape
    k_exp = k[:, :, None, :, :].expand(B, H_kv, n_rep, N, D).reshape(B, num_heads, N, D)
    v_exp = v[:, :, None, :, :].expand(B, H_kv, n_rep, N, D).reshape(B, num_heads, N, D)
    return k_exp, v_exp


# ===========================================================================
# UNIFIED DISPATCHER
# ===========================================================================


def triton_dispatch(kernel_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Unified router that executes high-performance Triton kernels when supported,
    or transparently falls back to optimized PyTorch implementations.
    """
    if kernel_name == "linear_attn":
        from neuralops.core.triton_kernels.linear_attn import fwd, is_supported

        q, k, v = args[0], args[1], args[2]
        head_dim = q.shape[-1]
        seq_len = q.shape[-2]
        suitable, reason = check_triton_suitability(
            q, kernel_name, head_dim=head_dim, seq_len=seq_len
        )
        if suitable and is_supported(q, k, v, **kwargs):
            try:
                return fwd(q, k, v, **kwargs)
            except Exception as e:
                log_fallback_reason(kernel_name, f"execution error: {e}")
                return linear_attn_pytorch_fallback(q, k, v, **kwargs)
        else:
            if not suitable:
                log_fallback_reason(kernel_name, reason or "unsupported configuration")
            else:
                log_fallback_reason(kernel_name, "kernel.is_supported returned False")
            return linear_attn_pytorch_fallback(q, k, v, **kwargs)

    elif kernel_name == "sparse_attn":
        from neuralops.core.triton_kernels.sparse_attn import fwd, is_supported

        q, k, v = args[0], args[1], args[2]
        head_dim = q.shape[-1]
        seq_len = q.shape[-2]
        suitable, reason = check_triton_suitability(
            q, kernel_name, head_dim=head_dim, seq_len=seq_len
        )
        if suitable and is_supported(q, k, v, **kwargs):
            try:
                return fwd(q, k, v, **kwargs)
            except Exception as e:
                log_fallback_reason(kernel_name, f"execution error: {e}")
                return sparse_attn_pytorch_fallback(q, k, v, **kwargs)
        else:
            if not suitable:
                log_fallback_reason(kernel_name, reason or "unsupported configuration")
            else:
                log_fallback_reason(kernel_name, "kernel.is_supported returned False")
            return sparse_attn_pytorch_fallback(q, k, v, **kwargs)

    elif kernel_name == "low_rank":
        from neuralops.core.triton_kernels.low_rank import fwd, is_supported

        kw = dict(kwargs)
        x = args[0]
        U = args[1] if len(args) > 1 else kw.pop("U", None)
        V = args[2] if len(args) > 2 else kw.pop("V", None)
        bias = args[3] if len(args) > 3 else kw.pop("bias", None)
        suitable, reason = check_triton_suitability(x, kernel_name)
        if suitable and is_supported(x, U, V, bias=bias, **kw):
            try:
                return fwd(x, U, V, bias=bias, **kw)
            except Exception as e:
                log_fallback_reason(kernel_name, f"execution error: {e}")
                return low_rank_pytorch_fallback(x, U, V, bias=bias, **kw)
        else:
            if not suitable:
                log_fallback_reason(kernel_name, reason or "unsupported configuration")
            else:
                log_fallback_reason(kernel_name, "kernel.is_supported returned False")
            return low_rank_pytorch_fallback(x, U, V, bias=bias, **kw)

    elif kernel_name == "state_update":
        from neuralops.core.triton_kernels.state_update import fwd, is_supported

        kw = dict(kwargs)
        u = args[0]
        delta = args[1] if len(args) > 1 else kw.pop("delta", None)
        A = args[2] if len(args) > 2 else kw.pop("A", None)
        B = args[3] if len(args) > 3 else kw.pop("B", None)
        C = args[4] if len(args) > 4 else kw.pop("C", None)
        D = args[5] if len(args) > 5 else kw.pop("D", None)
        suitable, reason = check_triton_suitability(u, kernel_name)
        if suitable and is_supported(u, delta, A, B, C, D=D, **kw):
            try:
                return fwd(u, delta, A, B, C, D=D, **kw)
            except Exception as e:
                log_fallback_reason(kernel_name, f"execution error: {e}")
                return state_update_pytorch_fallback(u, delta, A, B, C, D=D, **kw)
        else:
            if not suitable:
                log_fallback_reason(kernel_name, reason or "unsupported configuration")
            else:
                log_fallback_reason(kernel_name, "kernel.is_supported returned False")
            return state_update_pytorch_fallback(u, delta, A, B, C, D=D, **kw)

    elif kernel_name == "gqa":
        from neuralops.core.triton_kernels.gqa import fwd, is_supported

        kw = dict(kwargs)
        k = args[0]
        v = args[1]
        num_kv_heads = args[2] if len(args) > 2 else kw.pop("num_kv_heads", None)
        num_heads = args[3] if len(args) > 3 else kw.pop("num_heads", None)
        suitable, reason = check_triton_suitability(k, kernel_name)
        if suitable and is_supported(k, v, num_kv_heads, num_heads, **kw):
            try:
                return fwd(k, v, num_kv_heads, num_heads, **kw)
            except Exception as e:
                log_fallback_reason(kernel_name, f"execution error: {e}")
                return gqa_pytorch_fallback(k, v, num_kv_heads, num_heads, **kw)
        else:
            if not suitable:
                log_fallback_reason(kernel_name, reason or "unsupported configuration")
            else:
                log_fallback_reason(kernel_name, "kernel.is_supported returned False")
            return gqa_pytorch_fallback(k, v, num_kv_heads, num_heads, **kw)

    elif kernel_name is None:
        raise ValueError("triton_kernel cannot be None when calling triton_dispatch")
    else:
        raise ValueError(
            f"Unknown triton kernel: '{kernel_name}'. Supported kernels: "
            "['linear_attn', 'sparse_attn', 'low_rank', 'state_update', 'gqa']"
        )
