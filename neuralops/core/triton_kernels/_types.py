"""Shared base types, constants, detection, and PyTorch fallbacks for Triton kernels."""

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
    triton = type(
        "triton",
        (),
        {
            "jit": lambda f: f,
            "heuristics": lambda *a, **kw: lambda f: f,
            "autotune": lambda *a, **kw: lambda f: f,
            "cdiv": lambda a, b: (a + b - 1) // b,
            "next_power_of_2": lambda n: 1 << (n - 1).bit_length() if n > 0 else 1,
        },
    )
    tl = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_HEAD_DIMS: set[int] = {32, 64, 128, 256}
MAX_TRITON_SEQ_LEN: int = 16384
BLOCK_SIZE: int = 64

_WARNED_REASONS: set[str] = set()


def log_fallback_reason(kernel_name: str, reason: str) -> None:
    """Log the fallback reason once per unique (kernel, reason) pair."""
    key = f"{kernel_name}:{reason}"
    if key not in _WARNED_REASONS:
        _WARNED_REASONS.add(key)
        logger.info("Triton kernel '%s' using PyTorch fallback: %s", kernel_name, reason)


FORWARD_ONLY_KERNELS: set[str] = {
    "paged_attention",
    "flash_decoding",
}


TENSOR_CORE_DOT_KERNELS: set[str] = {
    "linear_attn",
    "low_rank",
    "sparse_attn",
    "gdn_scan",
    "gla_scan",
    "state_update",
    "cosformer",
    "fused_norm_linear",
    "varlen_attention",
    "block_sparse",
}


def check_triton_suitability(
    tensor: torch.Tensor,
    kernel_name: str,
    head_dim: int | None = None,
    seq_len: int | None = None,
    supports_backward: bool = True,
) -> tuple[bool, str | None]:
    """Validate whether the input tensor satisfies all requirements for Triton execution."""
    if not HAS_TRITON:
        return False, "triton package is not installed"

    if not torch.cuda.is_available():
        return False, "CUDA is not available on this system"

    if not tensor.is_cuda:
        return False, f"tensor is on {tensor.device}, expected CUDA"

    # Hardware Capability Profile Inspection
    try:
        from neuralops.core.triton_kernels.hardware_profile import get_device_hardware_profile

        profile = get_device_hardware_profile(tensor.device)
        if not profile.supports_triton_jit:
            return (
                False,
                f"GPU compute capability {profile.sm_major}.{profile.sm_minor} < 7.0 is unsupported by Triton (Pascal/older architecture requires PyTorch fallback)",
            )
        if kernel_name in TENSOR_CORE_DOT_KERNELS and not profile.supports_tensor_core_dot:
            return (
                False,
                f"Kernel '{kernel_name}' requires Tensor Core hardware (SM >= 7.5/8.0), device is {profile.sm_major}.{profile.sm_minor}",
            )
    except Exception:
        pass

    if tensor.dtype not in (torch.float16, torch.bfloat16):
        return (
            False,
            f"dtype {tensor.dtype} not in (float16, bfloat16); Triton kernels require fp16/bf16",
        )

    # If tensor is bfloat16 but device does not support native bf16 (requires Ampere SM 8.0+)
    if tensor.dtype == torch.bfloat16:
        try:
            from neuralops.core.triton_kernels.hardware_profile import get_device_hardware_profile
            profile = get_device_hardware_profile(tensor.device)
            if not profile.supports_bf16:
                return (
                    False,
                    f"bfloat16 is not natively supported on device {tensor.device} (requires Ampere SM 8.0+)",
                )
        except Exception:
            if not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported():
                return (
                    False,
                    f"bfloat16 is not natively supported on device {tensor.device} (requires Ampere SM 8.0+)",
                )

    if seq_len is not None and seq_len > MAX_TRITON_SEQ_LEN:
        return (
            False,
            f"sequence length {seq_len} exceeds max supported length {MAX_TRITON_SEQ_LEN}",
        )

    # If the kernel does not support autograd backward, fallback if gradients are requested
    if not supports_backward or kernel_name in FORWARD_ONLY_KERNELS:
        if getattr(tensor, "requires_grad", False) or torch.is_grad_enabled():
            return (
                False,
                f"kernel '{kernel_name}' is forward-only but gradient computation is enabled",
            )

    return True, None


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
    """Pure PyTorch fallback for linear attention with O(N) memory efficiency."""
    orig_dtype = q.dtype
    orig_dim = q.dim()

    if orig_dim == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    device_type = q.device.type
    with torch.autocast(device_type=device_type, enabled=False):
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

        has_modulation = cos_weight is not None and sin_weight is not None
        if has_modulation:
            cos_w = cos_weight.float()
            sin_w = sin_weight.float()
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

                    num_inter = torch.matmul(qc, S_cos) + torch.matmul(qs, S_sin)
                    denom_inter = torch.matmul(qc, Z_cos.unsqueeze(-1)).squeeze(-1) + torch.matmul(
                        qs, Z_sin.unsqueeze(-1)
                    ).squeeze(-1)

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

            res = out
        else:
            if has_modulation:
                num = torch.matmul(
                    q_c, torch.matmul(k_c.transpose(-1, -2), v_f)
                ) + torch.matmul(q_s, torch.matmul(k_s.transpose(-1, -2), v_f))
                denom = torch.matmul(
                    q_c, k_c.sum(dim=-2).unsqueeze(-1)
                ) + torch.matmul(q_s, k_s.sum(dim=-2).unsqueeze(-1))
                if phi in ("elu+1", "relu", "softmax_free"):
                    denom_safe = torch.clamp(denom, min=eff_eps)
                else:
                    denom_sign = torch.sign(denom)
                    denom_sign = torch.where(denom_sign == 0, torch.ones_like(denom_sign), denom_sign)
                    denom_safe = torch.where(denom.abs() < eff_eps, eff_eps * denom_sign, denom)
                res = num / denom_safe
            else:
                KV = torch.matmul(phi_k.transpose(-1, -2), v_f)
                num = torch.matmul(phi_q, KV)
                K_sum = phi_k.sum(dim=-2).unsqueeze(-1)
                denom = torch.matmul(phi_q, K_sum)
                if phi in ("elu+1", "relu", "softmax_free"):
                    denom_safe = torch.clamp(denom, min=eff_eps)
                else:
                    denom_sign = torch.sign(denom)
                    denom_sign = torch.where(denom_sign == 0, torch.ones_like(denom_sign), denom_sign)
                    denom_safe = torch.where(denom.abs() < eff_eps, eff_eps * denom_sign, denom)
                res = num / denom_safe

        if orig_dim == 3:
            res = res.squeeze(1)
        return res.to(orig_dtype)


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
    """Pure PyTorch fallback for factorized linear projection: y = (x @ V.T) @ U.T + bias."""
    orig_shape = x.shape
    D_in = orig_shape[-1]
    x_2d = x.contiguous().view(-1, D_in).float()

    V_mat = (V.t() if V.shape[-1] == D_in else V).float()
    Rank = V_mat.shape[1]
    U_mat = (U.t() if U.shape[-1] == Rank else U).float()
    D_out = U_mat.shape[1]

    h = torch.matmul(x_2d, V_mat)
    y_2d = torch.matmul(h, U_mat)
    if bias is not None:
        y_2d = y_2d + bias.float()
    out_shape = list(orig_shape[:-1]) + [D_out]
    return y_2d.to(x.dtype).view(*out_shape)


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
    """Pure PyTorch fallback for SSM state updates (Mamba/S4/RWKV recurrence)."""
    B_sz, L_sz, D_sz = u.shape
    N_sz = A.shape[-1] if A.dim() >= 2 else 16

    if state is None:
        state = torch.zeros((B_sz, D_sz, N_sz), dtype=torch.float32, device=u.device)
    else:
        state = state.float().clone()

    y_list = []
    for t in range(L_sz):
        u_t = u[:, t, :].float()
        delta_t = delta[:, t, :].float() if delta.dim() == 3 else delta.float()
        if delta_t.shape[-1] == 1:
            delta_t = delta_t.expand(-1, D_sz)

        dA = torch.exp(delta_t.unsqueeze(-1) * A.float())
        if B.dim() == 3:
            B_t = B[:, t, :].float()
            if B_t.dim() == 2 and B_t.shape[-1] == N_sz:
                dB = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
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
    """Pure PyTorch fallback for GQA repeat and reshape along the head dimension."""
    if num_kv_heads == num_heads:
        return k, v

    n_rep = num_heads // num_kv_heads
    B, H_kv, N, D = k.shape

    k_rep = (
        k[:, :, None, :, :]
        .expand(B, H_kv, n_rep, N, D)
        .reshape(B, H_kv * n_rep, N, D)
    )
    v_rep = (
        v[:, :, None, :, :]
        .expand(B, H_kv, n_rep, N, D)
        .reshape(B, H_kv * n_rep, N, D)
    )

    return k_rep, v_rep


__all__ = [
    "BLOCK_SIZE",
    "HAS_TRITON",
    "MAX_TRITON_SEQ_LEN",
    "SUPPORTED_HEAD_DIMS",
    "check_triton_suitability",
    "gqa_pytorch_fallback",
    "linear_attn_pytorch_fallback",
    "log_fallback_reason",
    "low_rank_pytorch_fallback",
    "sparse_attn_pytorch_fallback",
    "state_update_pytorch_fallback",
    "tl",
    "triton",
]
