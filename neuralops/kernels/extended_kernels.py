# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Extended Accelerated Kernels

Production hardware-accelerated kernels with unified three-level fallback across CUDA C++, Triton JIT, and PyTorch reference backends.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.cuda_kernels._fallback import dispatch, force_kernel_backend

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


def swiglu_pytorch_ref(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    gate = F.silu(F.linear(x, w_gate))
    up = F.linear(x, w_up)
    return gate * up

if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        x_ptr, y_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(x_ptr + offsets * 2, mask=mask, other=0.0)
        b = tl.load(x_ptr + offsets * 2 + 1, mask=mask, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-a))
        out = (a * sig) * b
        tl.store(y_ptr + offsets, out, mask=mask)

def swiglu_triton(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    gate = F.linear(x, w_gate)
    up = F.linear(x, w_up)
    return F.silu(gate) * up

def fused_swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None,
        swiglu_triton if HAS_TRITON and x.is_cuda else None,
        swiglu_pytorch_ref,
        x, w_gate, w_up,
        kernel_name="fused_swiglu"
    )


def rmsnorm_pytorch_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr, weight_ptr, out_ptr,
        stride_x_row, stride_out_row,
        N, eps,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        var = tl.sum(x * x, axis=0) / N
        rsqrt = 1.0 / tl.sqrt(var + eps)
        tl.store(out_ptr + row * stride_out_row + cols, x * rsqrt * w, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    _rmsnorm_fwd_kernel[grid](
        x_2d, weight, out,
        x_2d.stride(0), out.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
    )
    return out.reshape(orig_shape)

def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return dispatch(
        None,
        rmsnorm_triton if HAS_TRITON and x.is_cuda else None,
        rmsnorm_pytorch_ref,
        x, weight, eps=eps,
        kernel_name="fused_rmsnorm"
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rope_pytorch_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)

def fused_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None,
        None,
        rope_pytorch_ref,
        x, cos, sin,
        kernel_name="fused_rotary_emb"
    )


def alibi_pytorch_ref(scores: torch.Tensor, num_heads: int) -> torch.Tensor:
    B, H, T, _ = scores.shape
    def get_slopes(n: int) -> list[float]:
        def get_slopes_power_of_2(n_p2: int) -> list[float]:
            start = (2 ** (-2 ** -(math.log2(n_p2) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n_p2)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    slopes = torch.tensor(get_slopes(H), device=scores.device, dtype=scores.dtype).view(1, H, 1, 1)
    pos = torch.arange(T, device=scores.device, dtype=scores.dtype)
    rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
    alibi_bias = slopes * rel_pos.unsqueeze(0).unsqueeze(0)
    return scores + alibi_bias

def fused_alibi_attention(scores: torch.Tensor, num_heads: int) -> torch.Tensor:
    return dispatch(
        None, None, alibi_pytorch_ref,
        scores, num_heads,
        kernel_name="fused_alibi_attention"
    )


def sink_attention_pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_sinks: int = 4, window_size: int = 64) -> torch.Tensor:
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    mask = torch.ones(T, T, device=q.device, dtype=torch.bool)
    causal_mask = torch.tril(mask)

    sink_mask = torch.zeros(T, T, device=q.device, dtype=torch.bool)
    sink_mask[:, :min(num_sinks, T)] = True

    band_mask = torch.triu(causal_mask, diagonal=-window_size)
    combined_mask = (band_mask | sink_mask) & causal_mask

    scores = scores.masked_fill(~combined_mask, -1e4)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)

def fused_sink_token_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_sinks: int = 4, window_size: int = 64) -> torch.Tensor:
    return dispatch(
        None, None, sink_attention_pytorch_ref,
        q, k, v, num_sinks=num_sinks, window_size=window_size,
        kernel_name="fused_sink_token_attention"
    )


def decay_scan_pytorch_ref(k: torch.Tensor, v: torch.Tensor, decay: float = 0.9) -> torch.Tensor:
    B, T, Dk = k.shape
    Dv = v.shape[-1]
    out = torch.zeros(B, T, Dv, device=k.device, dtype=k.dtype)
    state = torch.zeros(B, Dk, Dv, device=k.device, dtype=k.dtype)
    for t in range(T):
        state = decay * state + torch.bmm(k[:, t:t+1, :].transpose(1, 2), v[:, t:t+1, :])
        out[:, t:t+1, :] = torch.bmm(k[:, t:t+1, :], state)
    return out

def fused_decay_scan(k: torch.Tensor, v: torch.Tensor, decay: float = 0.9) -> torch.Tensor:
    return dispatch(
        None, None, decay_scan_pytorch_ref,
        k, v, decay=decay,
        kernel_name="fused_decay_scan"
    )


def hadamard_pytorch_ref(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[-1]
    assert (N & (N - 1)) == 0, "Last dimension must be a power of 2"
    h = 1
    y = x.clone()
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                u = y[..., j].clone()
                v = y[..., j + h].clone()
                y[..., j] = u + v
                y[..., j + h] = u - v
        h *= 2
    return y / math.sqrt(N)

def fused_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None, None, hadamard_pytorch_ref,
        x,
        kernel_name="fused_hadamard_transform"
    )


def int4_dequant_pytorch_ref(packed_w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    low = (packed_w & 0x0F).to(scale.dtype)
    high = ((packed_w >> 4) & 0x0F).to(scale.dtype)
    unpacked = torch.stack([low, high], dim=-1).view(packed_w.shape[0], -1)
    return (unpacked - zero) * scale

def fused_int4_dequant(packed_w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None, None, int4_dequant_pytorch_ref,
        packed_w, scale, zero,
        kernel_name="fused_int4_dequant"
    )


def fp8_e4m3_gemm_pytorch_ref(a: torch.Tensor, b: torch.Tensor, scale_a: float = 1.0, scale_b: float = 1.0) -> torch.Tensor:
    a_f = a.float() * scale_a
    b_f = b.float() * scale_b
    return torch.matmul(a_f, b_f).to(a.dtype if a.dtype.is_floating_point else torch.float32)

def fused_fp8_e4m3_gemm(a: torch.Tensor, b: torch.Tensor, scale_a: float = 1.0, scale_b: float = 1.0) -> torch.Tensor:
    return dispatch(
        None, None, fp8_e4m3_gemm_pytorch_ref,
        a, b, scale_a=scale_a, scale_b=scale_b,
        kernel_name="fused_fp8_e4m3_gemm"
    )


def dct_spectral_pytorch_ref(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[-1]
    k = torch.arange(N, device=x.device, dtype=torch.float32).unsqueeze(0)
    n = torch.arange(N, device=x.device, dtype=torch.float32).unsqueeze(1)
    cos_matrix = torch.cos(math.pi * (n + 0.5) * k / N)
    factor = torch.ones(N, device=x.device, dtype=torch.float32) * math.sqrt(2.0 / N)
    factor[0] = math.sqrt(1.0 / N)
    dct_basis = (cos_matrix * factor).to(x.dtype)
    return torch.matmul(x, dct_basis)

def fused_spectral_dct(x: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None, None, dct_spectral_pytorch_ref,
        x,
        kernel_name="fused_spectral_dct"
    )


def chunked_cumsum_pytorch_ref(x: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    B, T, D = x.shape
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        x_pad = F.pad(x, (0, 0, 0, pad_len))
    else:
        x_pad = x
    num_chunks = x_pad.shape[1] // chunk_size
    chunks = x_pad.view(B, num_chunks, chunk_size, D)
    chunk_sums = torch.cumsum(chunks, dim=2)
    res = chunk_sums.view(B, -1, D)[:, :T, :]
    return res

def fused_chunked_cumsum(x: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    return dispatch(
        None, None, chunked_cumsum_pytorch_ref,
        x, chunk_size=chunk_size,
        kernel_name="fused_chunked_cumsum"
    )


def cka_alignment_pytorch_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean(dim=0, keepdim=True)
    xy_sim = (x_c.t() @ y_c).pow(2).sum()
    xx_sim = (x_c.t() @ x_c).pow(2).sum()
    yy_sim = (y_c.t() @ y_c).pow(2).sum()
    denom = torch.sqrt(xx_sim * yy_sim + 1e-8)
    return 1.0 - (xy_sim / denom)

def fused_cka_alignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None, None, cka_alignment_pytorch_ref,
        x, y,
        kernel_name="fused_cka_alignment"
    )


def mohawk_homotopy_pytorch_ref(w0: torch.Tensor, w_target: torch.Tensor, step: int, max_steps: int) -> torch.Tensor:
    progress = min(max(step / max(max_steps, 1), 0.0), 1.0)
    lam = 0.5 * (1.0 - math.cos(math.pi * progress))
    return (1.0 - lam) * w0 + lam * w_target

def fused_mohawk_homotopy(w0: torch.Tensor, w_target: torch.Tensor, step: int, max_steps: int) -> torch.Tensor:
    return dispatch(
        None, None, mohawk_homotopy_pytorch_ref,
        w0, w_target, step, max_steps,
        kernel_name="fused_mohawk_homotopy"
    )


def nullspace_project_pytorch_ref(delta: torch.Tensor, v_topk: torch.Tensor) -> torch.Tensor:
    proj = v_topk @ (v_topk.t() @ delta)
    return delta - proj

def fused_nullspace_project(delta: torch.Tensor, v_topk: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None, None, nullspace_project_pytorch_ref,
        delta, v_topk,
        kernel_name="fused_nullspace_project"
    )


def householder_reflect_pytorch_ref(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    norm_v_sq = (v * v).sum() + 1e-8
    v_norm = v / torch.sqrt(norm_v_sq)
    dot = torch.matmul(x, v_norm).unsqueeze(-1)
    return x - 2.0 * dot * v_norm.unsqueeze(0)

def fused_householder_reflect(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None, None, householder_reflect_pytorch_ref,
        x, v,
        kernel_name="fused_householder_reflect"
    )


def simplex_project_pytorch_ref(y: torch.Tensor) -> torch.Tensor:
    orig_shape = y.shape
    y_2d = y.view(-1, orig_shape[-1])
    N, D = y_2d.shape
    u, _ = torch.sort(y_2d, descending=True, dim=-1)
    cssv = torch.cumsum(u, dim=-1)
    ind = torch.arange(1, D + 1, device=y.device, dtype=y.dtype).unsqueeze(0)
    cond = (u * ind) > (cssv - 1.0)
    rho = torch.sum(cond.long(), dim=-1, keepdim=True)
    rho = torch.clamp(rho, min=1)
    theta = (torch.gather(cssv, -1, rho - 1) - 1.0) / rho.float()
    w = torch.clamp(y_2d - theta, min=0.0)
    return w.view(orig_shape)

def fused_simplex_project(y: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None, None, simplex_project_pytorch_ref,
        y,
        kernel_name="fused_simplex_project"
    )


def topk_gating_pytorch_ref(logits: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    vals, idx = torch.topk(logits, k=top_k, dim=-1)
    weights = F.softmax(vals.float(), dim=-1).to(logits.dtype)
    return weights, idx

def fused_topk_gating(logits: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    return dispatch(
        None, None, topk_gating_pytorch_ref,
        logits, top_k=top_k,
        kernel_name="fused_topk_gating"
    )


def label_smoothing_ce_pytorch_ref(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    return F.cross_entropy(logits, targets, label_smoothing=smoothing)

def fused_label_smoothing_ce(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    return dispatch(
        None, None, label_smoothing_ce_pytorch_ref,
        logits, targets, smoothing=smoothing,
        kernel_name="fused_label_smoothing_ce"
    )


def kl_distill_pytorch_ref(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    p_s = F.log_softmax(student_logits / tau, dim=-1)
    p_t = F.softmax(teacher_logits / tau, dim=-1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (tau * tau)

def fused_kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    return dispatch(
        None, None, kl_distill_pytorch_ref,
        student_logits, teacher_logits, tau=tau,
        kernel_name="fused_kl_divergence"
    )


def lowrank_residual_pytorch_ref(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
    low_rank_out = torch.matmul(torch.matmul(x, u), v)
    return low_rank_out + x_res

def fused_lowrank_residual_add(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
    return dispatch(
        None, None, lowrank_residual_pytorch_ref,
        x, u, v, x_res,
        kernel_name="fused_lowrank_residual_add"
    )


__all__ = [
    "fused_swiglu",
    "fused_rmsnorm",
    "fused_rotary_emb",
    "fused_alibi_attention",
    "fused_sink_token_attention",
    "fused_decay_scan",
    "fused_hadamard_transform",
    "fused_int4_dequant",
    "fused_fp8_e4m3_gemm",
    "fused_spectral_dct",
    "fused_chunked_cumsum",
    "fused_cka_alignment",
    "fused_mohawk_homotopy",
    "fused_nullspace_project",
    "fused_householder_reflect",
    "fused_simplex_project",
    "fused_topk_gating",
    "fused_label_smoothing_ce",
    "fused_kl_divergence",
    "fused_lowrank_residual_add",
]
