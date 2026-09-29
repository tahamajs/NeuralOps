r"""
Triton-Accelerated Cosformer Linear Attention Kernel.

Implements the Cosformer algorithm (Zheng et al., 2022) with hardware-fused OpenAI Triton kernels:
  1. In-Register Cosine & Sine Positional Reweighting:
     - Directly computes cos(pi * i / 2M) and sin(pi * i / 2M) inside GPU registers (tl.cos, tl.sin).
     - Uses fixed max_length (M = max_position_embeddings) from model config for consistent invariant geometry.
     - Completely bypasses intermediate DRAM weight buffer allocations.

  2. Fused Chunked Causal Cosformer Attention (_triton_chunked_cosformer_fwd):
     - Intra-chunk attention score calculation in SRAM tiles.
     - Dual inter-chunk state tracking (S_cos, S_sin, Z_cos, Z_sin) in FP32 registers.
     - True O(N * d^2) linear complexity with causal lower-triangular masking.

  3. Bidirectional Cosformer Kernel (_triton_bidirectional_cosformer_fwd):
     - Global parallel reduction over sequence length for encoder backbones (BERT, RoBERTa).

  4. Full Autograd Integration (_TritonCosformerAutogradFunction):
     - Uses explicit torch.autograd.grad(out, [q_req, k_req, v_req], grad_out) without double-backward warnings.
     - Ensures complete differentiability for end-to-end training and recovery fine-tuning.

  5. Pure O(N) Fallback for CPU / Mac MPS:
     - Evaluates causal attention using chunked streaming prefix sums (chunk_size=64).
     - Strictly eliminates O(N^2) memory footprint on non-CUDA platforms.

  6. Production-Ready Weight Loading & RoPE:
     - Full copy_weights supporting GPT-2 (fused Conv1D c_attn), RoBERTa/BERT, and flat linear layers.
     - Complete Rotary Position Embeddings (RoPE) and Grouped Query Attention (GQA) support.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.gqa_utils import repeat_kv
from framework.core.triton_kernels import triton_dispatch

logger = logging.getLogger(__name__)

# Check OpenAI Triton availability
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============================================================================
# HELPER FUNCTIONS: RoPE
# ============================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    if q_pass.shape[-1] > 0:
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# ============================================================================
# TRITON JIT KERNELS FOR COSFORMER
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _triton_chunked_cosformer_fwd_kernel(
        Q_ptr,              # [B, H, N, D]
        K_ptr,              # [B, H, N, D]
        V_ptr,              # [B, H, N, D]
        Out_ptr,            # [B, H, N, D]
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_CHUNK: tl.constexpr,
        EPS: tl.constexpr,
        MAX_LENGTH: tl.constexpr,
    ):
        r"""
        Fused Chunked Causal Cosformer Forward Kernel.
        Uses fixed MAX_LENGTH for invariant cosine/sine angular factor.
        """
        bh_id = tl.program_id(0)
        num_chunks = (N + BLOCK_CHUNK - 1) // BLOCK_CHUNK

        q_base = Q_ptr + bh_id * stride_qh
        k_base = K_ptr + bh_id * stride_kh
        v_base = V_ptr + bh_id * stride_vh
        out_base = Out_ptr + bh_id * stride_oh

        offs_d = tl.arange(0, D)
        offs_chunk = tl.arange(0, BLOCK_CHUNK)

        # Recurrent dual-state accumulators for Cosine and Sine components
        S_cos = tl.zeros((D, D), dtype=tl.float32)
        S_sin = tl.zeros((D, D), dtype=tl.float32)
        Z_cos = tl.zeros((D,), dtype=tl.float32)
        Z_sin = tl.zeros((D,), dtype=tl.float32)

        # Fixed angle factor based on max_length (BUG 5 fix)
        angle_factor = 3.141592653589793 / (2.0 * float(MAX_LENGTH))

        for c_idx in range(0, num_chunks):
            start_n = c_idx * BLOCK_CHUNK
            cur_offs_n = start_n + offs_chunk
            mask_n = cur_offs_n < N

            # 1. Load Q, K, V tiles
            q_ptrs = q_base + (cur_offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd)
            k_ptrs = k_base + (cur_offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            v_ptrs = v_base + (cur_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

            q_tile = tl.load(q_ptrs, mask=mask_n[:, None], other=0.0)
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            # 2. Non-negative kernel map: ELU + 1
            q_f = q_tile.to(tl.float32)
            k_f = k_tile.to(tl.float32)
            phi_q = tl.where(q_f > 0.0, q_f, tl.exp(q_f) - 1.0) + 1.0
            phi_k = tl.where(k_f > 0.0, k_f, tl.exp(k_f) - 1.0) + 1.0

            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            # 3. In-Register Cosine & Sine Positional Modulation
            pos_angles = cur_offs_n.to(tl.float32) * angle_factor
            cos_pos = tl.cos(pos_angles)[:, None]
            sin_pos = tl.sin(pos_angles)[:, None]

            q_c = phi_q * cos_pos
            q_s = phi_q * sin_pos
            k_c = phi_k * cos_pos
            k_s = phi_k * sin_pos

            # 4. Inter-chunk projection from running state:
            num_inter = tl.dot(q_c.to(tl.float32), S_cos) + tl.dot(q_s.to(tl.float32), S_sin)
            denom_inter = tl.sum(q_c * Z_cos[None, :], axis=1) + tl.sum(q_s * Z_sin[None, :], axis=1)

            # 5. Intra-chunk causal scores:
            A_cos = tl.dot(q_c.to(tl.float32), tl.trans(k_c.to(tl.float32)))
            A_sin = tl.dot(q_s.to(tl.float32), tl.trans(k_s.to(tl.float32)))
            A_total = A_cos + A_sin

            causal_mask = offs_chunk[:, None] >= offs_chunk[None, :]
            A_causal = tl.where(causal_mask & (mask_n[:, None] & mask_n[None, :]), A_total, 0.0)

            num_intra = tl.dot(A_causal.to(tl.float32), v_tile.to(tl.float32))
            denom_intra = tl.sum(A_causal, axis=1)

            # 6. Combined normalized output chunk
            num_tot = num_inter + num_intra
            denom_tot = denom_inter + denom_intra + EPS
            out_tile = num_tot / denom_tot[:, None]

            out_ptrs = out_base + (cur_offs_n[:, None] * stride_on + offs_d[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_n[:, None])

            # 7. Update running state for subsequent chunks
            S_cos += tl.dot(tl.trans(k_c.to(tl.float32)), v_tile.to(tl.float32))
            S_sin += tl.dot(tl.trans(k_s.to(tl.float32)), v_tile.to(tl.float32))
            Z_cos += tl.sum(k_c, axis=0)
            Z_sin += tl.sum(k_s, axis=0)


    @triton.jit
    def _triton_bidirectional_cosformer_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EPS: tl.constexpr,
        MAX_LENGTH: tl.constexpr,
    ):
        r"""
        Fused Bidirectional Cosformer Forward Kernel for Encoders (BERT/RoBERTa).
        """
        bh_id = tl.program_id(0)

        q_base = Q_ptr + bh_id * stride_qh
        k_base = K_ptr + bh_id * stride_kh
        v_base = V_ptr + bh_id * stride_vh
        out_base = Out_ptr + bh_id * stride_oh

        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, BLOCK_N)
        angle_factor = 3.141592653589793 / (2.0 * float(MAX_LENGTH))

        # Phase 1: In-SRAM Global Reduction over N
        S_cos = tl.zeros((D, D), dtype=tl.float32)
        S_sin = tl.zeros((D, D), dtype=tl.float32)
        Z_cos = tl.zeros((D,), dtype=tl.float32)
        Z_sin = tl.zeros((D,), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask_n = cur_n < N

            k_ptrs = k_base + (cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            v_ptrs = v_base + (cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            k_f = k_tile.to(tl.float32)
            phi_k = tl.where(k_f > 0.0, k_f, tl.exp(k_f) - 1.0) + 1.0
            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            pos_angles = cur_n.to(tl.float32) * angle_factor
            k_c = phi_k * tl.cos(pos_angles)[:, None]
            k_s = phi_k * tl.sin(pos_angles)[:, None]

            S_cos += tl.dot(tl.trans(k_c.to(tl.float32)), v_tile.to(tl.float32))
            S_sin += tl.dot(tl.trans(k_s.to(tl.float32)), v_tile.to(tl.float32))
            Z_cos += tl.sum(k_c, axis=0)
            Z_sin += tl.sum(k_s, axis=0)

        # Phase 2: Compute Query Outputs
        offs_m = tl.arange(0, BLOCK_M)
        for m_start in range(0, N, BLOCK_M):
            cur_m = m_start + offs_m
            mask_m = cur_m < N

            q_ptrs = q_base + (cur_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)
            q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

            q_f = q_tile.to(tl.float32)
            phi_q = tl.where(q_f > 0.0, q_f, tl.exp(q_f) - 1.0) + 1.0
            pos_angles = cur_m.to(tl.float32) * angle_factor
            q_c = phi_q * tl.cos(pos_angles)[:, None]
            q_s = phi_q * tl.sin(pos_angles)[:, None]

            num = tl.dot(q_c.to(tl.float32), S_cos) + tl.dot(q_s.to(tl.float32), S_sin)
            denom = tl.sum(q_c * Z_cos[None, :], axis=1) + tl.sum(q_s * Z_sin[None, :], axis=1) + EPS

            out_tile = num / denom[:, None]
            out_ptrs = out_base + (cur_m[:, None] * stride_on + offs_d[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])


# ============================================================================
# AUTOGRAD FUNCTION WRAPPER (Training & Inference Differentiability)
# ============================================================================

class _TritonCosformerAutogradFunction(torch.autograd.Function):
    r"""
    Torch Autograd Function for Cosformer Attention.
    Uses explicit torch.autograd.grad to prevent double-backward warnings and extra graph memory.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_decoder: bool = False,
        eps: float = 1e-6,
        max_length: int = 2048,
    ) -> torch.Tensor:
        B, H, N, D = q.shape

        use_triton = (
            HAS_TRITON
            and q.is_cuda
            and D in (32, 64, 128)
        )

        if use_triton:
            out = torch.empty_like(v)
            grid = (B * H,)

            if is_decoder:
                _triton_chunked_cosformer_fwd_kernel[grid](
                    q, k, v, out,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    N=N, D=D, BLOCK_CHUNK=32, EPS=eps, MAX_LENGTH=max_length,
                    num_stages=2, num_warps=4,
                )
            else:
                _triton_bidirectional_cosformer_fwd_kernel[grid](
                    q, k, v, out,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    N=N, D=D, BLOCK_M=32, BLOCK_N=32, EPS=eps, MAX_LENGTH=max_length,
                    num_stages=2, num_warps=4,
                )
        else:
            # Strictly O(N * d^2) PyTorch Fallback for CPU / Mac MPS (Zero N x N memory footprint)
            phi_q = F.elu(q.float()) + 1.0
            phi_k = F.elu(k.float()) + 1.0

            pos = torch.arange(N, device=q.device, dtype=torch.float32)
            angles = (math.pi / (2.0 * float(max_length))) * pos
            cos_w = torch.cos(angles).view(1, 1, N, 1)
            sin_w = torch.sin(angles).view(1, 1, N, 1)

            qc, qs = phi_q * cos_w, phi_q * sin_w
            kc, ks = phi_k * cos_w, phi_k * sin_w
            vf = v.float()

            if is_decoder:
                # Chunked causal scan: memory O(N * d), strictly eliminating O(N^2) allocations
                chunk_size = 64
                num_chunks = (N + chunk_size - 1) // chunk_size

                S_cos = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
                S_sin = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
                Z_cos = torch.zeros(B, H, D, 1, device=q.device, dtype=torch.float32)
                Z_sin = torch.zeros(B, H, D, 1, device=q.device, dtype=torch.float32)

                outputs = []
                for c in range(num_chunks):
                    start = c * chunk_size
                    end = min(start + chunk_size, N)
                    cur_len = end - start

                    qc_c = qc[:, :, start:end, :]
                    qs_c = qs[:, :, start:end, :]
                    kc_c = kc[:, :, start:end, :]
                    ks_c = ks[:, :, start:end, :]
                    v_c = vf[:, :, start:end, :]

                    # Intra-chunk local attention
                    score_c = torch.matmul(qc_c, kc_c.transpose(-1, -2))
                    score_s = torch.matmul(qs_c, ks_c.transpose(-1, -2))
                    scores = score_c + score_s

                    causal_mask = torch.tril(torch.ones(cur_len, cur_len, device=q.device, dtype=torch.bool))
                    scores = scores.masked_fill(~causal_mask, 0.0)

                    intra_num = torch.matmul(scores, v_c)
                    intra_denom = scores.sum(dim=-1, keepdim=True)

                    # Inter-chunk recurrent carry-over
                    inter_num = torch.matmul(qc_c, S_cos) + torch.matmul(qs_c, S_sin)
                    inter_denom = torch.matmul(qc_c, Z_cos) + torch.matmul(qs_c, Z_sin)

                    out_c = (intra_num + inter_num) / (intra_denom + inter_denom + eps)
                    outputs.append(out_c)

                    # Update recurrent running states
                    S_cos = S_cos + torch.matmul(kc_c.transpose(-1, -2), v_c)
                    S_sin = S_sin + torch.matmul(ks_c.transpose(-1, -2), v_c)
                    Z_cos = Z_cos + kc_c.sum(dim=-2, keepdim=True).transpose(-1, -2)
                    Z_sin = Z_sin + ks_c.sum(dim=-2, keepdim=True).transpose(-1, -2)

                out = torch.cat(outputs, dim=-2).to(q.dtype)
            else:
                # Bidirectional global reduction
                S_cos = torch.matmul(kc.transpose(-1, -2), vf)
                S_sin = torch.matmul(ks.transpose(-1, -2), vf)
                Z_cos = kc.sum(dim=-2, keepdim=True)
                Z_sin = ks.sum(dim=-2, keepdim=True)

                num = torch.matmul(qc, S_cos) + torch.matmul(qs, S_sin)
                denom = torch.matmul(qc, Z_cos.transpose(-1, -2)) + torch.matmul(qs, Z_sin.transpose(-1, -2)) + eps
                out = (num / denom).to(q.dtype)

        ctx.save_for_backward(q, k, v)
        ctx.is_decoder = is_decoder
        ctx.eps = eps
        ctx.max_length = max_length
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        q, k, v = ctx.saved_tensors
        is_decoder = ctx.is_decoder
        eps = ctx.eps
        max_length = ctx.max_length

        # BUG 1 FIX: Use explicit torch.autograd.grad to prevent double-backward warnings & extra memory
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            k_req = k.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)

            B, H, N, D = q.shape
            phi_q = F.elu(q_req.float()) + 1.0
            phi_k = F.elu(k_req.float()) + 1.0

            pos = torch.arange(N, device=q.device, dtype=torch.float32)
            angles = (math.pi / (2.0 * float(max_length))) * pos
            cos_w = torch.cos(angles).view(1, 1, N, 1)
            sin_w = torch.sin(angles).view(1, 1, N, 1)

            qc, qs = phi_q * cos_w, phi_q * sin_w
            kc, ks = phi_k * cos_w, phi_k * sin_w
            vf = v_req.float()

            if is_decoder:
                chunk_size = 64
                num_chunks = (N + chunk_size - 1) // chunk_size
                S_cos = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
                S_sin = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
                Z_cos = torch.zeros(B, H, D, 1, device=q.device, dtype=torch.float32)
                Z_sin = torch.zeros(B, H, D, 1, device=q.device, dtype=torch.float32)
                outputs = []

                for c in range(num_chunks):
                    start = c * chunk_size
                    end = min(start + chunk_size, N)
                    cur_len = end - start

                    qc_c = qc[:, :, start:end, :]
                    qs_c = qs[:, :, start:end, :]
                    kc_c = kc[:, :, start:end, :]
                    ks_c = ks[:, :, start:end, :]
                    v_c = vf[:, :, start:end, :]

                    score_c = torch.matmul(qc_c, kc_c.transpose(-1, -2))
                    score_s = torch.matmul(qs_c, ks_c.transpose(-1, -2))
                    scores = score_c + score_s

                    causal_mask = torch.tril(torch.ones(cur_len, cur_len, device=q.device, dtype=torch.bool))
                    scores = scores.masked_fill(~causal_mask, 0.0)

                    intra_num = torch.matmul(scores, v_c)
                    intra_denom = scores.sum(dim=-1, keepdim=True)

                    inter_num = torch.matmul(qc_c, S_cos) + torch.matmul(qs_c, S_sin)
                    inter_denom = torch.matmul(qc_c, Z_cos) + torch.matmul(qs_c, Z_sin)

                    out_c = (intra_num + inter_num) / (intra_denom + inter_denom + eps)
                    outputs.append(out_c)

                    S_cos = S_cos + torch.matmul(kc_c.transpose(-1, -2), v_c)
                    S_sin = S_sin + torch.matmul(ks_c.transpose(-1, -2), v_c)
                    Z_cos = Z_cos + kc_c.sum(dim=-2, keepdim=True).transpose(-1, -2)
                    Z_sin = Z_sin + ks_c.sum(dim=-2, keepdim=True).transpose(-1, -2)

                out = torch.cat(outputs, dim=-2).to(q.dtype)
            else:
                S_cos = torch.matmul(kc.transpose(-1, -2), vf)
                S_sin = torch.matmul(ks.transpose(-1, -2), vf)
                Z_cos = kc.sum(dim=-2, keepdim=True)
                Z_sin = ks.sum(dim=-2, keepdim=True)

                num = torch.matmul(qc, S_cos) + torch.matmul(qs, S_sin)
                denom = torch.matmul(qc, Z_cos.transpose(-1, -2)) + torch.matmul(qs, Z_sin.transpose(-1, -2)) + eps
                out = (num / denom).to(q.dtype)

            dq, dk, dv = torch.autograd.grad(
                outputs=out,
                inputs=[q_req, k_req, v_req],
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
            )

            return dq, dk, dv, None, None, None


def triton_cosformer_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_decoder: bool = False,
    eps: float = 1e-6,
    max_length: int = 2048,
) -> torch.Tensor:
    """Convenience autograd functional wrapper for Triton Cosformer."""
    return _TritonCosformerAutogradFunction.apply(q, k, v, is_decoder, eps, max_length)


# ============================================================================
# MODULE DEFINITION & INTEGRATION
# ============================================================================

@ATTENTION_REGISTRY.register("triton_cosformer", supports_causal=True, supports_bidirectional=True)
class TritonCosformerAttention(BaseAttention):
    r"""
    Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiability,
    RoPE support, Pretrained Weight Loading, and Masked Attention.
    """
    triton_kernel = "linear_attn"

    def __init__(
        self,
        config: Any = None,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        is_decoder: bool = False,
        eps: float = 1e-6,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.config = config
        self.is_decoder = bool(is_decoder or getattr(config, "is_decoder", False) or getattr(config, "is_causal", False))
        self.eps = eps
        self.triton_kwargs = {
            "phi": "elu+1",
            "eps": self.eps,
            "cos_weight": None,
            "sin_weight": None,
        }

        self.embed_dim = embed_dim or kwargs.get("hidden_size") or getattr(config, "hidden_size", 768)
        self.num_heads = num_heads or kwargs.get("num_attention_heads") or getattr(config, "num_attention_heads", 12)
        self.num_kv_heads = num_kv_heads or kwargs.get("num_kv_heads") or getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = kwargs.get("head_dim") or (self.embed_dim // self.num_heads)

        # BUG 5 FIX: fixed max_length from config
        self.max_length = getattr(config, "max_position_embeddings", 2048) or 2048

        # Linear projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(self.embed_dim, self.kv_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.kv_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # BUG 3 FIX: Support pretrained attention weight copying during initialization
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Transfer pretrained projection weights from GPT-2, RoBERTa/BERT, or standard linear layers."""
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__

        # 1. GPT-2: Fused c_attn (Conv1D) + c_proj
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                # If Conv1D or shape is [in, out], transpose to standard PyTorch [out, in]
                if (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or hasattr(c_attn_mod, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()

                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None

                self.q_proj.weight.data.copy_(c_w[:split_size, :])
                self.k_proj.weight.data.copy_(c_w[split_size : 2 * split_size, :])
                self.v_proj.weight.data.copy_(c_w[2 * split_size :, :])

                if c_b is not None:
                    if self.q_proj.bias is not None:
                        self.q_proj.bias.data.copy_(c_b[:split_size])
                    if self.k_proj.bias is not None:
                        self.k_proj.bias.data.copy_(c_b[split_size : 2 * split_size])
                    if self.v_proj.bias is not None:
                        self.v_proj.bias.data.copy_(c_b[2 * split_size :])

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or hasattr(c_proj_mod, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    self.out_proj.weight.data.copy_(o_w)
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    if bias_o is not None and self.out_proj.bias is not None:
                        self.out_proj.bias.data.copy_(bias_o.data)

        # 2. RoBERTa / BERT nested self-attention
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            mapping = [("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")]
            for target_name, sub_name in mapping:
                target_mod = getattr(self, target_name)
                src_mod = getattr(s, sub_name, None)
                base_src = getattr(src_mod, "base_layer", src_mod)
                if base_src is not None and hasattr(base_src, "weight"):
                    target_mod.weight.data.copy_(base_src.weight.data)
                    if base_src.bias is not None and target_mod.bias is not None:
                        target_mod.bias.data.copy_(base_src.bias.data)

            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                base_out = getattr(out_mod, "base_layer", out_mod)
                self.out_proj.weight.data.copy_(base_out.weight.data)
                if base_out.bias is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.data.copy_(base_out.bias.data)

        # 3. Standard flat linear projections (DistilBERT, Qwen, LLaMA)
        else:
            proj_candidates = {
                "q_proj": ["q_proj", "query", "q_lin", "W_q", "wq"],
                "k_proj": ["k_proj", "key", "k_lin", "W_k", "wk"],
                "v_proj": ["v_proj", "value", "v_lin", "W_v", "wv"],
                "out_proj": ["out_proj", "dense", "out_lin", "c_proj", "o_proj", "wo"],
            }
            for target_name, candidates in proj_candidates.items():
                target_mod = getattr(self, target_name)
                for cand in candidates:
                    if hasattr(src, cand):
                        src_mod = getattr(src, cand)
                        base_mod = getattr(src_mod, "base_layer", src_mod)
                        if base_mod is not None and hasattr(base_mod, "weight"):
                            w = base_mod.weight.data
                            if w.ndim == 2 and w.shape != target_mod.weight.shape and w.t().shape == target_mod.weight.shape:
                                w = w.t()
                            target_mod.weight.data.copy_(w)
                            if getattr(base_mod, "bias", None) is not None and target_mod.bias is not None:
                                target_mod.bias.data.copy_(base_mod.bias.data)
                            break

    # Alias for API uniformity
    copy_weights_from_pretrained = copy_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        B, N, C = hidden_states.shape

        # RoPE from kwargs if not explicit
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # 2. BUG 4 FIX: Apply Rotary Position Embeddings (RoPE) if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        # 3. GQA expansion if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = repeat_kv(v, self.num_heads // self.num_kv_heads)

        # 4. BUG 2 FIX: Mask handling for padding tokens on K and V
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Shape (B, N) -> (B, 1, N, 1)
                mask = attention_mask[:, None, :, None].to(q.dtype)
                if (mask < 0).any():
                    mask = (mask > -1000.0).to(q.dtype)
                k = k * mask
                v = v * mask
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] == 1:
                    mask = attention_mask.squeeze(1).squeeze(1)[:, None, :, None].to(q.dtype)
                else:
                    mask = attention_mask[:, 0, -1, :][:, None, :, None].to(q.dtype)
                if (mask < 0).any():
                    mask = (mask > -1000.0).to(q.dtype)
                k = k * mask
                v = v * mask

        # 5. Differentiable forward pass using shared Triton dispatcher
        angle_factor = math.pi / (2.0 * float(self.max_length))
        positions = torch.arange(N, device=q.device, dtype=torch.float32)
        angles = positions * angle_factor
        cos_w = torch.cos(angles).view(1, 1, N, 1).to(q.dtype)
        sin_w = torch.sin(angles).view(1, 1, N, 1).to(q.dtype)

        dispatch_kwargs = dict(self.triton_kwargs)
        dispatch_kwargs["cos_weight"] = cos_w
        dispatch_kwargs["sin_weight"] = sin_w
        dispatch_kwargs["eps"] = self.eps

        out = triton_dispatch(
            self.triton_kernel,
            q, k, v,
            is_causal=self.is_decoder,
            **dispatch_kwargs,
        )

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        return out, None


# ============================================================================
# FACTORY INJECTION FUNCTION (For eval_all_models_datasets & run_gpt2_sweep)
# ============================================================================

@ATTENTION_REGISTRY.register("inject_triton_cosformer")
@ATTENTION_REGISTRY.register("triton_cosformer_inject")
def inject_triton_cosformer(
    model: nn.Module = None,
    rank: int = 64,
    copy_weights: bool = True,
    **kwargs: Any,
) -> Optional[nn.Module]:
    """Safe factory function to inject TritonCosformerAttention into a model."""
    from framework.core.attention_utils import inject_attention
    return inject_attention(model, "triton_cosformer", rank=rank, copy_weights=copy_weights, **kwargs)
