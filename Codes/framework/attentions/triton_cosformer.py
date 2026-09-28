r"""
Triton-Accelerated Cosformer Linear Attention Kernel.

Implements the Cosformer algorithm (Zheng et al., 2022) with hardware-fused OpenAI Triton kernels:
  1. In-Register Cosine & Sine Positional Reweighting:
     - Directly computes cos(pi * i / 2M) and sin(pi * i / 2M) inside GPU registers (tl.cos, tl.sin).
     - Completely bypasses intermediate DRAM weight buffer allocations.

  2. Fused Chunked Causal Cosformer Attention (_triton_chunked_cosformer_fwd):
     - Intra-chunk attention score calculation in SRAM tiles.
     - Dual inter-chunk state tracking (S_cos, S_sin, Z_cos, Z_sin) in FP32 registers.
     - True O(N * d^2) linear complexity with causal lower-triangular masking.

  3. Bidirectional Cosformer Kernel (_triton_bidirectional_cosformer_fwd):
     - Global parallel reduction over sequence length for encoder backbones (BERT, RoBERTa).

  4. Full Autograd Integration (_TritonCosformerAutogradFunction):
     - Ensures complete differentiability for end-to-end training and recovery fine-tuning.
     - Implements exact gradient propagation for dQ, dK, and dV in the backward pass.

  5. Pure O(N) Fallback for CPU / Mac MPS:
     - Evaluates causal attention using chunked streaming prefix sums (chunk_size=64).
     - Strictly eliminates O(N^2) memory footprint on non-CUDA platforms.
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

logger = logging.getLogger(__name__)

# Check OpenAI Triton availability
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


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
    ):
        r"""
        Fused Chunked Causal Cosformer Forward Kernel.
        Computes positional cosine and sine modulation directly in registers,
        then evaluates chunked causal linear attention without intermediate score tensors.
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

        # Precompute angle step: pi / (2.0 * N)
        angle_factor = 3.141592653589793 / (2.0 * float(N))

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
            phi_q = tl.where(q_tile > 0.0, q_tile, tl.exp(q_tile) - 1.0) + 1.0
            phi_k = tl.where(k_tile > 0.0, k_tile, tl.exp(k_tile) - 1.0) + 1.0

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
            #    num_inter = q_c @ S_cos + q_s @ S_sin
            #    denom_inter = q_c @ Z_cos + q_s @ Z_sin
            num_inter = tl.dot(q_c.to(tl.float32), S_cos) + tl.dot(q_s.to(tl.float32), S_sin)
            denom_inter = tl.sum(q_c * Z_cos[None, :], axis=1) + tl.sum(q_s * Z_sin[None, :], axis=1)

            # 5. Intra-chunk causal scores:
            #    A_cos = q_c @ k_c^T
            #    A_sin = q_s @ k_s^T
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
        angle_factor = 3.141592653589793 / (2.0 * float(N))

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

            phi_k = tl.where(k_tile > 0.0, k_tile, tl.exp(k_tile) - 1.0) + 1.0
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

            phi_q = tl.where(q_tile > 0.0, q_tile, tl.exp(q_tile) - 1.0) + 1.0
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
    Ensures complete training differentiability by bridging forward kernel execution
    with exact analytical/autograd gradient propagation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_decoder: bool = False,
        eps: float = 1e-6,
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
                    N=N, D=D, BLOCK_CHUNK=64, EPS=eps,
                )
            else:
                _triton_bidirectional_cosformer_fwd_kernel[grid](
                    q, k, v, out,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    N=N, D=D, BLOCK_M=64, BLOCK_N=64, EPS=eps,
                )
        else:
            # Strictly O(N * d^2) PyTorch Fallback for CPU / Mac MPS (Zero N x N memory footprint)
            phi_q = F.elu(q.float()) + 1.0
            phi_k = F.elu(k.float()) + 1.0

            pos = torch.arange(N, device=q.device, dtype=torch.float32)
            angles = (math.pi / (2.0 * float(N))) * pos
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
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v = ctx.saved_tensors
        is_decoder = ctx.is_decoder
        eps = ctx.eps

        # Analytical autograd backward pass
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            k_req = k.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)

            B, H, N, D = q.shape
            phi_q = F.elu(q_req.float()) + 1.0
            phi_k = F.elu(k_req.float()) + 1.0

            pos = torch.arange(N, device=q.device, dtype=torch.float32)
            angles = (math.pi / (2.0 * float(N))) * pos
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

            # Evaluate exact vector-Jacobian product
            torch.autograd.backward(out, grad_out)

            return q_req.grad, k_req.grad, v_req.grad, None, None


def triton_cosformer_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_decoder: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convenience autograd functional wrapper for Triton Cosformer."""
    return _TritonCosformerAutogradFunction.apply(q, k, v, is_decoder, eps)


# ============================================================================
# MODULE DEFINITION & INTEGRATION
# ============================================================================

@ATTENTION_REGISTRY.register("triton_cosformer", supports_causal=True, supports_bidirectional=True)
class TritonCosformerAttention(BaseAttention):
    r"""
    Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiability.
    """

    def __init__(
        self,
        config: Any = None,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        is_decoder: bool = False,
        eps: float = 1e-6,
        **kwargs: Any,
    ):
        super().__init__()
        self.config = config
        self.is_decoder = bool(is_decoder or getattr(config, "is_decoder", False))
        self.eps = eps

        self.embed_dim = embed_dim or getattr(config, "hidden_size", 768)
        self.num_heads = num_heads or getattr(config, "num_attention_heads", 12)
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        B, N, C = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # Differentiable forward pass (bridges Triton kernel or chunked O(N) fallback)
        out = triton_cosformer_attention(
            q=q,
            k=k,
            v=v,
            is_decoder=self.is_decoder,
            eps=self.eps,
        )

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        return out, None
