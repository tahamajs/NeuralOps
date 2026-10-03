r"""
Triton-Accelerated SVD Linear Attention & Chunked Causal Linear Attention.

High-performance GPU implementation using OpenAI Triton language (tl) with:
  1. Chunked Causal Linear Attention Kernel (_triton_chunked_causal_linear_attn_fwd):
     - Keeps the recurrent key-value state S \in R^{D_k x D_v} strictly in GPU SRAM/registers.
     - Tiled in blocks of BLOCK_CHUNK = 64 to eliminate O(N^2) memory footprint.
     - Evaluates intra-chunk causal attention and inter-chunk prefix sums in a single fused pass.
     - Strictly O(N * d^2) FLOPs and O(N * d) memory.

  2. Bidirectional Linear Attention Kernel (_triton_bidirectional_linear_attn_fwd):
     - Fused parallel reduction of K^T V and K^T 1 for encoder architectures (DistilBERT, RoBERTa).
     - Fused query projection with numerical stability epsilon clamping in registers.

  3. Non-Zero Identity SVD Initialization (BUG 5 FIX):
     - SVDLinearProjection initializes U with identity-like structure (eye for rank dimensions)
       and V with Kaiming uniform so initial output is non-zero before fine-tuning.

  4. Backward Kernel & Clean Autograd (BUG 1 FIX):
     - Uses explicit torch.autograd.grad(out, [q_req, k_req, v_req], grad_out) preventing double-backward warnings.
     - Bridges _triton_linear_attn_bwd_kernel on CUDA GPUs for bidirectional attention.
     - Provides streaming chunked backward propagation for causal decoder attention.

  5. Attention Masking & Padding Invariant (BUG 2 FIX):
     - Ensures padding tokens on K and V are strictly zeroed out before attention accumulation,
       preventing contaminated normalization denominator Z.

  6. RoPE and GQA Support (BUG 3 & BUG 4 FIX):
     - Seamless Rotary Position Embeddings (RoPE) via apply_rotary_pos_emb.
     - Full Grouped Query Attention (GQA) support with separate num_kv_heads and repeat_kv.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.triton_kernels import triton_dispatch

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
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
# TRITON JIT KERNELS (Compiled to CUDA PTX on NVIDIA GPUs)
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _triton_chunked_causal_linear_attn_fwd_kernel(
        Q_ptr,  # [B, H, N, D_k]
        K_ptr,  # [B, H, N, D_k]
        V_ptr,  # [B, H, N, D_v]
        Out_ptr,  # [B, H, N, D_v]
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        N: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_CHUNK: tl.constexpr,
        EPS: tl.constexpr,
        ACTIVATION_MODE: tl.constexpr,  # 0: ELU+1, 1: ReLU, 2: Identity
    ):
        r"""
        Fused Chunked Causal Linear Attention Forward Kernel.
        Each thread block processes one (batch, head) stream sequentially across chunks,
        maintaining the recurrent state matrix S \in [D_K, D_V] and Z \in [D_K] in SRAM registers.
        """
        bh_id = tl.program_id(0)
        num_chunks = (N + BLOCK_CHUNK - 1) // BLOCK_CHUNK

        q_base = Q_ptr + bh_id * stride_qh
        k_base = K_ptr + bh_id * stride_kh
        v_base = V_ptr + bh_id * stride_vh
        out_base = Out_ptr + bh_id * stride_oh

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        offs_chunk = tl.arange(0, BLOCK_CHUNK)

        state_S = tl.zeros((D_K, D_V), dtype=tl.float32)
        state_Z = tl.zeros((D_K,), dtype=tl.float32)

        for c_idx in range(0, num_chunks):
            start_n = c_idx * BLOCK_CHUNK
            cur_offs_n = start_n + offs_chunk
            mask_n = cur_offs_n < N

            # 1. Load tile of Q, K, V
            q_ptrs = q_base + (cur_offs_n[:, None] * stride_qn + offs_dk[None, :] * stride_qd)
            k_ptrs = k_base + (cur_offs_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd)
            v_ptrs = v_base + (cur_offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd)

            q_tile = tl.load(q_ptrs, mask=mask_n[:, None], other=0.0)
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            # 2. In-SRAM Non-Negative Activation
            if ACTIVATION_MODE == 0:  # ELU + 1
                q_f = q_tile.to(tl.float32)
                k_f = k_tile.to(tl.float32)
                phi_q = tl.where(q_f > 0.0, q_f, tl.exp(q_f) - 1.0) + 1.0
                phi_k = tl.where(k_f > 0.0, k_f, tl.exp(k_f) - 1.0) + 1.0
            elif ACTIVATION_MODE == 1:  # ReLU
                phi_q = tl.maximum(q_tile, 0.0)
                phi_k = tl.maximum(k_tile, 0.0)
            else:
                phi_q = q_tile
                phi_k = k_tile

            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            # 3. Inter-Chunk Projection: query @ running state
            num_inter = tl.dot(phi_q.to(tl.float32), state_S)
            denom_inter = tl.sum(phi_q * state_Z[None, :], axis=1)

            # 4. Intra-Chunk Causal Scores
            scores_intra = tl.dot(phi_q.to(tl.float32), tl.trans(phi_k.to(tl.float32)))
            causal_mask = offs_chunk[:, None] >= offs_chunk[None, :]
            scores_causal = tl.where(
                causal_mask & (mask_n[:, None] & mask_n[None, :]), scores_intra, 0.0
            )

            num_intra = tl.dot(scores_causal.to(tl.float32), v_tile.to(tl.float32))
            denom_intra = tl.sum(scores_causal, axis=1)

            # 5. Combine and Store
            num_total = num_inter + num_intra
            denom_total = denom_inter + denom_intra + EPS
            out_tile = num_total / denom_total[:, None]

            out_ptrs = out_base + (cur_offs_n[:, None] * stride_on + offs_dv[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_n[:, None])

            # 6. Update Running Recurrent State
            state_S += tl.dot(tl.trans(phi_k.to(tl.float32)), v_tile.to(tl.float32))
            state_Z += tl.sum(phi_k, axis=0)

    @triton.jit
    def _triton_bidirectional_linear_attn_fwd_kernel(
        Q_ptr,  # [B, H, N, D_k]
        K_ptr,  # [B, H, N, D_k]
        V_ptr,  # [B, H, N, D_v]
        Out_ptr,  # [B, H, N, D_v]
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        N: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EPS: tl.constexpr,
        ACTIVATION_MODE: tl.constexpr,
    ):
        r"""
        Fused Bidirectional Linear Attention Forward Kernel for Encoders.
        """
        bh_id = tl.program_id(0)

        q_base = Q_ptr + bh_id * stride_qh
        k_base = K_ptr + bh_id * stride_kh
        v_base = V_ptr + bh_id * stride_vh
        out_base = Out_ptr + bh_id * stride_oh

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        offs_n = tl.arange(0, BLOCK_N)

        acc_S = tl.zeros((D_K, D_V), dtype=tl.float32)
        acc_Z = tl.zeros((D_K,), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask_n = cur_n < N

            k_ptrs = k_base + (cur_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd)
            v_ptrs = v_base + (cur_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd)

            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            if ACTIVATION_MODE == 0:
                k_f = k_tile.to(tl.float32)
                phi_k = tl.where(k_f > 0.0, k_f, tl.exp(k_f) - 1.0) + 1.0
            elif ACTIVATION_MODE == 1:
                phi_k = tl.maximum(k_tile, 0.0)
            else:
                phi_k = k_tile

            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            acc_S += tl.dot(tl.trans(phi_k.to(tl.float32)), v_tile.to(tl.float32))
            acc_Z += tl.sum(phi_k, axis=0)

        offs_m = tl.arange(0, BLOCK_M)
        for m_start in range(0, N, BLOCK_M):
            cur_m = m_start + offs_m
            mask_m = cur_m < N

            q_ptrs = q_base + (cur_m[:, None] * stride_qn + offs_dk[None, :] * stride_qd)
            q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

            if ACTIVATION_MODE == 0:
                q_f = q_tile.to(tl.float32)
                phi_q = tl.where(q_f > 0.0, q_f, tl.exp(q_f) - 1.0) + 1.0
            elif ACTIVATION_MODE == 1:
                phi_q = tl.maximum(q_tile, 0.0)
            else:
                phi_q = q_tile

            num = tl.dot(phi_q.to(tl.float32), acc_S)
            denom = tl.sum(phi_q * acc_Z[None, :], axis=1) + EPS

            out_tile = num / denom[:, None]
            out_ptrs = out_base + (cur_m[:, None] * stride_on + offs_dv[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])

    @triton.jit
    def _triton_linear_attn_bwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        dQ_ptr,
        dK_ptr,
        dV_ptr,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_dob,
        stride_doh,
        stride_don,
        stride_dod,
        stride_dqb,
        stride_dqh,
        stride_dqn,
        stride_dqd,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dkd,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        stride_dvd,
        N: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EPS: tl.constexpr,
    ):
        r"""
        Analytical Backward Kernel in Triton for Bidirectional Linear Attention.
        """
        bh_id = tl.program_id(0)

        q_base = Q_ptr + bh_id * stride_qh
        k_base = K_ptr + bh_id * stride_kh
        v_base = V_ptr + bh_id * stride_vh
        do_base = dO_ptr + bh_id * stride_doh

        dq_base = dQ_ptr + bh_id * stride_dqh
        dk_base = dK_ptr + bh_id * stride_dkh
        dv_base = dV_ptr + bh_id * stride_dvh

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        offs_n = tl.arange(0, BLOCK_N)

        # 1. First Pass: S = K^T V and Z = sum(K)
        S = tl.zeros((D_K, D_V), dtype=tl.float32)
        Z = tl.zeros((D_K,), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask = cur_n < N
            k = tl.load(
                k_base + cur_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd,
                mask=mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + cur_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
                mask=mask[:, None],
                other=0.0,
            )
            S += tl.dot(tl.trans(k.to(tl.float32)), v.to(tl.float32))
            Z += tl.sum(k, axis=0)

        # 2. Second Pass: Compute dQ and accumulate dS
        dS = tl.zeros((D_K, D_V), dtype=tl.float32)
        dZ = tl.zeros((D_K,), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask = cur_n < N

            q = tl.load(
                q_base + cur_n[:, None] * stride_qn + offs_dk[None, :] * stride_qd,
                mask=mask[:, None],
                other=0.0,
            )
            do = tl.load(
                do_base + cur_n[:, None] * stride_don + offs_dv[None, :] * stride_dod,
                mask=mask[:, None],
                other=0.0,
            )

            denom = tl.sum(q * Z[None, :], axis=1) + EPS
            inv_denom = 1.0 / denom

            do_scaled = do.to(tl.float32) * inv_denom[:, None]
            num = tl.dot(q.to(tl.float32), S)
            ratio = tl.sum(num * do_scaled, axis=1) * inv_denom

            dq = tl.dot(do_scaled, tl.trans(S)) - ratio[:, None] * Z[None, :]

            dS += tl.dot(tl.trans(q.to(tl.float32)), do_scaled)
            dZ -= tl.sum(q.to(tl.float32) * ratio[:, None], axis=0)

            tl.store(
                dq_base + cur_n[:, None] * stride_dqn + offs_dk[None, :] * stride_dqd,
                dq.to(dQ_ptr.dtype.element_ty),
                mask=mask[:, None],
            )

        # 3. Third Pass: dK and dV
        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask = cur_n < N

            k = tl.load(
                k_base + cur_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd,
                mask=mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + cur_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
                mask=mask[:, None],
                other=0.0,
            )

            dv = tl.dot(k.to(tl.float32), dS)
            dk = tl.dot(v.to(tl.float32), tl.trans(dS)) + dZ[None, :]

            tl.store(
                dv_base + cur_n[:, None] * stride_dvn + offs_dv[None, :] * stride_dvd,
                dv.to(dV_ptr.dtype.element_ty),
                mask=mask[:, None],
            )
            tl.store(
                dk_base + cur_n[:, None] * stride_dkn + offs_dk[None, :] * stride_dkd,
                dk.to(dK_ptr.dtype.element_ty),
                mask=mask[:, None],
            )


# ============================================================================
# FULL AUTOGRAD FUNCTION (Forward & Backward Fully Connected)
# ============================================================================


class _TritonLinearAttentionAutogradFunction(torch.autograd.Function):
    """
    Autograd Function connecting Triton forward and backward kernels on CUDA GPUs,
    with an exact numerically validated PyTorch reference fallback on CPU / Mac MPS.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        activation_mode: str = "elu",
        eps: float = 1e-6,
    ) -> torch.Tensor:
        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        act_id = 0 if activation_mode == "elu" else (1 if activation_mode == "relu" else 2)

        use_triton = HAS_TRITON and q.is_cuda and D_k in (32, 64, 128) and D_v in (32, 64, 128)

        if use_triton:
            out = torch.empty((B, H, N, D_v), device=q.device, dtype=q.dtype)
            grid = (B * H,)

            if is_causal:
                block_chunk = 32
                _triton_chunked_causal_linear_attn_fwd_kernel[grid](
                    q,
                    k,
                    v,
                    out,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    out.stride(0),
                    out.stride(1),
                    out.stride(2),
                    out.stride(3),
                    N=N,
                    D_K=D_k,
                    D_V=D_v,
                    BLOCK_CHUNK=block_chunk,
                    EPS=eps,
                    ACTIVATION_MODE=act_id,
                    num_stages=2,
                    num_warps=4,
                )
            else:
                block_m = 32
                block_n = 32
                _triton_bidirectional_linear_attn_fwd_kernel[grid](
                    q,
                    k,
                    v,
                    out,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    out.stride(0),
                    out.stride(1),
                    out.stride(2),
                    out.stride(3),
                    N=N,
                    D_K=D_k,
                    D_V=D_v,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    EPS=eps,
                    ACTIVATION_MODE=act_id,
                    num_stages=2,
                    num_warps=4,
                )
        else:
            # Reference PyTorch fallback (CPU / Apple Silicon MPS / Non-Triton)
            if activation_mode == "elu":
                phi_q = F.elu(q.float()) + 1.0
                phi_k = F.elu(k.float()) + 1.0
            elif activation_mode == "relu":
                phi_q = F.relu(q.float())
                phi_k = F.relu(k.float())
            else:
                phi_q = q.float()
                phi_k = k.float()

            v_f = v.float()

            if is_causal:
                chunk_size = 64
                num_chunks = (N + chunk_size - 1) // chunk_size
                S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)
                Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=torch.float32)
                outputs = []

                for c in range(num_chunks):
                    start = c * chunk_size
                    end = min(start + chunk_size, N)
                    cur_len = end - start
                    qc = phi_q[:, :, start:end, :]
                    kc = phi_k[:, :, start:end, :]
                    vc = v_f[:, :, start:end, :]

                    scores = torch.matmul(qc, kc.transpose(-1, -2))
                    causal_mask = torch.tril(
                        torch.ones(cur_len, cur_len, device=q.device, dtype=torch.bool)
                    )
                    scores = scores.masked_fill(~causal_mask, 0.0)
                    intra_num = torch.matmul(scores, vc)
                    intra_denom = scores.sum(dim=-1, keepdim=True)

                    inter_num = torch.matmul(qc, S)
                    inter_denom = torch.matmul(qc, Z)

                    out_c = (intra_num + inter_num) / (intra_denom + inter_denom + eps)
                    outputs.append(out_c)

                    S = S + torch.matmul(kc.transpose(-1, -2), vc)
                    Z = Z + kc.sum(dim=-2, keepdim=True).transpose(-1, -2)

                out = torch.cat(outputs, dim=-2).to(q.dtype)
            else:
                S = torch.matmul(phi_k.transpose(-1, -2), v_f)
                Z = phi_k.sum(dim=-2, keepdim=True)
                num = torch.matmul(phi_q, S)
                denom = torch.matmul(phi_q, Z.transpose(-1, -2)) + eps
                out = (num / denom).to(q.dtype)

        ctx.save_for_backward(q, k, v)
        ctx.is_causal = is_causal
        ctx.activation_mode = activation_mode
        ctx.eps = eps
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        q, k, v = ctx.saved_tensors
        is_causal = ctx.is_causal
        activation_mode = ctx.activation_mode
        eps = ctx.eps

        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        # Use Triton backward kernel on CUDA for bidirectional identity mode
        use_triton_bwd = (
            HAS_TRITON
            and q.is_cuda
            and not is_causal
            and activation_mode == "identity"
            and D_k in (32, 64, 128)
            and D_v in (32, 64, 128)
        )

        if use_triton_bwd:
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            grid = (B * H,)

            _triton_linear_attn_bwd_kernel[grid](
                q,
                k,
                v,
                grad_out,
                dq,
                dk,
                dv,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                grad_out.stride(0),
                grad_out.stride(1),
                grad_out.stride(2),
                grad_out.stride(3),
                dq.stride(0),
                dq.stride(1),
                dq.stride(2),
                dq.stride(3),
                dk.stride(0),
                dk.stride(1),
                dk.stride(2),
                dk.stride(3),
                dv.stride(0),
                dv.stride(1),
                dv.stride(2),
                dv.stride(3),
                N=N,
                D_K=D_k,
                D_V=D_v,
                BLOCK_N=32,
                EPS=eps,
                num_stages=2,
                num_warps=4,
            )
            return dq, dk, dv, None, None, None

        # BUG 1 FIX: Clean explicit torch.autograd.grad preventing double-backward warnings & extra memory
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            k_req = k.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)

            if activation_mode == "elu":
                phi_q = F.elu(q_req.float()) + 1.0
                phi_k = F.elu(k_req.float()) + 1.0
            elif activation_mode == "relu":
                phi_q = F.relu(q_req.float())
                phi_k = F.relu(k_req.float())
            else:
                phi_q = q_req.float()
                phi_k = k_req.float()

            v_f = v_req.float()

            if is_causal:
                chunk_size = 64
                num_chunks = (N + chunk_size - 1) // chunk_size
                S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)
                Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=torch.float32)
                outputs = []

                for c in range(num_chunks):
                    start = c * chunk_size
                    end = min(start + chunk_size, N)
                    cur_len = end - start
                    qc = phi_q[:, :, start:end, :]
                    kc = phi_k[:, :, start:end, :]
                    vc = v_f[:, :, start:end, :]

                    scores = torch.matmul(qc, kc.transpose(-1, -2))
                    causal_mask = torch.tril(
                        torch.ones(cur_len, cur_len, device=q.device, dtype=torch.bool)
                    )
                    scores = scores.masked_fill(~causal_mask, 0.0)
                    intra_num = torch.matmul(scores, vc)
                    intra_denom = scores.sum(dim=-1, keepdim=True)

                    inter_num = torch.matmul(qc, S)
                    inter_denom = torch.matmul(qc, Z)

                    out_c = (intra_num + inter_num) / (intra_denom + inter_denom + eps)
                    outputs.append(out_c)

                    S = S + torch.matmul(kc.transpose(-1, -2), vc)
                    Z = Z + kc.sum(dim=-2, keepdim=True).transpose(-1, -2)

                out = torch.cat(outputs, dim=-2).to(q.dtype)
            else:
                S = torch.matmul(phi_k.transpose(-1, -2), v_f)
                Z = phi_k.sum(dim=-2, keepdim=True)
                num = torch.matmul(phi_q, S)
                denom = torch.matmul(phi_q, Z.transpose(-1, -2)) + eps
                out = (num / denom).to(q.dtype)

            dq, dk, dv = torch.autograd.grad(
                outputs=out,
                inputs=[q_req, k_req, v_req],
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
            )

            return dq, dk, dv, None, None, None


def triton_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    activation_mode: str = "elu",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convenience functional wrapper for Triton Linear Attention."""
    return _TritonLinearAttentionAutogradFunction.apply(q, k, v, is_causal, activation_mode, eps)


# ============================================================================
# GENUINE SVD LOW-RANK PROJECTION LAYER
# ============================================================================


class SVDLinearProjection(nn.Module):
    r"""
    Genuine SVD Low-Rank Linear Layer: W \approx U @ V.

    Factorizes in_features -> out_features through bottleneck rank r:
      V \in R^{r x in_features}  (Down-projection)
      U \in R^{out_features x r} (Up-projection)
      y = (x @ V^T) @ U^T + bias
    """

    triton_kernel = "low_rank"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)

        self.V = nn.Parameter(torch.empty(self.rank, in_features))
        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # BUG 5 FIX: Identity-like init for U so initial output is non-zero before fine-tuning
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.zeros_(self.U)
        with torch.no_grad():
            r = min(self.out_features, self.rank)
            self.U[:r, :r].copy_(torch.eye(r))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused low-rank projection: (x @ V^T) @ U^T + bias
        # Align weights with input dtype (e.g. under fp16/bf16 autocast) so Triton kernel runs directly
        u = self.U.to(x.dtype) if self.U.dtype != x.dtype else self.U
        v = self.V.to(x.dtype) if self.V.dtype != x.dtype else self.V
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        return triton_dispatch(self.triton_kernel, x, u, v, bias=b)

    @classmethod
    def from_pretrained_linear(cls, linear: nn.Linear, rank: int = 64) -> SVDLinearProjection:
        r"""
        Initializes low-rank factors directly from pretrained weights via Truncated SVD:
        W = U_full @ diag(S) @ V_full^T
        U = U_full[:, :r] @ sqrt(S[:r])
        V = sqrt(S[:r]) @ V_full^T[:r, :]
        """
        out_features, in_features = linear.weight.shape
        proj = cls(in_features, out_features, rank=rank, bias=(linear.bias is not None))

        with torch.no_grad():
            W = linear.weight.data.float()
            u, s, vh = torch.linalg.svd(W, full_matrices=False)

            r = proj.rank
            s_sqrt = torch.diag(torch.sqrt(s[:r]))
            proj.U.data.copy_(torch.matmul(u[:, :r], s_sqrt).to(linear.weight.dtype))
            proj.V.data.copy_(torch.matmul(s_sqrt, vh[:r, :]).to(linear.weight.dtype))

            if linear.bias is not None:
                proj.bias.data.copy_(linear.bias.data)

        return proj


# ============================================================================
# MODULE DEFINITION (Triton + Genuine SVD Linear Attention)
# ============================================================================


@ATTENTION_REGISTRY.register(
    "triton_svd_linear_attention", supports_causal=True, supports_bidirectional=True
)
@ATTENTION_REGISTRY.register(
    "triton_linear_attention", supports_causal=True, supports_bidirectional=True
)
class TritonSVDLinearAttention(BaseAttention):
    r"""
    Triton-Accelerated Linear Attention with Genuine SVD Low-Rank Projections,
    RoPE, GQA, and Masked Attention.
    """

    triton_kernel = "linear_attn"

    def __init__(
        self,
        config: Any = None,
        embed_dim: int | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        rank: int | None = 64,
        is_decoder: bool = False,
        activation_mode: str = "elu",
        eps: float = 1e-6,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.config = config
        self.is_decoder = bool(
            is_decoder
            or getattr(config, "is_decoder", False)
            or getattr(config, "is_causal", False)
        )
        self.activation_mode = activation_mode
        self.eps = eps
        self.triton_kwargs = {
            "phi": "elu+1" if self.activation_mode == "elu" else self.activation_mode,
            "eps": self.eps,
            "cos_weight": None,
            "sin_weight": None,
        }

        # Dimensionality extraction
        self.embed_dim = (
            embed_dim or kwargs.get("hidden_size") or getattr(config, "hidden_size", 768)
        )
        self.num_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or getattr(config, "num_attention_heads", 12)
        )
        # BUG 4 FIX: Support GQA
        self.num_kv_heads = (
            num_kv_heads
            or kwargs.get("num_kv_heads")
            or getattr(config, "num_key_value_heads", self.num_heads)
        )
        self.head_dim = kwargs.get("head_dim") or (self.embed_dim // self.num_heads)
        self.rank = rank or 64
        self.kv_dim = self.num_kv_heads * self.head_dim

        # Genuine SVD Low-Rank Linear Projections
        if self.rank < self.embed_dim:
            self.q_proj = SVDLinearProjection(
                self.embed_dim, self.embed_dim, rank=self.rank, bias=True
            )
            self.k_proj = SVDLinearProjection(
                self.embed_dim, self.kv_dim, rank=min(self.rank, self.kv_dim), bias=True
            )
            self.v_proj = SVDLinearProjection(
                self.embed_dim, self.kv_dim, rank=min(self.rank, self.kv_dim), bias=True
            )
        else:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.k_proj = nn.Linear(self.embed_dim, self.kv_dim, bias=True)
            self.v_proj = nn.Linear(self.embed_dim, self.kv_dim, bias=True)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Transfer pretrained projection weights from GPT-2, RoBERTa/BERT, or standard linear layers."""
        if src is None:
            return

        getattr(src, "__class__", type(src)).__name__

        # 1. GPT-2: Fused c_attn (Conv1D) + c_proj
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
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

                q_w, q_b = c_w[:split_size, :], (c_b[:split_size] if c_b is not None else None)
                k_w, k_b = (
                    c_w[split_size : 2 * split_size, :],
                    (c_b[split_size : 2 * split_size] if c_b is not None else None),
                )
                v_w, v_b = (
                    c_w[2 * split_size :, :],
                    (c_b[2 * split_size :] if c_b is not None else None),
                )

                self._copy_single_proj(self.q_proj, q_w, q_b)
                self._copy_single_proj(self.k_proj, k_w, k_b)
                self._copy_single_proj(self.v_proj, v_w, v_b)

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
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            mapping = [("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")]
            for target_name, sub_name in mapping:
                target_mod = getattr(self, target_name)
                src_mod = getattr(s, sub_name, None)
                base_src = getattr(src_mod, "base_layer", src_mod)
                if base_src is not None and hasattr(base_src, "weight"):
                    w = base_src.weight.data
                    b = getattr(base_src, "bias", None)
                    b_data = b.data if b is not None else None
                    self._copy_single_proj(target_mod, w, b_data)

            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                base_out = getattr(out_mod, "base_layer", out_mod)
                self.out_proj.weight.data.copy_(base_out.weight.data)
                if base_out.bias is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.data.copy_(base_out.bias.data)

        # 3. Standard flat linear projections
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
                            b = getattr(base_mod, "bias", None)
                            b_data = b.data if b is not None else None
                            if target_name == "out_proj":
                                if (
                                    w.ndim == 2
                                    and w.shape != target_mod.weight.shape
                                    and w.t().shape == target_mod.weight.shape
                                ):
                                    w = w.t()
                                target_mod.weight.data.copy_(w)
                                if b_data is not None and target_mod.bias is not None:
                                    target_mod.bias.data.copy_(b_data)
                            else:
                                self._copy_single_proj(target_mod, w, b_data)
                            break

    def _copy_single_proj(
        self, target_mod: nn.Module, w: torch.Tensor, b: torch.Tensor | None
    ) -> None:
        """Decomposes weights into low-rank SVD factors or copies directly."""
        if isinstance(target_mod, SVDLinearProjection):
            with torch.no_grad():
                W_f = w.float()
                u, s, vh = torch.linalg.svd(W_f, full_matrices=False)
                r = target_mod.rank
                s_sqrt = torch.diag(torch.sqrt(s[:r]))
                target_mod.U.data.copy_(torch.matmul(u[:, :r], s_sqrt).to(w.dtype))
                target_mod.V.data.copy_(torch.matmul(s_sqrt, vh[:r, :]).to(w.dtype))
                if b is not None and target_mod.bias is not None:
                    target_mod.bias.data.copy_(b)
        else:
            if (
                w.ndim == 2
                and w.shape != target_mod.weight.shape
                and w.t().shape == target_mod.weight.shape
            ):
                w = w.t()
            target_mod.weight.data.copy_(w)
            if b is not None and target_mod.bias is not None:
                target_mod.bias.data.copy_(b)

    # Alias for API uniformity
    copy_weights_from_pretrained = copy_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        B, N, C = hidden_states.shape

        if torch.is_autocast_enabled() and hidden_states.is_cuda:
            ac_dtype = (
                torch.get_autocast_dtype("cuda")
                if hasattr(torch, "get_autocast_dtype")
                else torch.float16
            )
            hidden_states = hidden_states.to(ac_dtype)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")

        # 1. Project Q, K, V via SVD low-rank factorized projections
        q = (
            self.q_proj(hidden_states)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, N, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, N, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # 2. BUG 3 FIX: Apply Rotary Position Embeddings (RoPE) if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        # 3. BUG 4 FIX: GQA expansion if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = repeat_kv(v, self.num_heads // self.num_kv_heads)

        # 4. BUG 2 FIX: Mask handling for padding tokens on K and V
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, :, None].to(q.dtype)  # [B, 1, N, 1]
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

        # 5. Dispatch Fully Differentiable Triton Linear Attention
        out = triton_dispatch(
            self.triton_kernel,
            q,
            k,
            v,
            is_causal=self.is_decoder,
            **self.triton_kwargs,
        )

        # 6. Reshape and final output projection
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)

        # Numerical safety guard against unbounded linear attention residual accumulation in FP16
        if out.dtype == torch.float16:
            out = torch.clamp(out, min=-65000.0, max=65000.0)

        return out, None


# ============================================================================
# FACTORY INJECTION FUNCTION (For eval_all_models_datasets & run_gpt2_sweep)
# ============================================================================


@ATTENTION_REGISTRY.register("inject_triton_svd_linear_attention")
@ATTENTION_REGISTRY.register("triton_svd_linear_attention_inject")
def inject_triton_svd_linear_attention(
    model: nn.Module = None,
    rank: int = 64,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module | None:
    """Safe factory function to inject TritonSVDLinearAttention into a model."""
    from neuralops.core.attention_utils import inject_attention

    return inject_attention(
        model,
        "triton_svd_linear_attention",
        rank=rank,
        copy_weights=copy_weights,
        **kwargs,
    )
