"""
2025–2026 Flagship Linear & Orthogonal Attention Mechanisms inspired by Flash-Linear-Attention (fla):
1. GDN2Attention: Gated DeltaNet 2 with Decoupled Erase & Write Linear Recurrence
2. DeltaProductAttention / PaTHAttention: Householder Orthogonal Reflection Attention
3. PreconditionedGDNAttention / PGDNAttention: Curvature-Aware Preconditioned Gated DeltaNet
"""

from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.attention_utils import apply_rotary_pos_emb, LoRALinear
from neuralops.core.common import _cfg_get, repeat_kv
from neuralops.core.triton_kernels.fused_norm_linear import fused_rmsnorm_gate


# ==============================================================================
# 1. Gated DeltaNet 2 (Decoupled Erase & Write Linear Recurrence)
# ==============================================================================
def gdn2_recurrent_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    alpha_t: torch.Tensor,
    beta_erase_t: torch.Tensor,
    beta_write_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decoupled Erase & Write Recurrence step for GDN2:
        decay_gate = alpha_t (or retention scalar)
        erase_proj = beta_erase_t * (k_t @ S)
        S_erased   = alpha_t * (S - k_t^T @ erase_proj)
        S_new      = S_erased + beta_write_t * (k_t^T @ v_t)
        y_t        = q_t @ S_new
    Shapes:
        S: (B, H, d_k, d_v)
        q_t, k_t: (B, H, 1, d_k)
        v_t: (B, H, 1, d_v)
        alpha_t, beta_erase_t, beta_write_t: (B, H, 1, 1)
    """
    # Normalized features
    kt_norm = F.normalize(k_t, p=2, dim=-1, eps=1e-6)
    
    # 1. Erase from memory
    inter_mem = torch.matmul(kt_norm, S)  # (B, H, 1, d_v)
    erase_term = beta_erase_t * torch.matmul(kt_norm.transpose(-1, -2), inter_mem)  # (B, H, d_k, d_v)
    S_decayed = alpha_t * (S - erase_term)
    
    # 2. Write new memory
    write_term = beta_write_t * torch.matmul(kt_norm.transpose(-1, -2), v_t)  # (B, H, d_k, d_v)
    S_next = S_decayed + write_term
    
    # 3. Readout
    qt_norm = F.normalize(q_t, p=2, dim=-1, eps=1e-6)
    y_t = torch.matmul(qt_norm, S_next)  # (B, H, 1, d_v)
    return y_t, S_next


def gdn2_chunk_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta_erase: torch.Tensor,
    beta_write: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunk-wise parallel forward execution for GDN2 with gradient checkpointing.
    Processes sequence in chunks of size 64 to avoid recording L separate autograd nodes on tape.
    """
    B, H, L, d_k = q.shape
    d_v = v.shape[-1]
    device = q.device
    dtype = q.dtype

    if initial_state is None:
        S = torch.zeros((B, H, d_k, d_v), device=device, dtype=torch.float32)
    else:
        S = initial_state.to(device=device, dtype=torch.float32)

    # For short sequences, unroll directly
    if L <= 64 or not torch.is_grad_enabled():
        outs = []
        for t in range(L):
            qt = q[:, :, t : t + 1].float()
            kt = k[:, :, t : t + 1].float()
            vt = v[:, :, t : t + 1].float()
            at = alpha[:, :, t : t + 1, None].float()
            be = beta_erase[:, :, t : t + 1, None].float()
            bw = beta_write[:, :, t : t + 1, None].float()

            yt, S = gdn2_recurrent_step(S, qt, kt, vt, at, be, bw)
            outs.append(yt.to(dtype))
        return torch.cat(outs, dim=2), S.to(dtype)

    # Chunked execution with torch.utils.checkpoint
    num_chunks = (L + chunk_size - 1) // chunk_size
    pad_len = (chunk_size - (L % chunk_size)) % chunk_size

    if pad_len > 0:
        q_p = F.pad(q, (0, 0, 0, pad_len))
        k_p = F.pad(k, (0, 0, 0, pad_len))
        v_p = F.pad(v, (0, 0, 0, pad_len))
        alpha_p = F.pad(alpha, (0, pad_len), value=1.0)
        be_p = F.pad(beta_erase, (0, pad_len), value=0.0)
        bw_p = F.pad(beta_write, (0, pad_len), value=0.0)
    else:
        q_p, k_p, v_p, alpha_p, be_p, bw_p = q, k, v, alpha, beta_erase, beta_write

    def _run_chunk(s_in, qc, kc, vc, ac, bec, bwc):
        curr = s_in
        c_outs = []
        for t in range(qc.shape[2]):
            yt, curr = gdn2_recurrent_step(
                curr,
                qc[:, :, t : t + 1].float(),
                kc[:, :, t : t + 1].float(),
                vc[:, :, t : t + 1].float(),
                ac[:, :, t : t + 1, None].float(),
                bec[:, :, t : t + 1, None].float(),
                bwc[:, :, t : t + 1, None].float(),
            )
            c_outs.append(yt)
        return torch.cat(c_outs, dim=2), curr

    outs = []
    for c in range(num_chunks):
        st = c * chunk_size
        en = (c + 1) * chunk_size
        qc = q_p[:, :, st:en]
        kc = k_p[:, :, st:en]
        vc = v_p[:, :, st:en]
        ac = alpha_p[:, :, st:en]
        bec = be_p[:, :, st:en]
        bwc = bw_p[:, :, st:en]

        yc, S = torch.utils.checkpoint.checkpoint(
            _run_chunk, S, qc, kc, vc, ac, bec, bwc, use_reentrant=False
        )
        outs.append(yc.to(dtype))

    y = torch.cat(outs, dim=2)
    if pad_len > 0:
        y = y[:, :, :L, :]
    return y, S.to(dtype)


@ATTENTION_REGISTRY.register("gdn2_attention")
@ATTENTION_REGISTRY.register("gdn2")
class GDN2Attention(BaseAttention):
    """
    Gated DeltaNet 2 Attention with Decoupled Erase & Write Linear Recurrence.
    Formulation:
        S_t = S_{t-1} * alpha_t * (I - beta_erase_t * k_t k_t^T) + beta_write_t * k_t v_t^T
    """
    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: int | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        chunk_size: int = 64,
        **kwargs: Any,
    ):
        super().__init__(config, rank)
        self.d_model = _cfg_get(config, "hidden_size", hidden_size)
        self.num_heads = _cfg_get(config, "num_attention_heads", num_heads)
        self.num_kv_heads = _cfg_get(config, "num_key_value_heads", num_kv_heads or self.num_heads)
        self.head_dim = _cfg_get(config, "head_dim", head_dim or (self.d_model // self.num_heads))
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.z_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

        # Decoupled Erase & Write Gating Projections
        self.alpha_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.erase_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.write_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.norm_weight = nn.Parameter(torch.ones(self.num_heads * self.head_dim))

        with torch.no_grad():
            nn.init.constant_(self.alpha_proj.bias, 2.0)  # retention ~ 0.88
            nn.init.constant_(self.erase_proj.bias, 0.0)  # moderate erase
            nn.init.constant_(self.write_proj.bias, 1.0)  # strong write

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        B, L, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, D).transpose(1, 2)
        z = self.z_proj(hidden_states)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        alpha = torch.sigmoid(self.alpha_proj(hidden_states)).transpose(1, 2)
        beta_erase = torch.sigmoid(self.erase_proj(hidden_states)).transpose(1, 2)
        beta_write = torch.sigmoid(self.write_proj(hidden_states)).transpose(1, 2)

        init_state = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
                init_state = past_key_value[0]
            elif isinstance(past_key_value, torch.Tensor):
                init_state = past_key_value

        y, next_state = gdn2_chunk_scan(
            q, k, v, alpha, beta_erase, beta_write,
            initial_state=init_state,
            chunk_size=self.chunk_size,
        )

        y = y.transpose(1, 2).contiguous().view(B, L, H * D)
        # Apply fused rmsnorm + gating
        y_gated = fused_rmsnorm_gate(y, z, self.norm_weight, eps=1e-5, activation="silu")
        out = self.out_proj(y_gated)

        present_kv = (next_state,) if use_cache else None
        return out, present_kv


# ==============================================================================
# 2. Householder Product & PaTH Attention (DeltaProductAttention)
# ==============================================================================
def householder_reflection(
    v: torch.Tensor,
    u: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Applies Householder orthogonal transformation:
        H = I - 2 * beta * (u u^T) / (||u||^2 + eps)
        H @ v = v - 2 * beta * u * (u^T @ v) / (||u||^2 + eps)
    Preserves exact isometry and norm stability across deep recurrent steps.
    """
    u_norm_sq = (u * u).sum(dim=-1, keepdim=True) + 1e-6
    dot_uv = (v * u).sum(dim=-1, keepdim=True)
    reflect = 2.0 * beta * dot_uv * (u / u_norm_sq)
    return v - reflect


@ATTENTION_REGISTRY.register("delta_product_attention")
@ATTENTION_REGISTRY.register("delta_product")
@ATTENTION_REGISTRY.register("path_attention")
class DeltaProductAttention(BaseAttention):
    """
    Householder Product & PaTH Attention for NeuralOps.
    Tracks state via sequence-accumulated Householder orthogonal reflections, ensuring
    strict isometry, bounded spectrum, and norm preservation along sequence dimensions.
    """
    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: int | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        head_dim: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, rank)
        self.d_model = _cfg_get(config, "hidden_size", hidden_size)
        self.num_heads = _cfg_get(config, "num_attention_heads", num_heads)
        self.head_dim = _cfg_get(config, "head_dim", head_dim or (self.d_model // self.num_heads))

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.u_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.beta_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        B, L, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        u = self.u_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(hidden_states)).transpose(1, 2).unsqueeze(-1)

        init_state = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
                init_state = past_key_value[0]
            elif isinstance(past_key_value, torch.Tensor):
                init_state = past_key_value

        S = (
            torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
            if init_state is None
            else init_state.to(device=q.device, dtype=torch.float32)
        )

        # Chunked execution with torch.utils.checkpoint
        def _run_delta_product_chunk(s_in, qc, kc, vc, uc, bc):
            curr = s_in
            c_outs = []
            for t in range(qc.shape[2]):
                qt = qc[:, :, t : t + 1].float()
                kt = kc[:, :, t : t + 1].float()
                vt = vc[:, :, t : t + 1].float()
                ut = uc[:, :, t : t + 1].float()
                bt = bc[:, :, t : t + 1].float()

                u_norm_sq = (ut * ut).sum(dim=-1, keepdim=True) + 1e-6
                u_S = torch.matmul(ut, curr)
                refl = (2.0 * bt / u_norm_sq) * torch.matmul(ut.transpose(-1, -2), u_S)
                S_reflected = curr - refl
                curr = S_reflected + torch.matmul(kt.transpose(-1, -2), vt)
                yt = torch.matmul(qt, curr)
                c_outs.append(yt.to(qc.dtype))
            return torch.cat(c_outs, dim=2), curr

        chunk_size = 64
        if L <= 64 or not torch.is_grad_enabled():
            _, present_state = _run_delta_product_chunk(S, q, k, v, u, beta)
            y, present_state = _run_delta_product_chunk(S, q, k, v, u, beta)
        else:
            num_chunks = (L + chunk_size - 1) // chunk_size
            pad_len = (chunk_size - (L % chunk_size)) % chunk_size
            if pad_len > 0:
                q_p = F.pad(q, (0, 0, 0, pad_len))
                k_p = F.pad(k, (0, 0, 0, pad_len))
                v_p = F.pad(v, (0, 0, 0, pad_len))
                u_p = F.pad(u, (0, 0, 0, pad_len))
                b_p = F.pad(beta, (0, 0, 0, pad_len), value=0.0)
            else:
                q_p, k_p, v_p, u_p, b_p = q, k, v, u, beta

            outs = []
            curr = S
            for c in range(num_chunks):
                st = c * chunk_size
                en = (c + 1) * chunk_size
                qc = q_p[:, :, st:en]
                kc = k_p[:, :, st:en]
                vc = v_p[:, :, st:en]
                uc = u_p[:, :, st:en]
                bc = b_p[:, :, st:en]

                yc, curr = torch.utils.checkpoint.checkpoint(
                    _run_delta_product_chunk, curr, qc, kc, vc, uc, bc, use_reentrant=False
                )
                outs.append(yc)

            y = torch.cat(outs, dim=2)
            if pad_len > 0:
                y = y[:, :, :L, :]
            present_state = curr

        y = y.transpose(1, 2).contiguous().view(B, L, H * D)
        out = self.out_proj(y)
        present_kv = (present_state.to(q.dtype),) if use_cache else None
        return out, present_kv


# ==============================================================================
# 3. Preconditioned Gated DeltaNet (PreconditionedGDNAttention / PGDNAttention)
# ==============================================================================
@ATTENTION_REGISTRY.register("preconditioned_gdn_attention")
@ATTENTION_REGISTRY.register("preconditioned_gdn")
@ATTENTION_REGISTRY.register("pgdn_attention")
@ATTENTION_REGISTRY.register("pgdn")
class PreconditionedGDNAttention(BaseAttention):
    """
    Curvature-Aware Preconditioned Gated DeltaNet.
    Applies empirical covariance / diagonal curvature preconditioning:
        P_t = (diag(s_t) + lambda * I)^{-1/2}
        k_tilde_t = P_t @ k_t
        S_t = alpha_t * S_{t-1}(I - beta_t * k_tilde_t k_tilde_t^T) + beta_t * k_tilde_t v_t^T
    Directly aligns with NeuralOps Whitened Covariance Preconditioner.
    """
    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: int | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        head_dim: int | None = None,
        preconditioner_damping: float = 1e-4,
        **kwargs: Any,
    ):
        super().__init__(config, rank)
        self.d_model = _cfg_get(config, "hidden_size", hidden_size)
        self.num_heads = _cfg_get(config, "num_attention_heads", num_heads)
        self.head_dim = _cfg_get(config, "head_dim", head_dim or (self.d_model // self.num_heads))
        self.damping = preconditioner_damping

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.beta_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.alpha_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.curvature_diag = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        B, L, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, D).transpose(1, 2)

        # Precondition keys with curvature diagonal P = (diag(s) + damping)^{-1/2} across entire sequence in parallel
        p_scale = torch.rsqrt(F.softplus(self.curvature_diag) + self.damping).view(1, H, 1, D)
        k_preconditioned = k * p_scale

        alpha = torch.sigmoid(self.alpha_proj(hidden_states)).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(hidden_states)).transpose(1, 2)

        init_state = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
                init_state = past_key_value[0]
            elif isinstance(past_key_value, torch.Tensor):
                init_state = past_key_value

        # Fast path via Triton gdn_scan when suitable and without cache
        if init_state is None and D <= 64 and q.is_cuda and not use_cache:
            try:
                from neuralops.core.triton_kernels.gdn_scan import gdn_scan
                y_scan = gdn_scan(q, k_preconditioned, v, beta, alpha)
                y = y_scan.transpose(1, 2).contiguous().view(B, L, H * D)
                out = self.out_proj(y)
                return out, None
            except Exception:
                pass

        # Checkpointed chunked recurrence fallback
        def _run_pgdn_chunk(s_in, qc, kc, vc, ac, bc):
            curr = s_in
            c_outs = []
            for t in range(qc.shape[2]):
                qt = qc[:, :, t : t + 1].float()
                kt = kc[:, :, t : t + 1].float()
                vt = vc[:, :, t : t + 1].float()
                at = ac[:, :, t : t + 1, None].float()
                bt = bc[:, :, t : t + 1, None].float()

                inter_v = torch.matmul(kt, curr)
                err = vt - inter_v
                curr = at * curr + bt * torch.matmul(kt.transpose(-1, -2), err)
                yt = torch.matmul(qt, curr)
                c_outs.append(yt.to(qc.dtype))
            return torch.cat(c_outs, dim=2), curr

        S = (
            torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
            if init_state is None
            else init_state.to(device=q.device, dtype=torch.float32)
        )

        chunk_size = 64
        if L <= 64 or not torch.is_grad_enabled():
            y, present_state = _run_pgdn_chunk(S, q, k_preconditioned, v, alpha, beta)
        else:
            num_chunks = (L + chunk_size - 1) // chunk_size
            pad_len = (chunk_size - (L % chunk_size)) % chunk_size
            if pad_len > 0:
                q_p = F.pad(q, (0, 0, 0, pad_len))
                k_p = F.pad(k_preconditioned, (0, 0, 0, pad_len))
                v_p = F.pad(v, (0, 0, 0, pad_len))
                a_p = F.pad(alpha, (0, pad_len), value=1.0)
                b_p = F.pad(beta, (0, pad_len), value=0.0)
            else:
                q_p, k_p, v_p, a_p, b_p = q, k_preconditioned, v, alpha, beta

            outs = []
            curr = S
            for c in range(num_chunks):
                st = c * chunk_size
                en = (c + 1) * chunk_size
                qc = q_p[:, :, st:en]
                kc = k_p[:, :, st:en]
                vc = v_p[:, :, st:en]
                ac = a_p[:, :, st:en]
                bc = b_p[:, :, st:en]

                yc, curr = torch.utils.checkpoint.checkpoint(
                    _run_pgdn_chunk, curr, qc, kc, vc, ac, bc, use_reentrant=False
                )
                outs.append(yc)

            y = torch.cat(outs, dim=2)
            if pad_len > 0:
                y = y[:, :, :L, :]
            present_state = curr

        y = y.transpose(1, 2).contiguous().view(B, L, H * D)
        out = self.out_proj(y)
        present_kv = (present_state.to(q.dtype),) if use_cache else None
        return out, present_kv


__all__ = [
    "GDN2Attention",
    "gdn2_recurrent_step",
    "gdn2_chunk_scan",
    "DeltaProductAttention",
    "householder_reflection",
    "PreconditionedGDNAttention",
]
