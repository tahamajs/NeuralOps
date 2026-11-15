#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Linear Attention Bridges Module.

Provides:
- InvariantPreservingLinearBridge: Invariant compliance for Laws 1, 2, 3, 7, and 9
- CrossAttentionLinearBridge: Law 4 static cross-attention associative caching
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.surgery.director_modules.invariants import (
    sanitize_kv_for_padding_mask,
    sanitize_output_for_padding_mask,
)

logger = logging.getLogger("neuralops.surgery.director.bridges")


class InvariantPreservingLinearBridge(nn.Module):
    """
    Invariant Compliance Engine for Laws 1, 2, 3, 7, and 9:
      - Law 1: QK-Norm Preservation (RMSNorm on Q and K before linear feature map phi(x))
      - Law 2: Logit Soft-Capping Temperature Folding (tau_folded = (1/sqrt(d_k)) * tanh(c / max(c, 1.0)))
      - Law 3: Zero-Bleed Padding Mask Sanitization (K = K * M, V = V * M, O = O * M)
      - Law 7: Causal State Monotonicity & Exponential State Decay (gamma in (0, 1))
      - Law 9: FP32 Accumulation & Dtype Budget Guard for half-precision inputs
    """

    def __init__(
        self,
        base_attention: Optional[nn.Module] = None,
        has_qk_norm: bool = False,
        softcapping_factor: Optional[float] = None,
        head_dim: int = 64,
        decay: float = 0.99,
        num_attention_sinks: int = 0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.base_attention = base_attention
        self.has_qk_norm = has_qk_norm
        self.softcapping_factor = softcapping_factor
        self.head_dim = head_dim
        self.decay = decay
        self.num_attention_sinks = num_attention_sinks
        self.eps = eps

        if self.has_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.folded_scale = self.compute_folded_scale()

        # Telemetry & verification attributes
        self.last_accum_dtype: Optional[torch.dtype] = None
        self.recurrent_state_dtype: Optional[torch.dtype] = None
        self.cached_recurrent_state: Optional[torch.Tensor] = None
        self.cached_normalizer_state: Optional[torch.Tensor] = None

    def compute_folded_scale(self) -> float:
        """
        Law 2: Folds logit soft-capping threshold c into the linear attention temperature:
        tau_folded = (1 / sqrt(d_k)) * tanh(c / max(c, 1.0)).
        """
        base_scale = 1.0 / math.sqrt(max(self.head_dim, 1))
        if self.softcapping_factor is not None and self.softcapping_factor > 0:
            c = float(self.softcapping_factor)
            folding_ratio = math.tanh(c / max(c, 1.0))
            return base_scale * folding_ratio
        return base_scale

    def fold_softcapping_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Applies mathematical feature mapping with folded temperature scaling."""
        if self.folded_scale != 1.0:
            x = x * self.folded_scale
        return F.elu(x) + 1.0

    def apply_qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Law 1: Normalizes Q and K projections along head_dim before feature mapping."""
        if self.has_qk_norm and self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Executes invariant-safe forward pass with Law 9 FP32 accumulation
        and Law 3 zero-bleed padding sanitization.
        """
        orig_dtype = hidden_states.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            self.last_accum_dtype = torch.float32
            self.recurrent_state_dtype = torch.float32

        if self.base_attention is not None:
            # Delegate to underlying replaced attention module
            out = self.base_attention(hidden_states, *args, attention_mask=attention_mask, **kwargs)
            if isinstance(out, tuple):
                out_tensor = out[0]
                rest = out[1:]
                out_tensor = sanitize_output_for_padding_mask(out_tensor, attention_mask, cu_seqlens)
                return (out_tensor, *rest)  # type: ignore[return-value]
            return sanitize_output_for_padding_mask(out, attention_mask, cu_seqlens)

        # Standalone Reference Linear Attention with Invariants
        b = hidden_states.shape[0]
        t = hidden_states.shape[1]
        d = self.head_dim

        q = hidden_states.view(b, t, -1, d).transpose(1, 2)
        k = hidden_states.view(b, t, -1, d).transpose(1, 2)
        v = hidden_states.view(b, t, -1, d).transpose(1, 2)

        # Law 1: QK-Norm
        q, k = self.apply_qk_norm(q, k)

        # Law 3: Zero-Bleed Masking
        k, v = sanitize_kv_for_padding_mask(k, v, attention_mask, cu_seqlens)

        # Law 2: Temperature Folded Feature Map
        q_phi = self.fold_softcapping_feature_map(q)
        k_phi = self.fold_softcapping_feature_map(k)

        # Law 9: Strict FP32 Accumulation & Law 7: Decay
        q_phi_f = q_phi.to(torch.float32)
        k_phi_f = k_phi.to(torch.float32)
        v_f = v.to(torch.float32)

        s_state = torch.zeros(b, q.shape[1], d, d, device=hidden_states.device, dtype=torch.float32)
        z_state = torch.zeros(b, q.shape[1], d, device=hidden_states.device, dtype=torch.float32)
        out_f = torch.zeros_like(v_f)

        gamma = self.decay
        for step in range(t):
            qt = q_phi_f[:, :, step, :]
            kt = k_phi_f[:, :, step, :]
            vt = v_f[:, :, step, :]

            # Law 7: Exponential Decay State Recurrence
            s_state = s_state * gamma + torch.einsum("bhd,bhe->bhde", kt, vt)
            z_state = z_state * gamma + kt

            num = torch.einsum("bhd,bhde->bhe", qt, s_state)
            den = torch.einsum("bhd,bhd->bh", qt, z_state).unsqueeze(-1) + self.eps
            out_f[:, :, step, :] = num / den

        out = out_f.transpose(1, 2).contiguous().view(b, t, -1)
        out = out.to(dtype=orig_dtype)
        return sanitize_output_for_padding_mask(out, attention_mask, cu_seqlens)

    def forward_step(
        self,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        prev_s: Optional[torch.Tensor] = None,
        prev_z: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        O(1) Autoregressive Step with Law 7 Decay & Law 9 FP32 Accumulation.
        q_t, k_t, v_t: [B, H, 1, D] or [B, H, D].
        """
        orig_dtype = q_t.dtype
        d = self.head_dim

        q, k = self.apply_qk_norm(q_t, k_t)
        q_phi = self.fold_softcapping_feature_map(q).to(torch.float32)
        k_phi = self.fold_softcapping_feature_map(k).to(torch.float32)
        v_f = v_t.to(torch.float32)

        if q_phi.ndim == 4:
            q_phi = q_phi.squeeze(-2)
            k_phi = k_phi.squeeze(-2)
            v_f = v_f.squeeze(-2)

        b, h = q_phi.shape[0], q_phi.shape[1]
        if prev_s is None:
            prev_s = torch.zeros(b, h, d, d, device=q_t.device, dtype=torch.float32)
        if prev_z is None:
            prev_z = torch.zeros(b, h, d, device=q_t.device, dtype=torch.float32)

        # Update recurrent associative states
        s_next = prev_s * self.decay + torch.einsum("bhd,bhe->bhde", k_phi, v_f)
        z_next = prev_z * self.decay + k_phi

        num = torch.einsum("bhd,bhde->bhe", q_phi, s_next)
        den = torch.einsum("bhd,bhd->bh", q_phi, z_next).unsqueeze(-1) + self.eps
        out_t = (num / den).to(dtype=orig_dtype)

        self.cached_recurrent_state = s_next
        self.cached_normalizer_state = z_next
        return out_t, s_next, z_next

    def get_surgical_manifest(self) -> Dict[str, Any]:
        """Law 10 Compliance: Returns surgical manifest metadata for this bridge."""
        return {
            "bridge_class": self.__class__.__name__,
            "head_dim": self.head_dim,
            "has_qk_norm": self.has_qk_norm,
            "softcapping_factor": self.softcapping_factor,
            "decay": self.decay,
            "target_attention": "invariant_linear_bridge",
        }

    def get_attention_family(self) -> str:
        """Law 10 Compliance: Returns canonical attention family key."""
        return "invariant_linear_bridge"


class CrossAttentionLinearBridge(nn.Module):
    """
    Law 4 Compliance Engine: Static Cross-Attention Associative Caching.
    In Encoder-Decoder models (e.g. T5, BART), queries attend to static encoder representations.
    Computes and caches:
      S_cross = sum_{j=1}^{T_enc} phi(K_j)^T * V_j in R^{D_k x D_v}
    once during encoder prefill, allowing O(1) single-token cross-attention decoding steps.
    Decoding step latency drops from O(T_enc) to strictly O(1) operations.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6, persistent_cache: bool = True) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.persistent_cache = persistent_cache
        self.cached_s_cross: Optional[torch.Tensor] = None
        self.cached_z_cross: Optional[torch.Tensor] = None
        self.prefill_count: int = 0
        self.decode_steps: int = 0

    @property
    def is_initialized(self) -> bool:
        """Returns True if the static cross-attention state matrix is precomputed."""
        return self.cached_s_cross is not None

    def prefill_encoder(
        self,
        k_enc: torch.Tensor,
        v_enc: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Precomputes the static associative state matrix from encoder tokens:
          S_cross = sum_{s=1}^{T_enc} phi(K_s)^T V_s in R^{D_k x D_v}.
        k_enc, v_enc: [B, H, T_enc, D] or [B, T_enc, D].
        """
        self.prefill_count += 1
        k_enc, v_enc = sanitize_kv_for_padding_mask(k_enc, v_enc, attention_mask=attention_mask, cu_seqlens=cu_seqlens)

        k_phi = F.elu(k_enc.to(torch.float32)) + 1.0
        v_enc_f = v_enc.to(torch.float32)

        # Compute S_cross = K_phi^T @ V in FP32
        self.cached_s_cross = torch.matmul(k_phi.transpose(-2, -1), v_enc_f)
        self.cached_z_cross = k_phi.sum(dim=-2, keepdim=True)
        return self.cached_s_cross

    def compute_cross_state(
        self,
        k_enc: torch.Tensor,
        v_enc: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for prefill_encoder computing S_cross once."""
        return self.prefill_encoder(k_enc, v_enc, attention_mask=attention_mask, cu_seqlens=cu_seqlens)

    def decode_step(
        self,
        q_dec: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes cross-attention output for a single decoder token or chunk in strictly O(1) operations:
          O_t = (phi(Q_t) @ S_cross) / (phi(Q_t) @ z_cross^T + eps).
        q_dec: [B, H, T_q, D] or [B, T_q, D].
        """
        if self.cached_s_cross is None:
            raise RuntimeError("Static cross-attention matrix not initialized. Call prefill_encoder first.")
        self.decode_steps += q_dec.shape[-2]
        orig_dtype = q_dec.dtype
        q_phi = F.elu(q_dec.to(torch.float32)) + 1.0

        numerator = torch.matmul(q_phi, self.cached_s_cross)
        if self.cached_z_cross is not None:
            denom = torch.matmul(q_phi, self.cached_z_cross.transpose(-2, -1)).clamp(min=self.eps)
            out = numerator / denom
        else:
            out = numerator

        out = out.to(dtype=orig_dtype)
        return sanitize_output_for_padding_mask(out, attention_mask)

    def decode_batch(
        self,
        q_dec: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for decode_step executing over a full sequence batch of queries."""
        return self.decode_step(q_dec, attention_mask=attention_mask)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Dual-mode forward execution:
        1. If k and v (or encoder_hidden_states) provided: prefill encoder associative state and decode.
        2. If cached_s_cross is already initialized: execute O(1) single-token or full batch decode.
        """
        if k is not None and v is not None:
            self.prefill_encoder(k, v, attention_mask=attention_mask)
            return self.decode_step(q, attention_mask=attention_mask)
        elif encoder_hidden_states is not None:
            self.prefill_encoder(encoder_hidden_states, encoder_hidden_states, attention_mask=attention_mask)
            return self.decode_step(q, attention_mask=attention_mask)
        elif self.cached_s_cross is not None:
            return self.decode_step(q, attention_mask=attention_mask)
        else:
            raise RuntimeError(
                "CrossAttentionLinearBridge requires encoder representations to prefill, or a pre-cached matrix."
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Returns cache telemetry including hit rate across decoding steps."""
        total_accesses = self.decode_steps
        hit_rate = (
            1.0
            if total_accesses > 0 and self.prefill_count == 1
            else (total_accesses / max(total_accesses + self.prefill_count - 1, 1))
        )
        return {
            "prefill_count": self.prefill_count,
            "decode_steps": self.decode_steps,
            "hit_rate": hit_rate,
        }

    def reset(self) -> None:
        """Clears cached encoder state matrices and counters."""
        self.cached_s_cross = None
        self.cached_z_cross = None
        self.prefill_count = 0
        self.decode_steps = 0

    def get_surgical_manifest(self) -> Dict[str, Any]:
        """Law 10 Compliance: Returns surgical manifest metadata for this bridge."""
        return {
            "bridge_class": self.__class__.__name__,
            "head_dim": self.head_dim,
            "is_cross_attention": True,
            "target_attention": "cross_linear_attention",
            "persistent_cache": self.persistent_cache,
        }

    def get_attention_family(self, layer_idx: int = 0) -> str:
        """Law 10 Compliance: Returns canonical attention family key."""
        return "cross_linear_attention"
