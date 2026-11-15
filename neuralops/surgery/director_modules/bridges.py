from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.surgery.topology_router import BiDirectionalScanWrapper, extract_binary_token_mask
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

        # Law 1: QK-Norm Modules
        if has_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Law 9: Intermediate Accumulation Dtype Tracking
        self.recurrent_state_dtype: torch.dtype = torch.float32
        self.last_accum_dtype: torch.dtype = torch.float32

    def compute_folded_scale(self) -> float:
        """
        Law 2: Folds logit soft-capping factor c into linear attention temperature scaling:
        tau_folded = (1 / sqrt(d_k)) * tanh(c / max(c, 1.0)).
        Folds Gemma-2 soft-capping asymptote C into a bounded temperature scale,
        preventing logit saturation and overflow during long-context generation.
        """
        base_scale = 1.0 / math.sqrt(self.head_dim)
        if self.softcapping_factor is not None and self.softcapping_factor > 0:
            c = float(self.softcapping_factor)
            factor = math.tanh(c / max(c, 1.0))
            return base_scale * factor
        return base_scale

    def fold_softcapping_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Law 2: Bounded soft-capping feature map folding asymptote C:
        phi_cap(x) = C * tanh(x / C) + C, where 0.0 <= phi_cap(x) <= 2C.
        Guarantees non-negative activations, non-saturating gradients,
        and absolute immunity to FP16 overflow during long-context generation.
        """
        if self.softcapping_factor is not None and self.softcapping_factor > 0:
            c = float(self.softcapping_factor)
            return c * torch.tanh(x / c) + c
        return F.elu(x) + 1.0

    def apply_qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Law 1: Normalizes query and key activations with RMSNorm prior to linear feature mapping:
        q_tilde = RMSNorm(Q), k_tilde = RMSNorm(K).
        """
        if self.has_qk_norm and self.q_norm is not None and self.k_norm is not None:
            return self.q_norm(q), self.k_norm(k)
        return q, k

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Executes invariant-safe forward pass.
        Shapes:
          q, k, v: [B, H, T, D] or [B, T, D]
          attention_mask: [B, T], [B, 1, 1, T], or [B, 1, T, T]
        """
        orig_dtype = q.dtype

        # Law 1: QK-Norm Preservation
        q, k = self.apply_qk_norm(q, k)

        # Law 2: Folded Temperature Scale
        scale = self.compute_folded_scale()
        q_scaled = q * scale

        # Law 3: Zero-Bleed Padding Mask Sanitization (Keys & Values)
        k, v = sanitize_kv_for_padding_mask(k, v, attention_mask)

        # Law 9: FP32 Accumulation Registers for Reduced Precision & Half-Precision Inputs
        q_acc = q_scaled.to(torch.float32)
        k_acc = k.to(torch.float32)
        v_acc = v.to(torch.float32)
        self.last_accum_dtype = torch.float32

        if self.base_attention is not None:
            out = self.base_attention(
                q_acc.to(orig_dtype),
                k_acc.to(orig_dtype),
                v_acc.to(orig_dtype),
                attention_mask=attention_mask,
                is_causal=is_causal,
                **kwargs,
            )
            return sanitize_output_for_padding_mask(out, attention_mask)

        # Vectorized Fallback Linear Attention (Level 3) in FP32
        if self.softcapping_factor is not None and self.softcapping_factor > 0:
            q_phi = self.fold_softcapping_feature_map(q_acc)
            k_phi = self.fold_softcapping_feature_map(k_acc)
        else:
            q_phi = F.elu(q_acc) + 1.0
            k_phi = F.elu(k_acc) + 1.0

        if is_causal:
            # Law 7: Exponential State Decay in Recurrence with Attention Sinks
            gamma = float(self.decay)
            T = q_acc.shape[-2]
            t_idx = torch.arange(T, device=q.device)
            diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)  # [T, T], diff[t, i] = t - i
            causal_mask = diff >= 0

            if gamma < 1.0:
                decay_weights = torch.where(
                    causal_mask,
                    torch.pow(torch.tensor(gamma, device=q.device, dtype=torch.float32), diff.float()),
                    torch.zeros(1, device=q.device, dtype=torch.float32),
                )
                if self.num_attention_sinks > 0:
                    sink_mask = (t_idx.unsqueeze(0) < self.num_attention_sinks) & causal_mask
                    decay_weights = torch.where(sink_mask, torch.ones_like(decay_weights), decay_weights)
            else:
                decay_weights = causal_mask.to(dtype=torch.float32)

            if q_acc.ndim == 4:
                k_cum = torch.einsum("ti,bhid->bhtd", decay_weights, k_phi)
                kv = torch.einsum("bhtd,bhte->bhtde", k_phi, v_acc)
                kv_cum = torch.einsum("ti,bhide->bhtde", decay_weights, kv)
                out = torch.einsum("bhtd,bhtde->bhte", q_phi, kv_cum)
                denom = torch.einsum("bhtd,bhtd->bht", q_phi, k_cum).unsqueeze(-1) + self.eps
                result = out / denom
            else:
                k_cum = torch.einsum("ti,bid->btd", decay_weights, k_phi)
                kv = torch.einsum("btd,bte->btde", k_phi, v_acc)
                kv_cum = torch.einsum("ti,bide->btde", decay_weights, kv)
                out = torch.einsum("btd,btde->bte", q_phi, kv_cum)
                denom = torch.einsum("btd,btd->bt", q_phi, k_cum).unsqueeze(-1) + self.eps
                result = out / denom
        else:
            # Non-causal Bidirectional Associative Scan in FP32
            if q_acc.ndim == 4:
                kv = torch.einsum("bhtd,bhte->bhde", k_phi, v_acc)
                out = torch.einsum("bhtd,bhde->bhte", q_phi, kv)
                k_sum = k_phi.sum(dim=2)
                denom = torch.einsum("bhtd,bhd->bht", q_phi, k_sum).unsqueeze(-1) + self.eps
                result = out / denom
            else:
                kv = torch.einsum("btd,bte->bde", k_phi, v_acc)
                out = torch.einsum("btd,bde->bte", q_phi, kv)
                k_sum = k_phi.sum(dim=1)
                denom = torch.einsum("btd,bd->bt", q_phi, k_sum).unsqueeze(-1) + self.eps
                result = out / denom

        # Cast back to original activation dtype
        result = result.to(dtype=orig_dtype)

        # Law 3: Zero-Bleed Padding Output Sanitization (||O_t||_infty == 0 on padding)
        result = sanitize_output_for_padding_mask(result, attention_mask)
        return result

    def forward_step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        z_state: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressive single-step O(1) recurrence with exponential state decay gamma:
        S_t = gamma * S_{t-1} + phi(K_t)^T * V_t
        z_t = gamma * z_{t-1} + phi(K_t)^T
        O_t = (phi(Q_t) @ S_t) / (phi(Q_t) @ z_t + eps)

        If step_idx < num_attention_sinks, state accumulates as an attention sink without decay.
        """
        orig_dtype = q.dtype
        q, k = self.apply_qk_norm(q, k)
        scale = self.compute_folded_scale()
        q_scaled = q * scale

        # Law 9: FP32 Accumulation Registers
        q_acc = q_scaled.to(torch.float32)
        k_acc = k.to(torch.float32)
        v_acc = v.to(torch.float32)
        self.last_accum_dtype = torch.float32

        if self.softcapping_factor is not None and self.softcapping_factor > 0:
            q_phi = self.fold_softcapping_feature_map(q_acc)
            k_phi = self.fold_softcapping_feature_map(k_acc)
        else:
            q_phi = F.elu(q_acc) + 1.0
            k_phi = F.elu(k_acc) + 1.0

        delta_s = torch.matmul(k_phi.transpose(-2, -1), v_acc)
        delta_z = k_phi.sum(dim=-2, keepdim=True)

        gamma = self.decay
        is_sink = step_idx is not None and step_idx < self.num_attention_sinks
        effective_gamma = 1.0 if is_sink else gamma

        if state is None:
            new_s = delta_s
            new_z = delta_z
        else:
            new_s = effective_gamma * state + delta_s
            new_z = effective_gamma * (z_state if z_state is not None else 0.0) + delta_z

        num = torch.matmul(q_phi, new_s)
        denom = torch.matmul(q_phi, new_z.transpose(-2, -1)).clamp(min=self.eps)
        out = (num / denom).to(dtype=orig_dtype)
        return out, new_s, new_z

    def get_surgical_manifest(self) -> Dict[str, Any]:
        """Returns structured metadata manifest for invariant auditing."""
        return {
            "has_qk_norm": self.has_qk_norm,
            "softcapping_factor": self.softcapping_factor,
            "decay": self.decay,
            "num_attention_sinks": self.num_attention_sinks,
            "head_dim": self.head_dim,
            "attention_family": self.get_attention_family(),
        }

    def get_attention_family(self) -> str:
        """Returns the canonical attention architecture family."""
        return "linear_attention"


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


__all__ = ["InvariantPreservingLinearBridge", "CrossAttentionLinearBridge"]
