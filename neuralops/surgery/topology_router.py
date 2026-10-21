#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Topology Router

Routes attention topologies and sanitizes masks for causal, bidirectional, and cross-attention.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("NeuralOps.Surgery.TopologyRouter")


def extract_binary_token_mask(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """
    Normalizes any HuggingFace attention mask format into a canonical (B, 1, T, 1) float keep-mask
    where valid tokens are 1.0 and padding/masked tokens are 0.0.

    Handles:
      - 2D binary integer mask: (B, T) where 1=valid, 0=pad
      - 3D attention mask: (B, 1, T) or (B, T, T)
      - 4D additive float mask: (B, 1, 1, T) or (B, 1, T, T) where 0.0=valid, -10000.0/-inf=pad
    """
    if attention_mask is None:
        return None

    mask = attention_mask
    if device is not None and mask.device != device:
        mask = mask.to(device=device)

    def _keep_values(value: torch.Tensor) -> torch.Tensor:
        if value.dtype == torch.bool:
            return value
        if torch.is_floating_point(value):
            # HuggingFace uses both binary 0/1 masks and additive masks whose
            # blocked positions are negative (often -inf or -1e4).
            if bool(torch.any(value < 0)):
                return value > -100.0
        return value > 0.5

    def _collapse_key_axis(allowed: torch.Tensor) -> torch.Tensor:
        # Attention masks can vary by query and head. A key is padding only
        # when no query/head can attend to it; this preserves padding
        # suppression without treating a causal triangle as token padding.
        if allowed.ndim == 4:
            allowed = allowed.any(dim=-2).any(dim=1)
        elif allowed.ndim == 3:
            allowed = allowed.any(dim=-2)
        if allowed.ndim == 2:
            return allowed.unsqueeze(1).unsqueeze(-1).to(dtype=dtype)
        return allowed

    if mask.ndim == 4:
        if mask.shape[-1] != seq_len:
            return None
        return _collapse_key_axis(_keep_values(mask))

    if mask.ndim == 2:
        if mask.shape[-1] != seq_len:
            return None
        return _keep_values(mask).to(dtype=dtype).unsqueeze(1).unsqueeze(-1)

    if mask.ndim == 3:
        if mask.shape[-1] != seq_len:
            return None
        if mask.shape[1] == 1:
            return _keep_values(mask).to(dtype=dtype).unsqueeze(-1)
        if mask.shape[-2] == seq_len:
            return _collapse_key_axis(_keep_values(mask))

    return None


def sanitize_kv_for_padding_mask(
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multiplies Key and Value tensors by 0.0 at all padding positions.
    Guarantees that padded tokens do not pollute linear/recurrent memory states.
    k: (B, H, T, D) or (B, T, D)
    v: (B, H, T, D) or (B, T, D)
    """
    if attention_mask is None:
        return k, v

    if k.shape != v.shape:
        raise ValueError(f"Key/value shape mismatch: {tuple(k.shape)} != {tuple(v.shape)}")
    if k.ndim not in (3, 4):
        raise ValueError(f"Expected key/value tensors with rank 3 or 4, got rank {k.ndim}")

    seq_len = k.shape[-2]
    keep_mask = extract_binary_token_mask(
        attention_mask, seq_len=seq_len, device=k.device, dtype=k.dtype
    )
    if keep_mask is None:
        return k, v

    if k.ndim == 4:
        mask_shape = (k.shape[0], 1, k.shape[-2], 1)
    else:
        mask_shape = (k.shape[0], k.shape[-2], 1)
        keep_mask = keep_mask.squeeze(1)
    keep_mask = keep_mask.reshape(mask_shape)
    k_sanitized = k * keep_mask
    v_sanitized = v * keep_mask

    return k_sanitized, v_sanitized


class BiDirectionalScanWrapper(nn.Module):
    """
    Wraps any unidirectional causal recurrent/linear attention module to execute
    symmetrically on bidirectional encoder models (BERT, RoBERTa, DistilBERT, ViT).

    Computes:
        O_fwd = CoreAttention(Q, K, V)
        O_bwd = Flip(CoreAttention(Flip(Q), Flip(K), Flip(V)))
        O_bi  = 0.5 * (O_fwd + O_bwd)
    """

    def __init__(self, core_attention: nn.Module, d_model: Optional[int] = None) -> None:
        super().__init__()
        self.core_attention = core_attention
        self.d_model = d_model
        if d_model is not None:
            self.fusion_gate = nn.Linear(d_model * 2, d_model)
            with torch.no_grad():
                self.fusion_gate.weight.zero_()
                self.fusion_gate.bias.zero_()
        else:
            self.fusion_gate = None

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        B, T, D = hidden_states.shape
        attention_mask = kwargs.get("attention_mask", None)

        out_fwd = self.core_attention(hidden_states, *args, **kwargs)
        if isinstance(out_fwd, tuple):
            core_fwd = out_fwd[0]
            extra_fwd = out_fwd[1:]
        else:
            core_fwd = out_fwd
            extra_fwd = ()

        x_flipped = torch.flip(hidden_states, dims=[1])
        kwargs_flipped = dict(kwargs)
        if attention_mask is not None and attention_mask.ndim in (2, 3, 4):
            mask_dims = [-1]
            if attention_mask.ndim >= 3 and attention_mask.shape[-2] == T:
                mask_dims.append(-2)
            kwargs_flipped["attention_mask"] = torch.flip(attention_mask, dims=mask_dims)

        out_bwd = self.core_attention(x_flipped, *args, **kwargs_flipped)
        core_bwd = out_bwd[0] if isinstance(out_bwd, tuple) else out_bwd
        core_bwd_unflipped = torch.flip(core_bwd, dims=[1])

        if self.fusion_gate is not None:
            cat_feat = torch.cat([core_fwd, core_bwd_unflipped], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(cat_feat))
            fused_core = gate * core_fwd + (1.0 - gate) * core_bwd_unflipped
        else:
            fused_core = 0.5 * (core_fwd + core_bwd_unflipped)

        if len(extra_fwd) > 0:
            return (fused_core, *extra_fwd)
        return fused_core


class AsymmetricCrossAttentionRouter(nn.Module):
    """
    Handles cross-attention where Query length T_q differs from Key/Value length T_kv.
    Uses PyTorch SDPA for unequal query and key/value lengths while preserving
    the supplied padding or pairwise attention mask.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError(f"d_model={d_model} must be divisible by positive num_heads={num_heads}")
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        B, T_q, D = hidden_states.shape
        if D != self.d_model or encoder_hidden_states.ndim != 3:
            raise ValueError(
                "Cross-attention expects query and encoder states shaped "
                f"(batch, sequence, {self.d_model}); got query={tuple(hidden_states.shape)}, "
                f"encoder={tuple(encoder_hidden_states.shape)}"
            )
        if encoder_hidden_states.shape[0] != B or encoder_hidden_states.shape[-1] != self.d_model:
            raise ValueError("Query and encoder states must share batch and feature dimensions")
        T_kv = encoder_hidden_states.shape[1]
        H, D_h = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, T_q, H, D_h).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).view(B, T_kv, H, D_h).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).view(B, T_kv, H, D_h).transpose(1, 2)

        sdpa_mask = attention_mask
        if sdpa_mask is not None:
            sdpa_mask = sdpa_mask.to(device=q.device)
            if sdpa_mask.ndim == 2:
                if sdpa_mask.shape != (B, T_kv):
                    raise ValueError(
                        f"2D cross-attention mask must have shape {(B, T_kv)}, "
                        f"got {tuple(sdpa_mask.shape)}"
                    )
                if sdpa_mask.dtype != torch.bool and not torch.is_floating_point(sdpa_mask):
                    sdpa_mask = sdpa_mask > 0
                elif torch.is_floating_point(sdpa_mask) and bool(torch.any(sdpa_mask < 0)):
                    sdpa_mask = sdpa_mask[:, None, None, :]
                else:
                    sdpa_mask = sdpa_mask > 0.5
                if sdpa_mask.ndim == 2:
                    sdpa_mask = sdpa_mask[:, None, None, :]
            elif sdpa_mask.ndim == 3:
                if sdpa_mask.shape[0] != B or sdpa_mask.shape[-1] != T_kv:
                    raise ValueError("3D cross-attention mask must match batch and key/value length")
                sdpa_mask = sdpa_mask.unsqueeze(1)
                if torch.is_floating_point(sdpa_mask) and not bool(torch.any(sdpa_mask < 0)):
                    sdpa_mask = sdpa_mask > 0.5
            elif sdpa_mask.ndim == 4:
                if sdpa_mask.shape[0] != B or sdpa_mask.shape[-1] != T_kv:
                    raise ValueError("4D cross-attention mask must match batch and key/value length")
                if sdpa_mask.shape[-2] not in (1, T_q):
                    raise ValueError("4D cross-attention mask query dimension must be 1 or query length")
                if torch.is_floating_point(sdpa_mask) and not bool(torch.any(sdpa_mask < 0)):
                    sdpa_mask = sdpa_mask > 0.5
            else:
                raise ValueError(f"Unsupported cross-attention mask rank: {sdpa_mask.ndim}")

        is_causal = bool(kwargs.get("is_causal", False)) and sdpa_mask is None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(out), None


__all__ = [
    "extract_binary_token_mask",
    "sanitize_kv_for_padding_mask",
    "BiDirectionalScanWrapper",
    "AsymmetricCrossAttentionRouter",
]
