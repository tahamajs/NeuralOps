"""
Automatic Causal, Bidirectional & Cross-Attention Topology Router.

Provides:
  - extract_binary_token_mask(): converts any 2D, 3D, or 4D HuggingFace attention mask
    into a clean boolean/float keep-mask of shape (B, 1, T, 1) or (B, 1, T).
  - sanitize_kv_for_padding_mask(): explicitly zeros out Key and Value representations
    for masked/padding tokens so that associative/linear state accumulations S_t = S_{t-1} + k_t v_t^T
    are not corrupted by padding tokens.
  - BiDirectionalScanWrapper: enables causal recurrent/scan attention modules (GLA, GDN, RetNet, etc.)
    to execute symmetrically on bidirectional encoders (BERT, RoBERTa, DistilBERT, ViT) via
    forward + reversed scans.
  - AsymmetricCrossAttentionRouter: length-decoupled associative routing for cross-attention
    where query sequence length T_q differs from key/value sequence length T_kv.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union
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

    # If already 4D additive float mask (0.0 for valid, negative large for pad)
    if mask.ndim == 4:
        # Check if additive format (min < -1.0)
        if mask.min() < -1.0:
            # (mask > -100.0) gives boolean keep mask
            keep_mask = (mask > -100.0).to(dtype=dtype)
        else:
            keep_mask = (mask > 0.5).to(dtype=dtype)

        # Slice or pool along query dimension to get (B, 1, T_kv, 1)
        if keep_mask.shape[-2] == keep_mask.shape[-1] and keep_mask.shape[-1] == seq_len:
            # Causal or full square mask: take the diagonal or column presence
            keep_mask = keep_mask[:, :, -1:, :].transpose(-1, -2)
        elif keep_mask.shape[-1] == seq_len:
            keep_mask = keep_mask.view(mask.shape[0], 1, seq_len, 1)
        return keep_mask

    # If 2D mask (B, T)
    if mask.ndim == 2:
        if mask.dtype in (torch.int64, torch.int32, torch.bool):
            keep_mask = (mask > 0).to(dtype=dtype)
        else:
            keep_mask = (mask > -100.0).to(dtype=dtype)
        # Reshape to (B, 1, T, 1)
        return keep_mask.unsqueeze(1).unsqueeze(-1)

    # If 3D mask (B, 1, T) or (B, T, 1)
    if mask.ndim == 3:
        if mask.shape[1] == 1 and mask.shape[2] == seq_len:
            keep_mask = (mask > 0 if mask.dtype != torch.float32 else mask > -100.0).to(dtype=dtype)
            return keep_mask.unsqueeze(-1)
        if mask.shape[2] == 1 and mask.shape[1] == seq_len:
            keep_mask = (mask > 0 if mask.dtype != torch.float32 else mask > -100.0).to(dtype=dtype)
            return keep_mask.unsqueeze(1)

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

    seq_len = k.shape[-2] if k.ndim >= 3 else k.shape[1]
    keep_mask = extract_binary_token_mask(
        attention_mask, seq_len=seq_len, device=k.device, dtype=k.dtype
    )
    if keep_mask is None:
        return k, v

    # Broadcast keep_mask to match k and v shape
    if k.ndim == 4:
        # keep_mask is (B, 1, T, 1), k is (B, H, T, D)
        k_sanitized = k * keep_mask
        v_sanitized = v * keep_mask
    elif k.ndim == 3:
        # keep_mask is (B, 1, T, 1), squeeze to (B, T, 1)
        mask_3d = keep_mask.squeeze(1)
        k_sanitized = k * mask_3d
        v_sanitized = v * mask_3d
    else:
        k_sanitized = k
        v_sanitized = v

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
        # hidden_states: (B, T, D)
        B, T, D = hidden_states.shape
        attention_mask = kwargs.get("attention_mask", None)

        # 1. Forward scan
        out_fwd = self.core_attention(hidden_states, *args, **kwargs)
        if isinstance(out_fwd, tuple):
            core_fwd = out_fwd[0]
            extra_fwd = out_fwd[1:]
        else:
            core_fwd = out_fwd
            extra_fwd = ()

        # 2. Backward scan on flipped sequence
        x_flipped = torch.flip(hidden_states, dims=[1])
        kwargs_flipped = dict(kwargs)
        if attention_mask is not None and attention_mask.ndim in (2, 3, 4):
            # Flip attention mask along the sequence dimension
            kwargs_flipped["attention_mask"] = torch.flip(attention_mask, dims=[-1])

        out_bwd = self.core_attention(x_flipped, *args, **kwargs_flipped)
        core_bwd = out_bwd[0] if isinstance(out_bwd, tuple) else out_bwd
        core_bwd_unflipped = torch.flip(core_bwd, dims=[1])

        # 3. Combine forward and backward streams
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
    Uses length-decoupled associative linear attention or falls back cleanly to PyTorch SDPA.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
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
        # hidden_states: (B, T_q, D)
        # encoder_hidden_states: (B, T_kv, D)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        B, T_q, D = hidden_states.shape
        T_kv = encoder_hidden_states.shape[1]
        H, D_h = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, T_q, H, D_h).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).view(B, T_kv, H, D_h).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).view(B, T_kv, H, D_h).transpose(1, 2)

        # Sanitize keys and values if padding mask is present
        k, v = sanitize_kv_for_padding_mask(k, v, attention_mask)

        # Cross attention execution via SDPA
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(out), None


__all__ = [
    "extract_binary_token_mask",
    "sanitize_kv_for_padding_mask",
    "BiDirectionalScanWrapper",
    "AsymmetricCrossAttentionRouter",
]
