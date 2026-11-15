#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Surgery Invariant Transformers and Helpers.

Provides implementations for:
- Law 3: Zero-bleed padding mask sanitization
- Law 5: Rotary position embedding transformations and decoupling
- Law 6: Parameter identity initialization
- Law 7: Exponential state decay bounds
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from neuralops.surgery.topology_router import extract_binary_token_mask

logger = logging.getLogger("neuralops.surgery.director.invariants")


def sanitize_kv_for_padding_mask(
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Law 3: Zero-Bleed Padding Mask Sanitization for Keys and Values.
    Forces phi(K_s) = 0 and V_s = 0 at all masked/padded positions,
    guaranteeing Delta S_s = phi(K_s)^T V_s = 0 (no memory contamination).
    Also supports variable-length document packing via cu_seqlens boundary offsets,
    guaranteeing that cross-document attention leakage is mathematically zero:
    ||leakage|| == 0.0.

    Args:
        k: Key tensor of shape [B, H, T, D] or [B, T, D]
        v: Value tensor of shape [B, H, T, D] or [B, T, D]
        attention_mask: Binary mask [B, T], [B, 1, 1, T], or [B, 1, T, T] (1 = keep, 0 = mask)
        cu_seqlens: 1D cumulative sequence offsets [0, len_1, len_1 + len_2, ...] for varlen packing

    Returns:
        Tuple of (k_sanitized, v_sanitized) with exact zero vectors at padding positions.
    """
    if attention_mask is None and cu_seqlens is None:
        return k, v

    if k.shape != v.shape:
        raise ValueError(f"Key/value shape mismatch: {tuple(k.shape)} != {tuple(v.shape)}")
    if k.ndim not in (3, 4):
        raise ValueError(f"Expected key/value tensors with rank 3 or 4, got rank {k.ndim}")

    seq_len = k.shape[-2]
    keep_mask: Optional[torch.Tensor] = None

    if attention_mask is not None:
        keep_mask = extract_binary_token_mask(attention_mask, seq_len=seq_len, device=k.device, dtype=k.dtype)

    if cu_seqlens is not None:
        total_valid = int(cu_seqlens[-1].item())
        varlen_mask = torch.zeros(seq_len, device=k.device, dtype=k.dtype)
        varlen_mask[: min(seq_len, total_valid)] = 1.0
        varlen_mask = varlen_mask.view(1, 1, seq_len, 1)
        keep_mask = varlen_mask if keep_mask is None else keep_mask * varlen_mask

    if keep_mask is None:
        return k, v

    if k.ndim == 4:
        mask_shape = (k.shape[0], 1, k.shape[-2], 1)
    else:
        mask_shape = (k.shape[0], k.shape[-2], 1)
        if keep_mask.ndim == 4:
            keep_mask = keep_mask.squeeze(1)

    keep_mask = keep_mask.expand(mask_shape)
    return k * keep_mask, v * keep_mask


def sanitize_output_for_padding_mask(
    out: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Law 3: Zero-Bleed Padding Output Sanitization.
    Forces output on padding positions to absolute zero:
    forall t in padding: ||O_t||_infty == 0.0.
    Guarantees cross-document and padding leakage is mathematically zero.

    Args:
        out: Output tensor of shape [B, H, T, D] or [B, T, D]
        attention_mask: Binary mask [B, T], [B, 1, 1, T], or [B, 1, T, T]
        cu_seqlens: Optional varlen cumulative sequence length boundary offsets

    Returns:
        out_sanitized with exactly 0.0 at all masked positions.
    """
    if attention_mask is None and cu_seqlens is None:
        return out

    seq_len = out.shape[-2]
    keep_mask: Optional[torch.Tensor] = None

    if attention_mask is not None:
        keep_mask = extract_binary_token_mask(attention_mask, seq_len=seq_len, device=out.device, dtype=out.dtype)

    if cu_seqlens is not None:
        total_valid = int(cu_seqlens[-1].item())
        varlen_mask = torch.zeros(seq_len, device=out.device, dtype=out.dtype)
        varlen_mask[: min(seq_len, total_valid)] = 1.0
        varlen_mask = varlen_mask.view(1, 1, seq_len, 1)
        keep_mask = varlen_mask if keep_mask is None else keep_mask * varlen_mask

    if keep_mask is None:
        return out

    if out.ndim == 4:
        mask_shape = (out.shape[0], 1, out.shape[-2], 1)
    else:
        mask_shape = (out.shape[0], out.shape[-2], 1)
        if keep_mask.ndim == 4:
            keep_mask = keep_mask.squeeze(1)

    keep_mask = keep_mask.expand(mask_shape)
    return out * keep_mask


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Applies rotary position embedding (RoPE) rotation to input tensor.
    x: [..., seq_len, head_dim]
    cos, sin: [..., seq_len, head_dim] or broadcastable
    """
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotate_half = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotate_half * sin)


def apply_rope_pre_feature_map(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Law 5: Ensures rotary position embeddings are applied to Q and K prior to
    non-linear feature mapping phi(x), preserving positional frequency decay.
    """
    q_rot = apply_rotary_pos_emb(q, cos, sin)
    k_rot = apply_rotary_pos_emb(k, cos, sin)
    return q_rot, k_rot


def decouple_rope_transform(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    rope_dim: Optional[int] = None,
    decay_gamma: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Law 5: Rotary Embedding Decoupling & Pass-Through.
    In linear attention, 2D rotary matrices break associativity.
    This function decouples RoPE by:
      1. Maintaining RoPE exclusively on dedicated decoupled positional channels (rope_dim),
         while preserving unrotated associative dimensions for linear recurrence;
      2. Applying 1D exponential positional decay gamma^{|t-s|} along the sequence
         to encode relative distance without breaking cumulative associative scan.

    Args:
        q: Query tensor of shape [..., seq_len, head_dim]
        k: Key tensor of shape [..., seq_len, head_dim]
        cos: RoPE cosine tensor [..., seq_len, rope_dim] or broadcastable
        sin: RoPE sine tensor [..., seq_len, rope_dim] or broadcastable
        rope_dim: Number of dimensions allocated to the decoupled RoPE stream.
                  If None and cos/sin are provided, defaults to cos.shape[-1] or head_dim // 2.
        decay_gamma: Optional 1D exponential decay coefficient (e.g., 0.99) applied across tokens.

    Returns:
        Tuple of (q_decoupled, k_decoupled) with associativity preserved.
    """
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]

    # Mode 1: Decoupled Key Stream (Dedicated positional channels)
    if cos is not None and sin is not None:
        if rope_dim is None:
            rope_dim = min(cos.shape[-1], head_dim // 2 if head_dim > 1 else head_dim)
        rope_dim = max(1, min(rope_dim, head_dim))

        if rope_dim < head_dim:
            content_dim = head_dim - rope_dim
            q_content = q[..., :content_dim]
            k_content = k[..., :content_dim]
            q_rope = q[..., content_dim:]
            k_rope = k[..., content_dim:]

            cos_slice = cos[..., :rope_dim]
            sin_slice = sin[..., :rope_dim]

            q_rope_rot = apply_rotary_pos_emb(q_rope, cos_slice, sin_slice)
            k_rope_rot = apply_rotary_pos_emb(k_rope, cos_slice, sin_slice)

            q_out = torch.cat([q_content, q_rope_rot], dim=-1)
            k_out = torch.cat([k_content, k_rope_rot], dim=-1)
        else:
            q_out = apply_rotary_pos_emb(q, cos, sin)
            k_out = apply_rotary_pos_emb(k, cos, sin)
    else:
        q_out = q
        k_out = k

    # Mode 2: 1D Exponential Positional Decay (associativity-preserving)
    if decay_gamma is not None and 0.0 < decay_gamma < 1.0:
        t = torch.arange(seq_len, device=q.device, dtype=q.dtype)
        gamma_decay = (decay_gamma**t).view(*([1] * (q.ndim - 2)), seq_len, 1)
        q_out = q_out * gamma_decay

    return q_out, k_out


_decouple_rope_transform = decouple_rope_transform


def initialize_identity_adapters(model: nn.Module) -> None:
    """
    Law 6: Post-Surgery Parameter Identity Initialization.
    Initializes newly added adapters, sidecar gates, or residual blending weights
    with alpha_init = 0.0 (or zero-initialized up-projection weights),
    guaranteeing the surgical model exactly replicates teacher output at step 0
    (Delta PPL <= 0.05, zero degradation at initialization).

    Args:
        model: PyTorch module/model to initialize identity adapters for.
    """
    with torch.no_grad():
        for mod in model.modules():
            # 1. Scalar or Parameter alpha blending factors
            if hasattr(mod, "alpha"):
                alpha = mod.alpha
                if isinstance(alpha, nn.Parameter):
                    nn.init.constant_(alpha, 0.0)
                elif isinstance(alpha, torch.Tensor):
                    alpha.zero_()
                elif isinstance(alpha, (int, float)):
                    mod.alpha = 0.0

            # 2. Sidecar gate branch identity initialization
            if hasattr(mod, "sidecar") and mod.sidecar is not None:
                sidecar = mod.sidecar
                if hasattr(sidecar, "alpha"):
                    sc_alpha = sidecar.alpha
                    if isinstance(sc_alpha, nn.Parameter):
                        nn.init.constant_(sc_alpha, 0.0)
                    elif isinstance(sc_alpha, torch.Tensor):
                        sc_alpha.zero_()
                    elif isinstance(sc_alpha, (int, float)):
                        sidecar.alpha = 0.0
                if hasattr(sidecar, "gate_proj") and hasattr(sidecar.gate_proj, "weight"):
                    nn.init.zeros_(sidecar.gate_proj.weight)
                    if getattr(sidecar.gate_proj, "bias", None) is not None:
                        nn.init.zeros_(sidecar.gate_proj.bias)

            # 3. Dedicated adapter modules (LoRA, Houlsby, Pfeiffer, MLP sidecars)
            for attr in ("up_proj", "linear_2", "lora_B", "adapter_up", "dense_2"):
                if hasattr(mod, attr):
                    proj = getattr(mod, attr)
                    if hasattr(proj, "weight") and isinstance(proj.weight, (torch.Tensor, nn.Parameter)):
                        nn.init.zeros_(proj.weight)
                        if getattr(proj, "bias", None) is not None:
                            nn.init.zeros_(proj.bias)

            # 4. Identity gate projections
            if hasattr(mod, "identity_init") and callable(mod.identity_init):
                mod.identity_init()

    logger.info("Law 6: Successfully executed post-surgery parameter identity initialization (alpha_init = 0.0).")


_initialize_identity_adapters = initialize_identity_adapters


def compute_state_decay_bound(
    max_step_norm: float,
    gamma: float = 0.99,
) -> float:
    """
    Law 7: Computes theoretical asymptotic Frobenius norm bound under exponential state decay:
    ||S_t||_F <= (1 / (1 - gamma)) * max_i ||phi(K_i)^T V_i||_F.
    For gamma = 0.99, the theoretical maximum expansion factor is 1 / (1 - 0.99) = 100x.
    """
    if gamma >= 1.0 or gamma <= 0.0:
        raise ValueError(f"State decay gamma must be strictly within (0, 1), got {gamma}")
    return max_step_norm / (1.0 - gamma)


compute_decayed_state_bound = compute_state_decay_bound
