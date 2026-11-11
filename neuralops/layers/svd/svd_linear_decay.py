#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SVD Linear Decay Gated Attention

Causal linear attention combining singular value decomposition factorization with exponential decay gating.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.attention_utils import normalize_attention_mask as _normalize_mask
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.svd.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
    compute_exponential_decay,
    inject_svd_linear_attention,
)
from neuralops.ops.cuda.svd_linear_decay import svd_linear_decay

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "svd_linear_decay",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class DecayDecayLinearAttention(SVDLinearAttention):
    r"""
    Decay Gated SVD Linear Attention.

    Combines low-rank singular value decomposition factorization with an exponential
    temporal recency decay kernel. Implements linear-time O(N) recurrence:
        S_t = \lambda S_{t-1} + K_t^\top V_t
        O_t = Q_t S_t
    and chunk-wise cumulative decay prefix-scans, strictly avoiding dense N x N matrix
    materialization.
    """

    is_custom_attention: bool = True
    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_rope: bool = True

    def __init__(self, config: Any | None = None, rank: int | None = None, **kwargs: Any):
        kwargs["use_decay_gating"] = True
        kwargs.setdefault("mode", "decay_linear")
        super().__init__(config=config, rank=rank, **kwargs)
        self._return_state_requested: bool = False

    def forward(self, *args: Any, **kwargs: Any):
        self._return_state_requested = bool(kwargs.get("return_state", False))
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._return_state_requested = False

    def _forward_decay_gated_linear_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if past_state is not None or self._return_state_requested:
            return super()._forward_decay_gated_linear_kernel(q, k, v, hidden_states, mask=mask, past_state=past_state)

        B, H, N, D_k = q.shape
        compute_dtype = torch.float32

        q_f = q.to(compute_dtype).contiguous()
        k_f = k.to(compute_dtype).contiguous()
        v_f = v.to(compute_dtype).contiguous()

        mask_expanded = _normalize_mask(mask, B, H, N, compute_dtype, q.device)
        if mask_expanded is not None:
            k_f = torch.where(mask_expanded > 0, k_f, torch.full_like(k_f, -20.0))
            v_f = v_f * mask_expanded

        decay_logits = self.decay_gate(hidden_states).float()
        if self.decay_channel_wise:
            decay_logits = decay_logits.reshape(B, N, H, D_k).permute(0, 2, 1, 3).contiguous()
            lambda_decay = compute_exponential_decay(decay_logits, activation=self.decay_activation).to(compute_dtype)
        else:
            decay_logits = decay_logits.permute(0, 2, 1).contiguous().unsqueeze(-1)
            lambda_decay = compute_exponential_decay(decay_logits, activation=self.decay_activation).to(compute_dtype)

        if mask_expanded is not None:
            m_dec = mask_expanded
            while m_dec.dim() < lambda_decay.dim():
                m_dec = m_dec.unsqueeze(-1)
            while m_dec.dim() > lambda_decay.dim():
                m_dec = m_dec.squeeze(-1)
            lambda_decay = torch.where(m_dec > 0, lambda_decay, torch.ones_like(lambda_decay))

        context = svd_linear_decay(
            q_f,
            k_f.contiguous(),
            v_f.contiguous(),
            lambda_decay.contiguous(),
            eps=self.eps,
        ).to(q.dtype)
        D_v = v.shape[-1]
        S_final = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
        Z_final = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)
        return context, (S_final, Z_final)


SVDLinearDecayAttention = DecayDecayLinearAttention


def inject_svd_linear_decay(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_svd_linear_decay(
                layer=lyr,
                model_config=cfg,
                comp_config=(comp_config if "comp_config" in locals() or "comp_config" in kwargs else None),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    src_attn = old_attn
    src_attr_name: str | None = None
    src_is_nested: bool = False

    if src_attn is None:
        for attr in ("attn", "attention", "self_attn", "self"):
            candidate = getattr(layer, attr, None)
            if candidate is None:
                continue

            if isinstance(candidate, SVDLinearAttention) or "SVDLinear" in candidate.__class__.__name__:
                logger.debug(
                    "inject_svd_linear_decay: %s.%s is already SVDLinearAttention, skipping",
                    type(layer).__name__,
                    attr,
                )
                return None

            if hasattr(candidate, "self") and hasattr(candidate.self, "query"):
                src_attn = candidate.self
                src_attr_name = attr
                src_is_nested = True
            else:
                src_attn = candidate
                src_attr_name = attr
                src_is_nested = False
            break

    if src_attn is None:
        logger.warning(
            "inject_svd_linear_decay: no source attention found on %s " "(checked: attn, attention, self_attn, self)",
            type(layer).__name__,
        )
        return None

    kwargs["use_decay_gating"] = True
    kwargs.setdefault("mode", "decay_linear")
    try:
        result = inject_svd_linear_attention(
            model=model,
            layer=layer,
            model_config=model_config,
            old_attn=src_attn,
            _attn_cls=DecayDecayLinearAttention,
            **kwargs,
        )
    except Exception as e:
        logger.error("inject_svd_linear_attention delegate failed: %s", e)
        return None

    if result is None:
        return None

    if src_attr_name is not None and result is not layer:
        if src_is_nested:
            parent = getattr(layer, src_attr_name, None)
            if parent is not None and getattr(parent, "self", None) is src_attn:
                parent.self = result
                logger.debug(
                    "inject_svd_linear_decay: forced replacement of %s.%s.self",
                    type(layer).__name__,
                    src_attr_name,
                )
        else:
            if getattr(layer, src_attr_name, None) is src_attn:
                setattr(layer, src_attr_name, result)
                logger.debug(
                    "inject_svd_linear_decay: forced replacement of %s.%s",
                    type(layer).__name__,
                    src_attr_name,
                )

    return result


inject_svd_linear_decay_attention = inject_svd_linear_decay

__all__ = [
    "DecayDecayLinearAttention",
    "SVDLinearDecayAttention",
    "SVDLinearAttention",
    "SVDLinear",
    "inject_svd_linear_decay",
    "inject_svd_linear_decay_attention",
]
