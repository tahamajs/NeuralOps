#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Universal BERT Attention Adapter.

Wraps any NeuralOps attention module into a drop-in replacement for
Hugging Face transformers' BertSelfAttention, standardizing forward signature,
mask handling, and return shape contracts while enforcing the Single-W_O contract.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BertAttentionAdapter(nn.Module):
    """
    Universal adapter adapting any NeuralOps attention module into the
    exact forward contract expected by Hugging Face BertLayer.

    Single-W_O Contract:
      - BERT's native layer.attention.output.dense (nn.Linear(768, 768)) is preserved
        as the single canonical output projection.
      - If the injected core attention has an internal full-rank out_proj (nn.Linear(768, 768)),
        it is bypassed with nn.Identity() to avoid double output projection (W_O^2).
      - If the core module uses factorized low-rank projections (e.g. rank r -> 768),
        they map to 768, and layer.attention.output.dense performs the canonical output mapping.
      - Stage-0 Ridge calibration aligns layer.attention.output.dense directly.

    Expected contract:
      Input:  hidden_states: [batch_size, seq_len, embed_dim]
      Output: (context_layer, attention_probabilities_or_None)
    """

    def __init__(
        self,
        core_attention: nn.Module,
        embed_dim: int = 768,
        num_heads: int = 12,
        inherit_weights_from: Optional[nn.Module] = None,
        bypass_internal_out_proj: bool = True,
    ) -> None:
        super().__init__()
        self.core = core_attention
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Enforce Single-W_O Contract: bypass duplicate internal out_proj if present
        if bypass_internal_out_proj:
            self._enforce_single_wo_contract()

        if inherit_weights_from is not None:
            self._inherit_weights(inherit_weights_from)

    def _enforce_single_wo_contract(self) -> None:
        """
        Detects duplicate internal output projections inside core_attention and replaces
        them with nn.Identity() so that BERT's native layer.attention.output.dense
        remains the single authoritative output projection.
        """
        for attr_name in ("out_proj", "o_proj", "wo", "proj"):
            mod = getattr(self.core, attr_name, None)
            if mod is not None and isinstance(mod, nn.Linear):
                # If it's a standard square projection matching embed_dim, bypass it
                if mod.in_features == self.embed_dim and mod.out_features == self.embed_dim:
                    setattr(self.core, attr_name, nn.Identity())
                    logger.debug("Single-W_O Contract: Bypassed %s with nn.Identity() in %s", attr_name, type(self.core).__name__)
                    break

    def _inherit_weights(self, source_attn: nn.Module) -> None:
        """Attempt best-effort weight transplantation from BertSelfAttention."""
        try:
            # Standard Hugging Face BertSelfAttention has query, key, value
            q_src = getattr(source_attn, "query", None)
            k_src = getattr(source_attn, "key", None)
            v_src = getattr(source_attn, "value", None)

            # Check if core has matching linear projections
            if q_src is not None and hasattr(self.core, "q_proj") and isinstance(self.core.q_proj, nn.Linear):
                if self.core.q_proj.weight.shape == q_src.weight.shape:
                    self.core.q_proj.weight.data.copy_(q_src.weight.data)
                    if self.core.q_proj.bias is not None and q_src.bias is not None:
                        self.core.q_proj.bias.data.copy_(q_src.bias.data)

            if k_src is not None and hasattr(self.core, "k_proj") and isinstance(self.core.k_proj, nn.Linear):
                if self.core.k_proj.weight.shape == k_src.weight.shape:
                    self.core.k_proj.weight.data.copy_(k_src.weight.data)
                    if self.core.k_proj.bias is not None and k_src.bias is not None:
                        self.core.k_proj.bias.data.copy_(k_src.bias.data)

            if v_src is not None and hasattr(self.core, "v_proj") and isinstance(self.core.v_proj, nn.Linear):
                if self.core.v_proj.weight.shape == v_src.weight.shape:
                    self.core.v_proj.weight.data.copy_(v_src.weight.data)
                    if self.core.v_proj.bias is not None and v_src.bias is not None:
                        self.core.v_proj.bias.data.copy_(v_src.bias.data)

            # Some variants use in_proj_weight (e.g. combined QKV)
            if hasattr(self.core, "in_proj_weight") and q_src is not None and k_src is not None and v_src is not None:
                qkv_weight = torch.cat([q_src.weight.data, k_src.weight.data, v_src.weight.data], dim=0)
                if self.core.in_proj_weight.shape == qkv_weight.shape:
                    self.core.in_proj_weight.data.copy_(qkv_weight)

            # Family-specific forget-gate bias retention: sigma(b_g) >= 0.88 -> b_g in [2.0, 3.0]
            for gate_attr in ("gate_bias", "b_g", "forget_bias", "decay_bias"):
                if hasattr(self.core, gate_attr):
                    gate_param = getattr(self.core, gate_attr)
                    if isinstance(gate_param, (nn.Parameter, torch.Tensor)):
                        nn.init.uniform_(gate_param, 2.0, 3.0)
                        logger.debug("Initialized retention gate bias %s to [2.0, 3.0]", gate_attr)

        except Exception as e:
            logger.debug("Weight inheritance partial or skipped: %s", e)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass converting BERT inputs to core attention module inputs,
        then returning a tuple matching BertSelfAttention.
        """
        kwargs: dict[str, Any] = {}

        # Handle mask conversion if attention_mask provided
        # BERT attention mask is shape [batch, 1, 1, seq_len] or [batch, seq_len]
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                # HF BERT mask has large negative values for masked positions
                mask_2d = (attention_mask[:, 0, 0, :] > -1.0).long()
            elif attention_mask.dim() == 2:
                mask_2d = attention_mask
            else:
                mask_2d = None

            if mask_2d is not None:
                kwargs["mask"] = mask_2d
                kwargs["attention_mask"] = mask_2d

        # Forward pass through core module
        try:
            out = self.core(hidden_states, **kwargs)
        except TypeError:
            try:
                if "mask" in kwargs:
                    out = self.core(hidden_states, kwargs["mask"])
                else:
                    out = self.core(hidden_states)
            except TypeError:
                out = self.core(hidden_states)

        # Unpack return type
        attn_weights = None
        if isinstance(out, (tuple, list)):
            context = out[0]
            if len(out) > 1 and output_attentions:
                attn_weights = out[1]
        elif isinstance(out, dict):
            context = out.get("context", out.get("out", out.get("hidden_states", hidden_states)))
            if output_attentions:
                attn_weights = out.get("attn_weights", out.get("attentions", None))
        else:
            context = out

        # Validate context shape
        if context.shape[-1] != self.embed_dim:
            if not hasattr(self, "_dim_proj"):
                self._dim_proj = nn.Linear(context.shape[-1], self.embed_dim, device=context.device, dtype=context.dtype)
            context = self._dim_proj(context)

        return (context, attn_weights)
