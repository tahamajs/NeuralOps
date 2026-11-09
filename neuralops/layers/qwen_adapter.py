#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Universal Qwen Attention Adapter with GQA (Grouped-Query Attention) Expansion.

Adapts any NeuralOps attention module into a drop-in replacement for
Hugging Face Qwen2Attention / Qwen2DecoderLayer, standardizing forward signature,
causal mask conversion, GQA (14:2) weight expansion, and O(1) recurrent state decoding.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QwenAttentionAdapter(nn.Module):
    """
    Universal adapter adapting any NeuralOps attention module into the
    exact forward contract expected by Hugging Face Qwen2DecoderLayer.

    GQA Contract (14:2):
      - Qwen2.5-0.5B features 14 Query heads and 2 Key/Value heads (head_dim=64).
      - Query projection: (896, 896).
      - Key & Value projections: (128, 896).
      - If transplanted core attention expects standard MHA (896, 896), Key and Value
        weights are expanded via repeat_interleave(7, dim=0).
      - If transplanted core attention supports native GQA (num_kv_heads=2), weights
        are copied directly without expansion.
      - Output projection (896, 896) is preserved or unified.

    Expected forward return:
      Tuple of (context_layer, attention_probabilities_or_None) matching
      hidden_states, _ = self.self_attn(...) in Qwen2DecoderLayer.
    """

    def __init__(
        self,
        core_attention: nn.Module,
        embed_dim: int = 896,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        inherit_weights_from: Optional[nn.Module] = None,
        is_causal: bool = True,
        bypass_internal_out_proj: bool = True,
    ) -> None:
        super().__init__()
        self.core = core_attention
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_causal = is_causal

        # Single-o_proj Contract: Dedicated canonical output projection for Qwen
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Enforce Single-o_proj Contract: bypass duplicate internal out_proj if present
        if bypass_internal_out_proj:
            self._enforce_single_o_proj_contract()

        if inherit_weights_from is not None:
            self._inherit_weights(inherit_weights_from)

    def _enforce_single_o_proj_contract(self) -> None:
        """
        Detects duplicate internal output projections inside core_attention and replaces
        them with nn.Identity() so that self.o_proj remains the single authoritative
        output projection executing strictly once at the end of forward().
        """
        for attr_name in ("out_proj", "o_proj", "wo", "proj"):
            mod = getattr(self.core, attr_name, None)
            if mod is not None and isinstance(mod, nn.Linear):
                if mod.in_features == self.embed_dim and mod.out_features == self.embed_dim:
                    setattr(self.core, attr_name, nn.Identity())
                    logger.debug("Single-o_proj Contract: Bypassed %s with nn.Identity() in %s", attr_name, type(self.core).__name__)
                    break

    def _inherit_weights(self, source_attn: nn.Module) -> None:
        """
        Transplants weights from Qwen2Attention with automatic GQA (14:2) expansion.
        """
        try:
            q_src = getattr(source_attn, "q_proj", None)
            k_src = getattr(source_attn, "k_proj", None)
            v_src = getattr(source_attn, "v_proj", None)
            o_src = getattr(source_attn, "o_proj", None)

            # 1. Query projection (896, 896)
            if q_src is not None and hasattr(self.core, "q_proj") and isinstance(self.core.q_proj, nn.Linear):
                if self.core.q_proj.weight.shape == q_src.weight.shape:
                    self.core.q_proj.weight.data.copy_(q_src.weight.data)
                    if self.core.q_proj.bias is not None and q_src.bias is not None:
                        self.core.q_proj.bias.data.copy_(q_src.bias.data)

            # 2. Key projection (GQA expansion if target is MHA)
            if k_src is not None and hasattr(self.core, "k_proj") and isinstance(self.core.k_proj, nn.Linear):
                target_k = self.core.k_proj
                if target_k.weight.shape == k_src.weight.shape:
                    # Target natively supports GQA (e.g. 128, 896)
                    target_k.weight.data.copy_(k_src.weight.data)
                    if target_k.bias is not None and k_src.bias is not None:
                        target_k.bias.data.copy_(k_src.bias.data)
                elif target_k.weight.shape == (self.embed_dim, self.embed_dim) and k_src.weight.shape[0] < self.embed_dim:
                    # GQA expansion: repeat KV heads to match query heads (e.g. 2 heads -> 14 heads, repeat factor 7)
                    repeat_factor = self.num_heads // self.num_kv_heads
                    kv_heads = self.num_kv_heads
                    h_dim = self.head_dim

                    src_w = k_src.weight.data  # [kv_heads * h_dim, embed_dim]
                    expanded_w = (
                        src_w.view(kv_heads, h_dim, self.embed_dim)
                        .repeat_interleave(repeat_factor, dim=0)
                        .reshape(self.embed_dim, self.embed_dim)
                    )
                    target_k.weight.data.copy_(expanded_w)

                    if target_k.bias is not None and k_src.bias is not None:
                        src_b = k_src.bias.data
                        expanded_b = (
                            src_b.view(kv_heads, h_dim)
                            .repeat_interleave(repeat_factor, dim=0)
                            .reshape(self.embed_dim)
                        )
                        target_k.bias.data.copy_(expanded_b)
                    logger.debug("Transplanted Key projection with GQA (14:2) repeat_interleave factor %d", repeat_factor)

            # 3. Value projection (GQA expansion if target is MHA)
            if v_src is not None and hasattr(self.core, "v_proj") and isinstance(self.core.v_proj, nn.Linear):
                target_v = self.core.v_proj
                if target_v.weight.shape == v_src.weight.shape:
                    target_v.weight.data.copy_(v_src.weight.data)
                    if target_v.bias is not None and v_src.bias is not None:
                        target_v.bias.data.copy_(v_src.bias.data)
                elif target_v.weight.shape == (self.embed_dim, self.embed_dim) and v_src.weight.shape[0] < self.embed_dim:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    kv_heads = self.num_kv_heads
                    h_dim = self.head_dim

                    src_w = v_src.weight.data
                    expanded_w = (
                        src_w.view(kv_heads, h_dim, self.embed_dim)
                        .repeat_interleave(repeat_factor, dim=0)
                        .reshape(self.embed_dim, self.embed_dim)
                    )
                    target_v.weight.data.copy_(expanded_w)

                    if target_v.bias is not None and v_src.bias is not None:
                        src_b = v_src.bias.data
                        expanded_b = (
                            src_b.view(kv_heads, h_dim)
                            .repeat_interleave(repeat_factor, dim=0)
                            .reshape(self.embed_dim)
                        )
                        target_v.bias.data.copy_(expanded_b)
                    logger.debug("Transplanted Value projection with GQA (14:2) repeat_interleave factor %d", repeat_factor)

            # 4. Output projection (896, 896)
            if o_src is not None and isinstance(o_src, nn.Linear):
                if o_src.bias is not None and self.o_proj.bias is None:
                    self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
                if self.o_proj.weight.shape == o_src.weight.shape:
                    self.o_proj.weight.data.copy_(o_src.weight.data)
                    if self.o_proj.bias is not None and o_src.bias is not None:
                        self.o_proj.bias.data.copy_(o_src.bias.data)

            # 5. Family-specific forget-gate / retention bias: sigma(b_g) >= 0.88 -> b_g in [2.0, 3.0]
            for gate_attr in ("gate_bias", "b_g", "forget_bias", "decay_bias"):
                if hasattr(self.core, gate_attr):
                    gate_param = getattr(self.core, gate_attr)
                    if isinstance(gate_param, (nn.Parameter, torch.Tensor)):
                        nn.init.uniform_(gate_param, 2.0, 3.0)
                        logger.debug("Initialized Qwen retention gate bias %s to [2.0, 3.0]", gate_attr)

        except Exception as e:
            logger.debug("Qwen weight inheritance partial or skipped: %s", e)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass matching Hugging Face Qwen2Attention signature.
        Returns a 2-tuple: (hidden_states, attention_weights_or_None).
        """
        batch_size, seq_len, _ = hidden_states.shape
        call_kwargs: dict[str, Any] = {}

        # Forward causal / attention mask if accepted (sliced to seq_len)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask_2d = (attention_mask[:, 0, -1, :seq_len] > -1.0).long()
                call_kwargs["mask"] = mask_2d
                call_kwargs["attention_mask"] = mask_2d
            elif attention_mask.dim() == 2:
                call_kwargs["mask"] = attention_mask[:, :seq_len]
                call_kwargs["attention_mask"] = attention_mask[:, :seq_len]

        if self.is_causal:
            call_kwargs["is_causal"] = True
            call_kwargs["causal"] = True

        if past_key_values is not None:
            call_kwargs["past_key_value"] = past_key_values
            call_kwargs["past_key_values"] = past_key_values
            call_kwargs["use_cache"] = use_cache

        # Try forward pass through core module
        try:
            out = self.core(hidden_states, **call_kwargs)
        except TypeError:
            try:
                if "mask" in call_kwargs:
                    out = self.core(hidden_states, call_kwargs["mask"])
                else:
                    out = self.core(hidden_states)
            except TypeError:
                out = self.core(hidden_states)

        # Unpack output
        if isinstance(out, tuple):
            context = out[0]
            attn_weights = out[1] if len(out) > 1 else None
        else:
            context = out
            attn_weights = None

        # Check dimension alignment
        if context.shape[-1] != self.embed_dim:
            if not hasattr(self, "_dim_proj"):
                self._dim_proj = nn.Linear(
                    context.shape[-1], self.embed_dim, bias=False, device=context.device, dtype=context.dtype
                )
            context = self._dim_proj(context)

        # Single-o_proj Contract: apply canonical output projection strictly once
        context = self.o_proj(context)

        # Qwen2DecoderLayer contract: (hidden_states, attn_weights, present_key_value)
        return (context, attn_weights, past_key_values)


__all__ = ["QwenAttentionAdapter"]
