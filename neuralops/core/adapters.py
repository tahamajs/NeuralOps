#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parameter-Efficient Adapters

Low-rank adapter layers and residual bottleneck modules for efficient fine-tuning.
"""

from typing import Any

import torch

CANONICAL_PROJECTION_ALIASES = {
    "q": ["q", "query", "q_proj", "query_proj", "q_lin", "Wq", "wq"],
    "k": ["k", "key", "k_proj", "key_proj", "k_lin", "Wk", "wk"],
    "v": ["v", "value", "v_proj", "value_proj", "v_lin", "Wv", "wv"],
    "o": [
        "o",
        "out_proj",
        "o_proj",
        "out_lin",
        "dense",
        "c_proj",
        "wo",
        "W_o",
        "Wo",
        "output",
        "proj",
    ],
}


class UniversalAdapter:
    """Universal adapter supporting BERT, DistilBERT, RoBERTa, LLaMA, GPT2, DeBERTa, etc."""

    @staticmethod
    def get_weight_bias(old_attn: Any, proj: str) -> tuple:
        """Extract (weight, bias) for projection 'q', 'k', 'v', or 'o'.

        Handles:
        - RoBERTa/BERT: query/key/value attributes on .self sub-module
        - LLaMA/Qwen: q_proj/k_proj/v_proj on attention module directly
        - GPT-2: c_attn (fused, already handled by fused-QKV path); c_proj
        - DistilBERT: q_lin/k_lin/v_lin/out_lin
        - LoRALinear: unwraps .base_layer to get the frozen base weight
        - Conv1D: transposes weight from [in, out] to [out, in]
        """
        target = getattr(old_attn, "self", old_attn)
        candidate_names = CANONICAL_PROJECTION_ALIASES.get(proj, [])

        def _unwrap(mod: Any) -> Any:
            """Unwrap LoRALinear / merged adapter layers to their base_layer."""
            if mod is None:
                return mod
            if hasattr(mod, "base_layer") and hasattr(mod.base_layer, "weight"):
                return mod.base_layer
            return mod

        def _extract(mod: Any):
            """Return (weight_data, bias) from a module, or None if not a valid weight-holder."""
            mod = _unwrap(mod)
            if mod is None:
                return None
            if not hasattr(mod, "weight") or mod.weight is None:
                return None
            import torch
            import torch.nn as nn

            w = mod.weight
            if not (isinstance(w, nn.Parameter) or isinstance(w, torch.Tensor)):
                return None
            w = w.data if hasattr(w, "data") else w
            if w.ndim != 2:
                return None
            is_conv1d = (mod.__class__.__name__ == "Conv1D") or hasattr(mod, "nf")
            if is_conv1d or (w.shape[0] < w.shape[1]):
                w = w.t().contiguous()
            bias = getattr(mod, "bias", None)
            bias = bias.data if (bias is not None and hasattr(bias, "data")) else bias
            if proj == "q" and (
                hasattr(old_attn, "q_norm")
                or old_attn.__class__.__name__ == "Qwen3_5Attention"
                or hasattr(old_attn, "k_norm")
            ):
                cfg = getattr(old_attn, "config", None)
                tc = getattr(cfg, "text_config", cfg)
                h_dim = getattr(tc, "head_dim", 256)
                n_heads = getattr(tc, "num_attention_heads", getattr(old_attn, "num_heads", 8))
                if w.shape[0] == n_heads * h_dim * 2:
                    w = (
                        w.view(n_heads, 2, h_dim, -1)[:, 0, :, :]
                        .reshape(n_heads * h_dim, -1)
                        .contiguous()
                    )
                    if bias is not None:
                        bias = (
                            bias.view(n_heads, 2, h_dim)[:, 0, :]
                            .reshape(n_heads * h_dim)
                            .contiguous()
                        )
            return w, bias

        for name in candidate_names:
            if hasattr(target, name):
                result = _extract(getattr(target, name))
                if result is not None:
                    return result

        if proj == "o" and hasattr(old_attn, "output"):
            out_mod = old_attn.output
            if hasattr(out_mod, "dense"):
                result = _extract(out_mod.dense)
                if result is not None:
                    return result
            result = _extract(out_mod)
            if result is not None:
                return result

        for name in candidate_names:
            if hasattr(old_attn, name):
                result = _extract(getattr(old_attn, name))
                if result is not None:
                    return result

        if proj in ("q", "k", "v") and hasattr(old_attn, "c_attn"):
            fused_result = _extract(old_attn.c_attn)
            if fused_result is not None:
                w, b = fused_result
                split_w = w.size(0) // 3
                split_b = b.size(0) // 3 if b is not None else 0
                if proj == "q":
                    return w[:split_w, :], (b[:split_b] if b is not None else None)
                elif proj == "k":
                    return w[split_w : 2 * split_w, :], (
                        b[split_b : 2 * split_b] if b is not None else None
                    )
                elif proj == "v":
                    return w[2 * split_w :, :], (b[2 * split_b :] if b is not None else None)

        if proj in ("q", "k", "v") and hasattr(old_attn, "in_proj_qkv"):
            fused_result = _extract(old_attn.in_proj_qkv)
            if fused_result is not None:
                w, b = fused_result
                k_dim = getattr(old_attn, "key_dim", getattr(old_attn, "k_dim", w.size(0) // 3))
                v_dim = getattr(old_attn, "value_dim", getattr(old_attn, "v_dim", w.size(0) // 3))
                q_dim = w.size(0) - k_dim - v_dim if (k_dim + v_dim < w.size(0)) else k_dim
                if proj == "q":
                    return w[:q_dim, :], (b[:q_dim] if b is not None else None)
                elif proj == "k":
                    return w[q_dim : q_dim + k_dim, :], (
                        b[q_dim : q_dim + k_dim] if b is not None else None
                    )
                elif proj == "v":
                    return w[q_dim + k_dim : q_dim + k_dim + v_dim, :], (
                        b[q_dim + k_dim : q_dim + k_dim + v_dim] if b is not None else None
                    )

        for obj in (target, old_attn):
            if (
                proj in ("q", "k", "v")
                and hasattr(obj, "in_proj_weight")
                and obj.in_proj_weight is not None
            ):
                w = (
                    obj.in_proj_weight.data
                    if hasattr(obj.in_proj_weight, "data")
                    else obj.in_proj_weight
                )
                b = getattr(obj, "in_proj_bias", None)
                b = b.data if (b is not None and hasattr(b, "data")) else b
                split_w = w.size(0) // 3
                split_b = b.size(0) // 3 if b is not None else 0
                if proj == "q":
                    return w[:split_w, :].clone(), (b[:split_b].clone() if b is not None else None)
                elif proj == "k":
                    return w[split_w : 2 * split_w, :].clone(), (
                        b[split_b : 2 * split_b].clone() if b is not None else None
                    )
                elif proj == "v":
                    return w[2 * split_w :, :].clone(), (
                        b[2 * split_b :].clone() if b is not None else None
                    )

            if proj in ("q", "k", "v"):
                attr_w = f"{proj}_proj_weight"
                if hasattr(obj, attr_w) and getattr(obj, attr_w) is not None:
                    w = getattr(obj, attr_w)
                    w = w.data if hasattr(w, "data") else w
                    b = getattr(obj, f"{proj}_proj_bias", None)
                    b = b.data if (b is not None and hasattr(b, "data")) else b
                    return w.clone(), (b.clone() if b is not None else None)

        ordered_proj_map = {"q": 0, "k": 1, "v": 2}
        if proj in ordered_proj_map:
            children_with_weight = [
                (n, m) for n, m in target.named_children() if _extract(m) is not None
            ]
            if len(children_with_weight) >= 3:
                idx = ordered_proj_map[proj]
                if idx < len(children_with_weight):
                    return _extract(children_with_weight[idx][1])

        if proj == "o":
            for mod in target.modules():
                if (
                    hasattr(mod, "weight")
                    and isinstance(getattr(mod, "weight", None), torch.Tensor)
                    and mod.weight.ndim == 2
                ):
                    d = mod.weight.shape[0]
                    eye_w = torch.eye(d, dtype=mod.weight.dtype, device=mod.weight.device)
                    zero_b = torch.zeros(d, dtype=mod.weight.dtype, device=mod.weight.device)
                    return eye_w, zero_b

        raise ValueError(
            f"Could not extract '{proj}' projection weights from {type(old_attn).__name__}"
        )

    @staticmethod
    def get_rotary_emb(old_attn: Any) -> Any | None:
        """Return rotary embeddings if present."""
        return getattr(old_attn, "rotary_emb", None)

    @staticmethod
    def get_kv_heads(config: Any) -> int:
        """Return number of key-value heads for GQA/MQA."""
        return getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 12))


DistilBertAdapter = UniversalAdapter
BertAdapter = UniversalAdapter
LlamaAdapter = UniversalAdapter

ADAPTER_MAP = {}


def get_adapter(arch: str) -> Any:
    """Return universal adapter for any architecture."""
    return UniversalAdapter
