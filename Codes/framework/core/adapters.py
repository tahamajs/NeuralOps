"""
Model‑family adapters. Extracts projection weights and biases cleanly across all Transformer architectures.
"""

from typing import Any, Optional


CANONICAL_PROJECTION_ALIASES = {
    "q": ["q", "query", "q_proj", "query_proj", "q_lin", "Wq", "wq", "c_attn", "in_proj", "Wqkv", "qkv_proj", "qkv"],
    "k": ["k", "key", "k_proj", "key_proj", "k_lin", "Wk", "wk"],
    "v": ["v", "value", "v_proj", "value_proj", "v_lin", "Wv", "wv"],
    "o": ["o", "out_proj", "o_proj", "out_lin", "dense", "c_proj", "wo", "W_o", "Wo"],
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
            # LoRALinear stores the original frozen layer as base_layer
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
            # Must be either an nn.Parameter or a plain Tensor (Conv1D)
            if not (isinstance(w, nn.Parameter) or isinstance(w, torch.Tensor)):
                return None
            w = w.data if hasattr(w, "data") else w
            # Conv1D stores weights transposed: [in_features, out_features]
            if w.ndim == 2 and w.shape[0] < w.shape[1]:
                w = w.t().contiguous()
            bias = getattr(mod, "bias", None)
            bias = bias.data if (bias is not None and hasattr(bias, "data")) else bias
            return w, bias

        # 1. Search in target (old_attn.self or old_attn)
        for name in candidate_names:
            if hasattr(target, name):
                result = _extract(getattr(target, name))
                if result is not None:
                    return result

        # 2. Search in old_attn.output (BERT / RoBERTa output projection 'o')
        if proj == "o" and hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
            result = _extract(old_attn.output.dense)
            if result is not None:
                return result

        # 3. Search directly in old_attn (catches cases where there's no .self sub-module)
        for name in candidate_names:
            if hasattr(old_attn, name):
                result = _extract(getattr(old_attn, name))
                if result is not None:
                    return result

        # 4. Positional fallback: scan all Linear children of target in declaration order.
        #    q=first, k=second, v=third (the typical pattern for BERT/RoBERTa variants).
        ordered_proj_map = {"q": 0, "k": 1, "v": 2}
        if proj in ordered_proj_map:
            children_with_weight = [
                (n, m) for n, m in target.named_children()
                if _extract(m) is not None
            ]
            idx = ordered_proj_map[proj]
            if idx < len(children_with_weight):
                return _extract(children_with_weight[idx][1])

        raise ValueError(f"Could not extract '{proj}' projection weights from {type(old_attn).__name__}")

    @staticmethod
    def get_rotary_emb(old_attn: Any) -> Optional[Any]:
        """Return rotary embeddings if present."""
        return getattr(old_attn, "rotary_emb", None)

    @staticmethod
    def get_kv_heads(config: Any) -> int:
        """Return number of key-value heads for GQA/MQA."""
        return getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 12))


# Backward compatibility aliases
DistilBertAdapter = UniversalAdapter
BertAdapter = UniversalAdapter
LlamaAdapter = UniversalAdapter

ADAPTER_MAP = {}


def get_adapter(arch: str) -> Any:
    """Return universal adapter for any architecture."""
    return UniversalAdapter
