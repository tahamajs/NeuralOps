"""
Model‑family adapters. Extracts projection weights and biases cleanly across all Transformer architectures.
"""

from typing import Any, Optional


class UniversalAdapter:
    """Universal adapter supporting BERT, DistilBERT, RoBERTa, LLaMA, GPT2, DeBERTa, etc."""

    @staticmethod
    def get_weight_bias(old_attn: Any, proj: str) -> tuple:
        """Extract (weight, bias) for projection 'q', 'k', 'v', or 'o'."""
        target = getattr(old_attn, "self", old_attn)

        q_names = ["q", "query", "q_proj", "q_lin", "Wq", "c_attn"]
        k_names = ["k", "key", "k_proj", "k_lin", "Wk"]
        v_names = ["v", "value", "v_proj", "v_lin", "Wv"]
        o_names = ["o", "out_proj", "o_proj", "out_lin", "dense", "c_proj"]

        candidate_names = {"q": q_names, "k": k_names, "v": v_names, "o": o_names}.get(proj, [])

        # 1. Search in target (old_attn.self or old_attn)
        for name in candidate_names:
            if hasattr(target, name):
                mod = getattr(target, name)
                if hasattr(mod, "weight"):
                    return mod.weight.data, getattr(mod, "bias", None)

        # 2. Search in old_attn.output (BERT / RoBERTa output projection)
        if proj == "o" and hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
            mod = old_attn.output.dense
            if hasattr(mod, "weight"):
                return mod.weight.data, getattr(mod, "bias", None)

        # 3. Search directly in old_attn
        for name in candidate_names:
            if hasattr(old_attn, name):
                mod = getattr(old_attn, name)
                if hasattr(mod, "weight"):
                    return mod.weight.data, getattr(mod, "bias", None)

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
