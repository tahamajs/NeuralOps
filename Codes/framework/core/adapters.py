"""
Model‑family adapters. Each adapter knows how to copy weights, extract RoPE,
handle GQA, etc. for a specific architecture.
"""


class DistilBertAdapter:
    @staticmethod
    def get_weight_bias(old_attn, proj: str) -> Any:
        """Return the weight bias."""
        if proj == "q":
            return old_attn.q_lin.weight.data, getattr(old_attn.q_lin, "bias", None)
        if proj == "k":
            return old_attn.k_lin.weight.data, getattr(old_attn.k_lin, "bias", None)
        if proj == "v":
            return old_attn.v_lin.weight.data, getattr(old_attn.v_lin, "bias", None)
        if proj == "o":
            return old_attn.out_lin.weight.data, getattr(old_attn.out_lin, "bias", None)
        raise ValueError(f"Unknown projection: {proj}")


class LlamaAdapter:
    @staticmethod
    def get_weight_bias(old_attn, proj: str) -> Any:
        """Return the weight bias."""
        if proj == "q":
            return old_attn.q_proj.weight.data, getattr(old_attn.q_proj, "bias", None)
        if proj == "k":
            return old_attn.k_proj.weight.data, getattr(old_attn.k_proj, "bias", None)
        if proj == "v":
            return old_attn.v_proj.weight.data, getattr(old_attn.v_proj, "bias", None)
        if proj == "o":
            return old_attn.o_proj.weight.data, getattr(old_attn.o_proj, "bias", None)
        raise ValueError(f"Unknown projection: {proj}")

    @staticmethod
    def get_rotary_emb(old_attn) -> Any:
        """Return the rotary emb."""
        return getattr(old_attn, "rotary_emb", None)

    @staticmethod
    def get_kv_heads(config) -> Any:
        """Return the kv heads."""
        return getattr(config, "num_key_value_heads", config.num_attention_heads)


class BertAdapter:
    """Bert / RoBERTa style – uses self_attn with q/k/v weight names."""

    @staticmethod
    def get_weight_bias(old_attn, proj: str) -> Any:
        """Return the weight bias."""
        if proj == "q":
            return old_attn.query.weight.data, getattr(old_attn.query, "bias", None)
        if proj == "k":
            return old_attn.key.weight.data, getattr(old_attn.key, "bias", None)
        if proj == "v":
            return old_attn.value.weight.data, getattr(old_attn.value, "bias", None)
        if proj == "o":
            # BERT output projection is called 'dense' inside self_attn
            if hasattr(old_attn, "output"):
                return old_attn.output.dense.weight.data, getattr(
                    old_attn.output.dense, "bias", None
                )
            elif hasattr(old_attn, "dense"):
                return old_attn.dense.weight.data, getattr(old_attn.dense, "bias", None)
        raise ValueError(f"Unknown projection: {proj}")


# Map architecture name → adapter class
ADAPTER_MAP = {
    "distilbert": DistilBertAdapter,
    "bert": BertAdapter,
    "roberta": BertAdapter,
    "llama": LlamaAdapter,
    "mistral": LlamaAdapter,
    "gemma": LlamaAdapter,
    "phi": LlamaAdapter,
}


def get_adapter(arch: str) -> Any:
    """Return the adapter."""
    adapter = ADAPTER_MAP.get(arch)
    if adapter is None:
        raise ValueError(f"No adapter for architecture '{arch}'")
    return adapter
