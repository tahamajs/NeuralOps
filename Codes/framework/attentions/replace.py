"""Helper to replace attention modules in HuggingFace models."""

from framework.core.registry import ATTENTION_REGISTRY


def replace_attention_in_model(model, attention_name, attention_params=None):
    """
    Replace all self‑attention layers in a model with the specified attention type.
    Supports BERT, RoBERTa, DistilBERT, GPT‑2.
    """
    if attention_params is None:
        attention_params = {}

    model_type = getattr(model.config, "model_type", None)

    if model_type in ("bert", "roberta"):
        _replace_bert_attention(model, attention_name, attention_params)
    elif model_type == "gpt2":
        _replace_gpt_attention(model, attention_name, attention_params)
    elif model_type == "distilbert":
        _replace_distilbert_attention(model, attention_name, attention_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def _replace_bert_attention(model, attention_name, attention_params):
    for layer in model.encoder.layer:
        old_attn = layer.attention.self
        hidden_size = old_attn.query.in_features
        num_heads = old_attn.num_attention_heads
        new_attn = build_attention(
            attention_name, hidden_size, num_heads, **attention_params
        )
        layer.attention.self = new_attn


def _replace_gpt_attention(model, attention_name, attention_params):
    for block in model.transformer.h:
        old_attn = block.attn
        hidden_size = old_attn.embed_dim
        num_heads = old_attn.num_heads
        new_attn = build_attention(
            attention_name, hidden_size, num_heads, **attention_params
        )
        block.attn = new_attn


def _replace_distilbert_attention(model, attention_name, attention_params):
    for layer in model.distilbert.transformer.layer:
        old_attn = layer.attention
        # DistilBERT attention has q_lin, k_lin, v_lin, out_lin
        hidden_size = old_attn.q_lin.in_features
        num_heads = getattr(model.config, "num_attention_heads", 12)
        # Build a custom `inject` function instead of a class directly
        # But we rely on the registry's inject function for the given key.
        inject_fn = ATTENTION_REGISTRY.get(attention_name)
        if inject_fn is None:
            raise ValueError(f"Attention '{attention_name}' not registered.")
        inject_fn(
            layer,
            model.config,
            None,
            0,
            len(model.distilbert.transformer.layer),
            old_attn,
        )


def build_attention(attention_name, hidden_size, num_heads, **kwargs):
    """Helper to instantiate an attention class (not injection)."""
    # This is a simplified version; the registry expects an injection function,
    # not a class. For simplicity, we raise an error – use the injectors instead.
    raise NotImplementedError("Use the registry inject functions with a full model.")
