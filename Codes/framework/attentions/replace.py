# framework/attentions/replace.py
"""Replace attention modules in HuggingFace models."""

from framework.core.registry import build_attention

def replace_attention_in_model(model, attention_name, attention_params=None):
    """
    Replace all self‑attention layers in a model with the specified attention type.
    Supports BERT, RoBERTa, GPT‑2 out of the box.
    """
    if attention_params is None:
        attention_params = {}

    model_type = getattr(model.config, "model_type", None)

    if model_type in ("bert", "roberta"):
        _replace_bert_attention(model, attention_name, attention_params)
    elif model_type == "gpt2":
        _replace_gpt_attention(model, attention_name, attention_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model

def _replace_bert_attention(model, attention_name, attention_params):
    for layer in model.encoder.layer:
        old_attn = layer.attention.self
        hidden_size = old_attn.query.in_features
        num_heads = old_attn.num_attention_heads
        new_attn = build_attention(
            attention_name,
            hidden_size=hidden_size,
            num_heads=num_heads,
            **attention_params,
        )
        layer.attention.self = new_attn

def _replace_gpt_attention(model, attention_name, attention_params):
    for block in model.transformer.h:
        old_attn = block.attn
        hidden_size = old_attn.embed_dim
        num_heads = old_attn.num_heads
        new_attn = build_attention(
            attention_name,
            hidden_size=hidden_size,
            num_heads=num_heads,
            **attention_params,
        )
        block.attn = new_attn