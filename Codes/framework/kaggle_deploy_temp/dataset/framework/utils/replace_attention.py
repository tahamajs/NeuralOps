import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


def replace_attention(model, attention_key, layer_indices=None, **kwargs) -> Any:
    """
    Replace attention modules in a model with a registered attention block.

    Args:
        model: A PyTorch model (typically a Hugging Face transformer)
        attention_key: String key registered in ATTENTION_REGISTRY, e.g., "nha", "hybrid", etc.
        layer_indices: Optional list of layer indices to replace (0-indexed).
                       If None, replace all attention layers.
        **kwargs: Additional arguments passed to the attention block constructor
                  (e.g., rank, window_size, num_experts).

    Returns:
        Model with attention layers replaced.
    """
    # Get the attention class from registry
    attn_class = ATTENTION_REGISTRY.get(attention_key)
    if attn_class is None:
        raise ValueError(
            f"Attention key '{attention_key}' not found in registry. "
            f"Available: {ATTENTION_REGISTRY.keys()}"
        )

    # Check if model has a config (Hugging Face style)
    if hasattr(model, "config"):
        config = model.config
    else:
        # Try to infer or use default; raise error if missing
        raise AttributeError(
            "Model does not have a 'config' attribute. "
            "Please provide attention block arguments directly."
        )

    # Recursively replace attention modules
    replaced_count = 0
    for name, module in model.named_modules():
        # Common attention module names in HF models: 'attention', 'self_attn', 'self', 'attn'
        if isinstance(module, (nn.MultiheadAttention,)) or any(
            attn_name in name.lower()
            for attn_name in ["attention", "self_attn", "attn"]
        ):

            # Check if this layer should be replaced based on layer_indices
            # Extract layer index from name (e.g., "encoder.layer.3.attention")
            layer_idx = None
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part.isdigit():
                    layer_idx = int(part)
                    break

            if layer_indices is not None and layer_idx not in layer_indices:
                continue

            # Instantiate the new attention block
            try:
                new_attn = attn_class(config, **kwargs)
            except TypeError:
                # Fallback: try without config if block expects different signature
                new_attn = attn_class(**kwargs)

            # Find the parent module and replace
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            setattr(parent, child_name, new_attn)
            replaced_count += 1
            print(f"Replaced {name} with {attention_key}")

    print(f"Replaced {replaced_count} attention layers.")
    return model
