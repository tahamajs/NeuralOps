from typing import Any, Optional
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

    # Collect top-level attention module targets (avoiding submodules)
    targets = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if not parts or not parts[-1]:
            continue
        leaf = parts[-1].lower()
        if leaf in ["attention", "self_attn", "self", "attn"] or isinstance(module, (nn.MultiheadAttention,)):
            # Check if an ancestor was already selected
            if not any(name.startswith(t[0] + ".") for t in targets):
                # Check layer_indices if specified
                layer_idx = None
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                if layer_indices is None or layer_idx in layer_indices:
                    targets.append((name, module))

    replaced_count = 0
    for name, old_module in targets:
        # Instantiate the new attention block
        try:
            new_attn = attn_class(config, **kwargs)
        except TypeError:
            new_attn = attn_class(**kwargs)

        # Copy weights if copy_weights is supported
        if hasattr(new_attn, "copy_weights") and callable(new_attn.copy_weights):
            try:
                new_attn.copy_weights(old_module)
            except Exception:
                pass

        # Find parent module and replace
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        child_name = parts[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        setattr(parent, child_name, new_attn)
        replaced_count += 1
        print(f"Replaced {name} with {attention_key}")

    print(f"Replaced {replaced_count} attention layers.")
    return model
