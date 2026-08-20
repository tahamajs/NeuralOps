from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


class UniversalAttentionWrapper(nn.Module):
    """
    Universal forward adapter wrapping any custom attention module.
    Seamlessly adapts across all HuggingFace architectures:
      - DistilBERT: forward(query=x, key=x, value=x, mask=mask, head_mask=head_mask, output_attentions=...) -> tuple (out, weights)
      - BERT / RoBERTa: forward(hidden_states, attention_mask, ...) -> tuple (out, weights)
      - LLaMA / Mistral: forward(hidden_states, attention_mask, position_ids, ...) -> tuple (out, weights, ...)
      - GPT-2: forward(x, layer_past, ...) -> tuple (out, present)
    """

    def __init__(self, inner_attn: nn.Module) -> None:
        super().__init__()
        self.inner = inner_attn

    def forward(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Extract hidden states / input tensor
        hidden_states: Optional[torch.Tensor] = None
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
            args = args[1:]
        elif "hidden_states" in kwargs:
            hidden_states = kwargs.pop("hidden_states")
        elif "query" in kwargs:
            hidden_states = kwargs.pop("query")
        elif "x" in kwargs:
            hidden_states = kwargs.pop("x")

        # 2. Extract attention mask
        attention_mask: Optional[torch.Tensor] = None
        if len(args) > 0 and (isinstance(args[0], torch.Tensor) or args[0] is None):
            attention_mask = args[0]
            args = args[1:]
        elif "attention_mask" in kwargs:
            attention_mask = kwargs.pop("attention_mask")
        elif "mask" in kwargs:
            attention_mask = kwargs.pop("mask")
        elif "attn_mask" in kwargs:
            attention_mask = kwargs.pop("attn_mask")

        # Clean redundant or mismatched args
        kwargs.pop("key", None)
        kwargs.pop("value", None)
        head_mask = kwargs.pop("head_mask", None)
        output_attentions = kwargs.pop("output_attentions", False)

        # 3. Dynamic call dispatch
        out = None
        call_attempts = [
            lambda: self.inner(hidden_states, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, **kwargs),
            lambda: self.inner(hidden_states, attention_mask=attention_mask, **kwargs),
            lambda: self.inner(hidden_states, mask=attention_mask, **kwargs),
            lambda: self.inner(query=hidden_states, key=hidden_states, value=hidden_states, mask=attention_mask, **kwargs),
            lambda: self.inner(hidden_states, **kwargs),
            lambda: self.inner(hidden_states),
        ]

        last_err = None
        for attempt in call_attempts:
            try:
                out = attempt()
                break
            except TypeError as e:
                last_err = e
                continue

        if out is None:
            if last_err is not None:
                raise last_err
            raise RuntimeError(f"Failed to execute forward pass on {type(self.inner).__name__}")

        # 4. Standardize output format to a tuple for HuggingFace models
        if isinstance(out, tuple):
            return out
        return (out, None)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


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

        # Wrap in universal adapter to support any HuggingFace model signature (DistilBERT, BERT, etc.)
        wrapped_attn = UniversalAttentionWrapper(new_attn)

        # Find parent module and replace
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        child_name = parts[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        setattr(parent, child_name, wrapped_attn)
        replaced_count += 1
        print(f"Replaced {name} with {attention_key}")

    print(f"Replaced {replaced_count} attention layers.")
    return model
