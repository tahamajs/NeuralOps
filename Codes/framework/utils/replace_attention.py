from typing import Any, List, Optional, Tuple
import inspect
import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.qwen_compat import bypass_qwen_rope, qwen_attention_mask


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

        # Qwen supplies a prepared 4-D causal mask and rotary position
        # embeddings. Legacy replacements generally implement their own
        # masking and cannot consume that representation. The base Qwen model
        # is untouched; this applies only to an injected replacement.
        if "position_embeddings" in kwargs:
            # Native Qwen attention owns RoPE. A legacy replacement may opt
            # out explicitly; positional tensors are then removed only from
            # this injected call, never from the base model.
            if not getattr(self.inner, "supports_rope", False):
                kwargs = bypass_qwen_rope(kwargs, self.inner)
                kwargs.pop("qwen_rope_bypassed", None)
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
                attention_mask = qwen_attention_mask(
                    attention_mask,
                    causal=True,
                    expects_bidirectional=bool(getattr(self.inner, "expects_bidirectional", False)),
                )

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



def _try_instantiate(attn_fn: Any, config: Any, layer_idx: int, total_layers: int, old_attn: Any) -> nn.Module:
    """
    Instantiate a registered attention function / class using multiple
    call-signature strategies, from most-specific to most-permissive.

    Handles all 4 observed patterns across 126 registry entries:
      1. inject_fn(layer, model_config, comp_config, layer_idx, total_layers, old_attn)
      2. inject_fn(model_config, layer_idx=N, total_layers=N)
      3. AttnClass(config, layer_idx=N, total_layers=N)
      4. AttnClass(config)
    """
    # Enrich config with commonly missing attributes so modules don't crash
    for attr, default in [
        ("dropout", 0.1),
        ("hidden_dropout_prob", 0.1),
        ("attention_probs_dropout_prob", 0.1),
        ("intermediate_size", 3072),
        ("max_position_embeddings", 512),
        ("layer_norm_eps", 1e-12),
        ("is_decoder", False),
        ("add_cross_attention", False),
        ("chunk_size_feed_forward", 0),
        ("position_embedding_type", "absolute"),
    ]:
        if not hasattr(config, attr):
            setattr(config, attr, default)

    # Build a default ranks dict for spectron/svd_compressed which require it
    _default_ranks = {"q": 64, "k": 64, "v": 64, "o": 64}

    if isinstance(attn_fn, type) and issubclass(attn_fn, nn.Module):
        class_attempts = [
            lambda: attn_fn(config=config, pretrained_attn=old_attn, **kwargs),
            lambda: attn_fn(config=config, **kwargs),
            lambda: attn_fn(config, pretrained_attn=old_attn),
            lambda: attn_fn(config, layer_idx=layer_idx, total_layers=total_layers),
            lambda: attn_fn(config),
            lambda: attn_fn(config.hidden_size, getattr(config, 'num_attention_heads', 12)),
            lambda: attn_fn(config, config.hidden_size, getattr(config, 'num_attention_heads', 12)),
        ]
        for attempt in class_attempts:
            try:
                result = attempt()
                if isinstance(result, nn.Module):
                    return result
            except Exception:
                continue

    attempts = [
        # Pattern A: full positional (layer, model_config, comp_config, layer_idx, total_layers, old_attn)
        lambda: attn_fn(old_attn, config, None, layer_idx, total_layers, old_attn),
        # Pattern A2: 5-arg positional (layer, model_config, comp_config, layer_idx, total_layers)
        lambda: attn_fn(old_attn, config, None, layer_idx, total_layers),
        # Pattern A3: keyword comp_config
        lambda: attn_fn(old_attn, config, comp_config=None, layer_idx=layer_idx, total_layers=total_layers),
        # Pattern A4: layer + model_config only
        lambda: attn_fn(old_attn, config),
        # Pattern B: full inject keyword (comp_config + old_attn + layer aware)
        lambda: attn_fn(config, comp_config=None, old_attn=old_attn, layer_idx=layer_idx, total_layers=total_layers),
        # Pattern B2: inject with layer only
        lambda: attn_fn(config, layer_idx=layer_idx, total_layers=total_layers),
        # Pattern B3: inject with comp_config + old_attn, no layer
        lambda: attn_fn(config, comp_config=None, old_attn=old_attn),
        # Pattern C: spectron/svd_compressed: (model_config, pretrained_attn=..., ranks=...)
        lambda: attn_fn(config, pretrained_attn=old_attn, ranks=_default_ranks, layer_idx=layer_idx),
        lambda: attn_fn(config, pretrained_attn=old_attn, ranks=_default_ranks),
        # Pattern D: config + pretrained_attn (mla, nha, salt, vig, vspec, soa_fused_module)
        lambda: attn_fn(config, pretrained_attn=old_attn, layer_idx=layer_idx),
        lambda: attn_fn(config, pretrained_attn=old_attn),
        # Pattern E: soa_module: (model_config, k_rank=32, pretrained_attn=...)
        lambda: attn_fn(config, k_rank=32, pretrained_attn=old_attn),
        # Pattern F: orion_module/phantom_module: (model_config, None)
        lambda: attn_fn(config, None, pretrained_attn=old_attn),
        lambda: attn_fn(config, None),
        # Pattern G: nha_compressor: (model, tokenizer)
        lambda: attn_fn(None, None),
        # Pattern H: config + hidden_size derived
        lambda: attn_fn(config, config.hidden_size, getattr(config, 'num_attention_heads', 12)),
        # Pattern I: class(config) only
        lambda: attn_fn(config),
        # Pattern J: class(hidden_size, heads)
        lambda: attn_fn(config.hidden_size, getattr(config, 'num_attention_heads', 12)),
        # Pattern K: compressed_attention_layer(original_layer, rank_qkv, rank_out)
        lambda: attn_fn(old_attn, 32, 32),
        # Pattern L: compressed_attention_layer_energy(original_layer, energy_threshold)
        lambda: attn_fn(old_attn, 0.95),
        # Pattern A5: layer only (with defaults)
        lambda: attn_fn(old_attn),
        # Pattern M: no-arg fallback
        lambda: attn_fn(),
    ]

    last_err = None
    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, nn.Module):
                return result
        except (TypeError, AttributeError, ValueError) as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not instantiate {getattr(attn_fn, '__name__', str(attn_fn))} "
        f"with any known signature. Last error: {last_err}"
    )


def replace_attention(model, attention_key, layer_indices=None, **kwargs) -> Any:
    """
    Replace attention modules in a model with a registered attention block.

    Args:
        model: A PyTorch model (typically a Hugging Face transformer)
        attention_key: String key registered in ATTENTION_REGISTRY
        layer_indices: Optional list of layer indices to replace (0-indexed).
                       If None, replace all attention layers.
        **kwargs: Additional arguments passed to the attention block constructor.

    Returns:
        Model with attention layers replaced.
    """
    attn_fn = ATTENTION_REGISTRY.get(attention_key)
    if attn_fn is None:
        raise ValueError(
            f"Attention key '{attention_key}' not found in registry. "
            f"Available: {list(ATTENTION_REGISTRY.keys())}"
        )

    if not hasattr(model, "config"):
        raise AttributeError("Model does not have a 'config' attribute.")

    config = model.config
    total_layers = getattr(config, "num_hidden_layers", 12)

    # Collect top-level attention module targets (avoiding submodules)
    targets: List[Tuple[str, nn.Module, int]] = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if not parts or not parts[-1]:
            continue
        leaf = parts[-1].lower()
        if leaf in ["attention", "self_attn", "self", "attn"] or isinstance(module, nn.MultiheadAttention):
            if not any(name.startswith(t[0] + ".") for t in targets):
                layer_idx = 0
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                if layer_indices is None or layer_idx in layer_indices:
                    targets.append((name, module, layer_idx))

    replaced_count = 0
    failed_count = 0
    for name, old_module, layer_idx in targets:
        try:
            new_attn = _try_instantiate(attn_fn, config, layer_idx, total_layers, old_module)
        except Exception as e:
            print(f"[WARN] Could not replace {name} with '{attention_key}': {e}")
            failed_count += 1
            continue

        # Copy weights if supported
        if hasattr(new_attn, "copy_weights") and callable(new_attn.copy_weights):
            try:
                new_attn.copy_weights(old_module)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass

        # Wrap in universal adapter
        wrapped_attn = UniversalAttentionWrapper(new_attn)

        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        child_name = parts[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrapped_attn)
        replaced_count += 1

    if replaced_count == 0 and failed_count == 0:
        print(f"[WARN] No attention layers found to replace for key='{attention_key}'.")
    elif failed_count > 0:
        print(f"Replaced {replaced_count}/{replaced_count+failed_count} layers with '{attention_key}' ({failed_count} skipped).")
    else:
        print(f"Successfully replaced {replaced_count} attention layers with '{attention_key}'.")

    return model
