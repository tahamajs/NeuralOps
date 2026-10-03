"""Compatibility helpers for decoder-only Qwen attention blocks.

This module only exposes the names expected by legacy attention/compressor
implementations.  It does not replace Qwen's rotary or causal attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class QwenConfigProxy:
    """Small, read-through config view with the common attention fields."""

    source: Any
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    num_hidden_layers: int

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)


class QwenIncompatibleError(RuntimeError):
    """Raised when a module has no truthful Qwen execution path."""

    reason = "qwen_incompatible"


def is_qwen_model(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "text_config"):
        config = config.text_config
    model_type = str(getattr(config, "model_type", "")).lower()
    return model_type.startswith("qwen") or "qwen" in model.__class__.__name__.lower()


def is_qwen_attention(module: nn.Module) -> bool:
    """Detect a native Qwen attention block without importing transformers."""
    cfg = getattr(module, "config", None)
    if cfg is not None and hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    return str(getattr(cfg, "model_type", "")).lower().startswith("qwen") or (
        hasattr(module, "q_proj") and hasattr(module, "o_proj") and
        ("qwen" in module.__class__.__name__.lower() or hasattr(module, "rotary_emb"))
    )


def qwen_config_proxy(model_or_config: Any) -> QwenConfigProxy:
    config = getattr(model_or_config, "config", model_or_config)
    if hasattr(config, "text_config"):
        config = config.text_config
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    return QwenConfigProxy(
        source=config,
        hidden_size=hidden,
        num_attention_heads=heads,
        head_dim=int(getattr(config, "head_dim", hidden // heads)),
        num_hidden_layers=int(getattr(config, "num_hidden_layers", 0)),
    )


def install_qwen_aliases(model: nn.Module) -> nn.Module:
    """Install non-structural projection aliases on every Qwen attention block."""
    if not is_qwen_model(model):
        return model
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "text_config"):
        tc = cfg.text_config
        for attr in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "num_hidden_layers"):
            if hasattr(tc, attr) and not hasattr(cfg, attr):
                setattr(cfg, attr, getattr(tc, attr))
    for module in model.modules():
        if not hasattr(module, "q_proj"):
            continue
        # object.__setattr__ avoids registering duplicate parameter paths in
        # state_dict while still making getattr(module, "q_lin") work.
        for legacy, native in (("q_lin", "q_proj"), ("k_lin", "k_proj"),
                               ("v_lin", "v_proj"), ("out_lin", "o_proj")):
            if not hasattr(module, legacy) and hasattr(module, native):
                object.__setattr__(module, legacy, getattr(module, native))
    return model


def qwen_projection(module: nn.Module, legacy: str, native: str) -> nn.Module | None:
    """Return a projection under either its legacy or native spelling."""
    value = getattr(module, native, None)
    if value is None:
        value = getattr(module, legacy, None)
    return value


def qwen_attention_mask(mask: torch.Tensor | None, *, causal: bool = True,
                        expects_bidirectional: bool = False) -> torch.Tensor | None:
    """Adapt masks only for modules that explicitly request another contract.

    Native Qwen attention keeps its prepared causal 4-D additive mask. A
    replacement that declares ``expects_bidirectional`` receives a boolean
    visibility mask; other callers receive the original object unchanged.
    """
    if mask is None or not expects_bidirectional:
        return mask
    if mask.dim() == 2:
        return mask.bool()[:, None, None, :]
    if mask.dim() == 4:
        if not causal:
            return torch.ones(mask.shape, dtype=torch.bool, device=mask.device)
        return mask >= 0
    if mask.dim() == 3:
        return mask.bool()[:, None, :, :]
    return mask


def bypass_qwen_rope(kwargs: dict[str, Any], module: Any) -> dict[str, Any]:
    """Prepare kwargs for a module that does not implement RoPE.

    This does not modify the base model or its rotary module. The marker is
    consumed by wrappers and the positional tensors are omitted only from the
    injected replacement call.
    """
    if getattr(module, "supports_rope", False):
        return kwargs
    result = dict(kwargs)
    result.pop("position_embeddings", None)
    result.pop("position_ids", None)
    result["qwen_rope_bypassed"] = True
    return result


def fp32_linalg(fn, *tensors: torch.Tensor, **kwargs: Any):
    """Run QR/SVD/eigh-style operations in fp32 and restore output dtype."""
    dtype = next((t.dtype for t in tensors if isinstance(t, torch.Tensor) and t.is_floating_point()), torch.float32)
    prepared = [t.float() if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in tensors]
    result = fn(*prepared, **kwargs)
    if isinstance(result, tuple):
        return tuple(x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in result)
    return result.to(dtype) if isinstance(result, torch.Tensor) and result.is_floating_point() else result


def qwen_module_policy(module_name: str) -> str:
    """Classify modules whose semantics cannot be promised for causal RoPE Qwen."""
    name = module_name.lower()
    explicit_incompatible = {
        "aegis_v1", "aegis_v3", "aegis_v4", "aegis_v5", "aegis_v6",
        "bam_umoe", "cosformer", "cur", "cur_moe", "cur_moe_compressor",
        "decay_linear_attn", "dobi", "dynamic_rank",
        # These legacy replacements currently implement full-MHA, padding
        # masks, or absolute-position assumptions.  Until their forward pass
        # explicitly handles Qwen GQA + RoPE + causal masks, skipping is safer
        # than reporting a numerically valid but semantically invalid result.
        "a3", "aasvd", "adaptive_svd", "asvd", "cluster",
        "cluster_attention", "goldfinch_ahn", "griffin_ssd", "hgrn2_layer",
        "hgrn2_outer", "fusion_attn", "aegis_v2", "adaptive_svd_attention",
        "huffman", "block_transformer_hiba", "chela_slay", "fasa",
        "gated_deltanet_block", "granite_hybrid", "hgrn2_gated_deltanet",
    }
    if name in explicit_incompatible:
        return "incompatible"
    incompatible = ("linear", "rnn", "ssm", "mamba", "performer", "cosformer",
                    "random", "recurrent", "delta_net", "gated_linear", "span")
    return "incompatible" if any(token in name for token in incompatible) else "adapted"


def is_dtype_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, (RuntimeError, NotImplementedError)) and any(token in text for token in (
        "half", "float16", "geqrf", "dtype", "linalg", "qr decomposition", "not implemented",
    ))


def is_architecture_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in (
        "shape", "size mismatch", "unexpected keyword", "unexpected argument", "has no attribute",
        "not enough values to unpack", "too many values to unpack", "values to unpack",
        "attention mask", "position_embeddings", "mat1 and mat2", "cannot be multiplied",
        "unsupported operand type", "tensor' and 'tuple'",
    ))
