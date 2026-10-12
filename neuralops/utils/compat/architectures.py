"""Architectural utilities and family-aware adapters for model compatibility."""
from neuralops.core.common import repeat_kv

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.utils.model_family import CAUSAL_FAMILIES, ModelAdapter, detect_family
from neuralops.utils.qwen_compat import QwenIncompatibleError

logger = logging.getLogger(__name__)




def init_output_scale(module: nn.Module, name: str = "output_scale") -> None:
    """Register output_scale learnable parameter if not already present.

    Args:
        module: The target PyTorch module.
        name: Parameter attribute name to register.
    """
    if not hasattr(module, name):
        module.register_parameter(name, nn.Parameter(torch.ones(1)))


def is_decoder_family(family: str) -> bool:
    """Check whether a family (canonical or raw) is a causal decoder.

    Args:
        family: Model family name identifier.

    Returns:
        True if the family is in CAUSAL_FAMILIES.
    """
    return family in CAUSAL_FAMILIES


def is_decoder_projection_name(name: str) -> bool:
    """Whether a parameter name denotes a decoder attention projection.

    Args:
        name: Name of parameter or sub-module.

    Returns:
        True if the name matches standard decoder projection patterns.
    """
    name_low = name.lower()
    if any(
        p in name_low
        for p in [
            "c_attn",
            "c_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
        ]
    ):
        if "attn" in name_low or "attention" in name_low or "self_attn" in name_low:
            return True
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in {"weight", "bias", "base_layer"}:
        parts = name.rsplit(".", 2)
        leaf = parts[-2].lower() if len(parts) >= 2 else leaf
    return leaf in {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "c_attn",
        "c_proj",
    }


SENSITIVE_PATTERNS = [
    "attn",
    "attention",
    "delta",
    "delta_net",
    "gate",
    "expert",
    "router",
    "norm",
    "rope",
    "embed",
    "lm_head",
    "score",
    "classifier",
    "conv1d",
    "in_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "lora",
]

ALLOWED_FFN_PATTERNS = [
    "mlp",
    "ffn",
    "dense",
    "feed_forward",
    "down_proj",
    "up_proj",
    "wi",
    "wo",
]


def should_compress_qwen3_5(name: str) -> bool:
    """Check whether a layer in Qwen3.5 should undergo compression.

    Args:
        name: Full parameter or module name.

    Returns:
        True if the layer is an allowed FFN layer and not sensitive.
    """
    n = name.lower()
    if any(p in n for p in SENSITIVE_PATTERNS):
        return False
    return any(p in n for p in ALLOWED_FFN_PATTERNS)


def should_compress_linear(model: Any, name: str, family: str | None = None) -> bool:
    """Family-aware check for whether a linear layer should be compressed.

    For qwen3_5: sensitive layers (attention, DeltaNet, GQA, MoE routers/experts,
    norms, RoPE, embeddings, heads) are protected; only ordinary FFN dense layers
    are compressed. For other families, safe linear layer compression is preserved.

    Args:
        model: Target model instance or family string.
        name: Parameter or module path name.
        family: Optional explicit model family name.

    Returns:
        True if the linear layer can be safely compressed.
    """
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in {"weight", "bias", "base_layer"}:
        parts = name.rsplit(".", 2)
        leaf = parts[-2].lower() if len(parts) >= 2 else leaf

    name_low = name.lower()
    if leaf in {"classifier", "score", "lm_head", "pooler", "pre_classifier"} or any(
        k in name_low
        for k in [
            "classifier",
            "score",
            "lm_head",
            "pooler",
            "pre_classifier",
            "classification_head",
        ]
    ):
        return False

    if "c_attn" in name_low or "c_proj" in name_low:
        if "attn" in name_low or "attention" in name_low:
            return True

    fam = family or (detect_family(model) if not isinstance(model, str) else model)

    if fam == "qwen3_5":
        return should_compress_qwen3_5(name)

    recognized_attn = [
        "q_lin", "k_lin", "v_lin", "out_lin",  # DistilBERT
        "query", "key", "value", "dense", "output.dense",  # RoBERTa / BERT
        "c_attn", "c_proj",  # GPT-2
        "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",  # Qwen / LLaMA
    ]

    if any(leaf == k or k in name_low.split(".") for k in recognized_attn):
        return True

    if is_decoder_family(fam):
        return is_decoder_projection_name(name) or any(
            kw in name_low for kw in ["mlp", "ffn", "dense", "feed_forward"]
        )

    return True


def get_dynamic_compression_rank(
    in_features: int,
    out_features: int,
    model_or_family: Any = None,
    default_rank: int = 16,
    comp_config: dict | None = None,
) -> int:
    """Compute architecture-aware dynamic compression rank.

    For Qwen3.5: rank = min(256, max(128, min(in_features, out_features) // 4))
    For other families: preserves default fixed/configured rank.

    Args:
        in_features: Layer input features dimension.
        out_features: Layer output features dimension.
        model_or_family: Model instance or family string identifier.
        default_rank: Default fallback rank if not computed dynamically.
        comp_config: Optional dictionary or config object specifying rank overrides.

    Returns:
        Integer rank to use for compression.
    """
    if isinstance(comp_config, dict):
        if "qwen3_5_rank" in comp_config and comp_config["qwen3_5_rank"] is not None:
            return int(comp_config["qwen3_5_rank"])
        if "dynamic_rank" in comp_config and comp_config["dynamic_rank"] is not None:
            return int(comp_config["dynamic_rank"])
        if "rank" in comp_config and comp_config["rank"] is not None:
            return int(comp_config["rank"])
    elif comp_config is not None:
        if getattr(comp_config, "qwen3_5_rank", None) is not None:
            return int(comp_config.qwen3_5_rank)
        if getattr(comp_config, "dynamic_rank", None) is not None:
            return int(comp_config.dynamic_rank)
        if getattr(comp_config, "rank", None) is not None:
            return int(comp_config.rank)

    if isinstance(model_or_family, str):
        fam = model_or_family
    elif model_or_family is not None:
        fam = detect_family(model_or_family)
    else:
        fam = "unknown"
    if fam == "qwen3_5":
        target = min(in_features, out_features) // 4
        return max(128, min(256, target))
    return default_rank


def maybe_apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    adapter: ModelAdapter,
    rotary_emb: Any,
    position_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a native rotary module when required; never silently drop RoPE.

    Args:
        q: Query tensor.
        k: Key tensor.
        adapter: ModelAdapter instance with position_embedding metadata.
        rotary_emb: Rotary embedding module or callable.
        position_ids: Optional tensor of sequence position ids.

    Returns:
        Tuple of (rotated_q, rotated_k) tensors.
    """
    if adapter.position_embedding != "rotary":
        return q, k
    if rotary_emb is None:
        raise QwenIncompatibleError("rotary_position_embedding_required")
    try:
        rotary = rotary_emb(q, position_ids)
    except TypeError:
        try:
            rotary = rotary_emb(q, position_ids=position_ids)
        except TypeError as exc:
            raise QwenIncompatibleError(f"rotary_forward_signature: {exc}") from exc
    if hasattr(rotary, "cos") and hasattr(rotary, "sin"):
        cos, sin = rotary.cos, rotary.sin
    elif isinstance(rotary, (tuple, list)) and len(rotary) >= 2:
        cos, sin = rotary[0], rotary[1]
    else:
        raise QwenIncompatibleError("rotary_forward_invalid_output")

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]),
            dim=-1,
        )

    if cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.shape[-2] != q.shape[-2]:
        cos, sin = cos[..., : q.shape[-2], :], sin[..., : q.shape[-2], :]
    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )


def causal_mask_from_attention_mask(
    attention_mask: torch.Tensor | None,
    adapter: ModelAdapter,
) -> torch.Tensor | None:
    """Return the mask without discarding decoder causal visibility semantics.

    Args:
        attention_mask: Source attention mask tensor or None.
        adapter: ModelAdapter instance indicating whether model is causal.

    Returns:
        Adapted causal or non-causal boolean mask.
    """
    if attention_mask is None:
        return None
    if adapter.causal and attention_mask.dim() == 4:
        return attention_mask
    if adapter.causal and attention_mask.dim() == 2:
        bsz, seqlen = attention_mask.shape
        causal = torch.ones(
            (seqlen, seqlen),
            dtype=torch.bool,
            device=attention_mask.device,
        ).tril()
        return causal[None, None, :, :] & attention_mask.bool()[:, None, None, :]
    if not adapter.causal and attention_mask.dim() == 2:
        return attention_mask.bool()[:, None, None, :]
    if not adapter.causal:
        return attention_mask
    raise QwenIncompatibleError(f"causal_mask_format_{attention_mask.dim()}d_unsupported")


def convert_mask(
    mask: torch.Tensor | None,
    adapter: ModelAdapter,
    expects_bidirectional: bool = False,
) -> torch.Tensor | None:
    """Convert an attention mask to expected shape and bool dtype.

    Args:
        mask: Optional input mask tensor.
        adapter: ModelAdapter providing causal context.
        expects_bidirectional: Whether the target module expects bidirectional masking.

    Returns:
        Converted mask tensor or None.
    """
    if mask is None or not expects_bidirectional:
        return mask
    if mask.dim() == 2:
        return mask[:, None, None, :].bool()
    if mask.dim() == 3:
        return mask[:, None, :, :].bool()
    if mask.dim() == 4:
        return mask >= 0
    return mask


def fp32_linalg(op: Any, *tensors: torch.Tensor, **kwargs: Any) -> Any:
    """Execute a linear algebra operation with float32 precision and cast back.

    Args:
        op: Callable linear algebra function.
        *tensors: Floating point tensor operands.
        **kwargs: Additional keyword arguments for op.

    Returns:
        Result tensor(s) cast back to initial floating precision.
    """
    dtypes = [x.dtype for x in tensors if isinstance(x, torch.Tensor) and x.is_floating_point()]
    dtype = dtypes[0] if dtypes else torch.float32
    result = op(
        *[
            (x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x)
            for x in tensors
        ],
        **kwargs,
    )
    if isinstance(result, tuple):
        return tuple(
            x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
            for x in result
        )
    return (
        result.to(dtype)
        if isinstance(result, torch.Tensor) and result.is_floating_point()
        else result
    )


class FamilyModuleWrapper(nn.Module):
    """Wrapper that applies family adaptations such as mask conversions and RoPE checks."""

    def __init__(self, module: nn.Module, family: str) -> None:
        super().__init__()
        from .aliases import install_projection_aliases

        self.module = install_projection_aliases(module)
        self.adapter = ModelAdapter(module, family)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        mask = kwargs.get("attention_mask", kwargs.get("mask", kwargs.get("attn_mask")))
        if (
            self.adapter.position_embedding == "rotary"
            and not getattr(self.module, "supports_rope", False)
            and (
                "position_embeddings" in kwargs
                or "position_ids" in kwargs
                or getattr(self.module, "requires_rope", False)
            )
        ):
            raise QwenIncompatibleError("module_does_not_support_rope")
        if mask is not None:
            adapted = convert_mask(
                mask,
                self.adapter,
                bool(getattr(self.module, "expects_bidirectional", False)),
            )
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = adapted
            elif "mask" in kwargs:
                kwargs["mask"] = adapted
            elif "attn_mask" in kwargs:
                kwargs["attn_mask"] = adapted
        return self.module(*args, **kwargs)


def wrap_for_family(module: nn.Module, model_family: str | None = None) -> nn.Module:
    """Wrap a module with FamilyModuleWrapper if required by family rules.

    Args:
        module: PyTorch module to wrap.
        model_family: Family identifier string, or detected automatically if None.

    Returns:
        Wrapped module or original module if no wrapping is necessary.
    """
    from .aliases import install_projection_aliases

    family = model_family or detect_family(module)
    install_projection_aliases(module)
    family_aliases = globals().get("FAMILY_ALIASES", {})
    canonical = family_aliases.get(family, family) if isinstance(family_aliases, dict) else family
    return (
        module
        if canonical in {"bert", "roberta", "unknown"}
        else FamilyModuleWrapper(module, family)
    )


def module_family_support(name: str) -> dict[str, str]:
    """Return dictionary of compatibility support tags per model family.

    Args:
        name: Name of architecture/attention module.

    Returns:
        Dict mapping family names to compatibility status strings.
    """
    if name.lower() in {"mamba"}:
        return {
            "bert": "encoder_compatible",
            "roberta": "encoder_compatible",
            "gpt": "adapted",
            "qwen": "incompatible",
            "qwen3_5": "incompatible",
            "t5": "encoder_decoder_compatible",
            "ssm": "adapted",
            "unknown": "adapted",
        }
    return {
        "bert": "encoder_compatible",
        "roberta": "encoder_compatible",
        "gpt": "adapted",
        "qwen": "adapted",
        "qwen3_5": "adapted",
        "t5": "encoder_decoder_compatible",
        "ssm": "adapted",
        "unknown": "adapted",
    }
