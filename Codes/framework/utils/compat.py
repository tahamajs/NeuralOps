"""Family-aware, opt-in compatibility helpers for attention modules."""
from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn

from .model_family import ModelAdapter, detect_family
from .qwen_compat import QwenIncompatibleError


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query K/V heads: [B, Hkv, L, D] -> [B, H, L, D]."""
    if n_rep <= 1:
        return x
    if x.dim() != 4:
        raise ValueError(f"repeat_kv expects [B,H,L,D], got {tuple(x.shape)}")
    return x[:, :, None, :, :].expand(x.size(0), x.size(1), n_rep, x.size(2), x.size(3)).reshape(
        x.size(0), x.size(1) * n_rep, x.size(2), x.size(3)
    )


def is_decoder_family(family: str) -> bool:
    return family in {"gpt", "qwen"}


def is_decoder_projection_name(name: str) -> bool:
    """Whether a parameter name denotes a decoder attention projection."""
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in {"weight", "bias"}:
        parts = name.rsplit(".", 2)
        leaf = parts[-2].lower() if len(parts) >= 2 else leaf
    return leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "c_attn", "c_proj"}


def should_compress_linear(model: Any, name: str) -> bool:
    """Keep decoder compression restricted to attention projections while protecting task heads across all architectures."""
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in {"weight", "bias"}:
        parts = name.rsplit(".", 2)
        leaf = parts[-2].lower() if len(parts) >= 2 else leaf
    if leaf in {"classifier", "score", "lm_head", "pooler", "pre_classifier"} or any(
        k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "pre_classifier"]
    ):
        return False
    family = detect_family(model)
    return not is_decoder_family(family) or is_decoder_projection_name(name)


def maybe_apply_rotary(q: torch.Tensor, k: torch.Tensor, adapter: ModelAdapter,
                       rotary_emb: Any, position_ids: Optional[torch.Tensor]):
    """Apply a native rotary module when required; never silently drop RoPE."""
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
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    # HF rotary outputs are commonly [B,L,D] or [B,1,L,D], while custom
    # modules conventionally use [B,H,L,D].
    if cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.shape[-2] != q.shape[-2]:
        cos, sin = cos[..., :q.shape[-2], :], sin[..., :q.shape[-2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def causal_mask_from_attention_mask(attention_mask: Optional[torch.Tensor], adapter: ModelAdapter):
    """Return the mask without discarding decoder causal visibility semantics."""
    if attention_mask is None:
        return None
    if adapter.causal and attention_mask.dim() == 4:
        return attention_mask
    if adapter.causal and attention_mask.dim() == 2:
        bsz, seqlen = attention_mask.shape
        causal = torch.ones((seqlen, seqlen), dtype=torch.bool, device=attention_mask.device).tril()
        return causal[None, None, :, :] & attention_mask.bool()[:, None, None, :]
    if not adapter.causal and attention_mask.dim() == 2:
        return attention_mask.bool()[:, None, None, :]
    if not adapter.causal:
        return attention_mask
    raise QwenIncompatibleError(f"causal_mask_format_{attention_mask.dim()}d_unsupported")


def install_projection_aliases(module: nn.Module) -> nn.Module:
    for child in module.modules():
        for legacy, native in (("q_lin", "q_proj"), ("k_lin", "k_proj"),
                               ("v_lin", "v_proj"), ("out_lin", "o_proj")):
            if not hasattr(child, legacy) and hasattr(child, native):
                object.__setattr__(child, legacy, getattr(child, native))
    return module


def convert_mask(mask: Optional[torch.Tensor], adapter: ModelAdapter,
                 expects_bidirectional: bool = False) -> Optional[torch.Tensor]:
    if mask is None or not expects_bidirectional:
        return mask
    if mask.dim() == 2:
        return mask[:, None, None, :].bool()
    if mask.dim() == 3:
        return mask[:, None, :, :].bool()
    if mask.dim() == 4:
        # A causal additive mask cannot truthfully become bidirectional.
        # Preserve visibility semantics and let the caller decide whether to
        # skip a module whose contract requires non-causal attention.
        return mask >= 0
    return mask


def fp32_linalg(op, *tensors: torch.Tensor, **kwargs: Any):
    dtypes = [x.dtype for x in tensors if isinstance(x, torch.Tensor) and x.is_floating_point()]
    dtype = dtypes[0] if dtypes else torch.float32
    result = op(*[(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x) for x in tensors], **kwargs)
    if isinstance(result, tuple):
        return tuple(x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in result)
    return result.to(dtype) if isinstance(result, torch.Tensor) and result.is_floating_point() else result


class FamilyModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module, family: str):
        super().__init__()
        self.module = install_projection_aliases(module)
        self.adapter = ModelAdapter(module, family)

    def forward(self, *args: Any, **kwargs: Any):
        mask = kwargs.get("attention_mask", kwargs.get("mask", kwargs.get("attn_mask")))
        # A replacement that receives rotary tensors must explicitly support
        # them.  A bare legacy call with no positional inputs remains usable
        # for backwards-compatible unit/injection paths; the runner's strict
        # family policy is responsible for classifying such modules.
        if (self.adapter.position_embedding == "rotary"
                and not getattr(self.module, "supports_rope", False)
                and ("position_embeddings" in kwargs or "position_ids" in kwargs
                     or getattr(self.module, "requires_rope", False))):
            raise QwenIncompatibleError("module_does_not_support_rope")
        if mask is not None:
            adapted = convert_mask(mask, self.adapter, bool(getattr(self.module, "expects_bidirectional", False)))
            if "attention_mask" in kwargs: kwargs["attention_mask"] = adapted
            elif "mask" in kwargs: kwargs["mask"] = adapted
            elif "attn_mask" in kwargs: kwargs["attn_mask"] = adapted
        return self.module(*args, **kwargs)


def wrap_for_family(module: nn.Module, model_family: Optional[str] = None) -> nn.Module:
    family = model_family or detect_family(module)
    install_projection_aliases(module)
    # Keep BERT/DistilBERT's original call path byte-for-byte compatible.
    return module if family in {"bert", "roberta", "unknown"} else FamilyModuleWrapper(module, family)


def module_family_support(name: str) -> dict[str, str]:
    n = name.lower()
    incompatible = ("linear", "rnn", "ssm", "mamba", "performer", "cosformer", "random", "recurrent", "span", "delta_net")
    qwen = "incompatible" if any(x in n for x in incompatible) else "adapted"
    if n in {"full", "faq", "frandi_compressor", "astro"}:
        qwen = "adapted"
    return {
        "bert": "encoder_compatible",
        "roberta": "encoder_compatible",
        "gpt": qwen,
        "qwen": qwen,
        "t5": "encoder_decoder_compatible",
        "ssm": "incompatible",
        "unknown": "unknown",
    }
