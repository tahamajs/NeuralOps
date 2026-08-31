"""Family-aware, opt-in compatibility helpers for attention modules."""
from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn

from .model_family import ModelAdapter, detect_family


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
        kwargs.pop("position_embeddings", None) if self.adapter.position_embedding == "rotary" and not getattr(self.module, "supports_rope", False) else None
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
    if n in {"full", "faq", "frandi_compressor", "astro"}: qwen = "adapted"
    return {"bert": "encoder_compatible", "roberta": "encoder_compatible",
            "gpt": qwen, "qwen": qwen, "t5": "encoder_decoder_compatible",
            "ssm": "incompatible", "unknown": "unknown"}
