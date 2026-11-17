from __future__ import annotations
import logging
import torch
import torch.nn as nn
from typing import Any

logger = logging.getLogger(__name__)

def _copy_attention_weights(module: nn.Module, src: Any) -> bool:
    """Copies pretrained weights and biases into module and detects nested encoder."""
    if src is None:
        return False
    cls_name = getattr(src, "__class__", type(src)).__name__
    has_out = any(hasattr(src, a) for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))
    is_nested = False
    if (
        cls_name in (
            "BertSelfAttention",
            "RobertaSelfAttention",
            "XLMRobertaSelfAttention",
            "ElectraSelfAttention",
            "DebertaSelfAttention",
        )
        or (hasattr(src, "query") and not has_out)
        or (hasattr(src, "self") and hasattr(src, "output"))
    ):
        is_nested = True
    module.is_nested_encoder = bool(is_nested)

    extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "q": (None, None),
        "k": (None, None),
        "v": (None, None),
        "o": (None, None),
    }

    if hasattr(src, "c_attn") and src.c_attn is not None:
        c_attn_mod = src.c_attn
        c_w = getattr(c_attn_mod, "weight", None)
        if c_w is not None:
            c_w = c_w.data.clone()
            if "Conv1D" in type(c_attn_mod).__name__ or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                c_w = c_w.t()
            sz = c_w.size(0) // 3
            c_b = c_attn_mod.bias.data.clone() if getattr(c_attn_mod, "bias", None) is not None else None
            extracted["q"] = (c_w[:sz], c_b[:sz] if c_b is not None else None)
            extracted["k"] = (c_w[sz : 2 * sz], c_b[sz : 2 * sz] if c_b is not None else None)
            extracted["v"] = (c_w[2 * sz :], c_b[2 * sz :] if c_b is not None else None)
        if hasattr(src, "c_proj") and src.c_proj is not None:
            o_w = getattr(src.c_proj, "weight", None)
            if o_w is not None:
                o_w = o_w.data.clone()
                if "Conv1D" in type(src.c_proj).__name__ or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                    o_w = o_w.t()
                o_b = src.c_proj.bias.data.clone() if getattr(src.c_proj, "bias", None) is not None else None
                extracted["o"] = (o_w, o_b)
    elif hasattr(src, "self") and hasattr(src.self, "query"):
        for k, name in (("q", "query"), ("k", "key"), ("v", "value")):
            m = getattr(src.self, name, None)
            if m is not None and hasattr(m, "weight"):
                extracted[k] = (m.weight.data.clone(), m.bias.data.clone() if m.bias is not None else None)
        if hasattr(src, "output") and hasattr(src.output, "dense"):
            d = src.output.dense
            extracted["o"] = (d.weight.data.clone(), d.bias.data.clone() if d.bias is not None else None)
    else:
        aliases = {
            "q": ["q_proj", "query", "q_lin"],
            "k": ["k_proj", "key", "k_lin"],
            "v": ["v_proj", "value", "v_lin"],
            "o": ["out_proj", "dense", "out_lin", "o_proj"],
        }
        for k, names in aliases.items():
            for name in names:
                m = getattr(src, name, None)
                if m is not None and hasattr(m, "weight"):
                    extracted[k] = (m.weight.data.clone(), m.bias.data.clone() if m.bias is not None else None)
                    break

    missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not module.is_nested_encoder)]
    if missing:
        try:
            from neuralops.core.adapters import UniversalAdapter

            adapter = UniversalAdapter()
            for k in missing:
                w, b = adapter.get_weight_bias(src, k)
                if w is not None:
                    extracted[k] = (w.clone(), b.clone() if b is not None else None)
        except Exception as e:
            logger.debug("UniversalAdapter extraction fallback: %s", e)

    target_map = {
        "q": getattr(module, "q_proj", None),
        "k": getattr(module, "k_proj", None),
        "v": getattr(module, "v_proj", None),
        "o": getattr(module, "out_proj", None),
    }
    with torch.no_grad():
        for k, target in target_map.items():
            if target is None:
                continue
            w, b = extracted[k]
            if w is not None:
                if w.shape == target.weight.shape:
                    target.weight.copy_(w)
                elif w.t().shape == target.weight.shape:
                    target.weight.copy_(w.t())
                elif target.weight.dim() == 2 and w.dim() == 2 and target.weight.shape[1] == w.shape[1]:
                    if target.weight.shape[0] % w.shape[0] == 0:
                        g = target.weight.shape[0] // w.shape[0]
                        target.weight.copy_(w.repeat_interleave(g, dim=0))
                if b is not None and target.bias is not None:
                    if b.shape == target.bias.shape:
                        target.bias.copy_(b)
                    elif target.bias.shape[0] % b.shape[0] == 0:
                        g = target.bias.shape[0] // b.shape[0]
                        target.bias.copy_(b.repeat_interleave(g, dim=0))

        if module.is_nested_encoder and extracted["o"][0] is None and hasattr(module, "out_proj"):
            nn.init.eye_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
    return is_nested
