#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight copying and injection utilities for KDA Attention."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from neuralops.layers.gated.kda_attention import KDAAttention

logger = logging.getLogger(__name__)


def copy_kda_weights(self: Any, src: Any) -> None:
    if src is None:
        return

    cls_name = getattr(src, "__class__", type(src)).__name__
    has_out = (
        hasattr(src, "dense")
        or hasattr(src, "out_proj")
        or hasattr(src, "out_lin")
        or hasattr(src, "c_proj")
        or hasattr(src, "o_proj")
    )
    if cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (hasattr(src, "query") and not has_out):
        self.is_nested_encoder = True
    if hasattr(src, "self") and hasattr(src, "output"):
        self.is_nested_encoder = True

    extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "q": (None, None),
        "k": (None, None),
        "v": (None, None),
        "o": (None, None),
    }

    if hasattr(src, "c_attn") and src.c_attn is not None:
        c_attn_mod = src.c_attn
        base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
        c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
        if c_w is not None:
            c_w = c_w.data.clone()
            if (
                c_attn_mod.__class__.__name__ == "Conv1D"
                or base_c.__class__.__name__ == "Conv1D"
                or hasattr(c_attn_mod, "nf")
                or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
            ):
                c_w = c_w.t()
            split_size = c_w.size(0) // 3
            bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
            c_b = bias_mod.data.clone() if bias_mod is not None else None
            extracted["q"] = (
                c_w[:split_size, :],
                c_b[:split_size] if c_b is not None else None,
            )
            extracted["k"] = (
                c_w[split_size : 2 * split_size, :],
                c_b[split_size : 2 * split_size] if c_b is not None else None,
            )
            extracted["v"] = (
                c_w[2 * split_size :, :],
                c_b[2 * split_size :] if c_b is not None else None,
            )

        if hasattr(src, "c_proj") and src.c_proj is not None:
            c_proj_mod = src.c_proj
            base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
            o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
            if o_w is not None:
                o_w = o_w.data.clone()
                if (
                    c_proj_mod.__class__.__name__ == "Conv1D"
                    or base_o.__class__.__name__ == "Conv1D"
                    or hasattr(c_proj_mod, "nf")
                    or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                ):
                    o_w = o_w.t()
                bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (
                    o_w,
                    bias_o.data.clone() if bias_o is not None else None,
                )

    elif hasattr(src, "self") and hasattr(src.self, "query"):
        s = src.self
        for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
            mod = getattr(s, sub_name, None)
            if mod is not None and hasattr(mod, "weight"):
                extracted[proj_key] = (
                    mod.weight.data.clone(),
                    mod.bias.data.clone() if mod.bias is not None else None,
                )
        if hasattr(src, "output") and hasattr(src.output, "dense"):
            out_mod = src.output.dense
            extracted["o"] = (
                out_mod.weight.data.clone(),
                out_mod.bias.data.clone() if out_mod.bias is not None else None,
            )

    else:
        proj_candidates = {
            "q": ["q_lin", "q_proj", "query"],
            "k": ["k_lin", "k_proj", "key"],
            "v": ["v_lin", "v_proj", "value"],
            "o": ["out_lin", "out_proj", "dense", "o_proj"],
        }
        for proj_key, names in proj_candidates.items():
            for name in names:
                if hasattr(src, name):
                    mod = getattr(src, name)
                    if mod is not None and hasattr(mod, "weight"):
                        w_data = mod.weight.data.clone()
                        if (
                            mod.__class__.__name__ == "Conv1D"
                            or hasattr(mod, "nf")
                            or (w_data.ndim == 2 and proj_key == "o" and w_data.shape[0] < w_data.shape[1])
                        ):
                            w_data = w_data.t()
                        extracted[proj_key] = (
                            w_data,
                            mod.bias.data.clone() if mod.bias is not None else None,
                        )
                        break

    missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
    if missing:
        try:
            from neuralops.core.adapters import UniversalAdapter

            adapter = UniversalAdapter()
            for k in missing:
                w, b = adapter.get_weight_bias(src, k)
                if w is not None:
                    extracted[k] = (w.clone(), b.clone() if b is not None else None)
        except Exception as e:
            logger.debug("UniversalAdapter fallback: %s", e)

    missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
    if missing_after:
        raise RuntimeError(
            f"KDAAttention copy_weights: missing mandatory projections {missing_after} from {src.__class__.__name__}."
        )

    def _align_kv(
        w: torch.Tensor,
        b: torch.Tensor | None,
        target_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """GQA-aware alignment via mean-pooling when reducing head count."""
        if w.shape[0] == target_size:
            return w, b
        if w.shape[0] > target_size and w.shape[0] % target_size == 0:
            rep = w.shape[0] // target_size
            w_pooled = w.view(num_kv_heads, rep, head_dim, -1).mean(dim=1).view(target_size, -1)
            b_pooled = b.view(num_kv_heads, rep, head_dim).mean(dim=1).view(-1) if b is not None else None
            return w_pooled, b_pooled
        return w[:target_size, :], (b[:target_size] if b is not None else None)

    target_map = {
        "q": self.q_proj,
        "k": self.k_proj,
        "v": self.v_proj,
        "o": self.out_proj,
    }

    with torch.no_grad():
        for proj_key, target_mod in target_map.items():
            w, b = extracted[proj_key]
            if w is None:
                if proj_key == "o" and self.is_nested_encoder:
                    nn.init.eye_(target_mod.weight)
                    if target_mod.bias is not None:
                        nn.init.zeros_(target_mod.bias)
                continue

            if w.ndim != 2:
                w = w.reshape(w.size(0), -1)
            if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                w = w.t().contiguous()

            if proj_key in ("k", "v"):
                w, b = _align_kv(w, b, target_mod.weight.size(0), self.num_kv_heads, self.head_dim)

            out_d = min(target_mod.weight.size(0), w.size(0))
            in_d = min(target_mod.weight.size(1), w.size(1))

            target_mod.weight.data[:out_d, :in_d].copy_(
                w[:out_d, :in_d].to(
                    device=target_mod.weight.device,
                    dtype=target_mod.weight.dtype,
                )
            )

            if b is not None and target_mod.bias is not None:
                min_b = min(target_mod.bias.numel(), b.numel())
                target_mod.bias.data[:min_b].copy_(
                    b[:min_b].to(
                        device=target_mod.bias.device,
                        dtype=target_mod.bias.dtype,
                    )
                )
            elif target_mod.bias is not None:
                nn.init.zeros_(target_mod.bias)

            w_norm = float(torch.linalg.norm(target_mod.weight.data.float()).item())
            if w_norm < 1e-4:
                raise RuntimeError(f"KDA copy_weights: '{proj_key}' norm too small ({w_norm:.2e})")

            if self.freeze_base:
                target_mod.freeze_base_weights()

    if self.freeze_base:
        self.freeze_base_weights()


def inject_kda(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> KDAAttention | nn.Module | None:
    if _depth > 8:
        logger.warning("inject_kda: max recursion depth exceeded, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_kda(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    from neuralops.layers.gated.kda_attention import KDAAttention

    src_attn = old_attn
    if src_attn is None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                src_attn = target.attention.self
            else:
                src_attn = target.attention
        elif hasattr(target, "self"):
            src_attn = target.self
        elif hasattr(target, "attn"):
            src_attn = target.attn
        elif hasattr(target, "self_attn"):
            src_attn = target.self_attn

    if src_attn is None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None and cfg is None:
        return None

    if src_attn is not None and (isinstance(src_attn, KDAAttention) or "KDAAttention" in src_attn.__class__.__name__):
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 768))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 12))
    )

    try:
        new_attn = KDAAttention(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            num_heads=num_heads,
            old_attn=src_attn,
            layer_idx=layer_idx,
            **kwargs,
        )
    except Exception as e:
        logger.exception("KDA injection failed: %s", e)
        return None

    if src_attn is not None and src_attn is not target:
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    if target is not None:
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
                new_attn.is_nested_encoder = False
            spliced = True
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True
        elif hasattr(target, "attn"):
            target.attn = new_attn
            spliced = True
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            spliced = True

        if not spliced:
            logger.error(
                "inject_kda: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

    return new_attn
