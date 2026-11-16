#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight copying and injection utilities for MoE DeltaKV Paged Attention."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def copy_moe_deltakv_weights(self: Any, src: Any) -> None:
    if src is None:
        return

    from neuralops.core.adapters import UniversalAdapter

    if self._source_id is not None and id(src) == self._source_id:
        return

    extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "q": (None, None),
        "k": (None, None),
        "v": (None, None),
        "o": (None, None),
    }

    for proj in ("q", "k", "v", "o"):
        try:
            w, b = UniversalAdapter.get_weight_bias(src, proj)
            if w is not None:
                extracted[proj] = (w, b)
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

    if hasattr(src, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
        try:
            mod_c = src.c_attn
            base_c = getattr(mod_c, "base_layer", mod_c)
            weight = getattr(base_c, "weight", getattr(mod_c, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(mod_c, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                cls_c = mod_c.__class__.__name__
                base_cls = base_c.__class__.__name__
                if (
                    hasattr(mod_c, "nf")
                    or hasattr(base_c, "nf")
                    or "Conv1D" in cls_c
                    or "Conv1D" in base_cls
                    or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                ):
                    weight = weight.t().contiguous()
                split = weight.shape[0] // 3
                extracted["q"] = (
                    weight[:split],
                    bias_data[:split] if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split : 2 * split],
                    bias_data[split : 2 * split] if bias_data is not None else None,
                )
                extracted["v"] = (
                    weight[2 * split :],
                    bias_data[2 * split :] if bias_data is not None else None,
                )
        except Exception as e:
            logger.debug("GPT-2 c_attn extraction failed: %s", e)

    if hasattr(src, "c_proj") and extracted["o"][0] is None:
        try:
            mod_o = src.c_proj
            base_o = getattr(mod_o, "base_layer", mod_o)
            w_o = getattr(base_o, "weight", getattr(mod_o, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                cls_o = mod_o.__class__.__name__
                base_cls_o = base_o.__class__.__name__
                if (
                    hasattr(mod_o, "nf")
                    or hasattr(base_o, "nf")
                    or "Conv1D" in cls_o
                    or "Conv1D" in base_cls_o
                    or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                ):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(mod_o, "bias", None))
                extracted["o"] = (
                    w_o,
                    b_o.data.clone() if b_o is not None else None,
                )
        except Exception as e:
            logger.debug("GPT-2 c_proj extraction failed: %s", e)

    if hasattr(src, "self") and any(extracted[k][0] is None for k in ("q", "k", "v")):
        try:
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = (
                    s.query.weight.data.clone(),
                    s.query.bias.data.clone() if s.query.bias is not None else None,
                )
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = (
                    s.key.weight.data.clone(),
                    s.key.bias.data.clone() if s.key.bias is not None else None,
                )
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = (
                    s.value.weight.data.clone(),
                    s.value.bias.data.clone() if s.value.bias is not None else None,
                )
        except Exception as e:
            logger.debug("Nested self extraction failed: %s", e)

    if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
        try:
            dense = src.output.dense
            extracted["o"] = (
                dense.weight.data.clone(),
                dense.bias.data.clone() if dense.bias is not None else None,
            )
        except Exception as e:
            logger.debug("output.dense extraction failed: %s", e)

    named_map = {
        "q": ("q_proj", "q_lin", "query"),
        "k": ("k_proj", "k_lin", "key"),
        "v": ("v_proj", "v_lin", "value"),
        "o": ("o_proj", "out_proj", "out_lin", "dense"),
    }
    for proj, names in named_map.items():
        if extracted[proj][0] is None:
            for name in names:
                if hasattr(src, name):
                    try:
                        layer = getattr(src, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (
                                layer.weight.data.clone(),
                                (layer.bias.data.clone() if getattr(layer, "bias", None) is not None else None),
                            )
                            break
                    except Exception:
                        continue

    cls_name = src.__class__.__name__
    has_out_proj = any(
        hasattr(src, a)
        for a in (
            "dense",
            "out_proj",
            "c_proj",
            "out_lin",
            "o_proj",
            "wo",
            "o",
            "proj",
        )
    )
    is_nested = False
    if (
        hasattr(src, "self")
        and hasattr(src.self, "query")
        or cls_name
        in (
            "BertSelfAttention",
            "RobertaSelfAttention",
            "XLMRobertaSelfAttention",
            "ElectraSelfAttention",
        )
    ):
        is_nested = True
    elif cls_name in ("DistilBertAttention", "DistilBertSABlock"):
        is_nested = True
    elif cls_name.endswith("SelfAttention") and not has_out_proj or cls_name.endswith("Attention") and not has_out_proj:
        is_nested = True
    self.is_nested_encoder = bool(is_nested)

    missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
    if not self.is_nested_encoder and extracted["o"][0] is None:
        missing.append("o")
    if missing:
        logger.warning("MoEDeltaKVPaged: missing %s; aborting copy.", missing)
        return

    try:
        for p in src.parameters():
            self._orig_storage_bytes += p.numel() * p.element_size()
    except Exception:
        # allowed-swallow: best-effort-telemetry
        pass

    with torch.no_grad():
        for proj_key, target in (
            ("q", self.query),
            ("k", self.key),
            ("v", self.value),
        ):
            w, b = extracted[proj_key]
            if w is None:
                continue
            in_dim = self.hidden_size
            out_dim = target.weight.shape[0]

            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                w = w.t().contiguous()

            if w.shape[0] < out_dim and out_dim % w.shape[0] == 0:
                rep = out_dim // w.shape[0]
                w = w.repeat(rep, 1)
                if b is not None:
                    b = b.repeat(rep)
            elif w.shape[0] > out_dim and w.shape[0] % out_dim == 0:
                rep = w.shape[0] // out_dim
                w = w.view(out_dim, rep, -1).mean(dim=1)
                if b is not None:
                    b = b.view(out_dim, rep).mean(dim=1)

            target.weight.data.zero_()
            out_rows = min(out_dim, w.shape[0])
            in_cols = min(in_dim, w.shape[1])
            target.weight.data[:out_rows, :in_cols].copy_(
                w[:out_rows, :in_cols].to(target.weight.device, target.weight.dtype)
            )
            if b is not None and target.bias is not None:
                mb = min(target.bias.numel(), b.numel())
                target.bias.data.zero_()
                target.bias.data[:mb].copy_(b[:mb].to(target.bias.device, target.bias.dtype))

        w_o, b_o = extracted["o"]
        if w_o is not None and not self.is_nested_encoder:
            in_dim = self.all_head_size
            out_dim = self.hidden_size
            if w_o.dim() == 2 and w_o.shape[1] != in_dim and w_o.shape[0] == in_dim:
                w_o = w_o.t().contiguous()
            self.out_proj.weight.data.zero_()
            out_rows = min(out_dim, w_o.shape[0])
            in_cols = min(in_dim, w_o.shape[1])
            self.out_proj.weight.data[:out_rows, :in_cols].copy_(
                w_o[:out_rows, :in_cols].to(self.out_proj.weight.device, self.out_proj.weight.dtype)
            )
            if b_o is not None and self.out_proj.bias is not None:
                mb = min(self.out_proj.bias.numel(), b_o.numel())
                self.out_proj.bias.data.zero_()
                self.out_proj.bias.data[:mb].copy_(b_o[:mb].to(self.out_proj.bias.device, self.out_proj.bias.dtype))
        elif self.is_nested_encoder:
            pass

        for m in (self.query, self.key, self.value, self.out_proj):
            m.freeze_base_weights()

    try:
        for m in (self.query, self.key, self.value, self.out_proj):
            for p in m.parameters():
                self._comp_storage_bytes += p.numel() * p.element_size()
    except Exception:
        # allowed-swallow: best-effort-telemetry
        pass

    if hasattr(src, "is_decoder"):
        self.is_decoder = bool(src.is_decoder)
        self.is_causal = self.is_decoder

    self._source_id = id(src)


def inject_moe_deltakv_paged(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_moe_deltakv_paged: recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_moe_deltakv_paged(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is None:
        return None

    from neuralops.layers.sparse.moe_deltakv_paged import MoEDeltaKVPagedAttention

    if isinstance(src_attn, MoEDeltaKVPagedAttention) or "MoEDeltaKV" in src_attn.__class__.__name__:
        return None

    if cfg is None and hasattr(src_attn, "config"):
        cfg = src_attn.config

    page_size = int(kwargs.pop("page_size", getattr(comp_config, "page_size", 16) if comp_config else 16))
    k_experts = int(kwargs.pop("k_experts", getattr(comp_config, "k_experts", 4) if comp_config else 4))
    svd_rank = int(kwargs.pop("svd_rank", getattr(comp_config, "svd_rank", 32) if comp_config else 32))
    q_bits = int(kwargs.pop("q_bits", getattr(comp_config, "q_bits", 4) if comp_config else 4))
    max_delta_error = float(
        kwargs.pop(
            "max_delta_error",
            getattr(comp_config, "max_delta_error", 0.35) if comp_config else 0.35,
        )
    )
    lora_rank = int(kwargs.pop("lora_rank", getattr(comp_config, "lora_rank", 16) if comp_config else 16))
    enable_moe = bool(kwargs.pop("enable_moe", True))
    enable_deltakv = bool(kwargs.pop("enable_deltakv", True))

    try:
        new_attn = MoEDeltaKVPagedAttention(
            config=cfg,
            page_size=page_size,
            k_experts=k_experts,
            svd_rank=svd_rank,
            q_bits=q_bits,
            max_delta_error=max_delta_error,
            lora_rank=lora_rank,
            enable_moe=enable_moe,
            enable_deltakv=enable_deltakv,
            pretrained_attn=src_attn if src_attn is not target else None,
        )
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)
    except Exception as e:
        logger.error("MoEDeltaKVPaged construction failed: %s", e)
        return None

    spliced = False
    if hasattr(target, "attention") and hasattr(target.attention, "self"):
        new_attn.is_nested_encoder = True
        target.attention.self = new_attn
        spliced = True
    elif hasattr(target, "attention"):
        new_attn.is_nested_encoder = False
        target.attention = new_attn
        spliced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
        spliced = True

    if not spliced:
        logger.error(
            "inject_moe_deltakv_paged: no writable attribute on %s",
            type(target).__name__,
        )
        return None

    return new_attn
