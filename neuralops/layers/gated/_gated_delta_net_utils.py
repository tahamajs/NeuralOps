#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight copying and injection utilities for Gated DeltaNet."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def copy_gdn_weights(self: Any, old_attn: Any) -> None:
    if old_attn is None:
        return

    try:
        for p in old_attn.parameters():
            self._orig_storage_bytes += p.numel() * p.element_size()
    except Exception:
        # allowed-swallow: best-effort-telemetry
        pass

    cls_name = old_attn.__class__.__name__
    is_nested = False
    if (
        hasattr(old_attn, "self")
        and hasattr(old_attn.self, "query")
        and hasattr(old_attn, "output")
        and hasattr(old_attn.output, "dense")
    ) or (
        cls_name
        in (
            "BertSelfAttention",
            "RobertaSelfAttention",
            "XLMRobertaSelfAttention",
            "ElectraSelfAttention",
            "DebertaSelfAttention",
        )
        or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
    ):
        is_nested = True
    if not is_nested and getattr(self, "is_nested_encoder", False):
        is_nested = True
    self.is_nested_encoder = bool(is_nested)

    if hasattr(old_attn, "rotary_emb"):
        self.rotary_emb = old_attn.rotary_emb

    extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "q": (None, None),
        "k": (None, None),
        "v": (None, None),
        "o": (None, None),
    }

    if hasattr(old_attn, "c_attn") and old_attn.c_attn is not None:
        c = old_attn.c_attn
        if hasattr(c, "merge") and callable(c.merge) and not getattr(c, "merged", False):
            try:
                c.merge()
            except Exception as e:
                logger.debug("c_attn merge skipped: %s", e)
        base_c = getattr(c, "base_layer", c)
        cw = getattr(base_c, "weight", getattr(c, "weight", None))
        if cw is not None:
            cw = cw.data.clone()
            is_conv = (
                "Conv1D" in c.__class__.__name__
                or "Conv1D" in base_c.__class__.__name__
                or hasattr(c, "nf")
                or hasattr(base_c, "nf")
                or (cw.ndim == 2 and cw.shape[0] < cw.shape[1])
            )
            if is_conv:
                cw = cw.t().contiguous()
            split = cw.size(0) // 3
            bmod = getattr(base_c, "bias", getattr(c, "bias", None))
            cb = bmod.data.clone() if bmod is not None else None
            extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
            extracted["k"] = (
                cw[split : 2 * split],
                cb[split : 2 * split] if cb is not None else None,
            )
            extracted["v"] = (
                cw[2 * split :],
                cb[2 * split :] if cb is not None else None,
            )

        if hasattr(old_attn, "c_proj") and old_attn.c_proj is not None:
            o = old_attn.c_proj
            base_o = getattr(o, "base_layer", o)
            ow = getattr(base_o, "weight", getattr(o, "weight", None))
            if ow is not None:
                ow = ow.data.clone()
                is_conv_o = (
                    "Conv1D" in o.__class__.__name__
                    or "Conv1D" in base_o.__class__.__name__
                    or hasattr(o, "nf")
                    or hasattr(base_o, "nf")
                    or (ow.ndim == 2 and ow.shape[0] < ow.shape[1])
                )
                if is_conv_o:
                    ow = ow.t().contiguous()
                bo = getattr(base_o, "bias", getattr(o, "bias", None))
                extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

    elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
        s = old_attn.self
        for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
            m = getattr(s, sub, None)
            if m is not None and hasattr(m, "weight"):
                b = getattr(m, "bias", None)
                extracted[key] = (
                    m.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )
        if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
            d = old_attn.output.dense
            b = getattr(d, "bias", None)
            extracted["o"] = (
                d.weight.data.clone(),
                b.data.clone() if b is not None else None,
            )

    else:
        candidates = {
            "q": ("q_lin", "q_proj", "query", "W_q"),
            "k": ("k_lin", "k_proj", "key", "W_k"),
            "v": ("v_lin", "v_proj", "value", "W_v"),
            "o": ("out_lin", "out_proj", "dense", "o_proj", "output", "proj"),
        }
        for key, names in candidates.items():
            for n in names:
                if hasattr(old_attn, n):
                    m = getattr(old_attn, n)
                    if m is not None and hasattr(m, "weight"):
                        b = getattr(m, "bias", None)
                        extracted[key] = (
                            m.weight.data.clone(),
                            b.data.clone() if b is not None else None,
                        )
                        break

    missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
    if missing:
        try:
            from neuralops.core.adapters import UniversalAdapter

            adapter = UniversalAdapter()
            for k in missing:
                w, b = adapter.get_weight_bias(old_attn, k)
                if w is not None:
                    extracted[k] = (w.clone(), b.clone() if b is not None else None)
        except Exception as e:
            logger.debug("UniversalAdapter fallback failed: %s", e)

    missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
    if missing_after:
        logger.warning("GatedDeltaNet copy_weights: missing %s", missing_after)
        return

    with torch.no_grad():
        for key, target in (
            ("q", self.q_proj),
            ("k", self.k_proj),
            ("v", self.v_proj),
            ("o", self.out_proj),
        ):
            w, b = extracted[key]
            if w is None:
                if key == "o" and self.is_nested_encoder:
                    nn.init.eye_(target.weight)
                    if target.bias is not None:
                        nn.init.zeros_(target.bias)
                continue

            if w.ndim != 2:
                w = w.reshape(w.size(0), -1)
            if w.shape != target.weight.shape and w.t().contiguous().shape == target.weight.shape:
                w = w.t().contiguous()

            out_d = min(target.weight.size(0), w.size(0))
            in_d = min(target.weight.size(1), w.size(1))
            target.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(target.weight.device, target.weight.dtype))

            if b is not None and target.bias is not None:
                mb = min(target.bias.numel(), b.numel())
                target.bias.data[:mb].copy_(b[:mb].to(target.bias.device, target.bias.dtype))
            elif target.bias is not None:
                nn.init.zeros_(target.bias)

            n = float(torch.linalg.norm(target.weight.data.float()).item())
            if n < 1e-4:
                logger.warning("GatedDeltaNet: '%s' near-zero norm (%.2e)", key, n)

            if self.freeze_base:
                target.freeze_base_weights()

        w_q = extracted["q"][0]
        if w_q is not None:
            out_z = min(self.z_proj.weight.size(0), w_q.size(0))
            in_z = min(self.z_proj.weight.size(1), w_q.size(1))
            self.z_proj.weight.data[:out_z, :in_z].copy_(
                w_q[:out_z, :in_z].to(self.z_proj.weight.device, self.z_proj.weight.dtype)
            )
            if self.freeze_base:
                self.z_proj.freeze_base_weights()

    try:
        for m in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.z_proj,
            self.out_proj,
        ):
            for p in m.parameters():
                self._comp_storage_bytes += p.numel() * p.element_size()
    except Exception:
        # allowed-swallow: best-effort-telemetry
        pass

    if self.freeze_base:
        self.freeze_base_weights()


def inject_gated_delta_net(
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
        logger.warning("inject_gated_delta_net: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "transformer") and hasattr(target.transformer, "layer"):
        layer_list = target.transformer.layer
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_gated_delta_net(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    from neuralops.layers.gated.gated_delta_net import GatedDeltaNet

    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is not None and (isinstance(src_attn, GatedDeltaNet) or "GatedDeltaNet" in src_attn.__class__.__name__):
        return None

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (getattr(src_attn.q_proj, "in_features", None) if hasattr(src_attn, "q_proj") else None)
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    try:
        new_block = GatedDeltaNet(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            num_heads=num_heads,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )
    except Exception as e:
        logger.exception("GatedDeltaNet injection failed at layer %d: %s", layer_idx, e)
        return None

    if src_attn is not None and src_attn is not target:
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_block.to(device=p.device, dtype=p.dtype)

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_block.is_nested_encoder = True
                target.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                target.attention = new_block
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block
        else:
            logger.error(
                "inject_gated_delta_net: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

    return new_block
