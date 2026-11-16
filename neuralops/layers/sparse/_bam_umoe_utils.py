#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight copying and injection utilities for BAM-UMoE."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def copy_bam_umoe_weights(self: Any, pretrained_attn: nn.Module) -> None:
    if pretrained_attn is None:
        return

    try:
        for p in pretrained_attn.parameters():
            self._orig_storage_bytes += p.numel() * p.element_size()
    except Exception:
        # allowed-swallow: best-effort-telemetry
        pass

    cls_name = pretrained_attn.__class__.__name__
    has_out_lin = any(hasattr(pretrained_attn, a) for a in ("out_lin", "c_proj", "o_proj", "out_proj", "dense"))
    is_nested = False
    if (
        (
            hasattr(pretrained_attn, "self")
            and hasattr(pretrained_attn.self, "query")
            and hasattr(pretrained_attn, "output")
            and hasattr(pretrained_attn.output, "dense")
        )
        or (
            cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
                "DebertaSelfAttention",
            )
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        )
        and not has_out_lin
    ):
        is_nested = True
    self.is_nested_encoder = bool(is_nested)

    if hasattr(pretrained_attn, "rotary_emb") and pretrained_attn.rotary_emb is not None:
        self.rotary_emb = pretrained_attn.rotary_emb

    extracted = {
        "q": (None, None),
        "k": (None, None),
        "v": (None, None),
        "o": (None, None),
    }

    def _extract_mod_wb(mod: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if mod is None:
            return None, None
        if hasattr(mod, "merge") and callable(mod.merge) and not getattr(mod, "merged", False):
            try:
                mod.merge()
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass
        base_m = getattr(mod, "base_layer", mod)
        w_m = getattr(base_m, "weight", getattr(mod, "weight", None))
        if w_m is None or not isinstance(w_m, torch.Tensor) or w_m.ndim != 2:
            return None, None
        w_m = w_m.data.clone()
        cls_m = type(mod).__name__
        base_cls_m = type(base_m).__name__
        if ("Conv1D" in cls_m) or ("Conv1D" in base_cls_m) or hasattr(base_m, "nf") or hasattr(mod, "nf"):
            w_m = w_m.t().contiguous()
        b_mod = getattr(base_m, "bias", getattr(mod, "bias", None))
        b_m = b_mod.data.clone() if isinstance(b_mod, torch.Tensor) else None
        return w_m, b_m

    if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
        c = pretrained_attn.c_attn
        base_c = getattr(c, "base_layer", c)
        cw = getattr(base_c, "weight", getattr(c, "weight", None))
        if cw is not None:
            cw = cw.data.clone()
            is_conv = (
                c.__class__.__name__ == "Conv1D"
                or base_c.__class__.__name__ == "Conv1D"
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

        if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
            o = pretrained_attn.c_proj
            base_o = getattr(o, "base_layer", o)
            ow = getattr(base_o, "weight", getattr(o, "weight", None))
            if ow is not None:
                ow = ow.data.clone()
                is_conv_o = (
                    o.__class__.__name__ == "Conv1D"
                    or base_o.__class__.__name__ == "Conv1D"
                    or (ow.ndim == 2 and ow.shape[0] < ow.shape[1])
                )
                if is_conv_o:
                    ow = ow.t().contiguous()
                bo = getattr(base_o, "bias", getattr(o, "bias", None))
                extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

    elif hasattr(pretrained_attn, "self") and hasattr(pretrained_attn.self, "query"):
        s = pretrained_attn.self
        for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
            w_m, b_m = _extract_mod_wb(getattr(s, sub, None))
            if w_m is not None:
                extracted[key] = (w_m, b_m)
        if hasattr(pretrained_attn, "output") and hasattr(pretrained_attn.output, "dense"):
            w_m, b_m = _extract_mod_wb(pretrained_attn.output.dense)
            if w_m is not None:
                extracted["o"] = (w_m, b_m)

    else:
        candidates = {
            "q": ("q_lin", "q_proj", "query"),
            "k": ("k_lin", "k_proj", "key"),
            "v": ("v_lin", "v_proj", "value"),
            "o": ("out_lin", "out_proj", "dense", "o_proj"),
        }
        for key, names in candidates.items():
            for n in names:
                if hasattr(pretrained_attn, n):
                    w_m, b_m = _extract_mod_wb(getattr(pretrained_attn, n))
                    if w_m is not None:
                        extracted[key] = (w_m, b_m)
                        break

    missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
    if missing:
        try:
            from neuralops.core.adapters import UniversalAdapter

            adapter = UniversalAdapter()
            for k in missing:
                w, b = adapter.get_weight_bias(pretrained_attn, k)
                if w is not None:
                    extracted[k] = (w.clone(), b.clone() if b is not None else None)
        except Exception as e:
            logger.debug("UniversalAdapter fallback failed: %s", e)

    missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
    if missing_after:
        raise RuntimeError(f"BAMUMoE.copy_weights: missing {missing_after} from {pretrained_attn.__class__.__name__}")

    def _orient(w, in_dim):
        if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
            return w.t().contiguous()
        return w

    target_map = {
        "q": self.q_lin,
        "k": self.k_lin,
        "v": self.v_lin,
        "o": self.out_lin,
    }

    with torch.no_grad():
        for proj_key, target in target_map.items():
            w, b = extracted[proj_key]
            if w is None:
                if proj_key == "o" and self.is_nested_encoder:
                    eye = torch.eye(
                        target.out_features,
                        target.in_features,
                        device=target.U.device,
                        dtype=target.U.dtype,
                    )
                    target.set_svd_factors(eye, None)
                continue

            if w.ndim != 2:
                w = w.reshape(w.size(0), -1)

            in_dim = self.all_head_size if proj_key == "o" else self.dim
            w = _orient(w, in_dim)

            if proj_key in ("k", "v"):
                src_out = w.shape[0]
                tgt_out = target.out_features
                hd = self.head_dim
                if src_out > tgt_out and src_out % tgt_out == 0 and tgt_out % hd == 0:
                    rep = src_out // tgt_out
                    w = w.view(self.num_kv_heads, rep, hd, -1).mean(dim=1).reshape(tgt_out, -1)
                    if b is not None and b.numel() == src_out:
                        b = b.view(self.num_kv_heads, rep, hd).mean(dim=1).reshape(tgt_out)
                elif src_out < tgt_out and tgt_out % src_out == 0 and src_out % hd == 0:
                    rep = tgt_out // src_out
                    src_kv = src_out // hd
                    w = w.view(src_kv, hd, -1).repeat_interleave(rep, dim=0).reshape(tgt_out, -1)
                    if b is not None and b.numel() == src_out:
                        b = b.view(src_kv, hd).repeat_interleave(rep, dim=0).reshape(tgt_out)

            out_d = min(target.out_features, w.size(0))
            in_d = min(target.in_features, w.size(1))
            if (w.size(0), w.size(1)) != (target.out_features, target.in_features):
                padded = torch.zeros(
                    target.out_features,
                    target.in_features,
                    device=w.device,
                    dtype=w.dtype,
                )
                padded[:out_d, :in_d] = w[:out_d, :in_d]
                w = padded

            target.set_svd_factors(w, b)
            target.freeze_base_weights()

    if bool(getattr(pretrained_attn, "is_decoder", False)) or bool(getattr(pretrained_attn, "is_causal", False)):
        self.is_decoder = True
        self.is_causal = True

    try:
        for m in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            for p in m.parameters():
                self._comp_storage_bytes += p.numel() * p.element_size()
    except Exception:
        # allowed-swallow: best-effort-telemetry
        pass


def inject_bam_umoe(
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
        logger.warning("inject_bam_umoe: max recursion depth exceeded.")
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
            res = inject_bam_umoe(
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

    layer = target

    from neuralops.layers.sparse.bam_umoe import BAMUMoEBlock

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self", "attn", "self_attn"):
            if hasattr(layer, attr):
                cand = getattr(layer, attr)
                if isinstance(cand, nn.Module) and cand is not layer:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is not None and (isinstance(src_attn, BAMUMoEBlock) or "BAMUMoE" in src_attn.__class__.__name__):
        return None

    if cfg is None and src_attn is not None:
        cfg = getattr(src_attn, "config", None)

    hidden_size = kwargs.get("hidden_size")
    if hidden_size is None and src_attn is not None:
        hidden_size = getattr(src_attn, "hidden_size", None)
        if hidden_size is None and hasattr(src_attn, "q_proj"):
            q = getattr(src_attn, "q_proj", None)
            if q is not None and hasattr(q, "in_features"):
                hidden_size = q.in_features
    if hidden_size is None:
        hidden_size = getattr(cfg, "hidden_size", 768) if cfg else 768

    num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads"))
    if num_heads is None and src_attn is not None:
        num_heads = getattr(src_attn, "num_heads", getattr(src_attn, "num_attention_heads", None))
    if num_heads is None:
        num_heads = getattr(cfg, "num_attention_heads", 12) if cfg else 12

    moe_enabled = bool(kwargs.pop("moe_enabled", True))
    svd_rank = kwargs.pop("svd_rank", None)

    try:
        kwargs.pop("num_heads", None)
        kwargs.pop("hidden_size", None)
        kwargs.pop("d_model", None)
        new_attn = BAMUMoEBlock(
            config=cfg,
            old_attn=src_attn if src_attn is not layer else None,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_id=layer_idx,
            moe_enabled=moe_enabled,
            svd_rank=svd_rank,
            **kwargs,
        )
        if src_attn is not None and src_attn is not layer:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        spliced = False
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
            spliced = True
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = True
            layer.self = new_attn
            spliced = True
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
            spliced = True
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
            spliced = True

        if not spliced:
            logger.error(
                "inject_bam_umoe: no writable attention attribute on %s",
                layer.__class__.__name__,
            )
            return None
        return new_attn

    except Exception as e:
        logger.exception("inject_bam_umoe failed at layer %d: %s", layer_idx, e)
        return None
