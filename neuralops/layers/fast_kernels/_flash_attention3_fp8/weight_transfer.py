#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight transfer mixin for FlashAttention3FP8."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FlashAttention3FP8WeightTransferMixin:
    """Mixin providing copy_weights logic for FlashAttention3FP8."""

    def copy_weights(self, src: Any) -> None:
        r"""Copy projection weights from pretrained attention module with Frobenius norm verification."""
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if has_out_lin:
            is_nested = False
        elif (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            and hasattr(src, "output")
            and hasattr(src.output, "dense")
        ) or (
            cls_name
            in [
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
                "DebertaSelfAttention",
            ]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ):
            is_nested = True
        else:
            is_nested = False
        self.is_nested_encoder = bool(is_nested)

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    logger.warning(f"copy_weights c_attn merge warning: {e}")
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                cls_c = getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__
                base_cls_c = getattr(base_c, "__class__", type(base_c)).__name__
                if (
                    (cls_c == "Conv1D")
                    or (base_cls_c == "Conv1D")
                    or hasattr(c_attn_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (
                        c_w.ndim == 2
                        and c_w.shape[0] < c_w.shape[1]
                        and "Linear" not in cls_c
                        and "Linear" not in base_cls_c
                    )
                ):
                    c_w = c_w.t().contiguous()
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
                    cls_o = getattr(c_proj_mod, "__class__", type(c_proj_mod)).__name__
                    base_cls_o = getattr(base_o, "__class__", type(base_o)).__name__
                    if (
                        (cls_o == "Conv1D")
                        or (base_cls_o == "Conv1D")
                        or hasattr(c_proj_mod, "nf")
                        or hasattr(base_o, "nf")
                        or (
                            o_w.ndim == 2
                            and o_w.shape[0] < o_w.shape[1]
                            and "Linear" not in cls_o
                            and "Linear" not in base_cls_o
                        )
                    ):
                        o_w = o_w.t().contiguous()
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
                "q": ["q_lin", "q_proj", "query", "W_q"],
                "k": ["k_lin", "k_proj", "key", "W_k"],
                "v": ["v_lin", "v_proj", "value", "W_v"],
                "o": ["out_lin", "out_proj", "dense", "o_proj", "proj", "output"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
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
                logger.debug(f"UniversalAdapter fallback warning: {e}")

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"FlashAttention3FP8 copy_weights: missing mandatory projections {missing_after} "
                f"from source module {src.__class__.__name__}."
            )

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
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype)
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.linalg.norm(target_mod.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning(
                        f"FlashAttention3FP8 weight copy verification: norm too small ({w_norm:.2e}) for '{proj_key}'"
                    )

                if self.freeze_base:
                    target_mod.freeze_base_weights()

