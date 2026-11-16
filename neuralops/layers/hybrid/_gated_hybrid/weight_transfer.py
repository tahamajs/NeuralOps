#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight transfer mixin for GatedHybridKernelAttention."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.adapters import UniversalAdapter

logger = logging.getLogger(__name__)


class GatedHybridWeightTransferMixin:
    """Mixin providing weight copying logic for GatedHybridKernelAttention."""

    def copy_weights(self, old_attn):
        cls_name = old_attn.__class__.__name__
        has_out_proj_attr = any(
            hasattr(old_attn, a) for a in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj")
        )
        if hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj"):
            self.is_nested_encoder = False
        elif (
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
            )
            or (cls_name.endswith("SelfAttention") and not has_out_proj_attr)
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = bool(self.is_nested_encoder)

        extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}

        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter failed for %s: %s", proj, e)

        if hasattr(old_attn, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            c = old_attn.c_attn
            base_c = getattr(c, "base_layer", c)
            weight = getattr(base_c, "weight", getattr(c, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                is_conv1d = (
                    c.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or hasattr(c, "nf")
                    or hasattr(base_c, "nf")
                )
                if is_conv1d or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size):
                    weight = weight.t().contiguous()
                bias = getattr(base_c, "bias", getattr(c, "bias", None))
                bd = bias.data if bias is not None else None
                split = weight.shape[0] // 3
                extracted["q"] = (weight[:split, :], bd[:split] if bd is not None else None)
                extracted["k"] = (weight[split : 2 * split, :], bd[split : 2 * split] if bd is not None else None)
                extracted["v"] = (weight[2 * split :, :], bd[2 * split :] if bd is not None else None)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            p = old_attn.c_proj
            base_p = getattr(p, "base_layer", p)
            w_o = getattr(base_p, "weight", getattr(p, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    p.__class__.__name__ == "Conv1D"
                    or base_p.__class__.__name__ == "Conv1D"
                    or hasattr(p, "nf")
                    or hasattr(base_p, "nf")
                )
                if is_conv1d or (
                    w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size
                ):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_p, "bias", getattr(p, "bias", None))
                extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        if hasattr(old_attn, "self") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            s = old_attn.self
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if self.is_nested_encoder:
            extracted["o"] = (None, None)
        elif hasattr(old_attn, "output") and hasattr(old_attn.output, "dense") and extracted["o"][0] is None:
            d = old_attn.output.dense
            extracted["o"] = (d.weight.data, getattr(d, "bias", None))

        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(old_attn, name):
                        layer = getattr(old_attn, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"GatedHybridKernelAttention.copy_weights: missing {missing} " f"from {cls_name}.")

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    target.decompose_from_weight(_orient(w, in_dim), b)
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(
                        self.all_head_size,
                        device=self.out_proj.U.device,
                        dtype=self.out_proj.U.dtype,
                    )
                    target.decompose_from_weight(eye_w, None)

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
            self.is_decoder = True


