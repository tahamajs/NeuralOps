# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight copying and initialization logic for NEXUS Attention."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def copy_weights(self, src: Any) -> None:
    from neuralops.core.adapters import UniversalAdapter

    extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "q": (None, None),
        "k": (None, None),
        "v": (None, None),
        "o": (None, None),
    }

    for proj in ["q", "k", "v", "o"]:
        try:
            w, b = UniversalAdapter.get_weight_bias(src, proj)
            if w is not None:
                extracted[proj] = (w, b)
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

    if hasattr(src, "c_attn") and (
        extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
    ):
        weight = src.c_attn.weight.data
        bias = getattr(src.c_attn, "bias", None)
        bias_data = bias.data if bias is not None else None
        if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
            weight = weight.t().contiguous()
        split_size = weight.shape[0] // 3
        q_w = weight[:split_size, :]
        k_w = weight[split_size : 2 * split_size, :]
        v_w = weight[2 * split_size :, :]
        q_b = bias_data[:split_size] if bias_data is not None else None
        k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
        v_b = bias_data[2 * split_size :] if bias_data is not None else None
        extracted["q"] = (q_w, q_b)
        extracted["k"] = (k_w, k_b)
        extracted["v"] = (v_w, v_b)

    if hasattr(src, "c_proj") and extracted["o"][0] is None:
        w_o = src.c_proj.weight.data
        if w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size:
            w_o = w_o.t().contiguous()
        b_o = getattr(src.c_proj, "bias", None)
        extracted["o"] = (w_o, b_o.data if b_o is not None else None)

    if hasattr(src, "self") and (
        extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
    ):
        s = src.self
        if hasattr(s, "query"):
            extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
        if hasattr(s, "key"):
            extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
        if hasattr(s, "value"):
            extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

    if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
        dense = src.output.dense
        extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

    named_map = {
        "q": ["q_proj", "q_lin", "query"],
        "k": ["k_proj", "k_lin", "key"],
        "v": ["v_proj", "v_lin", "value"],
        "o": ["o_proj", "out_proj", "out_lin", "dense"],
    }
    for proj, names in named_map.items():
        if extracted[proj][0] is None:
            for name in names:
                if hasattr(src, name):
                    layer = getattr(src, name)
                    if hasattr(layer, "weight"):
                        extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                        break

    cls_name = src.__class__.__name__
    has_out_proj = any(
        hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj")
    )
    is_nested = False
    if (
        (hasattr(src, "self") and hasattr(src.self, "query"))
        or (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        )
    ) and not has_out_proj:
        is_nested = True
    self.is_nested_encoder = bool(is_nested or self.is_nested_encoder)
    if hasattr(self, "dss_attn"):
        self.dss_attn.is_nested_encoder = self.is_nested_encoder

    def _orient(w, in_dim):
        if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
            return w.t().contiguous()
        return w

    with torch.no_grad():
        for proj_key, target in [
            ("q", self.query),
            ("k", self.key),
            ("v", self.value),
            ("o", self.out_proj),
        ]:
            w, b = extracted[proj_key]
            if w is not None:
                in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                w_oriented = _orient(w, in_dim)
                target_w = target.weight
                out_d, in_d = target_w.shape
                src_out, src_in = w_oriented.shape

                if proj_key in ("k", "v"):
                    if src_out == out_d:
                        target_w.data.copy_(w_oriented)
                        if b is not None and target.bias is not None:
                            target.bias.data.copy_(b)
                    elif src_out > out_d and src_out % out_d == 0:
                        rep = src_out // out_d
                        target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
                        if b is not None and target.bias is not None:
                            target.bias.data.copy_(b.view(out_d, rep).mean(dim=1))
                    elif src_out < out_d and out_d % src_out == 0:
                        rep = out_d // src_out
                        target_w.data.copy_(w_oriented.repeat(rep, 1))
                        if b is not None and target.bias is not None:
                            target.bias.data.copy_(b.repeat(rep))
                    else:
                        target_w.data.zero_()
                        target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                            w_oriented[: min(out_d, src_out), : min(in_d, src_in)]
                        )
                else:
                    target_w.data.zero_()
                    target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                        w_oriented[: min(out_d, src_out), : min(in_d, src_in)]
                    )
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        target.bias.data.zero_()
                        target.bias.data[: min(len(target.bias), len(b_val))].copy_(
                            b_val[: min(len(target.bias), len(b_val))]
                        )
            elif proj_key == "o" and self.is_nested_encoder:
                min_d = min(target.weight.shape[0], target.weight.shape[1])
                target.weight.data.zero_()
                target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                if target.bias is not None:
                    target.bias.data.zero_()

            if hasattr(target, "lora_B") and target.lora_B is not None:
                target.lora_B.data.zero_()

    if self.freeze_base:
        self.freeze_base_weights()
    else:
        self.unfreeze_base_weights()

    if hasattr(src, "is_decoder"):
        self.is_decoder = bool(src.is_decoder)
        if hasattr(self, "dss_attn"):
            self.dss_attn.is_decoder = self.is_decoder

