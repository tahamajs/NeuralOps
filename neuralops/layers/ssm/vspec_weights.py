# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight copying and initialization logic for VSpec Attention."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def copy_weights(self, src: Any) -> None:
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
        c_mod = src.c_attn
        base_c = getattr(c_mod, "base_layer", c_mod)
        weight = getattr(base_c, "weight", getattr(c_mod, "weight", None))
        if weight is not None:
            weight = weight.data.clone()
            bias = getattr(base_c, "bias", getattr(c_mod, "bias", None))
            bias_data = bias.data.clone() if bias is not None else None
            is_conv1d = (
                "Conv1D" in c_mod.__class__.__name__
                or "Conv1D" in base_c.__class__.__name__
                or hasattr(c_mod, "nf")
                or hasattr(base_c, "nf")
                or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
            )
            if is_conv1d:
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
        p_mod = src.c_proj
        base_o = getattr(p_mod, "base_layer", p_mod)
        w_o = getattr(base_o, "weight", getattr(p_mod, "weight", None))
        if w_o is not None:
            w_o = w_o.data.clone()
            is_conv1d = (
                "Conv1D" in p_mod.__class__.__name__
                or "Conv1D" in base_o.__class__.__name__
                or hasattr(p_mod, "nf")
                or hasattr(base_o, "nf")
                or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
            )
            if is_conv1d:
                w_o = w_o.t().contiguous()
            b_o = getattr(base_o, "bias", getattr(p_mod, "bias", None))
            extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

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
        "q": ["q_proj", "q_lin", "query", "W_q"],
        "k": ["k_proj", "k_lin", "key", "W_k"],
        "v": ["v_proj", "v_lin", "value", "W_v"],
        "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
    }
    for proj, names in named_map.items():
        if extracted[proj][0] is None:
            for name in names:
                if hasattr(src, name):
                    layer = getattr(src, name)
                    base = getattr(layer, "base_layer", layer)
                    if hasattr(base, "weight"):
                        extracted[proj] = (
                            base.weight.data,
                            getattr(base, "bias", None),
                        )
                        break

    cls_name = getattr(src, "__class__", type(src)).__name__
    has_out = (
        hasattr(src, "dense")
        or hasattr(src, "out_proj")
        or hasattr(src, "out_lin")
        or hasattr(src, "c_proj")
        or hasattr(src, "o_proj")
    )
    is_nested = False
    if cls_name in [
        "BertSelfAttention",
        "RobertaSelfAttention",
        "XLMRobertaSelfAttention",
        "ElectraSelfAttention",
        "DebertaSelfAttention",
    ] or (hasattr(src, "query") and not has_out):
        is_nested = True
    if hasattr(src, "self") and hasattr(src, "output"):
        is_nested = True
    self.is_nested_encoder = bool(is_nested)

    missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
    if not self.is_nested_encoder and extracted["o"][0] is None:
        missing.append("o")
    if missing:
        logger.warning(
            "VSpecAttention missing required weights for projections: %s; keeping defaults",
            missing,
        )

    def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
        if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
            return w.t().contiguous()
        return w

    if extracted["q"][0] is not None:
        self.Pq.set_weight(_orient(extracted["q"][0], self.hidden_size), extracted["q"][1])
    if extracted["k"][0] is not None:
        self.Pk.set_weight(_orient(extracted["k"][0], self.hidden_size), extracted["k"][1])
    if extracted["v"][0] is not None:
        self.Pv.set_weight(_orient(extracted["v"][0], self.hidden_size), extracted["v"][1])

    if extracted["o"][0] is not None and not self.is_nested_encoder:
        self.Po.set_weight(_orient(extracted["o"][0], self.all_head_size), extracted["o"][1])
    elif self.is_nested_encoder:
        eye_w = torch.eye(self.all_head_size, self.all_head_size)
        self.Po.set_weight(eye_w, None)

    source_submodules = {
        "q": ["q_proj", "q_lin", "query", "W_q"],
        "k": ["k_proj", "k_lin", "key", "W_k"],
        "v": ["v_proj", "v_lin", "value", "W_v"],
        "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
    }
    for proj_key, target in [
        ("q", self.Pq),
        ("k", self.Pk),
        ("v", self.Pv),
        ("o", self.Po),
    ]:
        for name in source_submodules[proj_key]:
            if hasattr(src, name):
                sub = getattr(src, name)
                target.copy_lora_from(sub)
                break

    if hasattr(src, "is_decoder"):
        self.is_decoder = bool(src.is_decoder)

