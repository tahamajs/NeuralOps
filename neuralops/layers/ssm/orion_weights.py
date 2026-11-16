# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight copying logic for ORION Attention."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def copy_weights(self, src: Any) -> None:
    if src is None:
        return

    cls_name = src.__class__.__name__
    has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
    is_nested = False
    if (
        hasattr(src, "self")
        and hasattr(src.self, "query")
        or (
            cls_name
            in [
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
                "DebertaSelfAttention",
            ]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        )
        and not has_out_lin
    ):
        is_nested = True
    self.is_nested_encoder = bool(is_nested)

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
                or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
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
                if (
                    c_proj_mod.__class__.__name__ == "Conv1D"
                    or base_o.__class__.__name__ == "Conv1D"
                    or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
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
            logger.debug("UniversalAdapter fallback failed: %s", e)

    missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
    if missing_after:
        logger.warning(
            "ORIONAttention.copy_weights: missing projections %s from %s",
            missing_after,
            src.__class__.__name__,
        )
        return

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
                    target_mod.down.weight.data.zero_()
                    target_mod.up.weight.data.zero_()
                    r = min(target_mod.rank, self.hidden_size, self.all_head_size)
                    target_mod.down.weight.data[:r, :r].fill_diagonal_(1.0)
                    target_mod.up.weight.data[:r, :r].fill_diagonal_(1.0)
                    if target_mod.up.bias is not None:
                        target_mod.up.bias.data.zero_()
                continue

            if w.ndim != 2:
                w = w.reshape(w.size(0), -1)

            expected = (target_mod.out_features, target_mod.in_features)
            if w.shape != expected and w.t().contiguous().shape == expected:
                w = w.t().contiguous()

            target_mod.decompose_from_weight(
                w[: target_mod.out_features, : target_mod.in_features],
                b[: target_mod.out_features] if b is not None else None,
            )

            norm = float(torch.linalg.norm(target_mod.weight.float()).item())
            if norm < 1e-4:
                logger.warning(
                    "ORIONAttention: %s has near-zero Frobenius norm (%.2e)",
                    proj_key,
                    norm,
                )
