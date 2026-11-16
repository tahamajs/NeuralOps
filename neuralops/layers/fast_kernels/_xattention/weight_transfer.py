#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight transfer mixin for XAttention."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.adapters import UniversalAdapter

logger = logging.getLogger(__name__)


class XAttentionWeightTransferMixin:
    """Mixin providing copy_weights implementation for XAttention."""

    def copy_weights(self, src: Any) -> None:
        """Extract Q/K/V/O weights across multiple architectures."""
        if src is None:
            return

        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        try:
            adapter = UniversalAdapter()
            for proj in ["q", "k", "v", "o"]:
                try:
                    w, b = adapter.get_weight_bias(src, proj)
                    if w is not None:
                        extracted[proj] = (w, b)
                except Exception as e:
                    logger.debug("UniversalAdapter '%s' extraction failed: %s", proj, e)
        except Exception as e:
            logger.debug("UniversalAdapter init failed: %s", e)

        if hasattr(src, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            c_attn = getattr(src.c_attn, "base_layer", src.c_attn)
            weight = c_attn.weight.data
            bias = getattr(c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None

            is_conv1d = "Conv1D" in c_attn.__class__.__name__ or hasattr(c_attn, "nf")
            if is_conv1d or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size):
                weight = weight.t().contiguous()

            split_size = weight.shape[0] // 3
            extracted["q"] = (
                weight[:split_size, :],
                bias_data[:split_size] if bias_data is not None else None,
            )
            extracted["k"] = (
                weight[split_size : 2 * split_size, :],
                bias_data[split_size : 2 * split_size] if bias_data is not None else None,
            )
            extracted["v"] = (
                weight[2 * split_size :, :],
                bias_data[2 * split_size :] if bias_data is not None else None,
            )

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj = getattr(src.c_proj, "base_layer", src.c_proj)
            w_o = c_proj.weight.data
            is_conv1d = "Conv1D" in c_proj.__class__.__name__ or hasattr(c_proj, "nf")
            if is_conv1d or (w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size):
                w_o = w_o.t().contiguous()
            b_o = getattr(c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        if hasattr(src, "self") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            s = src.self
            for proj_key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                if hasattr(s, sub):
                    mod = getattr(s, sub)
                    base = getattr(mod, "base_layer", mod)
                    if hasattr(base, "weight"):
                        b = getattr(base, "bias", None)
                        extracted[proj_key] = (
                            base.weight.data,
                            b.data if b is not None else None,
                        )

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            dense = getattr(src.output.dense, "base_layer", src.output.dense)
            b = getattr(dense, "bias", None)
            extracted["o"] = (dense.weight.data, b.data if b is not None else None)

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
                        layer = getattr(layer, "base_layer", layer)
                        if hasattr(layer, "weight"):
                            b = getattr(layer, "bias", None)
                            extracted[proj] = (
                                layer.weight.data,
                                b.data if b is not None else None,
                            )
                            break

        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
                "out",
            )
        )
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        def _align_kv(
            w: torch.Tensor,
            b: torch.Tensor | None,
            target_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if w.shape[0] == target_size:
                return w, b
            if w.shape[0] > target_size and w.shape[0] % target_size == 0:
                rep = w.shape[0] // target_size
                w_pooled = w.view(target_size, rep, -1).mean(dim=1).contiguous()
                b_pooled = b.view(target_size, rep).mean(dim=1).contiguous() if b is not None else None
                return w_pooled, b_pooled
            return w[:target_size, :], (b[:target_size] if b is not None else None)

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
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
                if proj_key == "o" and self.is_nested_encoder:
                    self.out_proj = nn.Identity()
                    continue

                w, b = extracted[proj_key]
                if w is None:
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                w = _orient(w, in_dim)

                if proj_key in ("k", "v"):
                    w, b = _align_kv(w, b, target.weight.size(0))

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))

                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(
                        device=target.weight.device,
                        dtype=target.weight.dtype,
                    )
                )

                if b is not None and target.bias is not None:
                    b_data = b.data if isinstance(b, nn.Parameter) else b
                    mb = min(target.bias.numel(), b_data.numel())
                    target.bias.data[:mb].copy_(
                        b_data[:mb].to(
                            device=target.bias.device,
                            dtype=target.bias.dtype,
                        )
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                w_norm = float(torch.linalg.norm(target.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning(
                        "XAttention copy_weights: '%s' near-zero norm (%.2e)",
                        proj_key,
                        w_norm,
                    )

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

