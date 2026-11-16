#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight transfer and SVD projection helpers for SpectronFP8Attention."""

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.common import FactorizedLinear

logger = logging.getLogger(__name__)


class SpectronWeightTransferMixin:
    """Mixin class providing SVD weight extraction and copying for SpectronFP8Attention."""

    @staticmethod
    def _svd_init(w: torch.Tensor, b: torch.Tensor | None, proj: FactorizedLinear) -> None:
        with torch.no_grad():
            W = w.detach().float()
            if W.ndim != 2:
                raise ValueError(f"Expected 2D weight, got shape {tuple(W.shape)}")
            dev = W.device
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception as exc:
                logger.warning("torch.linalg.svd failed on device %s (%s); falling back to CPU float32", dev, exc)
                U, S, Vh = torch.linalg.svd(W.detach().cpu().float(), full_matrices=False)
                U = U.to(dev)
                S = S.to(dev)
                Vh = Vh.to(dev)
            r = min(proj.rank, U.size(1), Vh.size(0))
            proj.lin1.weight.data.zero_()
            proj.lin2.weight.data.zero_()
            in_copy = min(W.shape[1], proj.lin1.weight.size(1))
            out_copy = min(W.shape[0], proj.lin2.weight.size(0))
            proj.lin1.weight.data[:r, :in_copy].copy_(Vh[:r, :in_copy].to(proj.lin1.weight.dtype))
            US = U[:out_copy, :r] @ torch.diag(S[:r])
            proj.lin2.weight.data[:out_copy, :r].copy_(US.to(proj.lin2.weight.dtype))
            if b is not None and proj.lin2.bias is not None:
                n = min(b.numel(), proj.lin2.bias.numel())
                proj.lin2.bias.data[:n].copy_(b.detach()[:n].to(proj.lin2.bias.dtype))
            elif proj.lin2.bias is not None:
                proj.lin2.bias.data.zero_()
            proj.lin1.weight.requires_grad = False
            proj.lin2.weight.requires_grad = False
            if proj.lin2.bias is not None:
                proj.lin2.bias.requires_grad = False
            if proj.lora_A is not None:
                nn.init.kaiming_uniform_(proj.lora_A, a=math.sqrt(5))
                nn.init.normal_(proj.lora_B, mean=0.0, std=1e-4)

    def _extract_proj_weights(self, old_attn: nn.Module) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        if hasattr(old_attn, "c_attn"):
            mod = old_attn.c_attn
            if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                try:
                    mod.merge()
                except Exception as exc:
                    logger.debug("Failed calling mod.merge() on c_attn: %s", exc)
            base = getattr(mod, "base_layer", mod)
            c_w = getattr(base, "weight", getattr(mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data
                is_conv1d = (
                    mod.__class__.__name__ == "Conv1D"
                    or base.__class__.__name__ == "Conv1D"
                    or hasattr(mod, "nf")
                    or hasattr(base, "nf")
                )
                if is_conv1d or (c_w.dim() == 2 and c_w.shape[1] == 3 * self.hidden_size):
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_holder = getattr(base, "bias", getattr(mod, "bias", None))
                c_b = bias_holder.data if bias_holder is not None else None
                for i, p in enumerate(("q", "k", "v")):
                    extracted[p] = (
                        c_w[i * split : (i + 1) * split, :],
                        c_b[i * split : (i + 1) * split] if c_b is not None else None,
                    )

        if hasattr(old_attn, "c_proj"):
            mod = old_attn.c_proj
            if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                try:
                    mod.merge()
                except Exception as exc:
                    logger.debug("Failed calling mod.merge() on c_proj: %s", exc)
            base = getattr(mod, "base_layer", mod)
            w = getattr(base, "weight", getattr(mod, "weight", None))
            if w is not None:
                w = w.data
                is_conv1d = (
                    mod.__class__.__name__ == "Conv1D"
                    or base.__class__.__name__ == "Conv1D"
                    or hasattr(mod, "nf")
                    or hasattr(base, "nf")
                )
                if is_conv1d or (w.dim() == 2 and w.shape[0] != self.hidden_size and w.shape[1] == self.hidden_size):
                    w = w.t().contiguous()
                b = getattr(base, "bias", getattr(mod, "bias", None))
                extracted["o"] = (w, b.data if b is not None else None)

        if hasattr(old_attn, "self"):
            s = old_attn.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense") and extracted["o"][0] is None:
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
                for n in names:
                    if hasattr(old_attn, n):
                        m = getattr(old_attn, n)
                        if hasattr(m, "weight"):
                            extracted[proj] = (m.weight.data, getattr(m, "bias", None))
                            break

        for p in ("q", "k", "v", "o"):
            if extracted[p][0] is None:
                try:
                    w, b = UniversalAdapter.get_weight_bias(old_attn, p)
                    if w is not None:
                        extracted[p] = (w, b)
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed for %s: %s", p, e)

        return extracted

    def copy_weights(self, old_attn: nn.Module) -> None:
        extracted = self._extract_proj_weights(old_attn)

        cls_name = old_attn.__class__.__name__
        has_out = any(hasattr(old_attn, a) for a in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o"))
        if hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj"):
            self.is_nested_encoder = False
        elif (
            (
                hasattr(old_attn, "self")
                and hasattr(old_attn.self, "query")
                and hasattr(old_attn, "output")
                and hasattr(old_attn.output, "dense")
            )
            or cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
            )
            or (cls_name.endswith("SelfAttention") and not has_out)
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        missing = [p for p in ("q", "k", "v") if extracted[p][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AegisSpectronFP8Attention.copy_weights: missing {missing} from " f"{cls_name}. Refusing random init."
            )

        for proj, target in (
            ("q", self.q_proj),
            ("k", self.k_proj),
            ("v", self.v_proj),
        ):
            w, b = extracted[proj]
            if w is not None:
                self._svd_init(w, b, target)

        if self.is_nested_encoder:
            with torch.no_grad():
                self.out_proj.lin1.weight.data.zero_()
                self.out_proj.lin2.weight.data.zero_()
                r = min(self.out_proj.rank, self.hidden_size, self.all_head_size)
                self.out_proj.lin1.weight.data[:r, :r].fill_diagonal_(1.0)
                self.out_proj.lin2.weight.data[:r, :r].fill_diagonal_(1.0)
                if self.out_proj.lin2.bias is not None:
                    self.out_proj.lin2.bias.data.zero_()
        elif extracted["o"][0] is not None:
            self._svd_init(extracted["o"][0], extracted["o"][1], self.out_proj)

        with torch.no_grad():
            if extracted["q"][0] is not None:
                U, _, _ = torch.linalg.svd(extracted["q"][0].float(), full_matrices=False)
                r = min(self.rank, U.size(1), self.svd_U_q.size(1))
                self.svd_U_q.data[:, :r].copy_(U[: self.svd_U_q.size(0), :r].to(self.svd_U_q.dtype))
            if extracted["k"][0] is not None:
                U, _, _ = torch.linalg.svd(extracted["k"][0].float(), full_matrices=False)
                r = min(self.rank, U.size(1), self.svd_U_k.size(1))
                rows_k = min(self.svd_U_k.size(0), U.size(0))
                self.svd_U_k.data[:rows_k, :r].copy_(U[:rows_k, :r].to(self.svd_U_k.dtype))

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
            self.is_decoder = True

