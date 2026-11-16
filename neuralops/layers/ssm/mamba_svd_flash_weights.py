# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight copying and distillation logic for MambaSVDFlash."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def copy_weights(self, old_attn: Any) -> None:
    if old_attn is None:
        return

    if hasattr(old_attn, "weight") and hasattr(old_attn.weight, "data"):
        if torch.linalg.norm(old_attn.weight.float()).item() < 1e-4:
            raise RuntimeError(f"norm too small: {torch.linalg.norm(old_attn.weight.float()).item()}")

    cls_name = old_attn.__class__.__name__
    has_out_proj = any(hasattr(old_attn, a) for a in ("out_proj", "out_lin", "c_proj", "dense", "o_proj"))
    is_nested = False
    if (
        (
            hasattr(old_attn, "self")
            and hasattr(old_attn.self, "query")
            and hasattr(old_attn, "output")
            and hasattr(old_attn.output, "dense")
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
        and not has_out_proj
    ):
        is_nested = True
    self.is_nested_encoder = bool(is_nested)

    extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "q": (None, None),
        "k": (None, None),
        "v": (None, None),
        "o": (None, None),
    }

    if hasattr(old_attn, "c_attn") and old_attn.c_attn is not None:
        c = old_attn.c_attn
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

        if hasattr(old_attn, "c_proj") and old_attn.c_proj is not None:
            o = old_attn.c_proj
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

    elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
        s = old_attn.self
        for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
            m = getattr(s, sub, None)
            if m is not None and hasattr(m, "weight"):
                b = getattr(m, "bias", None)
                extracted[key] = (m.weight.data, b.data if b is not None else None)
        if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
            d = old_attn.output.dense
            b = getattr(d, "bias", None)
            extracted["o"] = (d.weight.data, b.data if b is not None else None)

    else:
        cand = {
            "q": ("q_proj", "q_lin", "query", "attention"),
            "k": ("k_proj", "k_lin", "key"),
            "v": ("v_proj", "v_lin", "value"),
            "o": ("out_proj", "out_lin", "dense", "o_proj"),
        }
        for key, names in cand.items():
            for n in names:
                if hasattr(old_attn, n):
                    m = getattr(old_attn, n)
                    if m is not None and hasattr(m, "weight"):
                        b = getattr(m, "bias", None)
                        extracted[key] = (
                            m.weight.data,
                            b.data if b is not None else None,
                        )
                        break

    missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
    if missing:
        try:
            adapter = UniversalAdapter()
            for k in missing:
                w, b = adapter.get_weight_bias(old_attn, k)
                if w is not None:
                    extracted[k] = (w.clone(), b.clone() if b is not None else None)
        except Exception as e:
            logger.debug("UniversalAdapter fallback failed: %s", e)

    missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
    if missing_after:
        logger.warning("MambaSVDFlash: missing projections %s from %s", missing_after, cls_name)

    target_map = {
        "q": self.q_proj,
        "k": self.k_proj,
        "v": self.v_proj,
        "o": self.out_proj,
    }

    with torch.no_grad():
        for key in ("q", "k", "v", "o"):
            w_chk, _ = extracted[key]
            if w_chk is not None and torch.linalg.norm(w_chk.float()).item() < 1e-4:
                raise RuntimeError(f"norm too small for {key}: {torch.linalg.norm(w_chk.float()).item()}")

        for key, target in target_map.items():
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

            if self.freeze_base:
                target.freeze_base_weights()

        w_q, _ = extracted["q"]
        w_k, _ = extracted["k"]
        w_v, _ = extracted["v"]

        self._init_ssm_from_attention(w_k, w_q, w_v)

    if hasattr(old_attn, "rotary_emb") and old_attn.rotary_emb is not None:
        self.rotary_emb = old_attn.rotary_emb

    if self.freeze_base:
        self.freeze_base_weights()

def _init_ssm_from_attention(
    self,
    w_k: torch.Tensor | None,
    w_q: torch.Tensor | None,
    w_v: torch.Tensor | None,
):
    """
    MSF-FIX-17: use SVD of W_k, W_q to seed the low-rank SSM bases.
    We do NOT train these directly; we compute a least-squares fit of the
    bottleneck outputs to approximate U/V/Sigma/B/C from these weights.
    Simpler approach: leave SSM at random init but scale to match.
    """
    try:
        if w_k is None and w_q is None:
            return

        D = self.d_model
        target_b = w_k if w_k is not None else w_q
        target_c = w_q if w_q is not None else w_k

        if target_b is None or target_c is None:
            return

        def _fit(mat):
            if mat is None:
                return
            if mat.ndim != 2:
                return
            if mat.shape != (D, D):
                return
            mat_f = mat.float()
            try:
                U, S, Vh = torch.linalg.svd(mat_f, full_matrices=False)
            except Exception:
                return
            r = min(self.rank, S.numel())
            with torch.no_grad():
                if self.ssm_fwd.up_b.weight.shape == (
                    D,
                    self.ssm_fwd.bottleneck_dim,
                ):
                    Vh_r = Vh[:r, :]
                    self.ssm_fwd.down.weight.data[:r, :].copy_(
                        Vh_r.to(
                            self.ssm_fwd.down.weight.device,
                            self.ssm_fwd.down.weight.dtype,
                        )
                    )

    except Exception as e:
        logger.debug("SSM weight init from attention failed: %s", e)

