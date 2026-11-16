# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Weight distillation and copying logic for MambaBlock."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def copy_weights(self, src: nn.Module) -> None:
    """
    Distill attention weights (W_q, W_k, W_v, W_o) into Mamba's S6
    parameters via low-rank mapping:
        W_v  → in_proj (SSM input path)
        W_k  → x_proj (B branch, via SVD)
        W_q  → x_proj (C branch, via SVD)
        W_o  → out_proj
        dt   → kept at initialization (learnable, input-dependent via x_proj)
    """
    if src is None:
        return

    extracted: dict[str, torch.Tensor | None] = {
        "q": None,
        "k": None,
        "v": None,
        "o": None,
    }

    try:
        from neuralops.core.adapters import UniversalAdapter

        for proj in ("q", "k", "v", "o"):
            try:
                w, _ = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = w
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass
    except Exception:
        # allowed-swallow: best-effort-telemetry
        pass

    if hasattr(src, "c_attn") and src.c_attn is not None and extracted["q"] is None:
        c = src.c_attn
        base = getattr(c, "base_layer", c)
        w = getattr(base, "weight", getattr(c, "weight", None))
        if w is not None:
            w = w.data.clone()
            is_conv = (
                c.__class__.__name__ == "Conv1D"
                or base.__class__.__name__ == "Conv1D"
                or (w.ndim == 2 and w.shape[1] == 3 * self.d_model)
            )
            if is_conv:
                w = w.t().contiguous()
            split = w.size(0) // 3
            extracted["q"] = w[:split, :]
            extracted["k"] = w[split : 2 * split, :]
            extracted["v"] = w[2 * split :, :]

    if hasattr(src, "c_proj") and src.c_proj is not None and extracted["o"] is None:
        o = src.c_proj
        base = getattr(o, "base_layer", o)
        w = getattr(base, "weight", getattr(o, "weight", None))
        if w is not None:
            w = w.data.clone()
            if w.ndim == 2 and w.shape[0] < w.shape[1]:
                w = w.t().contiguous()
            extracted["o"] = w

    named_map = {
        "q": ("q_proj", "q_lin", "query"),
        "k": ("k_proj", "k_lin", "key"),
        "v": ("v_proj", "v_lin", "value"),
        "o": ("o_proj", "out_proj", "out_lin", "dense"),
    }
    nested = getattr(src, "self", None)
    for proj, names in named_map.items():
        if extracted[proj] is not None:
            continue
        if nested is not None:
            for n in names:
                mod = getattr(nested, n, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj] = mod.weight.data
                    break
        if extracted[proj] is not None:
            continue
        for n in names:
            mod = getattr(src, n, None)
            if mod is not None and hasattr(mod, "weight"):
                extracted[proj] = mod.weight.data
                break
        if extracted["o"] is None and proj == "o":
            out = getattr(src, "output", None)
            dense = getattr(out, "dense", None) if out is not None else None
            if dense is not None and hasattr(dense, "weight"):
                extracted["o"] = dense.weight.data

    cls_name = src.__class__.__name__
    if cls_name in (
        "BertSelfAttention",
        "RobertaSelfAttention",
        "XLMRobertaSelfAttention",
    ) or (nested is not None and hasattr(nested, "query")):
        self.is_nested_encoder = True

    dtype = self.in_proj.weight.dtype
    device = self.in_proj.weight.device

    def _to_t(x: torch.Tensor | None, r: int, c: int) -> torch.Tensor | None:
        """Project W (any shape) to target shape (r, c) via SVD."""
        if x is None:
            return None
        x = x.float()
        if x.ndim != 2:
            return None
        M, N = x.shape
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        k = min(r, c, s.numel())

        u_r = u[: min(r, M), :k]
        vh_c = vh[:k, : min(c, N)]
        core = (u_r * s[:k].unsqueeze(0)) @ vh_c

        out = torch.zeros(r, c, dtype=torch.float32, device=x.device)
        out[: core.size(0), : core.size(1)] = core

        if r > core.size(0):
            rem_r = r - core.size(0)
            reps_r = (rem_r + core.size(0) - 1) // core.size(0)
            scale_r = 1.0 / math.sqrt(reps_r + 1)
            out[: core.size(0), :] *= scale_r
            tiled_r = (core * scale_r).repeat(reps_r, 1)[:rem_r, :]
            out[core.size(0) :, : core.size(1)] = tiled_r

        if c > core.size(1):
            rem_c = c - core.size(1)
            reps_c = (rem_c + core.size(1) - 1) // core.size(1)
            scale_c = 1.0 / math.sqrt(reps_c + 1)
            out[:, : core.size(1)] *= scale_c
            tiled_c = (out[:, : core.size(1)]).repeat(1, reps_c)[:, :rem_c]
            out[:, core.size(1) :] = tiled_c

        return out.to(device=device, dtype=dtype)

    with torch.no_grad():
        w_q = extracted["q"]
        w_k = extracted["k"]
        w_v = extracted["v"]
        w_o = extracted["o"]

        if w_v is not None:
            w_v_p = _to_t(w_v, self.d_inner, self.d_model)
            if w_v_p is not None:
                self.in_proj.weight.data[: self.d_inner, : self.d_model].copy_(w_v_p)
                self.in_proj.weight.data[self.d_inner :, : self.d_model].copy_(0.5 * w_v_p)

        with torch.no_grad():
            b_start = self.dt_rank
            b_end = self.dt_rank + self.d_state
            c_start = b_end
            c_end = c_start + self.d_state

            if w_k is not None:
                b_mat = _to_t(w_k, self.d_state, self.d_inner)
                if b_mat is not None:
                    self.x_proj.weight.data[b_start:b_end, :].copy_(b_mat)
            if w_q is not None:
                c_mat = _to_t(w_q, self.d_state, self.d_inner)
                if c_mat is not None:
                    self.x_proj.weight.data[c_start:c_end, :].copy_(c_mat)

        if w_o is not None:
            w_o_p = _to_t(w_o, self.d_model, self.d_inner)
            if w_o_p is not None:
                self.out_proj.weight.data.copy_(w_o_p)
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data.zero_()

        self.conv1d.weight.data.zero_()
        if self.conv1d.weight.dim() == 3:
            self.conv1d.weight.data[:, 0, -1] = 1.0
        if self.conv1d.bias is not None:
            self.conv1d.bias.data.zero_()

    for name, mod in (
        ("in_proj", self.in_proj),
        ("x_proj", self.x_proj),
        ("out_proj", self.out_proj),
    ):
        n = float(torch.linalg.norm(mod.weight.data.float()).item())
        if n < 1e-6:
            logger.warning("MambaBlock: '%s' near-zero norm (%.2e)", name, n)

    if self.freeze_base:
        self.freeze_base_weights()

