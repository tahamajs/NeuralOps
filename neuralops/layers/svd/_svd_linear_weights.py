#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Weight adaptation and HNC spectral compression mixin for SVDLinearAttention.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.common import _cfg_get, _unpack_weight_bias

logger = logging.getLogger(__name__)


class SVDLinearWeightsMixin:
    """Mixin providing copy_weights and HNC spectral alignment for SVDLinearAttention."""

    def hnc_spectral_compress_and_align(
        self,
        probe_inputs: torch.Tensor | None = None,
        num_steps: int = 15,
        target_rank: int | None = None,
        anchor_outputs: torch.Tensor | None = None,
        full_rank_weights: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """
        Run Hessian Null Space Continuation (HNC) on all linear projections (Q, K, V, Out)
        to steer tail singular values toward zero while preserving network function,
        then factorize into low-rank SVD projections with low-rank factor restoration.
        """
        self.sync_inherited_weights()
        results: dict[str, Any] = {}
        target_rank = target_rank or self.rank
        projs = [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj)]
        if self.out_proj is not None and isinstance(self.out_proj, SVDLinear):
            projs.append(("out", self.out_proj))

        for name, proj in projs:
            with torch.no_grad():
                if full_rank_weights and name in full_rank_weights:
                    W = full_rank_weights[name].to(proj.U.device)
                else:
                    W = torch.matmul(proj.U.data.float(), proj.V.data.float()).to(proj.U.device)
                bias = proj.bias.data if proj.bias is not None else None
                self._factorize_and_set(
                    proj,
                    W.to(proj.U.dtype),
                    bias=bias,
                    probe_inputs=probe_inputs,
                    use_hnc=True,
                    hnc_steps=num_steps,
                )
                results[name] = {"rank": proj.rank, "status": "hnc_aligned"}

        if probe_inputs is not None and anchor_outputs is not None and target_rank < self.hidden_size:
            self.eval()
            anchor_targets = anchor_outputs.detach().float()
            trainable_factors = []
            for _, proj in projs:
                trainable_factors.extend([proj.U, proj.V])
            opt = torch.optim.Adam(trainable_factors, lr=2e-3)
            for _ in range(12):
                opt.zero_grad()
                out = self(probe_inputs)
                out_t = out[0] if isinstance(out, tuple) else out
                loss = 0.5 * torch.mean((out_t.float() - anchor_targets) ** 2)
                loss.backward()
                opt.step()

        return results

    def copy_weights(self, old_attn: nn.Module, **kwargs: Any) -> None:
        """
        SVDLA-FIX-01: never raise. On missing projections, log a warning
        and fall back to the randomly-initialized state.
        """
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            src_cls_name = type(old_attn).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
                and not hasattr(old_attn, "o_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                src = old_attn.self
                self.is_nested_encoder = True
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            if hasattr(src, "in_proj_weight") and src.in_proj_weight is not None:
                in_proj_w = getattr(src, "in_proj_weight", None)
                in_proj_b = getattr(src, "in_proj_bias", None)
                if in_proj_w is not None:
                    w_q, w_k, w_v = in_proj_w.chunk(3, dim=0)
                if in_proj_b is not None:
                    b_q, b_k, b_v = in_proj_b.chunk(3, dim=0)
                if hasattr(src, "out_proj"):
                    w_o, b_o = _unpack_weight_bias(src.out_proj)
                self.is_decoder = False

            elif hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
                if c_attn_w is not None:
                    if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
                        c_attn_w = c_attn_w.t().contiguous()
                    q_dim = self.q_head_size
                    k_dim = self.kv_head_size
                    v_dim = self.kv_head_size
                    w_q = c_attn_w[:q_dim, :]
                    w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                    w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]
                    if c_attn_b is not None:
                        b_q = c_attn_b[:q_dim]
                        b_k = c_attn_b[q_dim : q_dim + k_dim]
                        b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]
                out_w, out_b = _unpack_weight_bias(src.c_proj)
                if out_w is not None and out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size:
                    out_w = out_w.t().contiguous()
                w_o, b_o = out_w, out_b

            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q, b_q = _unpack_weight_bias(src.query)
                w_k, b_k = _unpack_weight_bias(src.key)
                w_v, b_v = _unpack_weight_bias(src.value)
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                            break

            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q, b_q = _unpack_weight_bias(src.q_proj)
                w_k, b_k = _unpack_weight_bias(src.k_proj)
                w_v, b_v = _unpack_weight_bias(src.v_proj)
                if hasattr(src, "out_proj"):
                    w_o, b_o = _unpack_weight_bias(src.out_proj)
                elif hasattr(src, "o_proj"):
                    w_o, b_o = _unpack_weight_bias(src.o_proj)

            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q, b_q = _unpack_weight_bias(src.q_lin)
                w_k, b_k = _unpack_weight_bias(src.k_lin)
                w_v, b_v = _unpack_weight_bias(src.v_lin)
                if hasattr(src, "out_lin"):
                    w_o, b_o = _unpack_weight_bias(src.out_lin)

            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(old_attn, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(old_attn, "k")
                    if w_v is None:
                        w_v, b_v = adapter.get_weight_bias(old_attn, "v")
                    if w_o is None and not self.is_nested_encoder:
                        w_o, b_o = adapter.get_weight_bias(old_attn, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            missing = []
            if w_q is None:
                missing.append("q")
            if w_k is None:
                missing.append("k")
            if w_v is None:
                missing.append("v")
            if missing:
                if kwargs.get("_allow_missing_projections", False):
                    logger.warning(
                        "SVDLinearAttention: could not extract %s from %s; "
                        "keeping random init. This will likely reduce quality.",
                        missing,
                        type(old_attn).__name__,
                    )
                    return
                raise RuntimeError(
                    f"SVDLinearAttention: failed to extract required projections {missing} from {type(old_attn).__name__}"
                )

            for name, w in (("w_q", w_q), ("w_k", w_k), ("w_v", w_v)):
                if w is None:
                    continue
                try:
                    n = torch.norm(w.float(), p="fro").item()
                except Exception:
                    n = float("nan")
                if not math.isfinite(n) or n < 1e-4:
                    if kwargs.get("_allow_near_zero_norm", False):
                        logger.warning(
                            "SVDLinearAttention: %s has near-zero/NaN norm (%.2e); proceeding anyway.",
                            name,
                            n,
                        )
                    else:
                        raise RuntimeError(f"SVDLinearAttention: {name} has near-zero Frobenius norm ({n:.2e})")

            use_hnc = bool(kwargs.get("use_hnc", False))
            probe_inputs = kwargs.get("probe_inputs", None)
            hnc_steps = int(kwargs.get("hnc_steps", 15))

            try:
                self._factorize_and_set(
                    self.q_proj, w_q, b_q, probe_inputs=probe_inputs, use_hnc=use_hnc, hnc_steps=hnc_steps
                )
            except Exception as e:
                logger.warning("SVDLinearAttention: Q factorization failed (%s).", e)
                return

            def _adapt_to_kv_size(weight, bias, mode="mean"):
                w, b = weight, bias
                if w.shape[0] == self.kv_head_size:
                    return w, b
                if w.shape[0] > self.kv_head_size and w.shape[0] % self.kv_head_size == 0:
                    rep = w.shape[0] // self.kv_head_size
                    if mode == "slice":
                        w = w.contiguous().view(self.num_kv_heads, rep, -1)[:, 0, :].contiguous()
                        if b is not None:
                            b = b.contiguous().view(self.num_kv_heads, rep, -1)[:, 0, :].contiguous().view(-1)
                    elif mode == "first":
                        w = w[: self.kv_head_size, :].contiguous()
                        if b is not None:
                            b = b[: self.kv_head_size].contiguous()
                    else:
                        w = w.contiguous().view(self.num_kv_heads, rep, -1).mean(dim=1)
                        if b is not None:
                            b = b.contiguous().view(self.num_kv_heads, rep, -1).mean(dim=1).view(-1)
                elif w.shape[0] < self.kv_head_size and self.kv_head_size % w.shape[0] == 0:
                    rep = self.kv_head_size // w.shape[0]
                    src_heads = max(1, w.shape[0] // self.head_dim)
                    if src_heads * self.head_dim == w.shape[0]:
                        w = (
                            w.contiguous()
                            .view(src_heads, self.head_dim, -1)
                            .repeat_interleave(rep, dim=0)
                            .reshape(self.kv_head_size, -1)
                        )
                        if b is not None:
                            b = (
                                b.contiguous()
                                .view(src_heads, self.head_dim)
                                .repeat_interleave(rep, dim=0)
                                .reshape(self.kv_head_size)
                            )
                    else:
                        w = w.repeat(rep, 1)
                        if b is not None:
                            b = b.repeat(rep)
                else:
                    padded_w = torch.zeros(
                        self.kv_head_size,
                        w.shape[1],
                        device=w.device,
                        dtype=w.dtype,
                    )
                    min_rows = min(self.kv_head_size, w.shape[0])
                    padded_w[:min_rows, :] = w[:min_rows, :]
                    w = padded_w
                    if b is not None:
                        padded_b = torch.zeros(
                            self.kv_head_size,
                            device=b.device,
                            dtype=b.dtype,
                        )
                        padded_b[:min_rows] = b[:min_rows]
                        b = padded_b
                return w, b

            try:
                w_k_a, b_k_a = _adapt_to_kv_size(w_k, b_k, mode="slice")
                self._factorize_and_set(
                    self.k_proj, w_k_a, b_k_a, probe_inputs=probe_inputs, use_hnc=use_hnc, hnc_steps=hnc_steps
                )
            except Exception as e:
                logger.warning("SVDLinearAttention: K factorization failed (%s).", e)
                return

            try:
                w_v_a, b_v_a = _adapt_to_kv_size(w_v, b_v, mode="slice")
                self._factorize_and_set(
                    self.v_proj, w_v_a, b_v_a, probe_inputs=probe_inputs, use_hnc=use_hnc, hnc_steps=hnc_steps
                )
            except Exception as e:
                logger.warning("SVDLinearAttention: V factorization failed (%s).", e)
                return

            if w_o is not None and not self.is_nested_encoder:
                try:
                    n_o = torch.norm(w_o.float(), p="fro").item()
                except Exception:
                    n_o = float("nan")
                if not math.isfinite(n_o) or n_o < 1e-4:
                    logger.warning(
                        "SVDLinearAttention: out_proj has near-zero/NaN norm (%.2e); " "proceeding anyway.",
                        n_o,
                    )
                try:
                    self._factorize_and_set(
                        self.out_proj, w_o, b_o, probe_inputs=probe_inputs, use_hnc=use_hnc, hnc_steps=hnc_steps
                    )
                except Exception as e:
                    logger.warning("SVDLinearAttention: out_proj factorization failed (%s).", e)

            if use_hnc and probe_inputs is not None and self.rank < self.hidden_size:
                try:
                    with torch.enable_grad():
                        with torch.no_grad():
                            if hasattr(old_attn, "in_proj_weight"):
                                out_anchor = old_attn(probe_inputs, probe_inputs, probe_inputs)
                            else:
                                out_anchor = old_attn(probe_inputs)
                            if isinstance(out_anchor, tuple):
                                out_anchor = out_anchor[0]
                        self.eval()
                        anchor_targets = out_anchor.detach().float()
                        trainable_factors = [
                            self.q_proj.U,
                            self.q_proj.V,
                            self.k_proj.U,
                            self.k_proj.V,
                            self.v_proj.U,
                            self.v_proj.V,
                        ]
                        if self.out_proj is not None and isinstance(self.out_proj, SVDLinear):
                            trainable_factors.extend([self.out_proj.U, self.out_proj.V])
                        opt = torch.optim.Adam(trainable_factors, lr=1e-3)
                        for _ in range(25):
                            opt.zero_grad()
                            pred = self(probe_inputs)
                            out_t = pred[0] if isinstance(pred, tuple) else pred
                            loss = 0.5 * torch.mean((out_t.float() - anchor_targets) ** 2)
                            loss.backward()
                            opt.step()
                except Exception as e:
                    logger.debug("Automatic HNC post-alignment failed: %s", e)
