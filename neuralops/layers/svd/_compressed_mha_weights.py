#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Helper routines for weight copying and factor alignment in Compressed Multi-Head Attention.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def copy_compressed_mha_weights(self: Any, src: Any) -> None:
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
        except Exception as e:
            logger.debug(f"UniversalAdapter failed for {proj}: {e}")

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
        p_cls = getattr(src.c_proj, "__class__", type(src.c_proj)).__name__
        if (
            ("Conv1D" in p_cls)
            or hasattr(src.c_proj, "nf")
            or (
                w_o.dim() == 2
                and (
                    w_o.shape[0] < w_o.shape[1]
                    or (w_o.shape[0] == self.hidden_size and w_o.shape[1] == self.hidden_size)
                )
            )
        ):
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
                        extracted[proj] = (
                            layer.weight.data,
                            getattr(layer, "bias", None),
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
        )
    )
    if hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj"):
        self.is_nested_encoder = False
    elif (
        (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            and hasattr(src, "output")
            and hasattr(src.output, "dense")
        )
        or (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        )
        and not has_out_proj
    ):
        self.is_nested_encoder = True
    else:
        self.is_nested_encoder = False

    missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
    if not self.is_nested_encoder and extracted["o"][0] is None:
        missing.append("o")
    if missing:
        logger.warning(
            f"CompressedDistilBertAttention: missing weights for projections {missing}; proceeding with random init"
        )

    for proj_key in ["q", "k", "v"]:
        w = extracted[proj_key][0]
        if w is not None:
            norm = torch.linalg.norm(w.float())
            if norm < 1e-4:
                logger.warning(
                    f"CompressedDistilBertAttention: extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f}); continuing"
                )

    if extracted["o"][0] is not None:
        norm_o = torch.linalg.norm(extracted["o"][0].float())
        if norm_o < 1e-4:
            logger.warning(
                f"CompressedDistilBertAttention: extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f}); continuing"
            )

    def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
        if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
            return w.t().contiguous()
        return w

    with torch.no_grad():
        w_q = _orient(extracted["q"][0].float(), self.hidden_size)
        w_k = _orient(extracted["k"][0].float(), self.hidden_size)
        w_v = _orient(extracted["v"][0].float(), self.hidden_size)

        def _pool_kv_tensor(t: torch.Tensor | None, target_dim: int, head_dim: int) -> torch.Tensor | None:
            if t is None:
                return None
            if t.dim() == 1:
                if t.shape[0] == target_dim:
                    return t
                src_heads = max(1, t.shape[0] // head_dim)
                dst_heads = max(1, target_dim // head_dim)
                if src_heads > dst_heads and src_heads % dst_heads == 0:
                    group = src_heads // dst_heads
                    scale = math.sqrt(group)
                    return (t.view(dst_heads, group, head_dim).mean(dim=1) * scale).reshape(target_dim)
                return t[:target_dim]
            elif t.dim() == 2:
                if t.shape[0] == target_dim:
                    return t
                src_heads = max(1, t.shape[0] // head_dim)
                dst_heads = max(1, target_dim // head_dim)
                if src_heads > dst_heads and src_heads % dst_heads == 0:
                    group = src_heads // dst_heads
                    scale = math.sqrt(group)
                    return (t.view(dst_heads, group, head_dim, -1).mean(dim=1) * scale).reshape(target_dim, -1)
                return t[:target_dim]
            return t

        if w_q.shape[0] != self.all_head_size:
            w_q = _pool_kv_tensor(w_q, self.all_head_size, self.head_dim)
        if w_k.shape[0] != self.kv_head_size:
            w_k = _pool_kv_tensor(w_k, self.kv_head_size, self.head_dim)
        if w_v.shape[0] != self.kv_head_size:
            w_v = _pool_kv_tensor(w_v, self.kv_head_size, self.head_dim)

        self._staged_qkv = {"q": w_q.clone(), "k": w_k.clone(), "v": w_v.clone()}
        self._decompose_qkv_weights(w_q, w_k, w_v)

        if extracted["q"][1] is not None:
            b_q = extracted["q"][1]
            b_q_data = b_q.data if isinstance(b_q, nn.Parameter) else b_q
            b_q_data = _pool_kv_tensor(b_q_data.float(), self.all_head_size, self.head_dim)
            self.bias_q.data[: min(self.all_head_size, len(b_q_data))].copy_(
                b_q_data[: min(self.all_head_size, len(b_q_data))]
            )
        if extracted["k"][1] is not None:
            b_k = extracted["k"][1]
            b_k_data = b_k.data if isinstance(b_k, nn.Parameter) else b_k
            b_k_data = _pool_kv_tensor(b_k_data.float(), self.kv_head_size, self.head_dim)
            self.bias_k.data[: min(self.kv_head_size, len(b_k_data))].copy_(
                b_k_data[: min(self.kv_head_size, len(b_k_data))]
            )
        if extracted["v"][1] is not None:
            b_v = extracted["v"][1]
            b_v_data = b_v.data if isinstance(b_v, nn.Parameter) else b_v
            b_v_data = _pool_kv_tensor(b_v_data.float(), self.kv_head_size, self.head_dim)
            self.bias_v.data[: min(self.kv_head_size, len(b_v_data))].copy_(
                b_v_data[: min(self.kv_head_size, len(b_v_data))]
            )

        if extracted["o"][0] is not None:
            w_o = _orient(extracted["o"][0].float(), self.all_head_size)
            U_o, S_o, Vh_o = torch.linalg.svd(w_o, full_matrices=False)
            r_out = min(self.rank_out, U_o.size(1), Vh_o.size(0))

            self.U_out.data[: min(self.hidden_size, U_o.size(0)), :r_out].copy_(
                U_o[: min(self.hidden_size, U_o.size(0)), :r_out]
            )
            self.S_out.data[:r_out].copy_(S_o[:r_out])
            self.Vh_out.data[:r_out, : min(self.all_head_size, Vh_o.size(1))].copy_(
                Vh_o[:r_out, : min(self.all_head_size, Vh_o.size(1))]
            )

            if extracted["o"][1] is not None:
                b_o = extracted["o"][1]
                b_o_data = b_o.data if isinstance(b_o, nn.Parameter) else b_o
                self.bias_out.data[: min(self.hidden_size, len(b_o_data))].copy_(
                    b_o_data[: min(self.hidden_size, len(b_o_data))]
                )
        elif self.is_nested_encoder:
            self.U_out.data.zero_()
            min_d = min(self.hidden_size, self.rank_out)
            self.U_out.data[:min_d, :min_d].fill_diagonal_(1.0)
            self.S_out.data.fill_(1.0)
            self.Vh_out.data.zero_()
            self.Vh_out.data[:min_d, :min_d].fill_diagonal_(1.0)
            self.bias_out.data.zero_()

        self.U_qkv.requires_grad = False
        self.S_qkv.requires_grad = False
        self.Vh_qkv.requires_grad = False
        self.U_out.requires_grad = False
        self.S_out.requires_grad = False
        self.Vh_out.requires_grad = False
        self.bias_q.requires_grad = False
        self.bias_k.requires_grad = False
        self.bias_v.requires_grad = False
        self.bias_out.requires_grad = False

        if self.lora_rank > 0:
            self.reset_lora_parameters()

    if hasattr(src, "rotary_emb"):
        self.rotary_emb = src.rotary_emb
    if hasattr(src, "is_decoder"):
        self.is_decoder = bool(src.is_decoder)
