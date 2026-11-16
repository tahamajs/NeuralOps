#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused SVD-Linear Kernel Layer

Fused SRAM-resident dual-projection kernel layer executing low-rank factorized linear transformations.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.amp_utils import maybe_half
from neuralops.core.attention_mask import apply_attention_mask, get_token_valid_mask
from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv
from neuralops.layers.svd._fused_svd_linear import (
    HAS_TRITON,
    FusedSVDLinear,
    _FactorizedWeightProxy,
    _FusedSVDLinearAutogradFunction,
    fused_svd_linear,
)

try:
    from neuralops.layers.svd._fused_svd_linear import _fused_svd_linear_fwd_kernel
except ImportError:
    _fused_svd_linear_fwd_kernel = None

logger = logging.getLogger(__name__)

@ATTENTION_REGISTRY.register(
    "fused_svd_kernel",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register("fused_svd", is_alias=True)
class FusedSVDKernelAttention(BaseAttention):
    r"""Fused SVD-Kernel Linear Attention Module.
    Production-hardened across DistilBERT, RoBERTa/BERT nested, GPT-2 Conv1D, and Qwen/LLaMA.
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        old_attn: nn.Module | None = None,
        use_fused_kernel: bool = True,
        use_rms_norm: bool = False,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        h_dim = int(
            kwargs.get("hidden_size")
            or _cfg_get(config, "hidden_size", None)
            or _cfg_get(config, "dim", None)
            or _cfg_get(config, "d_model", None)
            or _cfg_get(config, "n_embd", None)
            or getattr(
                old_attn if old_attn is not None else kwargs.get("pretrained_attn"),
                "embed_dim",
                None,
            )
            or getattr(
                old_attn if old_attn is not None else kwargs.get("pretrained_attn"),
                "hidden_size",
                None,
            )
            or 768
        )
        self.hidden_size: int = h_dim
        actual_rank = rank or _cfg_get(config, "rank", _cfg_get(config, "svd_rank", self.hidden_size))
        super().__init__(config, rank=actual_rank)

        src_attn = old_attn if old_attn is not None else kwargs.get("pretrained_attn")
        self.num_heads: int = int(
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or _cfg_get(config, "num_attention_heads", None)
            or _cfg_get(config, "n_heads", None)
            or _cfg_get(config, "num_heads", None)
            or _cfg_get(config, "n_head", None)
            or getattr(src_attn, "num_attention_heads", None)
            or getattr(src_attn, "n_heads", None)
            or getattr(src_attn, "num_heads", None)
            or (self.hidden_size // 64 if self.hidden_size % 64 == 0 else 12)
        )
        self.head_dim: int = int(
            kwargs.get("head_dim")
            or _cfg_get(
                config,
                "head_dim",
                self.hidden_size // self.num_heads if self.num_heads > 0 else 64,
            )
        )
        cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.q_head_size: int = self.num_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        max_rank = min(self.hidden_size, self.all_head_size) // 2
        self.rank: int = min(actual_rank, max_rank)
        self.use_fused_kernel: bool = use_fused_kernel
        self.use_rms_norm: bool = use_rms_norm

        is_dec = (
            bool(_cfg_get(config, "is_decoder", False))
            or any(
                str(_cfg_get(config, "model_type", "")).lower().startswith(p)
                for p in (
                    "gpt",
                    "qwen",
                    "llama",
                    "mistral",
                    "gemma",
                    "phi",
                    "opt",
                    "bloom",
                    "falcon",
                    "olmo",
                    "mpt",
                    "stablelm",
                )
            )
            or (src_attn is not None and hasattr(src_attn, "c_attn"))
            or bool(kwargs.get("is_causal", False))
            or bool(kwargs.get("is_decoder", False))
        )
        self.is_decoder: bool = is_dec
        self.is_causal: bool = is_dec
        self.is_nested_encoder: bool = False

        self.dropout_rate: float = float(
            _cfg_get(
                config,
                "attention_probs_dropout_prob",
                _cfg_get(config, "dropout", dropout),
            )
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        kv_rank = min(self.rank, self.kv_head_size)
        self.q_proj = FusedSVDLinear(
            self.hidden_size,
            self.q_head_size,
            rank=self.rank,
            bias=True,
            use_rms_norm=self.use_rms_norm,
        )
        self.k_proj = FusedSVDLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=kv_rank,
            bias=True,
            use_rms_norm=self.use_rms_norm,
        )
        self.v_proj = FusedSVDLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=kv_rank,
            bias=True,
            use_rms_norm=self.use_rms_norm,
        )
        self.out_proj = FusedSVDLinear(
            self.q_head_size,
            self.hidden_size,
            rank=self.rank,
            bias=True,
            use_rms_norm=self.use_rms_norm,
        )
        self.rotary_emb = getattr(src_attn, "rotary_emb", None) if src_attn is not None else None

        if src_attn is not None:
            self.copy_weights(src_attn)

    @property
    def query(self) -> FusedSVDLinear:
        return self.q_proj

    @property
    def key(self) -> FusedSVDLinear:
        return self.k_proj

    @property
    def value(self) -> FusedSVDLinear:
        return self.v_proj

    def copy_weights(self, old_attn: Any) -> None:
        """Multi-architecture weight copying supporting GPT-2 Conv1D, RoBERTa nested, and flat projections."""
        if old_attn is None:
            return

        if hasattr(old_attn, "rotary_emb"):
            self.rotary_emb = old_attn.rotary_emb

        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_out = (
                hasattr(old_attn, "out_lin")
                or hasattr(old_attn, "dense")
                or hasattr(old_attn, "out_proj")
                or hasattr(old_attn, "c_proj")
                or hasattr(old_attn, "o_proj")
            )

            if hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj"):
                self.is_nested_encoder = False
            elif (
                (
                    hasattr(old_attn, "self")
                    and hasattr(old_attn.self, "query")
                    and hasattr(old_attn, "output")
                    and hasattr(old_attn.output, "dense")
                )
                or src_cls_name in ("BertSelfAttention", "RobertaSelfAttention")
                or (hasattr(old_attn, "query") and not has_out)
            ):
                self.is_nested_encoder = True
            else:
                self.is_nested_encoder = False

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                if (
                    hasattr(old_attn, "output")
                    and hasattr(old_attn.output, "dense")
                    and hasattr(old_attn.output.dense, "weight")
                ):
                    w_o = old_attn.output.dense.weight.data
                    b_o = getattr(old_attn.output.dense, "bias", None)
                    b_o = b_o.data if b_o is not None else None
            src = old_attn
            while hasattr(src, "self") and not any(
                hasattr(src, attr) for attr in ("query", "key", "value", "q_proj", "k_proj", "v_proj")
            ):
                self.is_nested_encoder = True
                src = src.self

            if hasattr(src, "c_attn") and src.c_attn is not None:
                self.is_decoder = True
                self.is_causal = True
                mod_c = src.c_attn
                base_c = getattr(mod_c, "base_layer", mod_c)
                w = getattr(base_c, "weight", getattr(mod_c, "weight", None))
                if w is not None:
                    w = w.data.clone()
                    cls_c = getattr(mod_c, "__class__", type(mod_c)).__name__
                    base_cls_c = getattr(base_c, "__class__", type(base_c)).__name__
                    if (
                        "Conv1D" in cls_c
                        or "Conv1D" in base_cls_c
                        or hasattr(mod_c, "nf")
                        or hasattr(base_c, "nf")
                        or (w.ndim == 2 and w.shape[0] < w.shape[1])
                    ):
                        w = w.t().contiguous()
                    q_dim = self.q_head_size
                    k_dim = self.kv_head_size
                    v_dim = self.kv_head_size
                    if w.shape[0] == q_dim + k_dim + v_dim:
                        w_q = w[:q_dim, :]
                        w_k = w[q_dim : q_dim + k_dim, :]
                        w_v = w[q_dim + k_dim :, :]
                    else:
                        split_size = w.shape[0] // 3
                        w_q = w[:split_size, :]
                        w_k = w[split_size : 2 * split_size, :]
                        w_v = w[2 * split_size :, :]
                    b = getattr(base_c, "bias", getattr(mod_c, "bias", None))
                    if b is not None:
                        b_data = b.data.clone()
                        if b_data.shape[0] == q_dim + k_dim + v_dim:
                            b_q = b_data[:q_dim]
                            b_k = b_data[q_dim : q_dim + k_dim]
                            b_v = b_data[q_dim + k_dim :]
                        else:
                            b_q = b_data[:split_size]
                            b_k = b_data[split_size : 2 * split_size]
                            b_v = b_data[2 * split_size :]

                if hasattr(src, "c_proj") and src.c_proj is not None:
                    mod_o = src.c_proj
                    base_o = getattr(mod_o, "base_layer", mod_o)
                    out_w = getattr(base_o, "weight", getattr(mod_o, "weight", None))
                    if out_w is not None:
                        out_w = out_w.data.clone()
                        cls_o = getattr(mod_o, "__class__", type(mod_o)).__name__
                        base_cls_o = getattr(base_o, "__class__", type(base_o)).__name__
                        if (
                            "Conv1D" in cls_o
                            or "Conv1D" in base_cls_o
                            or hasattr(mod_o, "nf")
                            or hasattr(base_o, "nf")
                            or out_w.ndim == 2
                            and out_w.shape[0] < out_w.shape[1]
                        ):
                            out_w = out_w.t().contiguous()
                        w_o = out_w
                        out_b = getattr(base_o, "bias", getattr(mod_o, "bias", None))
                        b_o = out_b.data.clone() if out_b is not None else None

            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.data.clone()
                b_q = src.q_lin.bias.data.clone() if getattr(src.q_lin, "bias", None) is not None else None
                w_k = src.k_lin.weight.data.clone()
                b_k = src.k_lin.bias.data.clone() if getattr(src.k_lin, "bias", None) is not None else None
                w_v = src.v_lin.weight.data.clone()
                b_v = src.v_lin.bias.data.clone() if getattr(src.v_lin, "bias", None) is not None else None
                if (
                    hasattr(src, "out_lin")
                    and getattr(src, "out_lin", None) is not None
                    and hasattr(src.out_lin, "weight")
                ):
                    w_o = src.out_lin.weight.data.clone()
                    b_o = src.out_lin.bias.data.clone() if getattr(src.out_lin, "bias", None) is not None else None

            elif (
                (hasattr(src, "in_proj_weight") and getattr(src, "in_proj_weight", None) is not None)
                or (
                    hasattr(src, "qkv_proj")
                    and getattr(src, "qkv_proj", None) is not None
                    and hasattr(src.qkv_proj, "weight")
                )
                or (hasattr(src, "qkv") and getattr(src, "qkv", None) is not None and hasattr(src.qkv, "weight"))
                or (hasattr(src, "Wqkv") and getattr(src, "Wqkv", None) is not None and hasattr(src.Wqkv, "weight"))
            ):
                qkv_mod = getattr(src, "qkv_proj", getattr(src, "qkv", getattr(src, "Wqkv", None)))
                if qkv_mod is not None:
                    in_w = qkv_mod.weight.data.clone()
                    in_b = qkv_mod.bias.data.clone() if getattr(qkv_mod, "bias", None) is not None else None
                else:
                    in_w = src.in_proj_weight.data.clone()
                    in_b = getattr(src, "in_proj_bias", None)
                    in_b = in_b.data.clone() if in_b is not None else None

                if in_w.ndim == 2 and in_w.shape[0] < in_w.shape[1]:
                    in_w = in_w.t().contiguous()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size
                if in_w.shape[0] == q_dim + k_dim + v_dim:
                    w_q = in_w[:q_dim, :]
                    w_k = in_w[q_dim : q_dim + k_dim, :]
                    w_v = in_w[q_dim + k_dim :, :]
                else:
                    split_sz = in_w.shape[0] // 3
                    w_q = in_w[:split_sz, :]
                    w_k = in_w[split_sz : 2 * split_sz, :]
                    w_v = in_w[2 * split_sz :, :]

                if in_b is not None:
                    if in_b.shape[0] == q_dim + k_dim + v_dim:
                        b_q = in_b[:q_dim]
                        b_k = in_b[q_dim : q_dim + k_dim]
                        b_v = in_b[q_dim + k_dim :]
                    else:
                        split_sz = in_b.shape[0] // 3
                        b_q = in_b[:split_sz]
                        b_k = in_b[split_sz : 2 * split_sz]
                        b_v = in_b[2 * split_sz :]

                for out_name in ("out_proj", "o_proj", "dense", "out_lin"):
                    if (
                        hasattr(src, out_name)
                        and getattr(src, out_name, None) is not None
                        and hasattr(getattr(src, out_name), "weight")
                    ):
                        m_out = getattr(src, out_name)
                        w_o = m_out.weight.data.clone()
                        b_o = m_out.bias.data.clone() if getattr(m_out, "bias", None) is not None else None
                        break

            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.data.clone()
                b_q = src.query.bias.data.clone() if getattr(src.query, "bias", None) is not None else None
                w_k = src.key.weight.data.clone()
                b_k = src.key.bias.data.clone() if getattr(src.key, "bias", None) is not None else None
                w_v = src.value.weight.data.clone()
                b_v = src.value.bias.data.clone() if getattr(src.value, "bias", None) is not None else None
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if (
                            hasattr(src, attr)
                            and getattr(src, attr) is not None
                            and hasattr(getattr(src, attr), "weight")
                        ):
                            m = getattr(src, attr)
                            w_o = m.weight.data.clone()
                            b_o = m.bias.data.clone() if getattr(m, "bias", None) is not None else None
                            break

            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.data.clone()
                b_q = src.q_proj.bias.data.clone() if getattr(src.q_proj, "bias", None) is not None else None
                w_k = src.k_proj.weight.data.clone()
                b_k = src.k_proj.bias.data.clone() if getattr(src.k_proj, "bias", None) is not None else None
                w_v = src.v_proj.weight.data.clone()
                b_v = src.v_proj.bias.data.clone() if getattr(src.v_proj, "bias", None) is not None else None
                if (
                    hasattr(src, "o_proj")
                    and getattr(src, "o_proj", None) is not None
                    and hasattr(src.o_proj, "weight")
                ):
                    w_o = src.o_proj.weight.data.clone()
                    b_o = src.o_proj.bias.data.clone() if getattr(src.o_proj, "bias", None) is not None else None

            if w_q is None or w_k is None or w_v is None:
                try:
                    from neuralops.core.adapters import UniversalAdapter

                    adapter = UniversalAdapter(old_attn)
                    extracted = adapter.extract_qkv_out()
                    if extracted:
                        w_q = extracted.get("q_weight")
                        b_q = extracted.get("q_bias")
                        w_k = extracted.get("k_weight")
                        b_k = extracted.get("k_bias")
                        w_v = extracted.get("v_weight")
                        b_v = extracted.get("v_bias")
                        if w_o is None:
                            w_o = extracted.get("out_weight")
                            b_o = extracted.get("out_bias")
                except Exception as ex:
                    logger.debug("UniversalAdapter fallback failed: %s", ex)

            missing = []
            if w_q is None:
                missing.append("q")
            if w_k is None:
                missing.append("k")
            if w_v is None:
                missing.append("v")
            if missing:
                logger.warning(f"Failed to extract required projections {missing} from {src_cls_name}")

            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if w is not None and torch.linalg.norm(w.float()).item() < 1e-4:
                    logger.warning(f"Copied weight {name} failed Frobenius norm check (< 1e-4)")
            if w_o is not None and not self.is_nested_encoder:
                if torch.linalg.norm(w_o.float()).item() < 1e-4:
                    logger.warning("Copied weight w_o failed Frobenius norm check (< 1e-4)")

            if w_q is not None:
                self.q_proj.decompose_from_weight(w_q, b_q)
            if w_k is not None:
                self.k_proj.decompose_from_weight(w_k, b_k)
            if w_v is not None:
                self.v_proj.decompose_from_weight(w_v, b_v)
            if w_o is not None:
                self.out_proj.decompose_from_weight(w_o, b_o)
            elif self.is_nested_encoder:
                with torch.no_grad():
                    U = self.out_proj.U.data
                    V = self.out_proj.V.data
                    U.zero_()
                    V.zero_()
                    d = min(U.shape)
                    U[:d, :d].fill_diagonal_(1.0)
                    d2 = min(V.shape)
                    V[:d2, :d2].fill_diagonal_(1.0)
                    if self.out_proj.bias is not None:
                        self.out_proj.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention is not supported in FusedSVDKernelAttention")

        x = hidden_states if hidden_states is not None else kwargs.get("query", kwargs.get("x"))
        if x is None and len(args) > 0:
            x = args[0]
        if attention_mask is None:
            attention_mask = kwargs.get("mask", kwargs.get("attn_mask"))

        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        position_ids = kwargs.get("position_ids")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif getattr(self, "rotary_emb", None) is not None and position_ids is not None:
            try:
                cos, sin = self.rotary_emb(v, position_ids)
            except TypeError:
                cos, sin = self.rotary_emb(v, seq_len=L)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = 1.0 / math.sqrt(max(int(self.head_dim), 1))
        causal_flag = bool(kwargs.get("is_causal", getattr(self, "is_decoder", False) or getattr(self, "is_causal", False)))

        if not output_attentions and head_mask is None:
            try:
                _qs = maybe_half(q)
                _ks = maybe_half(k)
                _vs = maybe_half(v)
                if attention_mask is None:
                    _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                        _qs,
                        _ks,
                        _vs,
                        is_causal=(causal_flag and L > 1),
                        dropout_p=self.dropout.p if self.training else 0.0,
                    )
                elif attention_mask.dim() == 2:
                    valid_mask = get_token_valid_mask(attention_mask, B, L, x.device)
                    if bool(valid_mask.all()):
                        _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                            _qs,
                            _ks,
                            _vs,
                            is_causal=(causal_flag and L > 1),
                            dropout_p=self.dropout.p if self.training else 0.0,
                        )
                    else:
                        attn_bool = valid_mask[:, None, None, :].expand(B, 1, L, L).clone()
                        if causal_flag and L > 1:
                            causal_tril = torch.tril(torch.ones((L, L), device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
                            attn_bool = attn_bool & causal_tril
                        row_empty = ~attn_bool.any(dim=-1, keepdim=True)
                        if bool(row_empty.any()):
                            diag = torch.eye(L, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
                            attn_bool = attn_bool | (row_empty & diag)
                        _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                            _qs,
                            _ks,
                            _vs,
                            attn_mask=attn_bool,
                            is_causal=False,
                            dropout_p=self.dropout.p if self.training else 0.0,
                        )
                else:
                    raise RuntimeError("Use explicit attention path for 3D/4D mask")
                _sdpa_out = _sdpa_out.to(x.dtype).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
                _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                return (self.out_proj(_sdpa_out) if not getattr(self, "is_nested_encoder", False) else _sdpa_out), None
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        if causal_flag and L > 1:
            causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if attention_mask is not None:
            scores = apply_attention_mask(scores, attention_mask, mask_value=-1e4)

        probs = torch.nan_to_num(F.softmax(scores, dim=-1, dtype=torch.float32), nan=0.0).to(v.dtype)
        probs = self.dropout(probs)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            probs = probs * head_mask

        context = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, L, self.q_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, (probs if output_attentions else None)


def fused_svd_kernel_factory(config: Any = None, **kwargs: Any) -> nn.Module:
    """Factory creating FusedSVDKernelAttention with fused SVD kernel execution."""
    return FusedSVDKernelAttention(config=config, **kwargs)


def inject_fused_svd_kernel(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> Any | None:
    """Safe injection hook for fused SVD kernel attention."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None)

    if (
        old_attn is None
        and target is not None
        and (
            hasattr(target, "transformer")
            or hasattr(target, "distilbert")
            or hasattr(target, "roberta")
            or hasattr(target, "bert")
            or (hasattr(target, "model") and hasattr(target.model, "layers"))
            or (hasattr(target, "encoder") and hasattr(target.encoder, "layer"))
        )
        and not (
            hasattr(target, "q_lin")
            or hasattr(target, "q_proj")
            or hasattr(target, "c_attn")
            or hasattr(target, "query")
        )
    ):
        try:
            layer_list = None
            if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
                layer_list = target.distilbert.transformer.layer
            elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
                layer_list = target.roberta.encoder.layer
            elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
                layer_list = target.bert.encoder.layer
            elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
                layer_list = target.transformer.h
            elif hasattr(target, "transformer") and hasattr(target.transformer, "layer"):
                layer_list = target.transformer.layer
            elif hasattr(target, "model") and hasattr(target.model, "layers"):
                layer_list = target.model.layers
            elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
                layer_list = target.encoder.layer

            if layer_list is not None:
                tot = len(layer_list)
                for idx, lyr in enumerate(layer_list):
                    inject_fused_svd_kernel(
                        layer=lyr,
                        model_config=cfg,
                        comp_config=comp_config,
                        layer_idx=idx,
                        total_layers=tot,
                        **kwargs,
                    )
                return target
        except Exception as e:
            logger.error("inject_fused_svd_kernel whole-model traversal failed: %s", e)
            return None

    if (
        old_attn is None
        and target is not None
        and hasattr(target, "named_modules")
        and not (
            hasattr(target, "attention")
            or hasattr(target, "self")
            or hasattr(target, "attn")
            or hasattr(target, "q_proj")
            or hasattr(target, "q_lin")
            or hasattr(target, "query")
            or hasattr(target, "c_attn")
        )
    ):
        try:
            replaced = 0
            for name, module in list(target.named_modules()):
                if (
                    hasattr(module, "q_lin")
                    or hasattr(module, "query")
                    or hasattr(module, "q_proj")
                    or hasattr(module, "c_attn")
                ):
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    child_name = name.rsplit(".", 1)[-1] if "." in name else name
                    parent = target.get_submodule(parent_name) if parent_name else target
                    new_attn = inject_fused_svd_kernel(
                        layer=None,
                        model_config=cfg,
                        comp_config=comp_config,
                        old_attn=module,
                        **kwargs,
                    )
                    if new_attn is not None:
                        setattr(parent, child_name, new_attn)
                        replaced += 1
            if replaced > 0:
                return target
        except Exception as e:
            logger.error("inject_fused_svd_kernel named_modules replacement failed: %s", e)
            return None

    try:
        src_attn = (
            old_attn
            if old_attn is not None
            else (
                getattr(
                    layer,
                    "attention",
                    getattr(layer, "self", getattr(layer, "attn", None)),
                )
                if layer is not None
                else None
            )
        )
        new_attn = FusedSVDKernelAttention(
            config=cfg,
            old_attn=src_attn,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        if layer is not None:
            target_param = next(layer.parameters(), None)
            if target_param is not None:
                new_attn.to(device=target_param.device, dtype=target_param.dtype)

            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                else:
                    layer.attention = new_attn
            elif hasattr(layer, "self"):
                layer.self = new_attn
                new_attn.is_nested_encoder = True
            elif hasattr(layer, "attn"):
                layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_fused_svd_kernel failed: %s", e)
        return None


__all__ = [
    "_fused_svd_linear_fwd_kernel",
    "_FusedSVDLinearAutogradFunction",
    "fused_svd_linear",
    "FusedSVDLinear",
    "FusedSVDKernelAttention",
    "fused_svd_kernel_factory",
    "inject_fused_svd_kernel",
    "HAS_TRITON",
]
