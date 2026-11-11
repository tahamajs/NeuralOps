#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Compressed Multi-Head Attention via Joint Truncated SVD

Joint truncated singular value decomposition compressing query, key, value, and output projections.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import apply_rotary_pos_emb

logger = logging.getLogger(__name__)

from neuralops.core.attention_mask import apply_attention_mask, get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv
from neuralops.utils.model_family import ModelAdapter


class _FactorizedSliceWeightTensor(torch.Tensor):
    """Tensor proxy forwarding `.data.copy_(src)` / `.copy_(src)` into CompressedDistilBertAttention SVD update."""

    @staticmethod
    def __new__(
        cls, tensor: torch.Tensor, owner: CompressedDistilBertAttention | None = None, kind: str = "q"
    ) -> _FactorizedSliceWeightTensor:
        proxy = tensor.as_subclass(cls)
        proxy._owner = owner
        proxy._kind = kind
        return proxy

    @property
    def data(self) -> _FactorizedSliceWeightTensor:
        return self

    def copy_(self, src: torch.Tensor, non_blocking: bool = False) -> _FactorizedSliceWeightTensor:
        owner = getattr(self, "_owner", None)
        kind = getattr(self, "_kind", "q")
        if owner is not None:
            owner._update_projection_from_weight(kind, src)
            return self
        return super().copy_(src, non_blocking=non_blocking)


class _ProjectionWeightProxy:
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
        owner: CompressedDistilBertAttention | None = None,
        kind: str = "q",
    ) -> None:
        self._weight = weight
        self.bias = bias
        self.in_features = in_features if in_features is not None else (weight.shape[1] if weight.ndim == 2 else 0)
        self.out_features = out_features if out_features is not None else (weight.shape[0] if weight.ndim >= 1 else 0)
        self._owner = owner
        self._kind = kind

    @property
    def weight(self) -> torch.Tensor:
        if self._owner is not None:
            return _FactorizedSliceWeightTensor(self._weight, owner=self._owner, kind=self._kind)
        return self._weight


@ATTENTION_REGISTRY.register(
    "compressed_mha_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class CompressedDistilBertAttention(BaseAttention):
    """CompressedDistilBertAttention neural operator.

    Compressed Multi-Head Attention for DistilBERT and Multi-Architecture via Joint Truncated SVD.

    Attributes:
        supports_causal (bool): Whether causal autoregressive masking is supported.
        is_custom_attention (bool): Flag indicating specialized non-standard attention mechanics.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        model_config: Any | None = None,
        rank: int | None = None,
        rank_qkv: int | None = None,
        rank_out: int | None = None,
        dropout: float = 0.1,
        lora_rank: int = 16,
        pretrained_attn: nn.Module | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config
        super().__init__(cfg, rank)
        self.adapter = ModelAdapter(cfg)
        self.hidden_size: int = self.adapter.hidden_size or kwargs.get("hidden_size", kwargs.get("d_model", 768))
        self.num_heads: int = self.adapter.num_heads or kwargs.get("num_heads", kwargs.get("num_attention_heads", 12))
        self.num_kv_heads: int = self.adapter.num_key_value_heads or kwargs.get(
            "num_kv_heads", kwargs.get("num_key_value_heads", self.num_heads)
        )
        self.head_dim: int = self.adapter.head_dim or (self.hidden_size // max(1, self.num_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        _mt = getattr(cfg, "model_type", "").lower() if cfg else ""
        _decoder_prefixes = (
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
        self.is_decoder: bool = bool(
            getattr(cfg, "is_decoder", False)
            or getattr(cfg, "is_causal", False)
            or kwargs.get("is_decoder", False)
            or kwargs.get("is_causal", False)
            or any(_mt.startswith(p) for p in _decoder_prefixes)
        )
        self.is_nested_encoder: bool = False
        self._bypass_out_proj: bool = False
        self._staged_qkv: dict[str, torch.Tensor | None] = {"q": None, "k": None, "v": None}
        self._out_proj_proxy: _ProjectionWeightProxy | None = None
        self._q_proxy: _ProjectionWeightProxy | None = None
        self._k_proxy: _ProjectionWeightProxy | None = None
        self._v_proxy: _ProjectionWeightProxy | None = None

        default_rank = max(256, self.hidden_size // 2)
        rank_val = rank if rank is not None else kwargs.get("rank_qkv", rank_qkv if rank_qkv is not None else default_rank)
        self.rank_qkv = min(int(rank_val), self.all_head_size + 2 * self.kv_head_size, self.hidden_size)
        out_r = rank if rank is not None else kwargs.get("rank_out", rank_out if rank_out is not None else default_rank)
        self.rank_out = min(int(out_r), self.all_head_size, self.hidden_size)
        self.lora_rank = getattr(cfg, "lora_rank", lora_rank)
        self.lora_scaling = (32.0 / self.lora_rank) if self.lora_rank > 0 else 1.0

        qkv_out_dim = self.all_head_size + 2 * self.kv_head_size

        self.U_qkv = nn.Parameter(torch.randn(qkv_out_dim, self.rank_qkv) * 0.02, requires_grad=False)
        self.S_qkv = nn.Parameter(torch.ones(self.rank_qkv), requires_grad=False)
        self.Vh_qkv = nn.Parameter(torch.randn(self.rank_qkv, self.hidden_size) * 0.02, requires_grad=False)

        self.U_out = nn.Parameter(torch.randn(self.hidden_size, self.rank_out) * 0.02, requires_grad=False)
        self.S_out = nn.Parameter(torch.ones(self.rank_out), requires_grad=False)
        self.Vh_out = nn.Parameter(torch.randn(self.rank_out, self.all_head_size) * 0.02, requires_grad=False)

        self.bias_q = nn.Parameter(torch.zeros(self.all_head_size), requires_grad=False)
        self.bias_k = nn.Parameter(torch.zeros(self.kv_head_size), requires_grad=False)
        self.bias_v = nn.Parameter(torch.zeros(self.kv_head_size), requires_grad=False)
        self.bias_out = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        if self.lora_rank > 0:
            self.lora_A_qkv = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.lora_B_qkv = nn.Parameter(torch.empty(qkv_out_dim, self.lora_rank))
            self.lora_A_out = nn.Parameter(torch.empty(self.lora_rank, self.all_head_size))
            self.lora_B_out = nn.Parameter(torch.empty(self.hidden_size, self.lora_rank))
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A_qkv", None)
            self.register_parameter("lora_B_qkv", None)
            self.register_parameter("lora_A_out", None)
            self.register_parameter("lora_B_out", None)

        self.dropout_p = getattr(cfg, "dropout", dropout) if cfg else dropout
        self.dropout = nn.Dropout(self.dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("out_proj", "o_proj", "dense", "out_lin", "c_proj") and isinstance(value, nn.Module):
            self.__dict__["_bypass_out_proj"] = isinstance(value, nn.Identity)
        super().__setattr__(name, value)

    @property
    def qkv_weight(self) -> torch.Tensor:
        """Reconstruct the effective base QKV weight from SVD factors."""
        return (self.U_qkv * self.S_qkv.view(1, -1)) @ self.Vh_qkv

    @property
    def out_weight(self) -> torch.Tensor:
        """Reconstruct the effective base Out weight from SVD factors."""
        return (self.U_out * self.S_out.view(1, -1)) @ self.Vh_out

    @property
    def query(self) -> _ProjectionWeightProxy:
        if getattr(self, "_q_proxy", None) is None:
            self._q_proxy = _ProjectionWeightProxy(
                self.qkv_weight[: self.all_head_size],
                self.bias_q,
                in_features=self.hidden_size,
                out_features=self.all_head_size,
                owner=self,
                kind="q",
            )
        return self._q_proxy

    @property
    def key(self) -> _ProjectionWeightProxy:
        if getattr(self, "_k_proxy", None) is None:
            self._k_proxy = _ProjectionWeightProxy(
                self.qkv_weight[self.all_head_size : self.all_head_size + self.kv_head_size],
                self.bias_k,
                in_features=self.hidden_size,
                out_features=self.kv_head_size,
                owner=self,
                kind="k",
            )
        return self._k_proxy

    @property
    def value(self) -> _ProjectionWeightProxy:
        if getattr(self, "_v_proxy", None) is None:
            self._v_proxy = _ProjectionWeightProxy(
                self.qkv_weight[self.all_head_size + self.kv_head_size : self.all_head_size + 2 * self.kv_head_size],
                self.bias_v,
                in_features=self.hidden_size,
                out_features=self.kv_head_size,
                owner=self,
                kind="v",
            )
        return self._v_proxy

    @property
    def out_proj(self) -> Any:
        if getattr(self, "_bypass_out_proj", False) or "out_proj" in self._modules:
            return self._modules.get("out_proj", nn.Identity())
        if getattr(self, "_out_proj_proxy", None) is None:
            self._out_proj_proxy = _ProjectionWeightProxy(
                self.out_weight,
                self.bias_out,
                in_features=self.all_head_size,
                out_features=self.hidden_size,
                owner=self,
                kind="o",
            )
        return self._out_proj_proxy

    @out_proj.setter
    def out_proj(self, value: nn.Module) -> None:
        if isinstance(value, nn.Identity):
            self._bypass_out_proj = True

    @property
    def q_proj(self) -> _ProjectionWeightProxy:
        return self.query

    @property
    def k_proj(self) -> _ProjectionWeightProxy:
        return self.key

    @property
    def v_proj(self) -> _ProjectionWeightProxy:
        return self.value

    @property
    def o_proj(self) -> Any:
        return self.out_proj

    @o_proj.setter
    def o_proj(self, value: nn.Module) -> None:
        if isinstance(value, nn.Identity):
            self._bypass_out_proj = True

    @property
    def dense(self) -> Any:
        return self.out_proj

    @dense.setter
    def dense(self, value: nn.Module) -> None:
        if isinstance(value, nn.Identity):
            self._bypass_out_proj = True

    def _decompose_qkv_weights(self, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor) -> None:
        """Decompose [W_q; W_k; W_v] via GQA-energy-balanced truncated SVD into U_qkv, S_qkv, Vh_qkv."""
        with torch.no_grad():
            wq = w_q.detach().float()
            wk = w_k.detach().float()
            wv = w_v.detach().float()

            scale_k = 1.0
            scale_v = 1.0
            if self.kv_head_size < self.all_head_size:
                nq = float(torch.linalg.norm(wq).item())
                nk = float(torch.linalg.norm(wk).item())
                nv = float(torch.linalg.norm(wv).item())
                if nk > 1e-6 and nq > 0.0:
                    scale_k = min(8.0, max(1.0, math.sqrt(nq / nk)))
                if nv > 1e-6 and nq > 0.0:
                    scale_v = min(8.0, max(1.0, math.sqrt(nq / nv)))

            w_qkv = torch.cat([wq, wk * scale_k, wv * scale_v], dim=0).cpu()
            U, S, Vh = torch.linalg.svd(w_qkv, full_matrices=False)
            r_qkv = min(self.rank_qkv, U.size(1), Vh.size(0))

            U_r = U[:, :r_qkv].clone()
            if scale_k != 1.0:
                U_r[self.all_head_size : self.all_head_size + self.kv_head_size] /= scale_k
            if scale_v != 1.0:
                U_r[self.all_head_size + self.kv_head_size :] /= scale_v

            qkv_out_dim = self.all_head_size + 2 * self.kv_head_size
            self.U_qkv.data.zero_()
            self.S_qkv.data.zero_()
            self.Vh_qkv.data.zero_()
            rows = min(qkv_out_dim, U_r.size(0))
            cols = min(self.hidden_size, Vh.size(1))
            self.U_qkv.data[:rows, :r_qkv].copy_(U_r[:rows, :r_qkv].to(device=self.U_qkv.device, dtype=self.U_qkv.dtype))
            self.S_qkv.data[:r_qkv].copy_(S[:r_qkv].to(device=self.S_qkv.device, dtype=self.S_qkv.dtype))
            self.Vh_qkv.data[:r_qkv, :cols].copy_(Vh[:r_qkv, :cols].to(device=self.Vh_qkv.device, dtype=self.Vh_qkv.dtype))

    def _update_projection_from_weight(self, kind: str, src: torch.Tensor) -> None:
        """Accept external `.weight.data.copy_(src)` (e.g. QwenAttentionAdapter) and re-factorize."""
        with torch.no_grad():
            if kind == "o":
                w_o = src.detach().float().cpu()
                U_o, S_o, Vh_o = torch.linalg.svd(w_o, full_matrices=False)
                r_out = min(self.rank_out, U_o.size(1), Vh_o.size(0))
                self.U_out.data.zero_()
                self.S_out.data.zero_()
                self.Vh_out.data.zero_()
                rows = min(self.hidden_size, U_o.size(0))
                cols = min(self.all_head_size, Vh_o.size(1))
                self.U_out.data[:rows, :r_out].copy_(U_o[:rows, :r_out].to(device=self.U_out.device, dtype=self.U_out.dtype))
                self.S_out.data[:r_out].copy_(S_o[:r_out].to(device=self.S_out.device, dtype=self.S_out.dtype))
                self.Vh_out.data[:r_out, :cols].copy_(Vh_o[:r_out, :cols].to(device=self.Vh_out.device, dtype=self.Vh_out.dtype))
                return

            self._staged_qkv[kind] = src.detach().clone()
            self._q_proxy = None
            self._k_proxy = None
            self._v_proxy = None
            cur_qkv = self.qkv_weight.detach()
            w_q = self._staged_qkv["q"] if self._staged_qkv["q"] is not None else cur_qkv[: self.all_head_size]
            w_k = (
                self._staged_qkv["k"]
                if self._staged_qkv["k"] is not None
                else cur_qkv[self.all_head_size : self.all_head_size + self.kv_head_size]
            )
            w_v = (
                self._staged_qkv["v"]
                if self._staged_qkv["v"] is not None
                else cur_qkv[self.all_head_size + self.kv_head_size :]
            )
            self._decompose_qkv_weights(w_q, w_k, w_v)

    def reset_lora_parameters(self) -> None:
        if self.lora_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A_qkv, a=math.sqrt(5))
            nn.init.normal_(self.lora_B_qkv, mean=0.0, std=1e-4)
            nn.init.kaiming_uniform_(self.lora_A_out, a=math.sqrt(5))
            nn.init.normal_(self.lora_B_out, mean=0.0, std=1e-4)
            self.lora_A_qkv.requires_grad = True
            self.lora_B_qkv.requires_grad = True
            self.lora_A_out.requires_grad = True
            self.lora_B_out.requires_grad = True

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            attention_mask is not None
            and attention_mask.dtype == torch.bool
            and attention_mask.shape[-1] == self.num_heads
        ):
            head_mask, attention_mask = attention_mask, None
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        qkv_w_up = self.U_qkv * self.S_qkv
        qkv_w_down = self.Vh_qkv
        if hidden_states.is_cuda:
            try:
                from neuralops.ops.triton.low_rank import fused_svd_lora_sparse_attn_triton

                qkv_comp = fused_svd_lora_sparse_attn_triton(
                    hidden_states,
                    w_down=qkv_w_down,
                    w_up=qkv_w_up,
                    lora_A=self.lora_A_qkv if self.lora_rank > 0 else None,
                    lora_B=self.lora_B_qkv if self.lora_rank > 0 else None,
                    lora_scale=self.lora_scaling,
                )
            except Exception:
                proj = torch.matmul(hidden_states, self.Vh_qkv.t().contiguous()) * self.S_qkv.unsqueeze(0).unsqueeze(0)
                qkv_comp = torch.matmul(proj, self.U_qkv.t().contiguous())
                if self.lora_rank > 0 and self.lora_A_qkv is not None and self.lora_B_qkv is not None:
                    qkv_comp = qkv_comp + (
                        F.linear(F.linear(hidden_states, self.lora_A_qkv), self.lora_B_qkv) * self.lora_scaling
                    )
        else:
            proj = torch.matmul(hidden_states, self.Vh_qkv.t().contiguous()) * self.S_qkv.unsqueeze(0).unsqueeze(0)
            qkv_comp = torch.matmul(proj, self.U_qkv.t().contiguous())
            if self.lora_rank > 0 and self.lora_A_qkv is not None and self.lora_B_qkv is not None:
                qkv_comp = qkv_comp + (
                    F.linear(F.linear(hidden_states, self.lora_A_qkv), self.lora_B_qkv) * self.lora_scaling
                )

        q_raw, k_raw, v_raw = torch.split(qkv_comp, [self.all_head_size, self.kv_head_size, self.kv_head_size], dim=-1)
        q = q_raw + self.bias_q
        k = k_raw + self.bias_k
        v = v_raw + self.bias_v

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        position_ids = kwargs.get("position_ids")
        position_embeddings = kwargs.get("position_embeddings")
        if (
            position_embeddings is not None
            and isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) >= 2
        ):
            cos, sin = position_embeddings[0], position_embeddings[1]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif getattr(self, "rotary_emb", None) is not None and position_ids is not None:
            try:
                cos, sin = self.rotary_emb(v, position_ids)
            except TypeError:
                cos, sin = self.rotary_emb(v, seq_len=N)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask")
        mt = str(getattr(self.config, "model_type", "") or "").lower()
        decoder_prefixes = (
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
        is_decoder = bool(
            kwargs.get(
                "is_causal",
                self.is_decoder
                or getattr(self.config, "is_decoder", False)
                or any(mt.startswith(p) for p in decoder_prefixes),
            )
        )

        context = None
        attn_probs = None
        if not output_attentions and head_mask is None:
            try:
                if mask is None:
                    context = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=(is_decoder and N > 1),
                        dropout_p=self.dropout.p if self.training else 0.0,
                    )
                elif mask.dim() == 2:
                    valid_m = get_token_valid_mask(mask, B, N, device)
                    if bool(valid_m.all()):
                        context = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            is_causal=(is_decoder and N > 1),
                            dropout_p=self.dropout.p if self.training else 0.0,
                        )
                    else:
                        attn_bool = valid_m[:, None, None, :].expand(B, 1, N, N).clone()
                        if is_decoder and N > 1:
                            causal_tril = torch.tril(torch.ones((N, N), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
                            attn_bool = attn_bool & causal_tril
                        row_empty = ~attn_bool.any(dim=-1, keepdim=True)
                        if bool(row_empty.any()):
                            diag = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
                            attn_bool = attn_bool | (row_empty & diag)
                        context = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=attn_bool,
                            is_causal=False,
                            dropout_p=self.dropout.p if self.training else 0.0,
                        )
            except Exception:
                context = None

        if context is None:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1).contiguous()) * scale

            if mask is not None:
                scores = apply_attention_mask(scores, mask, mask_value=-10000.0)

            if is_decoder and N > 1:
                causal_tril = torch.tril(torch.ones((N, N), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(~causal_tril, -10000.0)

            attn_probs = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
            attn_probs = self.dropout(attn_probs)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                attn_probs = attn_probs * head_mask

            context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder or getattr(self, "_bypass_out_proj", False):
            out = context
        else:
            out_w_up = self.U_out * self.S_out
            out_w_down = self.Vh_out
            if context.is_cuda:
                try:
                    from neuralops.ops.triton.low_rank import fused_svd_lora_sparse_attn_triton

                    out = fused_svd_lora_sparse_attn_triton(
                        context,
                        w_down=out_w_down,
                        w_up=out_w_up,
                        lora_A=self.lora_A_out if self.lora_rank > 0 else None,
                        lora_B=self.lora_B_out if self.lora_rank > 0 else None,
                        lora_scale=self.lora_scaling,
                        bias=self.bias_out,
                    )
                except Exception:
                    out_proj = torch.matmul(context, self.Vh_out.t().contiguous()) * self.S_out.unsqueeze(0).unsqueeze(
                        0
                    )
                    out = torch.matmul(out_proj, self.U_out.t().contiguous()) + self.bias_out
                    if self.lora_rank > 0 and self.lora_A_out is not None and self.lora_B_out is not None:
                        out = out + (F.linear(F.linear(context, self.lora_A_out), self.lora_B_out) * self.lora_scaling)
            else:
                out_proj = torch.matmul(context, self.Vh_out.t().contiguous()) * self.S_out.unsqueeze(0).unsqueeze(0)
                out = torch.matmul(out_proj, self.U_out.t().contiguous()) + self.bias_out
                if self.lora_rank > 0 and self.lora_A_out is not None and self.lora_B_out is not None:
                    out = out + (F.linear(F.linear(context, self.lora_A_out), self.lora_B_out) * self.lora_scaling)

        return out, (attn_probs if output_attentions else None)


def inject_compressed_mha(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_compressed_mha(
                layer=lyr,
                model_config=cfg,
                comp_config=(comp_config if "comp_config" in locals() or "comp_config" in kwargs else None),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    if old_attn is None and layer is not None:
        for attr in ("attn", "attention", "self_attn", "self"):
            if hasattr(layer, attr):
                cand = getattr(layer, attr)
                if cand is not None:
                    old_attn = getattr(cand, "self", cand)
                    break

    if model_config is None and old_attn is not None:
        model_config = getattr(old_attn, "config", None)

    try:
        h_sz = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768)) if model_config else 768
        default_rank = max(256, h_sz // 2) if h_sz >= 512 else max(1, h_sz // 2)
        rank_qkv = (
            getattr(comp_config, "rank_qkv", getattr(comp_config, "rank", default_rank))
            if comp_config is not None
            else default_rank
        )
        rank_out = (
            getattr(comp_config, "rank_out", getattr(comp_config, "rank", default_rank))
            if comp_config is not None
            else default_rank
        )
        dropout = getattr(comp_config, "dropout", 0.1) if comp_config is not None else 0.1
        new_attn = CompressedDistilBertAttention(
            model_config=model_config,
            rank_qkv=rank_qkv,
            rank_out=rank_out,
            dropout=dropout,
            pretrained_attn=old_attn,
        )
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        else:
            if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
        if old_attn is not None:
            new_attn.train(old_attn.training)
        elif layer is not None and hasattr(layer, "training"):
            new_attn.train(layer.training)
        return new_attn
    except Exception as e:
        logger.error(f"inject_compressed_mha: failed: {e}")
        return None


def compressed_mha_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) * scale
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, -10000.0)
            else:
                scores = scores + attention_mask.to(scores.dtype)
        else:
            valid_m = get_token_valid_mask(attention_mask, query.shape[0], query.shape[-2], query.device)
            if valid_m is not None:
                scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)
    if is_causal:
        seq_len = query.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=query.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -10000.0)
    attn_probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    if dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p)
    return torch.matmul(attn_probs, value)
