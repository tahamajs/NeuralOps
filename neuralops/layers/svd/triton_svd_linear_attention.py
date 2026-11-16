#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Triton-Accelerated SVD Linear Attention Layer

Hardware-accelerated linear attention module fusing SVD low-rank factorized projections
with high-throughput Triton tiled kernel execution.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv
from neuralops.ops.triton import triton_dispatch
from neuralops.layers.svd._triton_svd_linear_kernel import (
    HAS_TRITON,
    _TritonLinearAttentionAutogradFunction,
    triton_linear_attention,
)

logger = logging.getLogger(__name__)

class SVDLinearProjection(nn.Module):
    r"""
    Genuine SVD Low-Rank Linear Layer: W \approx U @ V.

    Factorizes in_features -> out_features through bottleneck rank r:
      V \in R^{r x in_features}  (Down-projection)
      U \in R^{out_features x r} (Up-projection)
      y = (x @ V^T) @ U^T + bias
    """

    triton_kernel = "low_rank"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)

        self.V = nn.Parameter(torch.empty(self.rank, in_features))
        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.zeros_(self.U)
        with torch.no_grad():
            r = min(self.out_features, self.rank)
            self.U[:r, :r].copy_(torch.eye(r))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.U.to(x.dtype) if self.U.dtype != x.dtype else self.U
        v = self.V.to(x.dtype) if self.V.dtype != x.dtype else self.V
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        return triton_dispatch(self.triton_kernel, x, u, v, bias=b)

    @classmethod
    def from_pretrained_linear(cls, linear: nn.Linear, rank: int = 64) -> SVDLinearProjection:
        r"""
        Initializes low-rank factors directly from pretrained weights via Truncated SVD:
        W = U_full @ diag(S) @ V_full^T
        U = U_full[:, :r] @ sqrt(S[:r])
        V = sqrt(S[:r]) @ V_full^T[:r, :]
        """
        out_features, in_features = linear.weight.shape
        proj = cls(in_features, out_features, rank=rank, bias=(linear.bias is not None))

        with torch.no_grad():
            W = linear.weight.data.float()
            u, s, vh = torch.linalg.svd(W, full_matrices=False)

            r = proj.rank
            s_sqrt = torch.diag(torch.sqrt(s[:r]))
            proj.U.data.copy_(torch.matmul(u[:, :r], s_sqrt).to(linear.weight.dtype))
            proj.V.data.copy_(torch.matmul(s_sqrt, vh[:r, :]).to(linear.weight.dtype))

            if linear.bias is not None:
                proj.bias.data.copy_(linear.bias.data)

        return proj


@ATTENTION_REGISTRY.register("triton_svd_linear_attention", supports_causal=True, supports_bidirectional=True)
class TritonSVDLinearAttention(BaseAttention):
    r"""
    Triton-Accelerated Linear Attention with Genuine SVD Low-Rank Projections,
    RoPE, GQA, and Masked Attention.
    """

    triton_kernel = "linear_attn"

    def __init__(
        self,
        config: Any = None,
        embed_dim: int | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        rank: int | None = 64,
        is_decoder: bool = False,
        activation_mode: str = "elu",
        eps: float = 1e-6,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.config = config
        self.is_decoder = bool(
            is_decoder or getattr(config, "is_decoder", False) or getattr(config, "is_causal", False)
        )
        self.activation_mode = activation_mode
        self.eps = eps
        self.triton_kwargs = {
            "phi": "elu+1" if self.activation_mode == "elu" else self.activation_mode,
            "eps": self.eps,
            "cos_weight": None,
            "sin_weight": None,
        }

        self.embed_dim = (
            embed_dim
            or kwargs.get("dim")
            or kwargs.get("d_model")
            or kwargs.get("hidden_size")
            or getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 768)))
        )
        self.num_heads = num_heads or kwargs.get("num_attention_heads") or getattr(config, "num_attention_heads", 12)
        self.num_kv_heads = (
            num_kv_heads or kwargs.get("num_kv_heads") or getattr(config, "num_key_value_heads", self.num_heads)
        )
        self.head_dim = kwargs.get("head_dim") or (self.embed_dim // self.num_heads)
        self.hidden_size = self.embed_dim
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank or 64
        max_rank = self.embed_dim // 2
        self.rank = min(self.rank, max_rank)
        self.kv_dim = self.num_kv_heads * self.head_dim

        if self.rank < self.embed_dim:
            self.q_proj = SVDLinearProjection(self.embed_dim, self.embed_dim, rank=self.rank, bias=True)
            self.k_proj = SVDLinearProjection(self.embed_dim, self.kv_dim, rank=min(self.rank, self.kv_dim), bias=True)
            self.v_proj = SVDLinearProjection(self.embed_dim, self.kv_dim, rank=min(self.rank, self.kv_dim), bias=True)
        else:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.k_proj = nn.Linear(self.embed_dim, self.kv_dim, bias=True)
            self.v_proj = nn.Linear(self.embed_dim, self.kv_dim, bias=True)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Transfer pretrained projection weights from GPT-2, RoBERTa/BERT, or standard linear layers."""
        if src is None:
            return

        getattr(src, "__class__", type(src)).__name__

        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or hasattr(c_attn_mod, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()

                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None

                q_w, q_b = c_w[:split_size, :], (c_b[:split_size] if c_b is not None else None)
                k_w, k_b = (
                    c_w[split_size : 2 * split_size, :],
                    (c_b[split_size : 2 * split_size] if c_b is not None else None),
                )
                v_w, v_b = (
                    c_w[2 * split_size :, :],
                    (c_b[2 * split_size :] if c_b is not None else None),
                )

                self._copy_single_proj(self.q_proj, q_w, q_b)
                self._copy_single_proj(self.k_proj, k_w, k_b)
                self._copy_single_proj(self.v_proj, v_w, v_b)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or hasattr(c_proj_mod, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    self.out_proj.weight.data.copy_(o_w)
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    if bias_o is not None and self.out_proj.bias is not None:
                        self.out_proj.bias.data.copy_(bias_o.data)

        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            mapping = [("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")]
            for target_name, sub_name in mapping:
                target_mod = getattr(self, target_name)
                src_mod = getattr(s, sub_name, None)
                base_src = getattr(src_mod, "base_layer", src_mod)
                if base_src is not None and hasattr(base_src, "weight"):
                    w = base_src.weight.data
                    b = getattr(base_src, "bias", None)
                    b_data = b.data if b is not None else None
                    self._copy_single_proj(target_mod, w, b_data)

            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                base_out = getattr(out_mod, "base_layer", out_mod)
                self.out_proj.weight.data.copy_(base_out.weight.data)
                if base_out.bias is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.data.copy_(base_out.bias.data)

        else:
            proj_candidates = {
                "q_proj": ["q_proj", "query", "q_lin", "W_q", "wq"],
                "k_proj": ["k_proj", "key", "k_lin", "W_k", "wk"],
                "v_proj": ["v_proj", "value", "v_lin", "W_v", "wv"],
                "out_proj": ["out_proj", "dense", "out_lin", "c_proj", "o_proj", "wo"],
            }
            for target_name, candidates in proj_candidates.items():
                target_mod = getattr(self, target_name)
                for cand in candidates:
                    if hasattr(src, cand):
                        src_mod = getattr(src, cand)
                        base_mod = getattr(src_mod, "base_layer", src_mod)
                        if base_mod is not None and hasattr(base_mod, "weight"):
                            w = base_mod.weight.data
                            b = getattr(base_mod, "bias", None)
                            b_data = b.data if b is not None else None
                            if target_name == "out_proj":
                                if (
                                    w.ndim == 2
                                    and w.shape != target_mod.weight.shape
                                    and w.t().shape == target_mod.weight.shape
                                ):
                                    w = w.t()
                                target_mod.weight.data.copy_(w)
                                if b_data is not None and target_mod.bias is not None:
                                    target_mod.bias.data.copy_(b_data)
                            else:
                                self._copy_single_proj(target_mod, w, b_data)
                            break

    def _copy_single_proj(self, target_mod: nn.Module, w: torch.Tensor, b: torch.Tensor | None) -> None:
        """Decomposes weights into low-rank SVD factors or copies directly."""
        if isinstance(target_mod, SVDLinearProjection):
            with torch.no_grad():
                W_f = w.float()
                u, s, vh = torch.linalg.svd(W_f, full_matrices=False)
                r = target_mod.rank
                s_sqrt = torch.diag(torch.sqrt(s[:r]))
                target_mod.U.data.copy_(torch.matmul(u[:, :r], s_sqrt).to(w.dtype))
                target_mod.V.data.copy_(torch.matmul(s_sqrt, vh[:r, :]).to(w.dtype))
                if b is not None and target_mod.bias is not None:
                    target_mod.bias.data.copy_(b)
        else:
            if w.ndim == 2 and w.shape != target_mod.weight.shape and w.t().shape == target_mod.weight.shape:
                w = w.t()
            target_mod.weight.data.copy_(w)
            if b is not None and target_mod.bias is not None:
                target_mod.bias.data.copy_(b)

    copy_weights_from_pretrained = copy_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        B, N, C = hidden_states.shape

        if torch.is_autocast_enabled() and hidden_states.is_cuda:
            ac_dtype = torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else torch.float16
            hidden_states = hidden_states.to(ac_dtype)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = repeat_kv(v, self.num_heads // self.num_kv_heads)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, :, None].to(q.dtype)
                if (mask < 0).any():
                    mask = (mask > -1000.0).to(q.dtype)
                k = k * mask
                v = v * mask
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] == 1:
                    mask = attention_mask.squeeze(1).squeeze(1)[:, None, :, None].to(q.dtype)
                else:
                    mask = attention_mask[:, 0, -1, :][:, None, :, None].to(q.dtype)
                if (mask < 0).any():
                    mask = (mask > -1000.0).to(q.dtype)
                k = k * mask
                v = v * mask

        out = triton_dispatch(
            self.triton_kernel,
            q,
            k,
            v,
            is_causal=self.is_decoder,
            **self.triton_kwargs,
        )

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)

        out = (
            torch.nan_to_num(out.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
            .clamp(-65000.0, 65000.0)
            .to(hidden_states.dtype)
        )

        return out, None


def inject_triton_svd_linear_attention(
    model: nn.Module = None,
    rank: int = 64,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module | None:
    """Safe factory function to inject TritonSVDLinearAttention into a model."""
    from neuralops.core.attention_utils import inject_attention

    return inject_attention(
        model,
        "triton_svd_linear_attention",
        rank=rank,
        copy_weights=copy_weights,
        **kwargs,
    )
