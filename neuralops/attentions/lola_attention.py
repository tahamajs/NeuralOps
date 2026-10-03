"""
LoLA: Low-Rank Linear Attention with Sparse Caching.

Combines factorized low-rank projections with selective multi-system cached retrieval,
trainable LoRA residual adapters, frozen SVD base weights, GQA, RoPE, and causal decoding.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (
            _rotate_half(q_rot) * sin.to(q_rot.dtype)
        )
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (
            _rotate_half(k_rot) * sin.to(k_rot.dtype)
        )
        q = (
            torch.cat([q_rot_embed, q_pass], dim=-1)
            if q_pass.numel() > 0
            else q_rot_embed
        )
        k = (
            torch.cat([k_rot_embed, k_pass], dim=-1)
            if k_pass.numel() > 0
            else k_rot_embed
        )
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


class LowRankCache(nn.Module):
    def __init__(self, max_len: int = 128, low_rank_k: int = 16) -> None:
        super().__init__()
        self.max_len: int = max_len
        self.low_rank_k: int = low_rank_k
        self.register_buffer("recent_k", torch.zeros(0), persistent=False)
        self.recent_k: torch.Tensor
        self.register_buffer("recent_v", torch.zeros(0), persistent=False)
        self.recent_v: torch.Tensor
        self.register_buffer("sparse_k", torch.zeros(0), persistent=False)
        self.sparse_k: torch.Tensor
        self.register_buffer("sparse_v", torch.zeros(0), persistent=False)
        self.sparse_v: torch.Tensor
        self.register_buffer("low_rank_k_buf", torch.zeros(0), persistent=False)
        self.low_rank_k_buf: torch.Tensor
        self.register_buffer("low_rank_v_buf", torch.zeros(0), persistent=False)
        self.low_rank_v_buf: torch.Tensor
        self.register_buffer("_cache_proj", torch.zeros(0), persistent=False)
        self._cache_proj: torch.Tensor
        self._update_step: int = 0

    def reset(self) -> None:
        self.recent_k = torch.zeros(
            0, device=self.recent_k.device, dtype=self.recent_k.dtype
        )
        self.recent_v = torch.zeros(
            0, device=self.recent_v.device, dtype=self.recent_v.dtype
        )
        self.sparse_k = torch.zeros(
            0, device=self.sparse_k.device, dtype=self.sparse_k.dtype
        )
        self.sparse_v = torch.zeros(
            0, device=self.sparse_v.device, dtype=self.sparse_v.dtype
        )
        self.low_rank_k_buf = torch.zeros(
            0, device=self.low_rank_k_buf.device, dtype=self.low_rank_k_buf.dtype
        )
        self.low_rank_v_buf = torch.zeros(
            0, device=self.low_rank_v_buf.device, dtype=self.low_rank_v_buf.dtype
        )
        self._update_step = 0

    def update(self, K: torch.Tensor, V: torch.Tensor) -> None:
        B, H, N, D = K.shape
        k_flat = K.detach().reshape(B * H, N, D)
        v_flat = V.detach().reshape(B * H, N, D)

        # Recent cache
        if (
            self.recent_k.numel() == 0
            or self.recent_k.size(0) != B * H
            or self.recent_k.size(-1) != D
        ):
            self.recent_k = k_flat[:, -self.max_len :, :].contiguous().clone()
            self.recent_v = v_flat[:, -self.max_len :, :].contiguous().clone()
        else:
            self.recent_k = (
                torch.cat([self.recent_k, k_flat], dim=1)[:, -self.max_len :, :]
                .contiguous()
                .clone()
            )
            self.recent_v = (
                torch.cat([self.recent_v, v_flat], dim=1)[:, -self.max_len :, :]
                .contiguous()
                .clone()
            )

        # Sparse cache (sample every 4th token)
        sparse_idx = torch.arange(0, N, 4, device=K.device)
        sparse_k = k_flat[:, sparse_idx, :]
        sparse_v = v_flat[:, sparse_idx, :]
        if (
            self.sparse_k.numel() == 0
            or self.sparse_k.size(0) != B * H
            or self.sparse_k.size(-1) != D
        ):
            self.sparse_k = sparse_k[:, -self.max_len // 2 :, :].contiguous().clone()
            self.sparse_v = sparse_v[:, -self.max_len // 2 :, :].contiguous().clone()
        else:
            self.sparse_k = (
                torch.cat([self.sparse_k, sparse_k], dim=1)[:, -self.max_len // 2 :, :]
                .contiguous()
                .clone()
            )
            self.sparse_v = (
                torch.cat([self.sparse_v, sparse_v], dim=1)[:, -self.max_len // 2 :, :]
                .contiguous()
                .clone()
            )

        # Low-rank subspace cache
        if self.recent_k.size(1) >= self.low_rank_k:
            step = getattr(self, "_update_step", 0) + 1
            self._update_step = step
            if (
                step % 50 == 1
                or self.low_rank_k_buf.numel() == 0
                or self.low_rank_k_buf.size(0) != B * H
            ):
                with torch.no_grad():
                    rk = self.recent_k.float()
                    rv = self.recent_v.float()
                    if (
                        self._cache_proj.numel() == 0
                        or self._cache_proj.size(0) != D
                        or self._cache_proj.size(1) != self.low_rank_k
                        or self._cache_proj.device != rk.device
                    ):
                        proj = torch.empty(
                            D, self.low_rank_k, device=rk.device, dtype=torch.float32
                        )
                        nn.init.orthogonal_(proj)
                        self._cache_proj = proj

                    try:
                        y_k = torch.matmul(rk, self._cache_proj)
                        q_orth_k, _ = torch.linalg.qr(y_k)
                        self.low_rank_k_buf = torch.matmul(
                            q_orth_k,
                            torch.matmul(q_orth_k.transpose(-2, -1).contiguous(), rk),
                        ).to(self.recent_k.dtype)

                        y_v = torch.matmul(rv, self._cache_proj)
                        q_orth_v, _ = torch.linalg.qr(y_v)
                        self.low_rank_v_buf = torch.matmul(
                            q_orth_v,
                            torch.matmul(q_orth_v.transpose(-2, -1).contiguous(), rv),
                        ).to(self.recent_v.dtype)
                    except Exception:
                        pass

    def get_cached(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.recent_k.numel() == 0:
            return None, None
        B_H = self.recent_k.size(0)
        tensors_k = [self.recent_k]
        tensors_v = [self.recent_v]
        if self.sparse_k.numel() > 0 and self.sparse_k.size(0) == B_H:
            tensors_k.append(self.sparse_k)
            tensors_v.append(self.sparse_v)
        if self.low_rank_k_buf.numel() > 0 and self.low_rank_k_buf.size(0) == B_H:
            tensors_k.append(self.low_rank_k_buf)
            tensors_v.append(self.low_rank_v_buf)
        return torch.cat(tensors_k, dim=1), torch.cat(tensors_v, dim=1)


class LowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        lora_rank: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        self.lora_rank: int = min(lora_rank, min(in_features, out_features))
        self.lora_scaling: float = (
            (32.0 / self.lora_rank) if self.lora_rank > 0 else 1.0
        )

        self.U: nn.Parameter = nn.Parameter(
            torch.empty(out_features, rank), requires_grad=False
        )
        self.V: nn.Parameter = nn.Parameter(
            torch.empty(rank, in_features), requires_grad=False
        )
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.empty(out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A: nn.Parameter | None = nn.Parameter(
                torch.empty(self.lora_rank, in_features)
            )
            self.lora_B: nn.Parameter | None = nn.Parameter(
                torch.empty(out_features, self.lora_rank)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            u_fp32 = torch.empty_like(self.U, dtype=torch.float32)
            v_fp32 = torch.empty_like(self.V, dtype=torch.float32)
            nn.init.orthogonal_(u_fp32)
            nn.init.orthogonal_(v_fp32)
            self.U.data.copy_(u_fp32.to(self.U.dtype))
            self.V.data.copy_(v_fp32.to(self.V.dtype))
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.U, self.V)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (
                F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            )
            out = out + lora_out
        if self.bias is not None:
            out = out + self.bias
        return out


@ATTENTION_REGISTRY.register("lola_attn", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("lola_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("lola", supports_causal=True, supports_gqa=True)
class LoLAAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        model_config: Any | None = None,
        rank: int | None = None,
        lora_rank: int = 16,
        dropout: float = 0.1,
        comp_config: Any | None = None,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = config if config is not None else model_config

        def _cfg_get(name: str, fallback: Any) -> Any:
            if cfg is not None:
                val = getattr(cfg, name, None)
                if val is not None:
                    return val
            return fallback

        actual_rank = rank if rank is not None else _cfg_get("rank", 32)
        super().__init__(cfg, actual_rank)

        cfg_head_dim = getattr(cfg, "head_dim", None) if cfg is not None else None
        if cfg is not None:
            self.adapter = ModelAdapter(cfg)
            self.num_heads = self.adapter.num_heads or getattr(
                cfg, "num_attention_heads", 12
            )
            self.num_kv_heads = self.adapter.num_key_value_heads or getattr(
                cfg, "num_key_value_heads", self.num_heads
            )
            self.hidden_size = self.adapter.hidden_size or getattr(
                cfg, "hidden_size", 768
            )
            self.head_dim = int(
                cfg_head_dim
                or kwargs.get("head_dim")
                or self.adapter.head_dim
                or (self.hidden_size // self.num_heads)
            )
            self.kv_repeat = self.adapter.kv_repeat or 1
        else:
            self.adapter = None
            self.num_heads = kwargs.get(
                "num_heads", kwargs.get("num_attention_heads", 12)
            )
            self.num_kv_heads = kwargs.get("num_key_value_heads", self.num_heads)
            self.hidden_size = kwargs.get("hidden_size", 768)
            self.head_dim = int(
                kwargs.get("head_dim") or (self.hidden_size // self.num_heads)
            )
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        _mt = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
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
        self.is_decoder = (
            (
                getattr(cfg, "is_decoder", False)
                or getattr(cfg, "causal", False)
                or any(_mt.startswith(p) for p in _decoder_prefixes)
                or bool(kwargs.get("is_decoder", False))
            )
            if cfg
            else bool(kwargs.get("is_decoder", False))
        )

        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.is_nested_encoder: bool = False

        self.rank: int = actual_rank
        self.lora_rank: int = _cfg_get("lora_rank", lora_rank)

        self.q_proj = LowRankLinear(
            self.hidden_size,
            self.all_head_size,
            self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )
        self.k_proj = LowRankLinear(
            self.hidden_size,
            self.kv_head_size,
            self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )
        self.v_proj = LowRankLinear(
            self.hidden_size,
            self.kv_head_size,
            self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )
        self.out_proj = LowRankLinear(
            self.all_head_size,
            self.hidden_size,
            self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )

        self.dropout: nn.Dropout = nn.Dropout(dropout)

        max_seq_len: int = (
            getattr(comp_config, "max_seq_len", 128) if comp_config is not None else 128
        )
        low_rank_cache_size: int = (
            getattr(comp_config, "low_rank_cache_size", 16)
            if comp_config is not None
            else 16
        )
        self.cache: LowRankCache = LowRankCache(
            max_len=max_seq_len, low_rank_k=low_rank_cache_size
        )
        self.LR_scale: nn.Parameter = nn.Parameter(torch.tensor(4.0))
        self.sparse_gate: nn.Parameter = nn.Parameter(torch.tensor(-3.0))
        self.rotary_emb = (
            getattr(pretrained_attn, "rotary_emb", None)
            if pretrained_attn is not None
            else None
        )

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def reset_cache(self) -> None:
        if hasattr(self, "cache") and hasattr(self.cache, "reset"):
            self.cache.reset()

    def _set_lowrank_factors(
        self,
        lowrank: LowRankLinear,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, U.size(1), Vt.size(0))
            sqrt_s = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            lowrank.U.data.zero_()
            lowrank.V.data.zero_()
            out_d, in_d = lowrank.U.size(0), lowrank.V.size(1)
            lowrank.U.data[: min(out_d, U.size(0)), :r].copy_(
                (U[: min(out_d, U.size(0)), :r] * sqrt_s).to(lowrank.U.dtype)
            )
            lowrank.V.data[:r, : min(in_d, Vt.size(1))].copy_(
                (sqrt_s.unsqueeze(1) * Vt[:r, : min(in_d, Vt.size(1))]).to(
                    lowrank.V.dtype
                )
            )
            if lowrank.bias is not None and bias is not None:
                b_val = bias.data if isinstance(bias, nn.Parameter) else bias
                lowrank.bias.data[: min(len(lowrank.bias), len(b_val))].copy_(
                    b_val[: min(len(lowrank.bias), len(b_val))]
                )

            lowrank.U.requires_grad = False
            lowrank.V.requires_grad = False
            if lowrank.bias is not None:
                lowrank.bias.requires_grad = False
            if lowrank.lora_rank > 0 and lowrank.lora_A is not None:
                nn.init.kaiming_uniform_(lowrank.lora_A, a=math.sqrt(5))
                lowrank.lora_A.requires_grad = True
            if lowrank.lora_rank > 0 and lowrank.lora_B is not None:
                nn.init.zeros_(lowrank.lora_B)
                lowrank.lora_B.requires_grad = True

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
            except Exception:
                pass

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None
            or extracted["k"][0] is None
            or extracted["v"][0] is None
        ):
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                if (
                    (c_attn_mod.__class__.__name__ == "Conv1D")
                    or (base_c.__class__.__name__ == "Conv1D")
                    or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                ):
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                q_w = weight[:split_size, :]
                k_w = weight[split_size : 2 * split_size, :]
                v_w = weight[2 * split_size :, :]
                q_b = bias_data[:split_size] if bias_data is not None else None
                k_b = (
                    bias_data[split_size : 2 * split_size]
                    if bias_data is not None
                    else None
                )
                v_b = bias_data[2 * split_size :] if bias_data is not None else None
                extracted["q"] = (q_w, q_b)
                extracted["k"] = (k_w, k_b)
                extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj_mod = src.c_proj
            base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                p_cls = getattr(c_proj_mod, "__class__", type(c_proj_mod)).__name__
                b_cls = getattr(base_o, "__class__", type(base_o)).__name__
                if (
                    ("Conv1D" in p_cls)
                    or ("Conv1D" in b_cls)
                    or hasattr(c_proj_mod, "nf")
                    or hasattr(base_o, "nf")
                    or (
                        w_o.dim() == 2
                        and (
                            w_o.shape[0] < w_o.shape[1]
                            or (
                                w_o.shape[0] == self.hidden_size
                                and w_o.shape[1] == self.hidden_size
                            )
                        )
                    )
                ):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

        # RoBERTa / BERT nested
        if hasattr(src, "self") and (
            extracted["q"][0] is None
            or extracted["k"][0] is None
            or extracted["v"][0] is None
        ):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None:
                    base_mod = getattr(mod, "base_layer", mod)
                    if hasattr(base_mod, "weight"):
                        extracted[proj_key] = (
                            base_mod.weight.data.clone(),
                            (
                                base_mod.bias.data.clone()
                                if getattr(base_mod, "bias", None) is not None
                                else None
                            ),
                        )

        if (
            hasattr(src, "output")
            and hasattr(src.output, "dense")
            and extracted["o"][0] is None
        ):
            dense = src.output.dense
            base_dense = getattr(dense, "base_layer", dense)
            if hasattr(base_dense, "weight"):
                extracted["o"] = (
                    base_dense.weight.data.clone(),
                    (
                        base_dense.bias.data.clone()
                        if getattr(base_dense, "bias", None) is not None
                        else None
                    ),
                )

        # Modern decoders + DistilBERT
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
                        base_l = getattr(layer, "base_layer", layer)
                        if hasattr(base_l, "weight"):
                            extracted[proj] = (
                                base_l.weight.data.clone(),
                                (
                                    base_l.bias.data.clone()
                                    if getattr(base_l, "bias", None) is not None
                                    else None
                                ),
                            )
                            break

        # Nested encoder detection
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
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                )
                or (cls_name.endswith("SelfAttention") and not has_out_proj)
            )
            and not has_out_proj
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(
                "Module copy_weights warning. Proceeding with initialized weights."
            )

        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(
                        f"LoLAAttention extracted weight for {proj_key} has "
                        f"near-zero Frobenius norm ({norm.item():.6f})"
                    )

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(
                    f"LoLAAttention extracted weight for output projection has "
                    f"near-zero Frobenius norm ({norm_o.item():.6f})"
                )

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in [
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    self._set_lowrank_factors(target, w_oriented, b)
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.U.size(0), target.V.size(1), self.rank)
                    target.U.data.zero_()
                    target.V.data.zero_()
                    target.U.data[:min_d, :min_d].fill_diagonal_(1.0)
                    target.V.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

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
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = (
            self.k_proj(kv_input)
            .view(B, N_k, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            self.v_proj(kv_input)
            .view(B, N_k, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        # RoPE
        position_embeddings = kwargs.get("position_embeddings")
        if (
            position_embeddings is not None
            and isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) >= 2
        ):
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = d**-0.5

        _cfg_mt = (
            str(getattr(self.config, "model_type", "") or "").lower()
            if self.config is not None
            else ""
        )
        is_decoder = (
            self.is_decoder
            or (
                getattr(self.config, "is_decoder", False)
                if self.config is not None
                else False
            )
            or any(
                _cfg_mt.startswith(p)
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
        )

        scores_lr = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale

        mask = attention_mask if attention_mask is not None else kwargs.get("mask")
        if mask is not None:
            if mask.dim() == 4:
                if mask.dtype == torch.bool:
                    scores_lr = scores_lr.masked_fill(~mask, -10000.0)
                else:
                    scores_lr = scores_lr + mask.to(scores_lr.dtype)
            elif mask.dim() == 3:
                scores_lr = scores_lr + mask.unsqueeze(1).to(scores_lr.dtype)
            elif mask.dim() == 2:
                valid_m = get_token_valid_mask(mask, B, N_k, device)
                if valid_m is not None:
                    scores_lr = scores_lr.masked_fill(
                        (~valid_m).unsqueeze(1).unsqueeze(2), -10000.0
                    )
            else:
                valid_m = get_token_valid_mask(mask, B, N_k, device)
                if valid_m is not None:
                    scores_lr = scores_lr.masked_fill(
                        (~valid_m).unsqueeze(1).unsqueeze(2), -10000.0
                    )

        if is_decoder and N_k >= N_q:
            causal_tril = (
                torch.tril(
                    torch.ones((N_q, N_k), device=device, dtype=torch.bool),
                    diagonal=N_k - N_q,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores_lr = scores_lr.masked_fill(~causal_tril, -10000.0)

        # =============================================================
        # LL-FIX-01/02/03: cache only for streaming decoders with past_key_value.
        # For encoders (bidirectional), the cache would mix tokens across batches
        # and hold large view-retained buffers, causing ~70% VRAM bloat.
        # =============================================================
        use_cache = bool(
            is_decoder and past_key_value is not None and not is_cross_attention
        )

        if use_cache:
            self.cache.update(K, V)
            cached_k, cached_v = self.cache.get_cached()
        else:
            # Ensure stale state is dropped so batches stay isolated.
            self.cache.reset()
            cached_k, cached_v = None, None

        if (
            cached_k is not None
            and cached_v is not None
            and cached_k.size(0) == (B * H)
        ):
            C = cached_k.size(1)
            K_cache = cached_k.view(B, H, C, d)
            V_cache = cached_v.view(B, H, C, d)
            scores_cache = (
                torch.matmul(Q, K_cache.transpose(-2, -1).contiguous())
                * scale
                * torch.sigmoid(torch.clamp(self.sparse_gate, min=-10.0, max=10.0))
            )
            scores = torch.cat(
                [
                    scores_lr
                    * torch.sigmoid(torch.clamp(self.LR_scale, min=-10.0, max=10.0)),
                    scores_cache,
                ],
                dim=-1,
            )
            V_comb = torch.cat([V, V_cache], dim=2)
        else:
            scores = scores_lr
            V_comb = V

        attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn = attn * head_mask

        context = torch.matmul(attn, V_comb)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("lola_attention_inject")
@ATTENTION_REGISTRY.register("lola_inject")
@ATTENTION_REGISTRY.register("inject_lola_attention")
@ATTENTION_REGISTRY.register("inject_lola")
def inject_lola_attention(
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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(
        target.layers, (torch.nn.ModuleList, list)
    ):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_lola_attention(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
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
        h_size = (
            getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
            if model_config
            else 768
        )
        rank = (
            getattr(comp_config, "rank", kwargs.pop("rank", h_size))
            if comp_config is not None
            else kwargs.pop("rank", h_size)
        )
        kwargs.pop("rank", None)
        new_attn = LoLAAttention(
            config=model_config,
            rank=rank,
            comp_config=comp_config,
            pretrained_attn=old_attn,
            **kwargs,
        )
        if old_attn is not None:
            p = next(old_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = True
            layer.self = new_attn
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
        if old_attn is not None:
            new_attn.train(old_attn.training)
        elif layer is not None and hasattr(layer, "training"):
            new_attn.train(layer.training)
        return new_attn
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"LoLAAttention injection failed: {e}")
        return None


lola_inject = inject_lola_attention
LoLA = LoLAAttention
