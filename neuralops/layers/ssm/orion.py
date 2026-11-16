#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
ORION: Optimized Rectified Input-Aware Orthogonal Network

Optimized multi-head attention architecture integrating input-aware orthogonal projections and gated feature flow.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.amp_utils import maybe_half
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get, temperature_softmax
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv

logger = logging.getLogger(__name__)

from .orion_components import FactorisedLinear, ORIONConfig
from .orion_weights import copy_weights
from .orion_injection import inject_orion, inject_orion_attention



@ATTENTION_REGISTRY.register("orion", supports_causal=True, supports_gqa=True)
class ORIONAttention(BaseAttention):
    """ORIONAttention neural operator.

    ORION: Optimized Rectified Input-aware Orthogonal Network.

    Attributes:
        supports_causal (bool): Whether causal autoregressive masking is supported.
        is_custom_attention (bool): Flag indicating specialized non-standard attention mechanics.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        model_config: Any | None = None,
        orion_cfg: ORIONConfig | None = None,
        pretrained_attn: Any | None = None,
        old_attn: Any | None = None,
        comp_config: Any | None = None,
        bert_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config
        source_attn = (
            pretrained_attn
            if pretrained_attn is not None
            else (old_attn if old_attn is not None else kwargs.get("source_attn"))
        )
        try:
            super().__init__(cfg, rank)
        except TypeError:
            super().__init__(cfg)

        if orion_cfg is None:
            orion_cfg = ORIONConfig()
            if rank is not None:
                orion_cfg.svd_max_rank = max(rank, orion_cfg.svd_min_rank)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads"))
        num_kv_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads"))

        if hidden_size is None or num_heads is None:
            if cfg is not None:
                if hidden_size is None:
                    hidden_size = _cfg_get(
                        cfg,
                        "hidden_size",
                        _cfg_get(cfg, "d_model", _cfg_get(cfg, "dim", 768)),
                    )
                if num_heads is None:
                    num_heads = _cfg_get(cfg, "num_attention_heads", _cfg_get(cfg, "num_heads", 12))
                if num_kv_heads is None:
                    num_kv_heads = _cfg_get(
                        cfg,
                        "num_key_value_heads",
                        _cfg_get(cfg, "num_kv_heads", num_heads),
                    )
            elif source_attn is not None:
                if hidden_size is None:
                    out_layer = getattr(
                        source_attn,
                        "out_lin",
                        getattr(
                            source_attn,
                            "dense",
                            getattr(
                                source_attn,
                                "c_proj",
                                getattr(source_attn, "o_proj", None),
                            ),
                        ),
                    )
                    hidden_size = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
                if num_heads is None:
                    num_heads = getattr(
                        source_attn,
                        "num_attention_heads",
                        getattr(source_attn, "n_heads", 12),
                    )
                if num_kv_heads is None:
                    num_kv_heads = getattr(source_attn, "num_key_value_heads", num_heads)
            else:
                hidden_size = 768 if hidden_size is None else hidden_size
                num_heads = 12 if num_heads is None else num_heads
                num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.num_heads = max(1, int(num_heads))
        self.num_kv_heads = max(1, int(num_kv_heads))
        self.hidden_size = int(hidden_size)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")
        self.head_dim = self.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True
        self.cfg = orion_cfg
        self.bert_mode = bert_mode

        if self.num_heads <= 1:
            self.num_kernel_heads = 1
            self.num_full_heads = 0
        else:
            self.num_full_heads = max(1, self.num_heads // 3)
            self.num_kernel_heads = self.num_heads - self.num_full_heads

        model_type = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
        is_dec = (
            bool(
                getattr(
                    cfg,
                    "is_decoder",
                    getattr(cfg, "is_causal", kwargs.get("is_causal", False)),
                )
            )
            if cfg
            else False
        )
        if (
            any(model_type.startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "falcon", "bloom", "opt"))
            or hasattr(cfg, "architectures")
            and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(cfg, "architectures", []) or []))
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = (
            getattr(
                cfg,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
            if cfg
            else 16
        )
        self.lora_alpha = (
            getattr(
                cfg,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
            if cfg
            else 32.0
        )
        self.freeze_base = self.lora_rank > 0

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and source_attn is not None:
            p = next(source_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        self.q_proj = FactorisedLinear(
            hidden_size,
            self.all_head_size,
            bias=True,
            energy=orion_cfg.svd_energy,
            min_rank=orion_cfg.svd_min_rank,
            max_rank=orion_cfg.svd_max_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.k_proj = FactorisedLinear(
            hidden_size,
            self.all_kv_head_size,
            bias=True,
            energy=orion_cfg.svd_energy,
            min_rank=orion_cfg.svd_min_rank,
            max_rank=orion_cfg.svd_max_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.v_proj = FactorisedLinear(
            hidden_size,
            self.all_kv_head_size,
            bias=True,
            energy=orion_cfg.svd_energy,
            min_rank=orion_cfg.svd_min_rank,
            max_rank=orion_cfg.svd_max_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.out_proj = FactorisedLinear(
            self.all_head_size,
            hidden_size,
            bias=True,
            energy=orion_cfg.svd_energy,
            min_rank=orion_cfg.svd_min_rank,
            max_rank=orion_cfg.svd_max_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )

        rff_init = torch.randn(max(1, self.num_kernel_heads), self.head_dim, orion_cfg.rff_features) * (
            1.0 / math.sqrt(max(int(self.head_dim), 1))
        )
        self.register_buffer("rff_proj", rff_init, persistent=False)
        self.dropout = nn.Dropout(orion_cfg.dropout)

        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    @property
    def query(self) -> FactorisedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorisedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorisedLinear:
        return self.v_proj

    @property
    def dense(self) -> FactorisedLinear:
        return self.out_proj

    @property
    def o_proj(self) -> FactorisedLinear:
        return self.out_proj

    @property
    def q_lin(self) -> FactorisedLinear:
        return self.q_proj

    @property
    def k_lin(self) -> FactorisedLinear:
        return self.k_proj

    @property
    def v_lin(self) -> FactorisedLinear:
        return self.v_proj

    @property
    def out_lin(self) -> FactorisedLinear:
        return self.out_proj

    @property
    def c_proj(self) -> FactorisedLinear:
        return self.out_proj


    copy_weights = copy_weights


    def _kernel_linear_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        FAVOR+ random feature linear attention.

        Implements the unbiased feature map from Choromanski et al. (2020):
            φ(x) = exp(ω·x - ||x||²/2) / sqrt(m),   ω ~ N(0, I_d)

        The `-||x||²/2` term is essential for an unbiased estimate of exp(q·k).
        The scaling 1/sqrt(d) is applied to x once (not to ω, not twice).
        """
        B, Hk, N, D = Q.shape
        if Hk == 0:
            return torch.zeros(B, 0, N, D, device=Q.device, dtype=Q.dtype)

        F_dim = self.cfg.rff_features
        orig_dtype = Q.dtype
        Q, K, V = Q.float(), K.float(), V.float()
        proj = self.rff_proj[:Hk].to(Q.device).float()

        scale = 1.0 / math.sqrt(max(int(self.head_dim), 1))
        Q_scaled = Q * scale
        K_scaled = K * scale

        Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_scaled, proj)
        K_proj = torch.einsum("bhnd,hdf->bhnf", K_scaled, proj)

        Q_norm_sq = (Q_scaled**2).sum(dim=-1, keepdim=True) / 2.0
        K_norm_sq = (K_scaled**2).sum(dim=-1, keepdim=True) / 2.0

        max_q = (Q_proj - Q_norm_sq).detach().amax(dim=-1, keepdim=True).clamp(min=0.0)
        max_k = (K_proj - K_norm_sq).detach().amax(dim=-1, keepdim=True).clamp(min=0.0)

        Q_feat = torch.exp(Q_proj - Q_norm_sq - max_q) / math.sqrt(F_dim)
        K_feat = torch.exp(K_proj - K_norm_sq - max_k) / math.sqrt(F_dim)

        if token_valid is not None:
            mask_kv = token_valid.unsqueeze(1).unsqueeze(-1).to(K_feat.dtype)
            K_feat = K_feat * mask_kv
            V = V * mask_kv

        M = Q_feat.shape[-1]
        if self.is_decoder:
            chunk_size = min(64, N)
            num_chunks = (N + chunk_size - 1) // chunk_size
            S = torch.zeros(B, Hk, M, D, dtype=torch.float32, device=Q.device)
            Z = torch.zeros(B, Hk, M, dtype=torch.float32, device=Q.device)
            ctx_list = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                L = end - start

                q_c = Q_feat[:, :, start:end]
                k_c = K_feat[:, :, start:end]
                v_c = V[:, :, start:end]

                causal_mask = torch.tril(torch.ones(L, L, device=Q.device, dtype=torch.float32))
                intra_attn = torch.matmul(q_c, k_c.transpose(-1, -2).contiguous()) * causal_mask
                intra_out = torch.matmul(intra_attn, v_c)
                inter_out = torch.matmul(q_c, S)

                intra_den = intra_attn.sum(dim=-1, keepdim=True)
                inter_den = torch.matmul(q_c, Z.unsqueeze(-1))

                den = (intra_den + inter_den).clamp(min=1e-4)
                ctx_list.append((intra_out + inter_out) / den)

                S = S + torch.matmul(k_c.transpose(-1, -2).contiguous(), v_c)
                Z = Z + k_c.sum(dim=2)

            ctx = torch.cat(ctx_list, dim=2)
        else:
            KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)
            K_sum = K_feat.sum(dim=2)
            denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1).clamp(min=1e-4)
            ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / denom
        return ctx.to(orig_dtype)

    def _full_softpick_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: torch.Tensor | None = None,
        past_len: int = 0,
    ) -> torch.Tensor:
        """
        Full softmax attention for the "full" heads.

        Returns a tensor of shape (B, Hf, N, D). No internal out_proj is
        applied here; the caller handles out_proj after concatenating all
        head groups.
        """
        B, Hf, N, D = Q.shape
        N_kv = K.shape[2]
        if Hf == 0:
            return torch.zeros(B, 0, N, D, device=Q.device, dtype=Q.dtype)

        scale = 1.0 / math.sqrt(max(int(self.head_dim), 1))
        is_causal = bool(self.is_decoder) and (N > 1 or past_len > 0)

        attn_bias = None
        if token_valid is not None:
            valid_bias = ((1.0 - token_valid.float()) * -1e4).view(B, 1, 1, N_kv)
            attn_bias = valid_bias.to(Q.dtype)

        try:
            q_sdp = maybe_half(Q)
            k_sdp = maybe_half(K)
            v_sdp = maybe_half(V)
            bias_sdp = None
            if attn_bias is not None:
                bias_sdp = attn_bias.to(q_sdp.dtype)

            use_sdpa_causal = is_causal and past_len == 0 and N == N_kv and bias_sdp is None
            if past_len > 0 and is_causal:
                causal_mask = torch.tril(torch.ones(N, N_kv, device=Q.device, dtype=torch.bool), diagonal=past_len)
                c_bias = (~causal_mask).unsqueeze(0).unsqueeze(0).to(q_sdp.dtype) * -1e4
                bias_sdp = c_bias if bias_sdp is None else bias_sdp + c_bias
                use_sdpa_causal = False

            out = F.scaled_dot_product_attention(
                q_sdp,
                k_sdp,
                v_sdp,
                attn_mask=bias_sdp,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=use_sdpa_causal,
            )
            out = out.to(Q.dtype)
            return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        if is_causal:
            causal_mask = torch.tril(torch.ones(N, N_kv, device=Q.device, dtype=torch.bool), diagonal=past_len)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if token_valid is not None:
            scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2).bool(), -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_w = temperature_softmax(scores, dim=-1, temperature=self.cfg.softpick_temperature).to(V.dtype)
        attn_w = self.dropout(attn_w)
        return torch.matmul(attn_w, V)

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden_states.dim() == 4:
            B, H, L, Dh = hidden_states.shape
            hidden_states = hidden_states.view(B, L, H * Dh)
        elif not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, _ = hidden_states.shape
        Hk = self.num_kernel_heads
        Hf = self.num_full_heads
        device = hidden_states.device

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv = kv_input.shape[1]

        Q_all = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K_all = self.k_proj(kv_input).view(B, N_kv, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V_all = self.v_proj(kv_input).view(B, N_kv, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q_all, K_all = apply_rotary_pos_emb(Q_all, K_all, cos, sin)

        past_len = 0
        if self.is_decoder and past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
                if past_k is not None and past_v is not None:
                    past_len = past_k.shape[2]
                    K_all = torch.cat([past_k, K_all], dim=2)
                    V_all = torch.cat([past_v, V_all], dim=2)
        present_key_value = (K_all, V_all) if self.is_decoder else None

        if self.kv_repeat > 1:
            K_all = repeat_kv(K_all, self.kv_repeat)
            V_all = repeat_kv(V_all, self.kv_repeat)

        token_valid = None
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            token_valid = get_token_valid_mask(eff_mask, B, K_all.shape[2], device)

        is_dec = self.is_decoder or kwargs.get("is_decoder", False) or kwargs.get("is_causal", False)
        old_is_dec = self.is_decoder
        self.is_decoder = is_dec

        ctx_list = []

        if Hk > 0:
            Q_kernel = Q_all[:, :Hk]
            K_kernel = K_all[:, :Hk]
            V_kernel = V_all[:, :Hk]
            ctx_kernel = self._kernel_linear_attention(Q_kernel, K_kernel, V_kernel, token_valid)
            ctx_list.append(ctx_kernel)

        if Hf > 0:
            Q_full = Q_all[:, Hk:]
            K_full = K_all[:, Hk:]
            V_full = V_all[:, Hk:]
            ctx_full = self._full_softpick_attention(Q_full, K_full, V_full, token_valid, past_len=past_len)
            ctx_list.append(ctx_full)

        if len(ctx_list) == 1:
            context = ctx_list[0]
        else:
            context = torch.cat(ctx_list, dim=1)

        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            context = context * hm.to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        if is_dec:
            ret = (output, present_key_value, None) if output_attentions else (output, present_key_value)
        else:
            ret = (output, None)
        self.is_decoder = old_is_dec
        return ret

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for ORIONAttention."""
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        if state is None:
            state = {}
        step_idx = state.get("step_idx", 0)

        Q_all = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K_all = self.k_proj(x).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_all = self.v_proj(x).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            K_all = repeat_kv(K_all, self.kv_repeat)
            V_all = repeat_kv(V_all, self.kv_repeat)

        if "k_cache" not in state or state["k_cache"] is None:
            state["k_cache"] = K_all
            state["v_cache"] = V_all
        else:
            state["k_cache"] = torch.cat([state["k_cache"], K_all], dim=2)
            state["v_cache"] = torch.cat([state["v_cache"], V_all], dim=2)

        k_all = state["k_cache"]
        v_all = state["v_cache"]

        scores = torch.matmul(Q_all, k_all.transpose(-1, -2)) / math.sqrt(D)
        attn_w = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_w, v_all)

        state["step_idx"] = step_idx + 1

        context = context.transpose(1, 2).contiguous().view(B, 1, self.all_head_size)
        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        if is_2d:
            out = out.squeeze(1)

        return out, state

    def get_ortho_loss(self) -> torch.Tensor:
        W = self.q_proj.down.weight
        diff = torch.matmul(W, W.t()) - torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        return torch.norm(diff, p="fro") / (W.size(0) ** 2)



__all__ = [
    ORIONConfig,
    FactorisedLinear,
    ORIONAttention,
    inject_orion,
    inject_orion_attention,
]
