#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NEXUS: Neural Efficient X-Scoring Unified Sparse Attention

Unified sparse attention framework combining dynamic token scoring with structured block-level sparsity.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.common import softpick

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv


from .nexus_components import (
    DualStageStripeAttention,
    NEXUSConfig,
    SeerGating,
    SpectraformerRectify,
    XAttentionScorer,
)
from .nexus_chunked import _chunked_causal_nexus
from .nexus_weights import copy_weights
from .nexus_injection import inject_nexus, inject_nexus_attention, inject_nexus_into_distilbert


@ATTENTION_REGISTRY.register("nexus", supports_causal=True, supports_gqa=True)
class NEXUSAttention(BaseAttention):
    """NEXUSAttention neural operator.

    NEXUS: Neural Efficient X-scoring Unified Sparse Attention.

    Attributes:
        supports_causal (bool): Whether causal autoregressive masking is supported.
        is_custom_attention (bool): Flag indicating specialized non-standard attention mechanics.
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        model_config: Any | None = None,
        nexus_cfg: NEXUSConfig | None = None,
        layer_idx: int = 0,
        total_layers: int = 6,
        pretrained_attn: nn.Module | None = None,
        bert_mode: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if model_config is None:
            model_config = config
        super().__init__(model_config)
        self.num_heads: int = getattr(
            model_config,
            "num_attention_heads",
            getattr(model_config, "num_heads", kwargs.get("num_attention_heads", kwargs.get("num_heads", 4))),
        )
        self.hidden_size: int = getattr(
            model_config,
            "hidden_size",
            getattr(model_config, "dim", kwargs.get("hidden_size", kwargs.get("dim", 768))),
        )
        cfg_head_dim = getattr(model_config, "head_dim", None) if model_config else None
        adapter_head_dim = getattr(getattr(model_config, "adapter", None), "head_dim", None) if model_config else None
        self.head_dim: int = int(
            cfg_head_dim
            or adapter_head_dim
            or kwargs.get("head_dim")
            or (self.hidden_size // self.num_heads if self.num_heads > 0 else 64)
        )
        self.all_head_size: int = self.num_heads * self.head_dim

        cfg_kv = (
            getattr(
                model_config,
                "num_key_value_heads",
                getattr(model_config, "num_kv_heads", None),
            )
            if model_config
            else None
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.cfg: NEXUSConfig = nexus_cfg if nexus_cfg is not None else NEXUSConfig()
        self.layer_idx: int = layer_idx
        self.bert_mode: bool = bert_mode
        self.lora_rank: int = kwargs.get("lora_rank", getattr(model_config, "lora_rank", lora_rank))
        self.rank = kwargs.get("rank", getattr(model_config, "rank", None))
        self.lora_alpha: float = lora_alpha

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(getattr(model_config, "model_type", "") or "").lower() if model_config else ""
            self.is_decoder = (
                bool(getattr(model_config, "is_decoder", False) or getattr(model_config, "causal", False))
                or any(
                    _mt.startswith(p)
                    for p in (
                        "gpt",
                        "qwen",
                        "llama",
                        "mistral",
                        "falcon",
                        "bloom",
                        "opt",
                        "gpt_neo",
                        "gpt2",
                    )
                )
                if model_config
                else False
            )

        self.is_nested_encoder: bool = bool(kwargs.get("is_nested_encoder", False))
        self.freeze_base: bool = bool(kwargs.get("freeze_base", bool(getattr(model_config, "freeze_base", False) or (self.lora_rank > 0 and getattr(model_config, "lora_rank", None) is not None))))

        src_p = next(pretrained_attn.parameters(), None) if pretrained_attn is not None else None
        target_dtype = kwargs.get("dtype", src_p.dtype if src_p is not None else torch.float32)
        target_device = kwargs.get("device", src_p.device if src_p is not None else None)

        self.query = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            device=target_device,
            dtype=target_dtype,
        )
        self.key = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            device=target_device,
            dtype=target_dtype,
        )
        self.value = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            device=target_device,
            dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            device=target_device,
            dtype=target_dtype,
        )
        self.dropout = nn.Dropout(self.cfg.dropout)

        self.seer_beta = nn.Parameter(torch.zeros(1, device=target_device, dtype=target_dtype))

        self.dss_attn = DualStageStripeAttention(
            self.cfg,
            self.head_dim,
            num_heads=self.num_heads,
            is_decoder=self.is_decoder,
            is_nested_encoder=self.is_nested_encoder,
            seer_beta=self.seer_beta,
        )

        self.seer_gate: SeerGating | None = None
        if self.cfg.seer_enabled:
            self.seer_gate = SeerGating(
                self.head_dim,
                self.num_heads,
                self.cfg.seer_gate_rank,
                self.cfg.seer_top_k_blocks,
                self.cfg.seer_temperature,
            )

        self.rect_interval = self.cfg.rectify_interval
        self.is_rectify = (
            (self.rect_interval > 0)
            and (layer_idx % self.rect_interval == 0)
            and ((pretrained_attn is None) or getattr(self.cfg, "enable_spectraformer", False))
        )
        self.spec_rectify: SpectraformerRectify | None = None
        if self.is_rectify:
            self.spec_rectify = SpectraformerRectify(self.head_dim, self.cfg.rff_features)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        if target_device is not None and target_dtype is not None:
            self.to(device=target_device, dtype=target_dtype)
        elif target_device is not None:
            self.to(device=target_device)
        elif target_dtype is not None:
            self.to(target_dtype)

    @property
    def q_proj(self):
        return self.query

    @property
    def k_proj(self):
        return self.key

    @property
    def v_proj(self):
        return self.value

    @property
    def q_lin(self):
        return self.query

    @property
    def k_lin(self):
        return self.key

    @property
    def v_lin(self):
        return self.value

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    def _reshape(self, x: torch.Tensor, n_heads: int | None = None) -> torch.Tensor:
        B, N, _ = x.shape
        heads = n_heads if n_heads is not None else self.num_heads
        return x.view(B, N, heads, self.head_dim).transpose(1, 2).contiguous()

    def _get_token_valid_mask(self, mask, B, N, device):
        if mask is None:
            return None
        if mask.dim() == 2:
            if mask.dtype == torch.bool:
                return mask.to(device)
            if mask.dtype.is_floating_point:
                return (mask > -1e4).to(device)
            return (mask > 0).to(device)
        if mask.dim() == 3:
            if mask.dtype == torch.bool:
                v = mask
            elif mask.dtype.is_floating_point:
                v = mask > -1e4
            else:
                v = mask > 0
            return v.any(dim=1).to(device)
        if mask.dim() == 4:
            if mask.dtype == torch.bool:
                v = mask
            elif mask.dtype.is_floating_point:
                v = mask > -1e4
            else:
                v = mask > 0
            return v.any(dim=-2).squeeze(1).to(device)
        return torch.ones(B, N, device=device, dtype=torch.bool)


    _chunked_causal_nexus = _chunked_causal_nexus
    copy_weights = copy_weights

    def freeze_base_weights(self) -> None:
        for mod in (self.query, self.key, self.value, self.out_proj):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    def unfreeze_base_weights(self) -> None:
        for mod in (self.query, self.key, self.value, self.out_proj):
            if hasattr(mod, "unfreeze_base_weights"):
                mod.unfreeze_base_weights()

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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        device = hidden_states.device

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv = kv_input.shape[1]

        q_in = hidden_states
        kv_in = kv_input
        if not torch.is_autocast_enabled() and hasattr(self.query, "weight"):
            w_dtype = self.query.weight.dtype
            if q_in.dtype != w_dtype:
                q_in = q_in.to(w_dtype)
            if kv_in.dtype != w_dtype:
                kv_in = kv_in.to(w_dtype)

        Q = self._reshape(self.query(q_in), self.num_heads)
        K = self._reshape(self.key(kv_in), self.num_kv_heads)
        V = self._reshape(self.value(kv_in), self.num_kv_heads)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len : past_len + N, :]
            sin = sin[..., past_len : past_len + N, :]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat).contiguous()
            V = repeat_kv(V, self.kv_repeat).contiguous()

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None:
            eff_mask = kwargs.get("attn_mask")
        token_valid = self._get_token_valid_mask(eff_mask, B, N_kv, device)

        is_causal = bool(self.is_decoder and not is_cross)
        compute_dtype = Q.dtype

        if is_causal and not is_cross:
            context = self._chunked_causal_nexus(
                Q,
                K,
                V,
                token_valid,
                chunk_size=self.cfg.causal_chunk_size,
                num_landmarks=self.cfg.causal_landmarks,
            )
            if token_valid is not None:
                context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        elif is_cross or (self.cfg.sparse_min_seq_len >= N) or (self.dss_attn is None):
            scale = 1.0 / math.sqrt(max(int(D), 1))
            with torch.amp.autocast(device_type=device.type, enabled=False):
                scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

                if eff_mask is not None:
                    if eff_mask.dim() == 2:
                        scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
                    elif eff_mask.dim() == 3:
                        scores = scores + eff_mask.unsqueeze(1).float().clamp(min=-1e4)
                    elif eff_mask.dim() == 4:
                        scores = scores + eff_mask.float().clamp(min=-1e4)

                scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                if head_mask is not None:
                    hm = head_mask.float()
                    if hm.dim() == 1:
                        hm = hm.view(1, -1, 1, 1)
                    elif hm.dim() == 2:
                        hm = hm.unsqueeze(-1).unsqueeze(-1)
                    elif hm.dim() == 3:
                        hm = hm[:, :, :1].unsqueeze(-1)
                    attn_weights = attn_weights * hm

                context = torch.matmul(attn_weights, V.float()).to(compute_dtype)

        else:
            if self.is_rectify and self.spec_rectify is not None:
                context = self.spec_rectify(Q, K, V)
            else:
                seer_mask = None
                if self.seer_gate is not None:
                    bs = self.cfg.block_size
                    nb = (N + bs - 1) // bs
                    seer_mask = self.seer_gate(Q, K, nb, training=self.training)

                context = self.dss_attn(Q, K, V, token_valid, seer_mask=seer_mask)

            if token_valid is not None:
                context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if not self.is_nested_encoder and self.is_decoder:
            v_flat = V.transpose(1, 2).contiguous().view(B, N, H * D)
            v_rms = v_flat.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            c_rms = context.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            context = context * (v_rms / c_rms).clamp(min=0.25, max=4.0).to(context.dtype)

        if self.is_nested_encoder:
            output = context
        else:
            ctx_in = self.dropout(context)
            if not torch.is_autocast_enabled() and hasattr(self.out_proj, "weight") and ctx_in.dtype != self.out_proj.weight.dtype:
                ctx_in = ctx_in.to(self.out_proj.weight.dtype)
            output = self.out_proj(ctx_in)

        return (output, None)

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for NEXUSAttention."""
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        device = x.device

        if state is None:
            state = {}
        step_idx = state.get("step_idx", 0)

        x_in = x
        if not torch.is_autocast_enabled() and hasattr(self.q_proj, "weight") and x_in.dtype != self.q_proj.weight.dtype:
            x_in = x_in.to(self.q_proj.weight.dtype)

        Q = self.q_proj(x_in).view(B, 1, H, D).transpose(1, 2)
        K = self.k_proj(x_in).view(B, 1, self.num_kv_heads, D).transpose(1, 2)
        V = self.v_proj(x_in).view(B, 1, self.num_kv_heads, D).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        if "k_cache" not in state or state["k_cache"] is None:
            state["k_cache"] = K
            state["v_cache"] = V
        else:
            state["k_cache"] = torch.cat([state["k_cache"].to(K.dtype), K], dim=2)
            state["v_cache"] = torch.cat([state["v_cache"].to(V.dtype), V], dim=2)

        k_all = state["k_cache"]
        v_all = state["v_cache"]
        step_dtype = Q.dtype

        with torch.amp.autocast(device_type=device.type, enabled=False):
            scores = torch.matmul(Q.float(), k_all.float().transpose(-1, -2)) / math.sqrt(D)
            if getattr(self.cfg, "softpick_enabled", False):
                attn_w = softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature)
            else:
                scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
                attn_w = F.softmax(scores, dim=-1)
            context = torch.matmul(attn_w, v_all.float()).to(step_dtype)

        state["step_idx"] = step_idx + 1

        context = context.transpose(1, 2).contiguous().view(B, 1, H * D)
        v_flat = V.transpose(1, 2).contiguous().view(B, 1, H * D)
        v_rms = v_flat.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        c_rms = context.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        context = context * (v_rms / c_rms).clamp(min=0.25, max=4.0).to(step_dtype)

        if self.is_nested_encoder:
            out = context
        else:
            if not torch.is_autocast_enabled() and hasattr(self.out_proj, "weight") and context.dtype != self.out_proj.weight.dtype:
                context = context.to(self.out_proj.weight.dtype)
            out = self.out_proj(context)

        if is_2d:
            out = out.squeeze(1)

        return out, state

    def get_ortho_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.query.weight.device, dtype=self.query.weight.dtype)
        for proj in (self.query, self.key):
            if hasattr(proj, "weight") and proj.weight is not None:
                W = proj.weight
                diff = torch.matmul(W, W.t()) - torch.eye(W.size(0), device=W.device, dtype=W.dtype)
                loss = loss + torch.norm(diff, p="fro") / (W.size(0) ** 2)
        return loss




def softpick(scores: torch.Tensor, dim: int = -1, tau: float = 1.0) -> torch.Tensor:
    return F.softmax(scores / tau, dim=dim)


__all__ = [
    NEXUSConfig,
    NEXUSAttention,
    DualStageStripeAttention,
    SeerGating,
    SpectraformerRectify,
    XAttentionScorer,
    softpick,
    apply_rotary_pos_emb,
    inject_nexus,
    inject_nexus_attention,
]
