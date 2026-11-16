#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
xAttention Fast Kernel Module

High-throughput accelerated attention kernel utilizing block-sparse tiling and hardware-level fused operations.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv
from neuralops.core.utils.rope import _rotate_half

try:
    from neuralops.core.triton_sparse_attn import HAS_TRITON, triton_block_sparse_attention
except Exception:
    triton_block_sparse_attention = None
    HAS_TRITON = False

from neuralops.layers.fast_kernels._xattention.masks import _extract_valid_token_mask, _extract_valid_query_mask
from neuralops.layers.fast_kernels._xattention.weight_transfer import XAttentionWeightTransferMixin
from neuralops.layers.fast_kernels._xattention.injection import inject_xattention

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "xattention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class XAttention(XAttentionWeightTransferMixin, BaseAttention):
    """
    Block-sparse attention with coarse-to-fine routing.

    Pipeline:
        1. Project Q, K, V (with LoRA)
        2. Apply RoPE (with past_len slicing)
        3. Concatenate KV-cache
        4. If N_k is small or keep_ratio == 1 → dense attention
           Otherwise → block-level coarse routing → top-k blocks
        5. Fine-grained attention within selected blocks
        6. Optional Triton kernel acceleration
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "sparse_attn"

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        block_size: int = 32,
        sparsity: float = 0.5,
        lora_rank: int = 16,
        pretrained_attn: nn.Module | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        d_model = (
            kwargs.get("dim")
            or kwargs.get("hidden_size")
            or _cfg_get(config, "hidden_size")
            or _cfg_get(config, "d_model")
            or _cfg_get(config, "dim")
            or 768
        )
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        num_heads = (
            kwargs.get("num_heads")
            or _cfg_get(config, "num_attention_heads")
            or _cfg_get(config, "num_heads")
            or _cfg_get(config, "n_heads")
            or 12
        )
        num_kv_heads = (
            kwargs.get("num_kv_heads")
            or _cfg_get(config, "num_key_value_heads")
            or _cfg_get(config, "num_kv_heads")
            or num_heads
        )

        self.hidden_size = int(d_model)
        self.num_heads = int(num_heads)
        self.num_kv_heads = max(1, min(int(num_kv_heads), self.num_heads))
        self.head_dim = self.hidden_size // max(1, self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        mt = str(_cfg_get(config, "model_type", "") or "").lower()
        _dec_pfx = (
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
        is_dec = (
            bool(_cfg_get(config, "is_decoder", False))
            or bool(_cfg_get(config, "is_causal", False))
            or any(mt.startswith(p) for p in _dec_pfx)
            or kwargs.get("is_decoder", False)
            or kwargs.get("is_causal", False)
        )
        if pretrained_attn is not None:
            if getattr(pretrained_attn, "is_decoder", False) or hasattr(pretrained_attn, "c_attn"):
                is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_nested_encoder: bool = False
        self.layer_idx = int(layer_idx if layer_idx is not None else kwargs.get("layer_idx", 0))

        self.block_size: int = max(
            0,
            int(kwargs.get("block_size", _cfg_get(config, "xattn_block_size", block_size))),
        )
        raw_keep = kwargs.get("keep_ratio", _cfg_get(config, "xattn_keep_ratio", None))
        if raw_keep is not None:
            self.keep_ratio: float = float(raw_keep)
            self.sparsity: float = 1.0 - self.keep_ratio
        else:
            self.sparsity = float(kwargs.get("sparsity", _cfg_get(config, "xattn_sparsity", sparsity)))
            self.keep_ratio = 1.0 - self.sparsity
        self.keep_ratio = max(0.0, min(1.0, self.keep_ratio))

        self.lora_rank = int(kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank)))

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and pretrained_attn is not None:
            try:
                target_dtype = next(pretrained_attn.parameters()).dtype
            except StopIteration:
                target_dtype = torch.float32
        if target_dtype is None:
            target_dtype = torch.float32

        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.out_proj: nn.Module = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )

        self.lora_output_scale = nn.Parameter(torch.ones(1, dtype=target_dtype))

        dropout_p = _cfg_get(
            config,
            "attention_dropout",
            _cfg_get(
                config,
                "dropout",
                _cfg_get(
                    config,
                    "hidden_dropout_prob",
                    _cfg_get(config, "attention_probs_dropout_prob", 0.0 if is_dec else 0.1),
                ),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)
        self.triton_kwargs = {"block_size": self.block_size, "eps": 1e-6}

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        self.to(target_dtype)

    @property
    def output_scale(self) -> nn.Parameter:
        return self.lora_output_scale

    @property
    def query(self) -> nn.Module:
        return self.q_proj

    @property
    def key(self) -> nn.Module:
        return self.k_proj

    @property
    def value(self) -> nn.Module:
        return self.v_proj

    @property
    def W_q(self):
        return self.q_proj

    @property
    def W_k(self):
        return self.k_proj

    @property
    def W_v(self):
        return self.v_proj

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

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
    ) -> tuple[torch.Tensor, Any]:
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        if key_value_states is not None:
            raise NotImplementedError("XAttention does not support cross-attention.")

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N_q, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        past_len = 0

        if past_key_value is not None and past_key_value is not False:
            if isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) >= 2 and isinstance(past_key_value[0], torch.Tensor):
                    past_k, past_v = past_key_value[0], past_key_value[1]
                    past_len = past_k.shape[-2] if past_k is not None else 0
            elif hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
                if len(past_key_value.key_cache) > self.layer_idx:
                    past_k = past_key_value.key_cache[self.layer_idx]
                    past_v = past_key_value.value_cache[self.layer_idx]
                    past_len = past_k.shape[-2] if past_k is not None else 0

        proj_dtype = self.q_proj.weight.dtype if hasattr(self.q_proj, "weight") else dtype
        hs = (
            hidden_states
            if (torch.is_autocast_enabled() or hidden_states.dtype == proj_dtype)
            else hidden_states.to(proj_dtype)
        )
        Q = self.q_proj(hs).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K_new = self.k_proj(hs).view(B, N_q, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V_new = self.v_proj(hs).view(B, N_q, self.num_kv_heads, d).transpose(1, 2).contiguous()

        pos_emb = kwargs.get("position_embeddings") if "position_embeddings" in kwargs else None
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            Q, K_new = apply_rotary_pos_emb(
                Q,
                K_new,
                cos,
                sin,
                position_ids=kwargs.get("position_ids"),
                past_len=past_len,
            )
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                position_ids = kwargs.get("position_ids")
                if position_ids is None and past_len > 0:
                    position_ids = torch.arange(past_len, past_len + N_q, device=device).unsqueeze(0).expand(B, -1)
                cos, sin = self.rotary_emb(K_new, seq_len=past_len + N_q)
                Q, K_new = apply_rotary_pos_emb(
                    Q,
                    K_new,
                    cos,
                    sin,
                    position_ids=position_ids,
                    past_len=past_len,
                )
            except Exception as e:
                logger.debug("XAttention rotary_emb failed: %s", e)

        if past_k is not None and past_v is not None:
            K_cat = torch.cat([past_k.to(K_new.dtype), K_new], dim=-2)
            V_cat = torch.cat([past_v.to(V_new.dtype), V_new], dim=-2)
        else:
            K_cat, V_cat = K_new, V_new

        new_past_key_value: Any = None
        use_cache = kwargs.get("use_cache", False) or (past_key_value is not None and past_key_value is not False)
        if use_cache:
            if hasattr(past_key_value, "update") and past_key_value is not None:
                try:
                    past_key_value.update(K_new, V_new, self.layer_idx)
                    new_past_key_value = past_key_value
                except Exception:
                    new_past_key_value = (K_cat, V_cat)
            else:
                new_past_key_value = (K_cat, V_cat)

        N_k = K_cat.shape[-2]

        if self.kv_repeat > 1:
            K_full = repeat_kv(K_cat, self.kv_repeat)
            V_full = repeat_kv(V_cat, self.kv_repeat)
        else:
            K_full, V_full = K_cat, V_cat

        scale = 1.0 / math.sqrt(max(int(d), 1))
        attn: torch.Tensor | None = None
        context: torch.Tensor | None = None

        use_dense = (
            self.block_size <= 0
            or self.keep_ratio >= 1.0
            or N_k <= self.block_size
            or N_q <= self.block_size
            or N_q == 1
        )

        valid_m = _extract_valid_token_mask(attention_mask, B, N_k, device) if attention_mask is not None else None
        block_mask: torch.Tensor | None = None
        bsz = self.block_size

        if not use_dense:
            pad_q = (bsz - (N_q % bsz)) % bsz
            pad_k = (bsz - (N_k % bsz)) % bsz

            Q_f_pad = F.pad(Q.float(), (0, 0, 0, pad_q)) if pad_q > 0 else Q.float()
            K_f_pad = F.pad(K_full.float(), (0, 0, 0, pad_k)) if pad_k > 0 else K_full.float()

            total_q = Q_f_pad.shape[-2]
            total_k = K_f_pad.shape[-2]
            num_q_blocks = total_q // bsz
            num_k_blocks = total_k // bsz

            q_valid = torch.zeros((B, 1, total_q, 1), device=device, dtype=torch.float32)
            q_valid[:, :, :N_q, :] = 1.0
            k_valid = torch.zeros((B, 1, total_k, 1), device=device, dtype=torch.float32)
            k_valid[:, :, :N_k, :] = 1.0

            if valid_m is not None:
                k_valid[:, :, :N_k, 0] = k_valid[:, :, :N_k, 0] * valid_m.unsqueeze(1).float()

            q_valid_blk = q_valid.view(B, 1, num_q_blocks, bsz, 1)
            Q_blocks = (Q_f_pad.view(B, H, num_q_blocks, bsz, d) * q_valid_blk).sum(dim=3) / q_valid_blk.sum(
                dim=3
            ).clamp(min=1.0)

            k_valid_blk = k_valid.view(B, 1, num_k_blocks, bsz, 1)
            k_blk_counts = k_valid_blk.sum(dim=3)
            K_blocks = (K_f_pad.view(B, H, num_k_blocks, bsz, d) * k_valid_blk).sum(dim=3) / k_blk_counts.clamp(
                min=1.0
            )
            k_blk_has_valid = (k_blk_counts > 0).view(B, 1, 1, num_k_blocks)

            coarse_scores = torch.matmul(Q_blocks * scale, K_blocks.transpose(-1, -2).contiguous())
            neg_inf_coarse = -65500.0 if coarse_scores.dtype == torch.float16 else -1e4
            coarse_scores = coarse_scores.masked_fill(~k_blk_has_valid, neg_inf_coarse).clamp(min=-65500.0, max=65500.0)

            if self.is_decoder:
                q_pos_max = past_len + torch.clamp(
                    (torch.arange(num_q_blocks, device=device) + 1) * bsz - 1,
                    max=N_q - 1,
                )
                q_pos_min = past_len + torch.arange(num_q_blocks, device=device) * bsz
                q_blk_max = (q_pos_max // bsz).unsqueeze(1)
                q_blk_min = (q_pos_min // bsz).unsqueeze(1)
                k_blk = torch.arange(num_k_blocks, device=device).unsqueeze(0)
                causal_blk = (k_blk <= q_blk_max).unsqueeze(0).unsqueeze(0)
                diag_blk = ((k_blk >= q_blk_min) & (k_blk <= q_blk_max)).unsqueeze(0).unsqueeze(0)
                coarse_scores = coarse_scores.masked_fill(~causal_blk, neg_inf_coarse).clamp(min=-65500.0, max=65500.0)
            else:
                causal_blk = None
                diag_blk = None

            k_blocks_keep = max(
                1,
                min(num_k_blocks, int(math.ceil(num_k_blocks * self.keep_ratio))),
            )
            _, topk_idx = torch.topk(coarse_scores, k=k_blocks_keep, dim=-1)

            block_mask = torch.zeros((B, H, num_q_blocks, num_k_blocks), device=device, dtype=torch.bool)
            block_mask.scatter_(-1, topk_idx, True)

            if self.is_decoder and causal_blk is not None and diag_blk is not None:
                block_mask = (block_mask & causal_blk) | diag_blk

            from neuralops.ops.triton._types import FORCE_PYTORCH_FALLBACK

            # Triton block-sparse kernel ignores token_mask and uses FP16 atomic_add in backward
            # which overflows under AMP GradScaler; restrict Triton fast-path to unmasked inference.
            can_use_triton = (
                not FORCE_PYTORCH_FALLBACK
                and HAS_TRITON
                and Q.is_cuda
                and triton_block_sparse_attention is not None
                and not torch.is_grad_enabled()
                and valid_m is None
                and head_mask is None
                and past_len == 0
                and Q.dtype in (torch.float16, torch.bfloat16)
                and bsz in (16, 32, 64, 128)
            )
            if can_use_triton:
                try:
                    cand_ctx = triton_block_sparse_attention(
                        q=Q,
                        k=K_full,
                        v=V_full,
                        block_mask=block_mask,
                        block_size=bsz,
                        token_mask=None,
                        is_causal=self.is_decoder,
                    )
                    if torch.isfinite(cand_ctx).all():
                        context = cand_ctx
                except (RuntimeError, NotImplementedError, ValueError) as e:
                    from neuralops.utils.logging_config import warn_once

                    warn_once(
                        "xattention_triton_sparse_fallback",
                        "Triton block-sparse attention failed (%s); falling back to PyTorch.",
                        e,
                    )
                    context = None

        if context is None:
            # Compute scaled Q @ K^T and softmax in FP32 to prevent FP16 pre-scale overflow
            # and preserve exact gradient flow under AMP GradScaler.
            Q_f = Q.float() * scale
            K_f = K_full.float()
            V_f = V_full.float()
            scores = torch.matmul(Q_f, K_f.transpose(-2, -1))

            attn_valid = torch.ones((1, 1, N_q, N_k), device=device, dtype=torch.bool)
            if block_mask is not None:
                expanded_bm = block_mask.repeat_interleave(bsz, dim=2).repeat_interleave(bsz, dim=3)[
                    :, :, :N_q, :N_k
                ]
                attn_valid = attn_valid & expanded_bm

            if self.is_decoder and (N_q > 1 or past_len > 0):
                q_pos = (past_len + torch.arange(N_q, device=device)).unsqueeze(1)
                k_pos = torch.arange(N_k, device=device).unsqueeze(0)
                causal = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
                attn_valid = attn_valid & causal

            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    am = attention_mask.to(device=device)
                    if am.dtype == torch.bool:
                        attn_valid = attn_valid & am
                    elif am.dtype.is_floating_point:
                        if (am < -1.0).any():
                            attn_valid = attn_valid & (am >= -1.0)
                        elif not (am <= 0.0).all():
                            attn_valid = attn_valid & (am > 0.5)
                    else:
                        attn_valid = attn_valid & (am > 0)
                elif valid_m is not None:
                    attn_valid = attn_valid & valid_m.unsqueeze(1).unsqueeze(2)

            row_has_valid = attn_valid.any(dim=-1, keepdim=True)
            neg_inf_scores = -65500.0 if scores.dtype == torch.float16 else -1e4
            scores = scores.masked_fill(~attn_valid, neg_inf_scores).clamp(min=-65500.0, max=65500.0)
            attn = F.softmax(scores, dim=-1)
            attn = attn * row_has_valid.to(attn.dtype)
            attn = self.dropout(attn)

            if head_mask is not None:
                hm = head_mask.to(device=device, dtype=attn.dtype)
                if hm.dim() == 1:
                    hm = hm.view(1, -1, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(-1)
                attn = attn * hm

            context = torch.matmul(attn, V_f).to(dtype)

        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder or isinstance(self.out_proj, nn.Identity):
            output = context.to(dtype) * self.lora_output_scale.to(dtype)
        else:
            out_w_dtype = self.out_proj.weight.dtype if hasattr(self.out_proj, "weight") else dtype
            ctx_in = (
                context
                if (torch.is_autocast_enabled() or context.dtype == out_w_dtype)
                else context.to(out_w_dtype)
            )
            scale_dt = ctx_in.dtype if not torch.is_autocast_enabled() else dtype
            output = self.out_proj(ctx_in) * self.lora_output_scale.to(scale_dt)

        if dtype == torch.float16:
            fmax = 65000.0
            output = output.clamp(min=-fmax, max=fmax)

        if attention_mask is not None:
            v_m = _extract_valid_query_mask(attention_mask, B, N_q, device)
            output = output * v_m.unsqueeze(-1).to(output.dtype)

        if output_attentions:
            return output.to(dtype), (new_past_key_value if self.is_decoder else None), (
                attn.to(dtype) if attn is not None else None
            )
        return output.to(dtype), (new_past_key_value if self.is_decoder else None)



__all__ = [
    XAttention,
    LoRALinear,
    inject_xattention,
    apply_rotary_pos_emb,
]
