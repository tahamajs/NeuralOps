#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FlashAttention-3 FP8 Kernel Module

Low-precision FP8 FlashAttention-3 implementation maximizing tensor core throughput and memory efficiency.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.common import repeat_kv

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

from neuralops.layers.fast_kernels._flash_attention3_fp8.weight_transfer import FlashAttention3FP8WeightTransferMixin
from neuralops.layers.fast_kernels._flash_attention3_fp8.injection import inject_flash3_fp8


from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

# Denormalized float flush-to-zero threshold
_FTZ_THRESHOLD: float = 1e-15


def _flush_denormals(t: torch.Tensor) -> torch.Tensor:
    r"""Flush denormalized floating-point values to zero to prevent hardware pipeline stalls.

    Applies flush-to-zero (FTZ) semantics on floating-point tensors with magnitude below
    1e-15 to ensure numerical stability and avoid subnormal microcode traps.
    """
    if t.is_floating_point():
        return torch.where(t.abs() < _FTZ_THRESHOLD, torch.zeros_like(t), t)
    return t


try:
    from flash_attn_interface import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    try:
        from flash_attn import flash_attn_func

        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False


@ATTENTION_REGISTRY.register(
    "flash_attention3_fp8",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class FlashAttention3FP8(FlashAttention3FP8WeightTransferMixin, BaseAttention):
    r"""FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention.

    Accelerates attention using FP8 (e4m3fn) precision with FlashAttention-3 kernels on
    Hopper (SM 9.0+) GPUs. Automatically falls back to native PyTorch SDPA on pre-Hopper
    hardware or CPU.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any = 768,
        n_heads: int = 12,
        use_fp8: bool = False,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if config is None:

            class DummyConfig:
                pass

            config = DummyConfig()
            if source_attn is not None:
                out_layer = getattr(
                    source_attn,
                    "out_lin",
                    getattr(
                        source_attn,
                        "dense",
                        getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None)),
                    ),
                )
                q_layer = getattr(
                    source_attn,
                    "q_lin",
                    getattr(
                        source_attn,
                        "query",
                        getattr(source_attn, "c_attn", getattr(source_attn, "q_proj", None)),
                    ),
                )
                if out_layer is not None and hasattr(out_layer, "out_features"):
                    config.hidden_size = out_layer.out_features
                elif q_layer is not None and hasattr(q_layer, "in_features"):
                    config.hidden_size = q_layer.in_features
                else:
                    config.hidden_size = kwargs.get("hidden_size", kwargs.get("d_model", d_model))
                config.num_attention_heads = getattr(
                    source_attn,
                    "n_heads",
                    getattr(
                        source_attn,
                        "num_attention_heads",
                        getattr(
                            source_attn,
                            "num_heads",
                            kwargs.get("num_heads", kwargs.get("num_attention_heads", n_heads)),
                        ),
                    ),
                )
            else:
                config.hidden_size = kwargs.get("hidden_size", kwargs.get("d_model", d_model))
                config.num_attention_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads", n_heads))

        super().__init__(config, rank)
        self.num_heads: int = (
            getattr(
                config,
                "num_attention_heads",
                getattr(
                    config,
                    "num_heads",
                    kwargs.get("num_heads", kwargs.get("num_attention_heads", n_heads)),
                ),
            )
            if config is not None
            else kwargs.get("num_heads", kwargs.get("num_attention_heads", n_heads))
        )
        self.num_kv_heads: int = kwargs.get(
            "num_kv_heads",
            kwargs.get(
                "num_key_value_heads",
                (
                    getattr(
                        config,
                        "num_key_value_heads",
                        getattr(config, "num_kv_heads", self.num_heads),
                    )
                    if config is not None
                    else self.num_heads
                ),
            ),
        )
        hidden_size: int = (
            getattr(
                config,
                "hidden_size",
                getattr(
                    config,
                    "d_model",
                    kwargs.get("hidden_size", kwargs.get("d_model", d_model)),
                ),
            )
            if config is not None
            else kwargs.get("hidden_size", kwargs.get("d_model", d_model))
        )

        cfg_head_dim = getattr(config, "head_dim", None) if config is not None else None
        if cfg_head_dim is None and source_attn is not None:
            for attr in ("q_proj", "query", "q_lin", "W_q"):
                mod = getattr(source_attn, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    cfg_head_dim = mod.weight.shape[0] // max(1, self.num_heads)
                    break
        self.head_dim: int = (
            int(cfg_head_dim)
            if cfg_head_dim is not None
            else kwargs.get("head_dim", hidden_size // max(1, self.num_heads))
        )
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.use_fp8: bool = bool(use_fp8)
        if self.use_fp8:
            hopper_capable = False
            if torch.cuda.is_available() and hasattr(torch.cuda, "get_device_capability"):
                try:
                    cc = torch.cuda.get_device_capability()
                    hopper_capable = cc[0] >= 9
                except Exception:
                    hopper_capable = False
            if not hopper_capable or not FLASH_ATTN_AVAILABLE or not hasattr(torch, "float8_e4m3fn"):
                logger.info(
                    "FlashAttention3FP8: Hopper architecture (SM 9.0+), FlashAttention, or torch.float8_e4m3fn "
                    "not available; disabling FP8 and falling back to native SDPA."
                )
                self.use_fp8 = False

        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (layer_idx if layer_idx is not None else kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        )
        self.layer_idx: int = int(resolved_layer_id)

        self.lora_rank: int = getattr(
            config,
            "lora_rank",
            getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
        )
        self.lora_alpha: float = getattr(
            config,
            "lora_alpha",
            getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
        )
        self.freeze_base: bool = self.lora_rank > 0

        _mt = str(getattr(config, "model_type", "") or "").lower() if config is not None else ""
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
        is_dec = (
            (
                getattr(config, "is_decoder", False)
                or getattr(config, "is_causal", False)
                or any(_mt.startswith(p) for p in _decoder_prefixes)
            )
            if config is not None
            else False
        )
        if hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)

        is_nested = kwargs.get("is_nested_encoder")
        if is_nested is None and source_attn is not None:
            src_cls = source_attn.__class__.__name__
            if (
                src_cls
                in [
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (src_cls.endswith("SelfAttention") and "distil" not in src_cls.lower())
                or (hasattr(source_attn, "self") and hasattr(source_attn, "output"))
            ):
                is_nested = True
            elif hasattr(source_attn, "out_lin") or hasattr(source_attn, "c_proj") or hasattr(source_attn, "o_proj"):
                is_nested = False
        if is_nested is None:
            is_nested = False
        self.is_nested_encoder: bool = bool(is_nested)
        self.gradient_checkpointing: bool = (
            getattr(config, "gradient_checkpointing", False) if config is not None else False
        )

        self.q_proj = LoRALinear(
            hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = None
        self.comp_config: Any | None = comp_config

        # Delayed scaling history buffers for FP8 quantization
        # Maintains delayed maximum scales over window to prevent sudden scale jumps
        self.register_buffer("_delayed_q_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("_delayed_k_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("_delayed_v_scale", torch.tensor(1.0, dtype=torch.float32))
        self.delayed_scaling_momentum: float = 0.9

        if source_attn is not None:
            self.copy_weights(source_attn)

    @property
    def query(self) -> LoRALinear:
        return self.q_proj

    @property
    def key(self) -> LoRALinear:
        return self.k_proj

    @property
    def value(self) -> LoRALinear:
        return self.v_proj

    @property
    def dense(self) -> LoRALinear:
        return self.out_proj

    @property
    def o_proj(self) -> LoRALinear:
        return self.out_proj

    @property
    def q_lin(self) -> LoRALinear:
        return self.q_proj

    @property
    def k_lin(self) -> LoRALinear:
        return self.k_proj

    @property
    def v_lin(self) -> LoRALinear:
        return self.v_proj

    @property
    def out_lin(self) -> LoRALinear:
        return self.out_proj

    @property
    def W_q(self) -> LoRALinear:
        return self.q_proj

    @property
    def W_k(self) -> LoRALinear:
        return self.k_proj

    @property
    def W_v(self) -> LoRALinear:
        return self.v_proj

    @property
    def c_proj(self) -> LoRALinear:
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
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        if position_embeddings is not None and not is_cross_attention:
            if isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
            else:
                cos, sin = position_embeddings, None
            if sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross_attention:
            if position_ids is not None:
                cos, sin = self.rotary_emb(v, seq_len=kv_len)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # Enforce flush-to-zero handling on projection activations
        q = _flush_denormals(q)
        k = _flush_denormals(k)
        v = _flush_denormals(v)

        causal_flag = bool(self.is_decoder and not is_cross_attention and kv_len == L)

        out = None
        if FLASH_ATTN_AVAILABLE and self.use_fp8 and q.is_cuda and hasattr(torch, "float8_e4m3fn"):
            is_hopper = False
            if hasattr(torch.cuda, "get_device_capability"):
                try:
                    is_hopper = torch.cuda.get_device_capability(device)[0] >= 9
                except Exception:
                    is_hopper = False

            if is_hopper:
                try:
                    q_perm = q.transpose(1, 2).contiguous()
                    k_perm = k.transpose(1, 2).contiguous()
                    v_perm = v.transpose(1, 2).contiguous()

                    fp8_max = 448.0
                    cur_q_scale = (q_perm.abs().amax() / fp8_max).clamp(min=1e-6)
                    cur_k_scale = (k_perm.abs().amax() / fp8_max).clamp(min=1e-6)
                    cur_v_scale = (v_perm.abs().amax() / fp8_max).clamp(min=1e-6)

                    # Update delayed scaling factors
                    if self.training:
                        self._delayed_q_scale.mul_(self.delayed_scaling_momentum).add_(
                            cur_q_scale.detach().to(self._delayed_q_scale.device),
                            alpha=1.0 - self.delayed_scaling_momentum,
                        )
                        self._delayed_k_scale.mul_(self.delayed_scaling_momentum).add_(
                            cur_k_scale.detach().to(self._delayed_k_scale.device),
                            alpha=1.0 - self.delayed_scaling_momentum,
                        )
                        self._delayed_v_scale.mul_(self.delayed_scaling_momentum).add_(
                            cur_v_scale.detach().to(self._delayed_v_scale.device),
                            alpha=1.0 - self.delayed_scaling_momentum,
                        )
                        q_scale = self._delayed_q_scale.clamp(min=1e-6).to(device)
                        k_scale = self._delayed_k_scale.clamp(min=1e-6).to(device)
                        v_scale = self._delayed_v_scale.clamp(min=1e-6).to(device)
                    else:
                        q_scale = cur_q_scale
                        k_scale = cur_k_scale
                        v_scale = cur_v_scale

                    q_fa = (q_perm / q_scale).to(torch.float8_e4m3fn)
                    k_fa = (k_perm / k_scale).to(torch.float8_e4m3fn)
                    v_fa = (v_perm / v_scale).to(torch.float8_e4m3fn)

                    out_fp8 = flash_attn_func(q_fa, k_fa, v_fa, causal=causal_flag)

                    out_tensor = (out_fp8.float() * v_scale).to(dtype)

                    if head_mask is not None:
                        out_tensor_h = out_tensor.transpose(1, 2)
                        if head_mask.dim() == 1:
                            hm = head_mask.view(1, -1, 1, 1)
                        elif head_mask.dim() == 2:
                            hm = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
                        elif head_mask.dim() == 3:
                            hm = head_mask.unsqueeze(-1)
                        else:
                            hm = head_mask
                        out_tensor_h = out_tensor_h * hm
                        out = out_tensor_h.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
                    else:
                        out = out_tensor.contiguous().view(B, L, self.all_head_size)
                except (RuntimeError, NotImplementedError, ValueError) as e:
                    from neuralops.utils.logging_config import warn_once

                    warn_once(
                        "fa3_fp8_execution_fallback",
                        f"FlashAttention-3 FP8 execution failed: {e}; falling back to native SDPA.",
                    )
            elif not is_hopper and hasattr(torch.cuda, "get_device_capability"):
                try:
                    from neuralops.ops.triton._types import FORCE_PYTORCH_FALLBACK

                    major, minor = torch.cuda.get_device_capability(device)
                    if not FORCE_PYTORCH_FALLBACK and (major > 8 or (major == 8 and minor >= 9)):
                        from neuralops.ops.triton.low_rank import fp8_tensorcore_matmul_triton

                        scale = 1.0 / math.sqrt(max(int(Dh), 1))
                        scores = torch.matmul(q.float(), k.float().transpose(-1, -2).contiguous()) * scale
                        if causal_flag:
                            causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
                            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
                        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
                        if eff_mask is not None:
                            if eff_mask.dim() == 2:
                                scores = scores.masked_fill((~eff_mask.bool()).unsqueeze(1).unsqueeze(2), -1e4)
                        attn = F.softmax(scores, dim=-1)
                        if self.dropout.p > 0.0 and self.training:
                            attn = self.dropout(attn)
                        out_heads = []
                        for h_idx in range(self.num_heads):
                            ah = attn[:, h_idx].contiguous()
                            vh = v[:, h_idx].contiguous()
                            oh = fp8_tensorcore_matmul_triton(ah, vh, out_dtype=dtype)
                            out_heads.append(oh.unsqueeze(1))
                        out_tensor_h = torch.cat(out_heads, dim=1)
                        if head_mask is not None:
                            out_tensor_h = out_tensor_h * head_mask.view(1, -1, 1, 1).to(out_tensor_h.dtype)
                        out = out_tensor_h.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
                except Exception as e:
                    logger.debug("Ada FP8 Tensor Core path failed: %s; falling back to SDPA", e)

                    out = None

        if out is None:
            # Enforce FP32 accumulation registers for FP16 and BF16 inputs to maintain numerical precision
            accum_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
            q_acc = q.to(accum_dtype)
            k_acc = k.to(accum_dtype)
            v_acc = v.to(accum_dtype)

            eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
            attn_mask = None
            if eff_mask is not None:
                if eff_mask.dim() == 2:
                    min_val = torch.finfo(accum_dtype).min if accum_dtype.is_floating_point else -1e4
                    attn_mask = (1.0 - eff_mask.unsqueeze(1).unsqueeze(2).to(accum_dtype)) * min_val
                elif eff_mask.dim() == 3:
                    attn_mask = eff_mask.unsqueeze(1).to(accum_dtype)
                elif eff_mask.dim() == 4:
                    attn_mask = eff_mask.to(accum_dtype)

            if causal_flag and attn_mask is None:
                out_sdpa = F.scaled_dot_product_attention(
                    q_acc,
                    k_acc,
                    v_acc,
                    attn_mask=None,
                    is_causal=True,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
            else:
                if causal_flag:
                    causal_m = torch.triu(
                        torch.full(
                            (L, kv_len),
                            float("-inf"),
                            device=device,
                            dtype=accum_dtype,
                        ),
                        diagonal=1,
                    )
                    attn_mask = (
                        (attn_mask + causal_m.unsqueeze(0).unsqueeze(0))
                        if attn_mask is not None
                        else causal_m.unsqueeze(0).unsqueeze(0)
                    )

                out_sdpa = F.scaled_dot_product_attention(
                    q_acc,
                    k_acc,
                    v_acc,
                    attn_mask=attn_mask,
                    is_causal=False,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )

            # Cast back from FP32 accumulator to target output dtype and flush denormals
            out_sdpa = _flush_denormals(out_sdpa.to(dtype))

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                out_sdpa = out_sdpa * head_mask.to(out_sdpa.dtype)

            out = out_sdpa.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            final_out = out
        else:
            final_out = self.out_proj(out)

        return final_out, None



__all__ = [
    FlashAttention3FP8,
    LoRALinear,
    inject_flash3_fp8,
    apply_rotary_pos_emb,
]
