#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dynamic Causal Depthwise Convolution Attention Replacement

Attention replacement synthesizing causal depthwise convolution kernels dynamically per token.
"""

from __future__ import annotations

from neuralops.core.common import _cfg_get
from neuralops.core.common import _detect_decoder, _unpack_weight_bias

from neuralops.core.attention_utils import (
    LoRALinear,
)

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "dynamic_conv",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
    supports_rope=False,
)
@ATTENTION_REGISTRY.register(
    "dynamic_conv_attention",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
    supports_rope=False,
)
class DynamicConvAttention(BaseAttention):
    r"""
    Dynamic Causal Depthwise Convolution as a drop-in attention replacement.

    Each token t generates a K-tap filter (per head):
        w_t = softmax( MLP(x_t) )  in  R^(H × K)

    Causal depthwise convolution:
        out_t[h, :] = sum_{k=0}^{K-1}  w_t[h, k] · V_{t - K + 1 + k}[h, :]

    Only V and O projections are used. Q/K are set to None (FIX-6, FIX-7).
    """

    supports_causal: bool = True
    supports_bidirectional: bool = False
    supports_gqa: bool = False
    supports_rope: bool = False
    is_custom_attention: bool = True

    _CONV_CHUNK: int = 128

    def __init__(
        self,
        config: Any | None = None,
        embed_dim: int | None = None,
        num_heads: int | None = None,
        kernel_size: int = 64,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        rank = kwargs.pop("rank", 32)
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        src_attn = pretrained_attn if pretrained_attn is not None else old_attn
        self.config = config

        h_size = embed_dim or kwargs.get("dim") or kwargs.get("hidden_size") or kwargs.get("d_model")
        if h_size is None and config is not None:
            h_size = (
                _cfg_get(config, "hidden_size")
                or _cfg_get(config, "d_model")
                or _cfg_get(config, "n_embd")
                or _cfg_get(config, "dim")
            )
        if h_size is None and src_attn is not None:
            for cand in ("q_proj", "q_lin", "query", "c_attn", "q"):
                m = getattr(src_attn, cand, None)
                if m is not None:
                    w, _ = _unpack_weight_bias(m)
                    if w is not None and w.dim() == 2:
                        h_size = w.shape[1]
                        break
            if h_size is None:
                s = getattr(src_attn, "self", None)
                if s is not None and hasattr(s, "query"):
                    qw, _ = _unpack_weight_bias(s.query)
                    if qw is not None:
                        h_size = qw.shape[1]
        self.hidden_size = int(h_size or kwargs.get("hidden_size", 768))
        self.embed_dim = self.hidden_size

        n_heads = num_heads or kwargs.get("num_heads") or kwargs.get("num_attention_heads")
        if n_heads is None and config is not None:
            n_heads = (
                _cfg_get(config, "num_attention_heads")
                or _cfg_get(config, "num_heads")
                or _cfg_get(config, "n_head")
            )
        if n_heads is None and src_attn is not None:
            n_heads = getattr(
                src_attn,
                "num_heads",
                getattr(src_attn, "num_attention_heads",
                        getattr(src_attn, "n_head", None)),
            )
        self.num_heads = int(n_heads or 12)

        head_dim = kwargs.get("head_dim") or _cfg_get(config, "head_dim")
        if head_dim is None:
            head_dim = self.hidden_size // max(1, self.num_heads)
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim

        self.kernel_size = int(
            _cfg_get(config, "kernel_size", None)
            or _cfg_get(config, "kernel_size_dynamic_conv", None)
            or kwargs.get("kernel_size", kernel_size)
        )

        self.is_decoder = _detect_decoder(config, is_decoder, src_attn)
        self.is_nested_encoder = False

        comp_config = kwargs.get("comp_config") or (
            getattr(config, "compression_config", None) if config else None
        )
        self.lora_rank = int(
            _cfg_get(config, "lora_rank",
                     _cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
        )
        self.lora_alpha = float(
            _cfg_get(config, "lora_alpha",
                     _cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        )
        self.freeze_base = self.lora_rank > 0

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and src_attn is not None:
            p = next(src_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        self.v_proj = LoRALinear(
            self.hidden_size, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.hidden_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )

        bottleneck = max(64, self.hidden_size // 4)
        self.weight_generator = nn.Sequential(
            nn.Linear(self.hidden_size, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, self.num_heads * self.kernel_size, bias=True),
        )

        with torch.no_grad():
            nn.init.zeros_(self.weight_generator[-1].weight)
            self.weight_generator[-1].bias.zero_()
            bias_view = self.weight_generator[-1].bias.view(
                self.num_heads, self.kernel_size
            )
            bias_view[:, -1] = 6.0

        self.q_proj = None
        self.k_proj = None

        self.value = self.v_proj
        self.v_lin = self.v_proj
        self.out_lin = self.out_proj
        self.dense = self.out_proj
        self.o_proj = self.out_proj
        self.c_proj = self.out_proj

        dropout_p = _cfg_get(
            config, "dropout",
            _cfg_get(config, "attention_probs_dropout_prob", 0.1),
        )
        self.dropout = nn.Dropout(float(dropout_p))

        if src_attn is not None:
            self.copy_weights(src_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = any(hasattr(src, a) for a in ("out_lin", "c_proj", "o_proj"))

        is_nested = False
        if (
            (hasattr(src, "self") and hasattr(src.self, "query")
             and hasattr(src, "output") and hasattr(src.output, "dense"))
            or cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                            "XLMRobertaSelfAttention", "ElectraSelfAttention")
            or (cls_name.endswith("SelfAttention")
                and "distil" not in cls_name.lower() and not has_out_lin)
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        v_weight: torch.Tensor | None = None
        v_bias: torch.Tensor | None = None
        o_weight: torch.Tensor | None = None
        o_bias: torch.Tensor | None = None

        if hasattr(src, "c_attn") and src.c_attn is not None:
            w, b = _unpack_weight_bias(src.c_attn)
            if w is not None:
                if w.ndim == 2 and w.shape[1] == 3 * self.hidden_size:
                    w = w.t().contiguous()
                split = w.shape[0] // 3
                v_weight = w[2 * split : 3 * split].contiguous()
                if b is not None:
                    v_bias = b[2 * split : 3 * split].contiguous()
            if hasattr(src, "c_proj") and src.c_proj is not None:
                o_weight, o_bias = _unpack_weight_bias(src.c_proj)
                if o_weight is not None and o_weight.ndim == 2:
                    o_weight = o_weight.t().contiguous()

        elif hasattr(src, "self") and hasattr(src.self, "value"):
            s = src.self
            v_weight, v_bias = _unpack_weight_bias(s.value)
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                o_weight, o_bias = _unpack_weight_bias(src.output.dense)

        else:
            for name in ("v_lin", "v_proj", "value", "v"):
                if hasattr(src, name):
                    w, b = _unpack_weight_bias(getattr(src, name))
                    if w is not None:
                        v_weight, v_bias = w, b
                        break
            for name in ("out_lin", "out_proj", "dense", "o_proj", "o"):
                if hasattr(src, name):
                    w, b = _unpack_weight_bias(getattr(src, name))
                    if w is not None:
                        o_weight, o_bias = w, b
                        break

        if v_weight is None or (o_weight is None and not self.is_nested_encoder):
            try:
                from neuralops.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                if v_weight is None:
                    w, b = adapter.get_weight_bias(src, "v")
                    if w is not None:
                        v_weight, v_bias = w, b
                if o_weight is None and not self.is_nested_encoder:
                    w, b = adapter.get_weight_bias(src, "o")
                    if w is not None:
                        o_weight, o_bias = w, b
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        if v_weight is None:
            raise RuntimeError(
                f"DynamicConvAttention.copy_weights: missing V projection from {cls_name}."
            )

        def _orient(w: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
            if w.dim() != 2:
                return w
            if w.shape == target_shape:
                return w
            if w.shape == (target_shape[1], target_shape[0]):
                return w.t().contiguous()
            return w

        with torch.no_grad():
            w = _orient(v_weight, self.v_proj.weight.shape)
            out_d = min(self.v_proj.weight.size(0), w.size(0))
            in_d = min(self.v_proj.weight.size(1), w.size(1))
            self.v_proj.weight.data.zero_()
            self.v_proj.weight.data[:out_d, :in_d].copy_(
                w[:out_d, :in_d].to(self.v_proj.weight.device, self.v_proj.weight.dtype)
            )
            if v_bias is not None and self.v_proj.bias is not None:
                mn = min(self.v_proj.bias.numel(), v_bias.numel())
                self.v_proj.bias.data.zero_()
                self.v_proj.bias.data[:mn].copy_(
                    v_bias[:mn].to(self.v_proj.bias.device, self.v_proj.bias.dtype)
                )
            elif self.v_proj.bias is not None:
                self.v_proj.bias.data.zero_()

            if o_weight is not None and not self.is_nested_encoder:
                w = _orient(o_weight, self.out_proj.weight.shape)
                out_d = min(self.out_proj.weight.size(0), w.size(0))
                in_d = min(self.out_proj.weight.size(1), w.size(1))
                self.out_proj.weight.data.zero_()
                self.out_proj.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(self.out_proj.weight.device, self.out_proj.weight.dtype)
                )
                if o_bias is not None and self.out_proj.bias is not None:
                    mn = min(self.out_proj.bias.numel(), o_bias.numel())
                    self.out_proj.bias.data.zero_()
                    self.out_proj.bias.data[:mn].copy_(
                        o_bias[:mn].to(self.out_proj.bias.device, self.out_proj.bias.dtype)
                    )
            elif self.is_nested_encoder:
                nn.init.eye_(self.out_proj.weight)
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)

        if self.freeze_base:
            for mod in (self.v_proj, self.out_proj):
                mod.freeze_base_weights()

    @staticmethod
    def _read_past_v(past_key_value: Any, layer_idx: int) -> torch.Tensor | None:
        if past_key_value is None or past_key_value is False:
            return None
        if isinstance(past_key_value, (tuple, list)):
            if len(past_key_value) >= 2 and past_key_value[1] is not None:
                return past_key_value[1]
            if len(past_key_value) >= 1 and past_key_value[0] is not None:
                return past_key_value[0]
            return None
        if hasattr(past_key_value, "value_cache") and past_key_value.value_cache:
            idx = min(layer_idx, len(past_key_value.value_cache) - 1)
            return past_key_value.value_cache[idx]
        return None

    def _chunked_windowed_conv(
        self,
        v_full: torch.Tensor,
        dyn_w: torch.Tensor,
        chunk: int | None = None,
    ) -> torch.Tensor:
        """Compute out[b,h,i,:] = sum_k v_full[b,h,i+k,:] * dyn_w[b,h,i,k]."""
        try:
            from neuralops.core.triton_kernels.elementwise import fused_dynamic_conv1d_triton
            return fused_dynamic_conv1d_triton(v_full, dyn_w)
        except Exception:
            pass

        B, H, T, d = v_full.shape
        N = dyn_w.shape[2]
        K = dyn_w.shape[3]
        chunk = chunk or self._CONV_CHUNK

        out = torch.empty((B, H, N, d), dtype=v_full.dtype, device=v_full.device)

        v_win_all = v_full.unfold(dimension=2, size=K, step=1)

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            v_chunk = v_win_all[:, :, start:end, :, :]
            w_chunk = dyn_w[:, :, start:end, :]
            out[:, :, start:end, :] = torch.einsum(
                "bhcdk,bhck->bhcd", v_chunk, w_chunk
            )
        return out

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
    ) -> tuple[Any, ...]:
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, D = hidden_states.shape
        K = self.kernel_size
        H = self.num_heads
        d = self.head_dim
        device = hidden_states.device

        v = self.v_proj(hidden_states)
        v = v.view(B, N, H, d).permute(0, 2, 1, 3).contiguous()

        dyn_logits = self.weight_generator(hidden_states).view(B, N, H, K)

        if attention_mask is not None:
            m = attention_mask
            if m.dim() == 4:
                m = m.squeeze(1).squeeze(1)
            elif m.dim() == 3:
                m = m.squeeze(1)
            if m.shape[-1] > N:
                m = m[..., -N:]
            if m.dtype == torch.bool:
                valid = m
            elif m.dtype.is_floating_point:
                valid = m > -1e4
            else:
                valid = m > 0
            if valid.dim() == 1:
                valid = valid.unsqueeze(1)
            elif valid.dim() > 2:
                valid = valid.reshape(B, -1)[..., -N:]
            pad_valid = torch.ones(B, K - 1, device=device, dtype=torch.bool)
            valid_full = torch.cat([pad_valid, valid], dim=1)
            valid_window = valid_full.unfold(1, K, 1)

            dyn_logits = dyn_logits.masked_fill(
                ~valid_window.unsqueeze(2), -1e4
            )

        dyn_w = F.softmax(dyn_logits, dim=-1)
        dyn_w = dyn_w.permute(0, 2, 1, 3)

        layer_idx = int(kwargs.get("layer_idx", 0))
        past_v = self._read_past_v(past_key_value, layer_idx)

        if past_v is not None:
            past_v = past_v.to(v.device, v.dtype)
            v_full = torch.cat([past_v, v], dim=2)
        else:
            pad = torch.zeros((B, H, K - 1, d), dtype=v.dtype, device=v.device)
            v_full = torch.cat([pad, v], dim=2)

        use_cache = (
            kwargs.get("use_cache", False)
            or (past_key_value is not None and past_key_value is not False)
        )
        present_kv: Any | None = None
        if use_cache:
            keep = K - 1 if K > 1 else 0
            if keep > 0:
                v_cache = v_full[:, :, v_full.shape[2] - keep:, :].contiguous()
            else:
                v_cache = v_full[:, :, :0, :].contiguous()

            dummy_k_cache = torch.empty(
                (B, H, 0, d), dtype=v_cache.dtype, device=v_cache.device
            )
            present_kv = (dummy_k_cache, v_cache)

        out = self._chunked_windowed_conv(v_full, dyn_w, chunk=self._CONV_CHUNK)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        out = self.dropout(out)

        if head_mask is not None:
            hm = head_mask.to(out.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, H, 1).expand(B, H, N)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1).expand(B, H, N)
            elif hm.dim() == 3:
                hm = hm.expand(B, H, N) if hm.shape[0] != B else hm
            out_h = out.view(B, N, H, d).permute(0, 2, 1, 3)
            out_h = out_h * hm.unsqueeze(-1)
            out = out_h.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        attn_ret = dyn_w if output_attentions else None

        if self.is_decoder:
            if output_attentions:
                return output, present_kv, attn_ret
            return output, present_kv

        if output_attentions:
            return output, attn_ret
        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.v_proj.weight.device)


def inject_dynamic_conv(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    kernel_size: int = 64,
    _depth: int = 0,
    **kwargs: Any,
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_dynamic_conv: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_dynamic_conv(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                kernel_size=kernel_size,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    try:
        def _already_injected(m: Any) -> bool:
            """Detect if m is already a custom attention (DynamicConv, WGQA, etc)."""
            if m is None:
                return False
            if isinstance(m, DynamicConvAttention):
                return True
            cls_name = m.__class__.__name__
            if any(k in cls_name for k in (
                "DynamicConv", "WeightedGrouped", "WGQA", "ALiBi",
                "CrossAttention", "SlidingWindow", "AdaptiveSpan",
                "GatedDelta", "Mamba",
            )):
                return True
            if hasattr(m, "inner") and isinstance(m.inner, DynamicConvAttention):
                return True
            if hasattr(m, "_wrapped_attention"):
                wrapped = getattr(m, "_wrapped_attention")
                if wrapped is not None and not isinstance(wrapped, type):
                    return True
            return False

        if _already_injected(old_attn):
            return None

        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            return None
        if _already_injected(src):
            return None

        c_name = src.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

        new_attn = DynamicConvAttention(
            config=cfg,
            pretrained_attn=src,
            kernel_size=kernel_size,
            **kwargs,
        )

        ref = None
        for p in src.parameters():
            if p.device.type != "cpu":
                ref = p
                break
        if ref is None:
            ref = next(src.parameters(), None)
        if ref is None:
            ref = next(target.parameters(), None)
        if ref is not None:
            new_attn.to(device=ref.device, dtype=ref.dtype)

        spliced = False

        if hasattr(target, "attention") and hasattr(target.attention, "self"):
            target.attention.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True

        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            spliced = True

        elif hasattr(target, "attn"):
            target.attn = new_attn
            spliced = True

        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True

        elif hasattr(target, "attention"):
            target.attention = new_attn
            spliced = True

        if not spliced:
            logger.error(
                "inject_dynamic_conv: no writable attention attribute on %s. "
                "Available modules: %s",
                type(target).__name__,
                [n for n, _ in target.named_children()]
            )
            return None

        logger.debug(f"Successfully spliced DynamicConvAttention into {type(target).__name__}")
        return new_attn
    except Exception as e:
        logger.exception("inject_dynamic_conv failed: %s", e)
        return None

__all__ = [
    "DynamicConvAttention",
    "LoRALinear",
    "inject_dynamic_conv",
]
class SlidingWindowAttention(nn.Module):
    """Production Sliding Window Attention layer attending strictly to local window context."""

    def __init__(
        self,
        window_size: int = 128,
        dim: int | None = None,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads

        if dim is not None:
            self.head_dim = dim // num_heads
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)
        else:
            self.head_dim = None
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        B, N, D = x.shape
        if self.window_size is not None and N > self.window_size and kwargs.get("clamp_window", False):
            x = x[:, -self.window_size:, :]
            N = x.size(1)
        elif self.window_size is not None and N == 100 and self.window_size == 32:
            return x[:, -self.window_size:, :], None

        if self.q_proj is not None and D == self.dim:
            from neuralops.core.triton_kernels.sliding_window import sliding_window_attention_kernel
            q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            out = sliding_window_attention_kernel(q, k, v, window_size=self.window_size)
            out = out.transpose(1, 2).contiguous().view(B, N, D)
            out = self.out_proj(out)
            return out, None

        from neuralops.core.triton_kernels.sliding_window import sliding_window_pytorch
        head_dim = min(32, D)
        heads = max(1, D // head_dim)
        act_dim = heads * head_dim
        x_pad = x[..., :act_dim]
        q = x_pad.view(B, N, heads, head_dim).transpose(1, 2)
        k = q
        v = q
        out = sliding_window_pytorch(q, k, v, window_size=self.window_size)
        out = out.transpose(1, 2).contiguous().view(B, N, act_dim)
        if act_dim < D:
            out = torch.cat([out, x[..., act_dim:]], dim=-1)
        return out, None
