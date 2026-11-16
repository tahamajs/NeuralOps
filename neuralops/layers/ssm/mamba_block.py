#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Mamba Block with SVD-Based Attention Weight Distillation

Selective state space sequence model enhanced with SVD-based weight distillation for compact deployment.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.common import _detect_return_format, _infer_d_model
from neuralops.core.exceptions import IncompatibleArchitectureError
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.surgery.director import CrossAttentionLinearBridge

try:
    from neuralops.core.selective_scan import HAS_MAMBA_CUDA, selective_scan
except Exception as e:
    logging.getLogger(__name__).warning("selective_scan import failed: %s", e)
    selective_scan = None
    HAS_MAMBA_CUDA = False

try:
    from neuralops.ops.cuda.selective_scan.selective_scan_autograd import (
        _get_module as _get_cuda_ss_module,
        selective_scan as cuda_selective_scan,
    )
except Exception as _e:
    logging.getLogger(__name__).warning("CUDA selective_scan unavailable: %s", _e)
    cuda_selective_scan = None
    _get_cuda_ss_module = None

_WARNED_CUDA_SS_UNAVAILABLE = False

try:
    from neuralops.ops.triton.selective_scan import selective_scan_fn as triton_selective_scan

    HAS_TRITON_MAMBA = True
except Exception as _e:
    logging.getLogger(__name__).warning("Triton selective_scan unavailable: %s", _e)
    triton_selective_scan = None
    HAS_TRITON_MAMBA = False

logger = logging.getLogger(__name__)


from .mamba_block_scans import _parallel_scan, _read_state, _recurrent_scan
from .mamba_block_weights import copy_weights
from .mamba_injection import inject_mamba

@ATTENTION_REGISTRY.register("mamba_block")
@ATTENTION_REGISTRY.register("mamba", is_alias=True)
class MambaBlock(BaseAttention):
    """Faithful S6 / Mamba block, production-hardened."""

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = False

    def __init__(
        self,
        config: Any | None = None,
        d_model: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        conv_bias: bool = True,
        bias: bool = False,
        bidirectional: bool = False,
        is_decoder_override: bool | None = None,
        freeze_base: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, (int, float)) and d_model is None:
            d_model = int(config)
            config = None

        super().__init__(config=config, **kwargs)
        self.config = config

        if d_model is None:
            d_model = _infer_d_model(config, kwargs.get("source_attn"), None, **kwargs)
        self.d_model = int(d_model)

        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = int(dt_rank or math.ceil(self.d_model / 16))
        self.bidirectional = bool(bidirectional)
        self.freeze_base = bool(freeze_base)

        if is_decoder_override is not None:
            self.is_decoder = bool(is_decoder_override)
        else:
            _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
            _dec_prefixes = (
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
            )
            self.is_decoder = (
                bool(getattr(config, "is_decoder", False) or getattr(config, "causal", False)) if config else False
            ) or any(_mt.startswith(p) for p in _dec_prefixes)

        if not self.is_decoder and "bidirectional" not in kwargs:
            self.bidirectional = True

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=0,
            bias=conv_bias,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt_init_inv = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_init_inv)
            bound = 1.0 / math.sqrt(self.dt_rank)
            nn.init.uniform_(self.dt_proj.weight, -bound, bound)
            nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
            if self.conv1d.bias is not None:
                nn.init.zeros_(self.conv1d.bias)

        if self.bidirectional:
            self.A_log_bwd = nn.Parameter(torch.log(A.clone() + 0.01 * torch.randn_like(A).abs()))
            self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            with torch.no_grad():
                self.dt_proj_bwd.bias.copy_(dt_init_inv)
                nn.init.uniform_(self.dt_proj_bwd.weight, -bound, bound)
            self.x_proj_bwd = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

        source_attn = kwargs.get("source_attn") or kwargs.get("old_attn")
        self.is_nested_encoder = False
        self._return_format = _detect_return_format(source_attn, False)
        self.cross_attn_bridge: CrossAttentionLinearBridge | None = None

        if freeze_base:
            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        """MB-04: match the framework's LoRA-freeze convention."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_base_weights(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


    copy_weights = copy_weights
    _read_state = _read_state
    _recurrent_scan = _recurrent_scan
    _parallel_scan = _parallel_scan

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        past_state: Any | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        """Forward pass for MambaBlock supporting autoregressive recurrence and cross-attention bridge.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional attention mask tensor.
            *args: Positional arguments for interface compatibility.
            past_state: Past recurrent and convolutional state tuple.
            past_key_value: Alias for past_state.
            output_attentions: Whether to output attention weights.
            encoder_hidden_states: Optional cross-sequence conditioning tensor.
            encoder_attention_mask: Optional mask for cross-attention encoder sequence.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple according to model return format: (output, present_state, attn_weights) or subset.

        Raises:
            IncompatibleArchitectureError: If cross-attention conditioning invariants are violated.
        """
        enc_states = encoder_hidden_states if encoder_hidden_states is not None else kwargs.get("encoder_hidden_states")
        if enc_states is not None:
            if not isinstance(enc_states, torch.Tensor):
                raise IncompatibleArchitectureError(
                    f"encoder_hidden_states must be a torch.Tensor, got {type(enc_states).__name__}"
                )
            if enc_states.dim() != 3:
                raise IncompatibleArchitectureError(
                    f"encoder_hidden_states must have 3 dimensions (B, L_enc, D), got shape {enc_states.shape}"
                )
            if self.cross_attn_bridge is None:
                self.cross_attn_bridge = CrossAttentionLinearBridge(
                    head_dim=self.d_model,
                    persistent_cache=True,
                ).to(device=hidden_states.device, dtype=hidden_states.dtype)

            enc_mask = (
                encoder_attention_mask
                if encoder_attention_mask is not None
                else kwargs.get("encoder_attention_mask")
            )
            cross_out = self.cross_attn_bridge(
                hidden_states,
                k=enc_states,
                v=enc_states,
                encoder_hidden_states=enc_states,
                attention_mask=enc_mask,
            )
            if cross_out.shape[-1] != self.d_model:
                cross_out = (
                    F.pad(cross_out, (0, self.d_model - cross_out.shape[-1]))
                    if cross_out.shape[-1] < self.d_model
                    else cross_out[..., : self.d_model]
                )

            fmt = self._return_format
            attn_weights = None
            present_state = None
            if fmt == "bert":
                return (cross_out, attn_weights)
            if fmt == "gpt2":
                if output_attentions:
                    return (cross_out, present_state, attn_weights)
                return (cross_out, present_state)
            if self.is_decoder:
                if output_attentions:
                    return (cross_out, attn_weights, present_state)
                return (cross_out, present_state)
            return (cross_out, attn_weights)

        if past_state is None and past_key_value is not None:
            past_state = past_key_value

        B, L, _ = hidden_states.shape
        device = hidden_states.device
        target_dtype = hidden_states.dtype

        h_prev, conv_prev, _, past_len = self._read_state(past_state)

        xz = self.in_proj(hidden_states.to(self.in_proj.weight.dtype)).to(target_dtype)
        x_ssm, z = xz.chunk(2, dim=-1)

        if conv_prev is not None:
            conv_buf = conv_prev.transpose(1, 2).contiguous()
            x_cat = torch.cat([conv_buf, x_ssm], dim=1)
        else:
            x_cat = F.pad(x_ssm.transpose(1, 2), (self.d_conv - 1, 0)).transpose(1, 2)

        x_conv_full = self.conv1d(x_cat.transpose(1, 2).to(self.conv1d.weight.dtype))
        x_conv = x_conv_full[..., -L:].transpose(1, 2).to(target_dtype)

        if self.d_conv - 1 <= L:
            conv_next = x_cat[:, -(self.d_conv - 1) :, :].transpose(1, 2).contiguous()
        else:
            pad = self.d_conv - 1 - L
            conv_next = (
                torch.cat(
                    [
                        torch.zeros(B, pad, self.d_inner, device=device, dtype=x_cat.dtype),
                        x_cat,
                    ],
                    dim=1,
                )
                .transpose(1, 2)
                .contiguous()
            )

        x_ssm = F.silu(x_conv)

        x_dbl = self.x_proj(x_ssm.to(self.x_proj.weight.dtype))
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt.to(self.dt_proj.weight.dtype))

        A = -torch.exp(self.A_log)

        u_t = x_ssm.transpose(1, 2)
        dt_t = dt.transpose(1, 2)
        B_t = B_ssm.transpose(1, 2)
        C_t = C_ssm.transpose(1, 2)

        if h_prev is None:
            y = self._parallel_scan(u_t, dt_t, A, B_t, C_t, self.D)
            h_next = None
        else:
            y, h_next = self._recurrent_scan(u_t, dt_t, A, B_t, C_t, self.D, h_prev)

        y = y.transpose(1, 2)

        if self.bidirectional and not self.is_decoder:
            x_rev = torch.flip(x_ssm, dims=[1])
            x_dbl_rev = self.x_proj_bwd(x_rev.to(self.x_proj_bwd.weight.dtype))
            dt_b, B_b, C_b = x_dbl_rev.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_bwd(dt_b.to(self.dt_proj_bwd.weight.dtype))
            A_b = -torch.exp(self.A_log_bwd)
            y_b = self._parallel_scan(
                x_rev.transpose(1, 2),
                dt_b.transpose(1, 2),
                A_b,
                B_b.transpose(1, 2),
                C_b.transpose(1, 2),
                self.D_bwd,
            )
            y_b = torch.flip(y_b.transpose(1, 2), dims=[1])
            y = 0.5 * (y + y_b)

        y_gated = y.float() * F.silu(z.float())
        y = (
            torch.nan_to_num(y_gated, nan=0.0, posinf=65000.0, neginf=-65000.0)
            .clamp(-65000.0, 65000.0)
            .to(target_dtype)
        )

        out = self.out_proj(y.to(self.out_proj.weight.dtype)).to(target_dtype)
        out = (
            torch.nan_to_num(out.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
            .clamp(-65000.0, 65000.0)
            .to(target_dtype)
        )

        if attention_mask is not None:
            m = attention_mask
            if m.dtype.is_floating_point:
                m = (m > -1e4).to(out.dtype)
            else:
                m = m.to(out.dtype)
            if m.dim() == 2:
                out = out * m.unsqueeze(-1)
            elif m.dim() == 3:
                m_red = m.diagonal(dim1=-2, dim2=-1) if m.size(-1) == L else m[..., -1]
                out = out * m_red.unsqueeze(-1)
            elif m.dim() == 4:
                m_sq = m.squeeze(1)
                m_red = m_sq.diagonal(dim1=-2, dim2=-1) if m_sq.size(-1) == L else m_sq[..., -1]
                out = out * m_red.unsqueeze(-1)

        present_state = (h_next, conv_next)
        attn_weights = None

        fmt = self._return_format
        if fmt == "bert":
            return (out, attn_weights)
        if fmt == "gpt2":
            if output_attentions:
                return (out, present_state, attn_weights)
            return (out, present_state)
        if self.is_decoder:
            if output_attentions:
                return (out, attn_weights, present_state)
            return (out, present_state)
        return (out, attn_weights)

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Single-step autoregressive recurrent update for MambaBlock."""
        from neuralops.ops.triton.state_update import causal_conv1d_step, fused_recurrent_ssm_step

        if state is None:
            state = {}

        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_ssm_1d = x_ssm.squeeze(1) if x_ssm.dim() == 3 else x_ssm

        conv_state = state.get("conv_state", None)
        if conv_state is None:
            conv_state = torch.zeros(B, self.d_inner, self.d_conv - 1, device=device, dtype=dtype)
        x_conv, next_conv_state = causal_conv1d_step(x_ssm_1d, conv_state, self.conv1d.weight, self.conv1d.bias)
        x_act = F.silu(x_conv)

        x_dbl = self.x_proj(x_act)
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        ssm_state = state.get("ssm_state", None)
        if ssm_state is None:
            ssm_state = torch.zeros(B, self.d_inner, self.d_state, device=device, dtype=dtype)
        A = -torch.exp(self.A_log)
        y_ssm, next_ssm_state = fused_recurrent_ssm_step(x_act, dt, A, B_ssm, C_ssm, self.D, ssm_state)

        z_1d = z.squeeze(1) if z.dim() == 3 else z
        y_gated = y_ssm * F.silu(z_1d)
        out = self.out_proj(y_gated).unsqueeze(1)

        new_state = {
            "conv_state": next_conv_state,
            "ssm_state": next_ssm_state,
        }
        return out, new_state


Mamba = MambaBlock




Mamba = MambaBlock

__all__ = [MambaBlock, Mamba, inject_mamba]
