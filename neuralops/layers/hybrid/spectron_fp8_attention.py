#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Aegis-Spectron-FP8 Hybrid Attention

Three-branch hybrid attention featuring local windows, low-rank linear pathways, and FP8 quantized execution.
"""

import logging
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.common import FactorizedLinear, _detect_decoder

logger = logging.getLogger(__name__)

from neuralops.core.base import BaseAttention
from neuralops.core.common import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.rope import _rotate_half

from neuralops.layers.hybrid._spectron_fp8.kernels import (
    fp8_cast_simulate,
    AegisSpectronFP8Config,
    SpectronSpectralEntropyRouter,
    _local_window_attention,
    _causal_global_linear_attention,
    _noncausal_global_linear_attention,
    _state_recurrence_jit,
)
from neuralops.layers.hybrid._spectron_fp8.weight_transfer import SpectronWeightTransferMixin
from neuralops.layers.hybrid._spectron_fp8.injection import inject_aegis_spectron_fp8

@ATTENTION_REGISTRY.register("spectron_fp8", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("aegis_spectron_fp8", is_alias=True, supports_causal=True, supports_gqa=True)
class SpectronFP8Attention(SpectronWeightTransferMixin, BaseAttention):
    """SpectronFP8Attention neural operator.

    Aegis-Spectron-FP8 Hybrid Attention .

    Attributes:
        supports_causal (bool): Whether causal autoregressive masking is supported.
        is_custom_attention (bool): Flag indicating specialized non-standard attention mechanics.
    """

    is_custom_attention = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ):
        if not isinstance(config, AegisSpectronFP8Config):
            cfg_src = config if config is not None else kwargs
            if isinstance(cfg_src, dict):
                h_size = hidden_size or cfg_src.get("hidden_size", cfg_src.get("dim", 768))
                n_heads = num_heads or cfg_src.get("num_attention_heads", cfg_src.get("num_heads", 12))
                n_kv = cfg_src.get("num_key_value_heads", n_heads)
                h_dim = cfg_src.get("head_dim", h_size // max(1, n_heads))
            else:
                h_size = hidden_size or kwargs.get(
                    "hidden_size",
                    getattr(cfg_src, "hidden_size", getattr(cfg_src, "d_model", 768)),
                )
                n_heads = num_heads or kwargs.get(
                    "num_heads",
                    getattr(
                        cfg_src,
                        "num_attention_heads",
                        getattr(cfg_src, "num_heads", 12),
                    ),
                )
                n_kv = getattr(
                    cfg_src,
                    "num_key_value_heads",
                    getattr(cfg_src, "num_kv_heads", n_heads),
                )
                h_dim = getattr(
                    cfg_src,
                    "head_dim",
                    getattr(cfg_src, "kv_channels", h_size // max(1, n_heads)),
                )
            config = AegisSpectronFP8Config(
                hidden_size=h_size,
                num_heads=n_heads,
                num_key_value_heads=n_kv,
                head_dim=h_dim,
                window_size=getattr(cfg_src, "window_size", getattr(cfg_src, "local_window", 64)),
                svd_rank=getattr(cfg_src, "svd_rank", getattr(cfg_src, "rank", min(128, h_size))),
                lora_rank=getattr(cfg_src, "lora_rank", getattr(cfg_src, "lora_r", 32)),
                dropout=getattr(cfg_src, "dropout", 0.1),
                fp8_format=getattr(cfg_src, "fp8_format", "e4m3"),
                is_decoder=_detect_decoder(cfg_src, is_decoder, pretrained_attn),
            )
        super().__init__(config)
        self.config: AegisSpectronFP8Config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        if self.num_heads % max(1, self.num_kv_heads) != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by " f"num_kv_heads ({self.num_kv_heads})."
            )
        self.kv_repeat = self.num_heads // max(1, self.num_kv_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.supports_gqa = True
        self.is_nested_encoder = False
        self.is_decoder = _detect_decoder(config, is_decoder, pretrained_attn)
        self.fp8_format = config.fp8_format
        self.rank = config.svd_rank
        self.lora_rank = config.lora_rank
        self.window_size = getattr(config, "window_size", 64)

        self.router = SpectronSpectralEntropyRouter(config)

        self.q_proj = FactorizedLinear(
            self.hidden_size,
            self.all_head_size,
            rank=self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )
        self.k_proj = FactorizedLinear(
            self.hidden_size,
            self.all_kv_head_size,
            rank=self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )
        self.v_proj = FactorizedLinear(
            self.hidden_size,
            self.all_kv_head_size,
            rank=self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )
        self.out_proj = FactorizedLinear(
            self.all_head_size,
            self.hidden_size,
            rank=self.rank,
            lora_rank=self.lora_rank,
            bias=True,
        )

        self.svd_U_q = nn.Parameter(torch.randn(self.all_head_size, self.rank) * 0.02)
        self.svd_U_k = nn.Parameter(torch.randn(self.all_head_size, self.rank) * 0.02)

        rec_scale = 1.0 / math.sqrt(max(1, config.window_size))
        self.linear_B = nn.Parameter(torch.full((self.all_head_size,), rec_scale))
        self.linear_C = nn.Parameter(torch.full((self.all_head_size,), rec_scale))
        self.decay_param = nn.Parameter(torch.zeros(self.all_head_size))
        self.state_norm = nn.LayerNorm(self.all_head_size)
        self.bidir_gate = nn.Parameter(torch.zeros(self.num_heads))
        self.linear_in = (
            nn.Linear(self.hidden_size, self.all_head_size, bias=False)
            if self.hidden_size != self.all_head_size
            else None
        )

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
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

    @property
    def q(self):
        return self.q_proj

    @property
    def k(self):
        return self.k_proj

    @property
    def v(self):
        return self.v_proj

    @property
    def o(self):
        return self.out_proj

    @staticmethod
    def _prepare_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        N_kv: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    def _local_branch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: torch.Tensor | None,
        past_len: int,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return _local_window_attention(
            Q,
            K,
            V,
            window=self.config.window_size,
            token_valid=token_valid,
            is_decoder=self.is_decoder,
            past_len=past_len,
            dropout_p=self.config.dropout,
            training=self.training,
            return_attn=return_attn,
        )

    def _global_branch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Returns context with the same shape as Q: (B, h, N, d).

        The low-rank projection collapses the head axis, so all linear
        attention math happens on (B, N, rank) / (B, N, h*d) tensors.
        Head structure is restored at the very end via a final view.
        """
        B, h, N, d = Q.shape
        D = h * d

        Q_flat = Q.transpose(1, 2).contiguous().view(B, N, D)
        K_flat = K.transpose(1, 2).contiguous().view(B, N, D)
        V_flat = V.transpose(1, 2).contiguous().view(B, N, D)

        Q_lr = torch.matmul(Q_flat, self.svd_U_q.to(Q_flat.dtype))
        K_lr = torch.matmul(K_flat, self.svd_U_k.to(K_flat.dtype))

        phi_Q = F.elu(Q_lr) + 1.0
        phi_K = F.elu(K_lr) + 1.0

        with torch.autocast(device_type=Q.device.type, enabled=False):
            phi_Q_f = phi_Q.float()
            phi_K_f = phi_K.float()
            V_f = V_flat.float()

            if token_valid is not None:
                m = token_valid[:, :, None].to(phi_K_f.dtype)
                phi_K_f = phi_K_f * m
                V_f = V_f * m

            if self.is_decoder:
                out = _causal_global_linear_attention(
                    phi_Q_f,
                    phi_K_f,
                    V_f,
                    chunk_size=self.config.chunk_size,
                )
            else:
                out = _noncausal_global_linear_attention(phi_Q_f, phi_K_f, V_f)

            return torch.nan_to_num(out, nan=0.0).to(Q.dtype).view(B, N, h, d).transpose(1, 2).contiguous()

    def _linear_branch(
        self,
        X: torch.Tensor,
        token_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.linear_in is not None:
            X = self.linear_in(X)
        B, N, D = X.shape

        if token_valid is not None:
            X = X * token_valid.unsqueeze(-1).to(X.dtype)

        u = X * self.linear_B.view(1, 1, D).to(X.dtype)
        gamma = torch.sigmoid(self.decay_param.to(X.dtype)).clamp(min=1e-4, max=1.0 - 1e-4)

        out_fwd = _state_recurrence_jit(u, gamma, reverse=False)

        if self.is_decoder:
            out_flat = self.state_norm(out_fwd)
            out_h = out_flat.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            out_bwd = _state_recurrence_jit(u, gamma, reverse=True)
            gate = torch.sigmoid(self.bidir_gate.to(X.dtype)).view(1, self.num_heads, 1, 1)
            out_fwd_h = self.state_norm(out_fwd).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            out_bwd_h = self.state_norm(out_bwd).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            out_h = gate * out_fwd_h + (1.0 - gate) * out_bwd_h

        out_h = out_h * self.linear_C.view(1, self.num_heads, 1, self.head_dim).to(X.dtype)
        return out_h.contiguous()

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
    ):
        B, N, _ = hidden_states.shape
        device = hidden_states.device

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid_new = self._prepare_valid_mask(eff_mask, B, N_kv_new, device)

        alpha = self.router(hidden_states, token_valid=token_valid_new)
        force_branch = os.environ.get("AEGIS_FORCE_BRANCH", "").lower()
        if force_branch == "loc":
            alpha = torch.zeros_like(alpha)
            alpha[:, :, 0] = 1.0
        elif force_branch == "glob":
            alpha = torch.zeros_like(alpha)
            alpha[:, :, 1] = 1.0
        elif force_branch == "lin":
            alpha = torch.zeros_like(alpha)
            alpha[:, :, 2] = 1.0

        alpha_loc = alpha[:, :, 0].view(B, 1, N, 1)
        alpha_glob = alpha[:, :, 1].view(B, 1, N, 1)
        alpha_lin = alpha[:, :, 2].view(B, 1, N, 1)

        fp8_fmt = self.fp8_format
        use_ste = self.training

        Q_new = fp8_cast_simulate(self.q_proj(hidden_states), format=fp8_fmt, use_ste=use_ste)
        K_new = fp8_cast_simulate(self.k_proj(kv_input), format=fp8_fmt, use_ste=use_ste)
        V_new = fp8_cast_simulate(self.v_proj(kv_input), format=fp8_fmt, use_ste=use_ste)

        Q = Q_new.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K_new_h = K_new.view(B, N_kv_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V_new_h = V_new.view(B, N_kv_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len_rope = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len_rope = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len_rope = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len_rope : past_len_rope + N, :]
            sin = sin[..., past_len_rope : past_len_rope + N, :]
            Q, K_new_h = self._apply_rope(Q, K_new_h, cos, sin)

        past_k = past_v = None
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(getattr(self, "layer_idx", 0), len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]

        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k, K_new_h], dim=-2)
            V_cat = torch.cat([past_v, V_new_h], dim=-2)
        else:
            K_cat = K_new_h
            V_cat = V_new_h

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            idx = getattr(self, "layer_idx", 0)
            past_key_value.update(K_new_h, V_new_h, idx)

        N_kv = K_cat.shape[2]
        past_len = N_kv - N if (self.is_decoder and not is_cross) else 0

        if past_len > 0 and token_valid_new is not None:
            past_valid = torch.ones(B, past_len, device=device, dtype=torch.bool)
            token_valid = torch.cat([past_valid, token_valid_new], dim=1)
        else:
            token_valid = token_valid_new

        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        out_loc, attn_loc = self._local_branch(
            Q, K_full, V_full, token_valid, past_len, return_attn=bool(output_attentions)
        )
        out_glob = self._global_branch(Q, K_full, V_full, token_valid)
        out_lin = self._linear_branch(hidden_states, token_valid_new)
        target_dt = hidden_states.dtype
        if out_loc.is_cuda:
            try:
                from neuralops.ops.cuda.hybrid_local_kernel import hybrid_three_branch_blend

                alpha_h = alpha.unsqueeze(1).expand(B, self.num_heads, N, 3).contiguous()
                fused = hybrid_three_branch_blend(
                    out_loc.to(target_dt),
                    out_glob.to(target_dt),
                    out_lin.to(target_dt),
                    alpha_h.to(target_dt),
                )
            except Exception as exc:
                logger.warning("CUDA hybrid_three_branch_blend failed (%s); falling back to PyTorch", exc)
                fused = (
                    alpha_loc.to(target_dt) * out_loc.to(target_dt)
                    + alpha_glob.to(target_dt) * out_glob.to(target_dt)
                    + alpha_lin.to(target_dt) * out_lin.to(target_dt)
                )
        else:
            fused = (
                alpha_loc.to(target_dt) * out_loc.to(target_dt)
                + alpha_glob.to(target_dt) * out_glob.to(target_dt)
                + alpha_lin.to(target_dt) * out_lin.to(target_dt)
            )

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            fused = fused * head_mask.to(fused.dtype)

        out = fused.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            final = out
        else:
            final = self.out_proj(out)
        final = torch.nan_to_num(final, nan=0.0, posinf=1e4, neginf=-1e4)

        attn_ret = attn_loc if output_attentions else None
        return final.to(hidden_states.dtype), attn_ret

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for SpectronFP8Attention."""
        from neuralops.ops.triton.state_update import fused_recurrent_linear_step

        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        if state is None:
            state = {}
        step_idx = state.get("step_idx", 0)

        alpha = self.router(x)
        alpha_loc = alpha[:, :, 0].view(B, 1, 1, 1)
        alpha_glob = alpha[:, :, 1].view(B, 1, 1, 1)
        alpha_lin = alpha[:, :, 2].view(B, 1, 1, 1)

        Q_new = fp8_cast_simulate(self.q_proj(x), format=self.fp8_format, use_ste=False)
        K_new = fp8_cast_simulate(self.k_proj(x), format=self.fp8_format, use_ste=False)
        V_new = fp8_cast_simulate(self.v_proj(x), format=self.fp8_format, use_ste=False)

        Q = Q_new.view(B, 1, H, d).transpose(1, 2)
        K = K_new.view(B, 1, self.num_kv_heads, d).transpose(1, 2)
        V = V_new.view(B, 1, self.num_kv_heads, d).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        if "k_cache" not in state or state["k_cache"] is None:
            state["k_cache"] = K
            state["v_cache"] = V
        else:
            state["k_cache"] = torch.cat([state["k_cache"], K], dim=2)[:, :, -self.window_size :]
            state["v_cache"] = torch.cat([state["v_cache"], V], dim=2)[:, :, -self.window_size :]

        scores = torch.matmul(Q, state["k_cache"].transpose(-1, -2)) / math.sqrt(d)
        attn_loc = F.softmax(scores, dim=-1)
        out_loc = torch.matmul(attn_loc, state["v_cache"])

        q_elu = F.elu(Q.float() / math.sqrt(d)) + 1.0
        k_elu = F.elu(K.float()) + 1.0
        v_float = V.float()

        if "kv_state" not in state or state["kv_state"] is None:
            state["kv_state"] = torch.zeros((B, H, d, d), device=device, dtype=torch.float32)
            state["z_state"] = torch.zeros((B, H, d), device=device, dtype=torch.float32)

        kv_state = state["kv_state"]
        z_state = state["z_state"]

        out_glob_t, kv_state, z_state = fused_recurrent_linear_step(
            q_elu.squeeze(2), k_elu.squeeze(2), v_float.squeeze(2), kv_state, z_state, eps=1e-5
        )
        state["kv_state"] = kv_state
        state["z_state"] = z_state
        out_glob = out_glob_t.unsqueeze(2).to(dtype)

        gamma = torch.sigmoid(self.decay_param.to(device=device, dtype=torch.float32)).view(1, H, 1, d)
        if "lin_state" not in state or state["lin_state"] is None:
            state["lin_state"] = torch.zeros((B, H, 1, d), device=device, dtype=torch.float32)

        lin_state = state["lin_state"]
        u = (
            (x.float() * self.linear_B.view(1, 1, -1).to(device=device, dtype=torch.float32))
            .view(B, 1, H, d)
            .transpose(1, 2)
        )
        lin_state = gamma * lin_state + u
        state["lin_state"] = lin_state

        out_lin_flat = self.state_norm(lin_state.transpose(1, 2).contiguous().view(B, 1, -1).to(dtype))
        out_lin = out_lin_flat.view(B, 1, H, d).transpose(1, 2)

        fused = alpha_loc * out_loc + alpha_glob * out_glob + alpha_lin * out_lin.to(dtype)
        state["step_idx"] = step_idx + 1

        out = fused.transpose(1, 2).contiguous().view(B, 1, self.all_head_size)
        final = self.out_proj(out)

        if is_2d:
            final = final.squeeze(1)

        return final.to(dtype), state

    @staticmethod
    def _apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cos.dim() == 3:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        while cos.dim() < q.dim():
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

        if cos.shape[-2] != q.shape[-2]:
            raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}")

        rotary_dim = cos.shape[-1]
        head_dim = q.shape[-1]
        if rotary_dim <= head_dim:
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
            k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
            q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
            k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
        else:
            cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
            q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
            k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))
        return q, k

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.out_proj.lin1.weight.device)


AegisSpectronFP8Attention = SpectronFP8Attention


AegisSpectronFP8Attention = SpectronFP8Attention

__all__ = [
    "SpectronFP8Attention",
    "AegisSpectronFP8Attention",
    "AegisSpectronFP8Config",
    "SpectronSpectralEntropyRouter",
    "fp8_cast_simulate",
    "inject_aegis_spectron_fp8",
]
