#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Simple SSM: Simplified State Space Attention Block

Simplified state space attention block providing fast chunked recurrent updates,
GQA weight alignment, and residual value skip for multi-architecture transformers.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.common import LoRALinear
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv

logger = logging.getLogger(__name__)


def _extract_valid_token_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a boolean validity mask (B, seq_len) where True = valid token, or None."""
    if mask is None or not isinstance(mask, torch.Tensor):
        return None

    mask = mask.to(device=device)
    if mask.dim() == 2:
        if mask.dtype == torch.bool:
            return mask
        if mask.dtype.is_floating_point:
            if (mask < -1.0).any():
                return mask >= -1.0
            if (mask <= 0.0).all():
                return torch.ones((batch_size, mask.size(-1)), dtype=torch.bool, device=device)
            return mask > 0.5
        return mask > 0

    if mask.dim() == 3:
        m = mask[:, -1, :]
        if m.dtype == torch.bool:
            return m
        if m.dtype.is_floating_point:
            if (m < -1.0).any():
                return m >= -1.0
            if (m <= 0.0).all():
                return torch.ones((batch_size, m.size(-1)), dtype=torch.bool, device=device)
            return m > 0.5
        return m > 0

    if mask.dim() == 4:
        if mask.size(2) == 1:
            m = mask[:, 0, 0, :]
            if m.dtype == torch.bool:
                return m
            if m.dtype.is_floating_point:
                if (m < -1.0).any():
                    return m >= -1.0
                if (m <= 0.0).all():
                    return torch.ones((batch_size, m.size(-1)), dtype=torch.bool, device=device)
                return m > 0.5
            return m > 0
        m4 = mask[:, 0]
        if m4.dtype == torch.bool:
            return m4.any(dim=1)
        if m4.dtype.is_floating_point:
            if (m4 < -1.0).any():
                return (m4 >= -1.0).any(dim=1)
            if (m4 <= 0.0).all():
                return torch.ones((batch_size, m4.size(-1)), dtype=torch.bool, device=device)
            return (m4 > 0.5).any(dim=1)
        return (m4 > 0).any(dim=1)

    return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)


@ATTENTION_REGISTRY.register(
    "simple_ssm",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class SimpleSSM(BaseAttention):
    """SimpleSSM neural operator.

    Simplified state space attention block with GQA-aligned Q/K/V/O projections,
    selective bounded exponential decay, chunked parallel causal scan, and
    residual value skip initialization.

    Attributes:
        supports_causal (bool): Whether causal autoregressive masking is supported.
        supports_bidirectional (bool): Whether bidirectional encoding is supported.
        supports_gqa (bool): Whether grouped-query attention projections are native.
        is_custom_attention (bool): Flag indicating specialized non-standard attention mechanics.
    """

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    triton_kernel: str = "state_update"

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 64,
        d_model: int | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        old_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.config = config
        source_attn = old_attn if old_attn is not None else kwargs.get("pretrained_attn")

        def _get(k: str, default: Any = None) -> Any:
            if k in kwargs and kwargs[k] is not None:
                return kwargs[k]
            for obj in (config, getattr(config, "compression_config", None)):
                if obj is not None:
                    if isinstance(obj, dict) and k in obj and obj[k] is not None:
                        return obj[k]
                    elif hasattr(obj, k):
                        val = getattr(obj, k)
                        if val is not None:
                            return val
            return default

        dim = (
            d_model
            or kwargs.get("dim")
            or kwargs.get("hidden_size")
            or kwargs.get("embed_dim")
            or _get("hidden_size")
            or _get("d_model")
            or _get("dim")
            or 64
        )
        self.dim = int(dim)
        self.hidden_size = self.dim

        heads = num_heads or _get("num_attention_heads") or _get("num_heads") or _get("n_head") or 4
        self.num_heads = int(heads)

        kv_heads = num_kv_heads or _get("num_key_value_heads") or _get("num_kv_heads") or self.num_heads
        self.num_kv_heads = max(1, min(int(kv_heads), self.num_heads))

        explicit_dim = d_model is not None or kwargs.get("dim") is not None or kwargs.get("hidden_size") is not None
        if explicit_dim:
            self.head_dim = max(1, self.dim // max(1, self.num_heads))
        else:
            self.head_dim = int(_get("head_dim", self.dim // max(1, self.num_heads)))
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.lora_rank = int(_get("lora_rank", lora_rank))
        self.lora_alpha = float(_get("lora_alpha", lora_alpha))
        self.freeze_base = self.lora_rank > 0
        self.layer_idx = int(_get("layer_idx", 0))

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and source_attn is not None:
            try:
                target_dtype = next(source_attn.parameters()).dtype
            except StopIteration:
                target_dtype = torch.float32
        if target_dtype is None:
            target_dtype = torch.float32

        self.q_proj = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.out_proj: nn.Module = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )

        # Bounded state-space parameters:
        #   decay_param ~ -1.5 => baseline retention sigmoid(-1.5) ~ 0.18
        #   gate_scale  ~  0.5 => smooth Q/K selectivity modulation
        #   skip_logit  ~  1.0 => initial skip weight sigmoid(1.0) ~ 0.73 on V + 0.27 on normalized state
        self.decay_param = nn.Parameter(
            torch.full((self.num_heads, 1, 1), -1.5, dtype=target_dtype)
            + torch.randn(self.num_heads, 1, 1, dtype=target_dtype) * 0.05
        )
        self.gate_scale = nn.Parameter(torch.full((self.num_heads, 1, 1), 0.5, dtype=target_dtype))
        self.skip_logit = nn.Parameter(torch.full((self.num_heads, 1, 1), 1.0, dtype=target_dtype))

        model_type = str(_get("model_type", "") or "").lower()
        dec_prefixes = (
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
        is_dec = bool(_get("is_decoder", False)) or bool(_get("is_causal", False))
        if any(model_type.startswith(pfx) for pfx in dec_prefixes):
            is_dec = True
        if source_attn is not None and (
            getattr(source_attn, "is_decoder", False) or hasattr(source_attn, "c_attn")
        ):
            is_dec = True
        self.is_decoder = bool(is_dec)

        self.is_nested_encoder = False
        self.rotary_emb = getattr(source_attn, "rotary_emb", None)
        self.triton_kwargs = {}

        if source_attn is not None:
            self.copy_weights(source_attn)

        self.to(target_dtype)

    @property
    def in_proj(self) -> nn.Module:
        return self.v_proj

    @property
    def o_proj(self) -> nn.Module:
        return self.out_proj

    @property
    def query(self) -> nn.Module:
        return self.q_proj

    @property
    def key(self) -> nn.Module:
        return self.k_proj

    @property
    def value(self) -> nn.Module:
        return self.v_proj

    def copy_weights(self, old_attn: Any) -> None:
        """Inherit Q/K/V/O weights across BERT, DistilBERT, GPT-2, Llama, and GQA Qwen models."""
        if old_attn is None:
            return

        src_cls = old_attn.__class__.__name__
        has_out_proj = any(
            hasattr(old_attn, attr)
            for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj", "out")
        ) or (hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"))
        self.is_nested_encoder = (
            src_cls in ("BertSelfAttention", "RobertaSelfAttention")
            or (src_cls.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        try:
            adapter = UniversalAdapter()
            for proj in ("q", "k", "v", "o"):
                if proj == "o" and self.is_nested_encoder:
                    continue
                try:
                    w, b = adapter.get_weight_bias(old_attn, proj)
                    if w is not None:
                        extracted[proj] = (w, b)
                except Exception:
                    # allowed-swallow: fallback to explicit attribute lookup below
                    pass
        except Exception:
            # allowed-swallow: fallback to explicit attribute lookup below
            pass

        if hasattr(old_attn, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            c_attn = getattr(old_attn.c_attn, "base_layer", old_attn.c_attn)
            if hasattr(c_attn, "weight") and c_attn.weight is not None:
                weight = c_attn.weight.data
                bias = getattr(c_attn, "bias", None)
                bias_data = bias.data if bias is not None else None
                is_conv1d = "Conv1D" in c_attn.__class__.__name__ or hasattr(c_attn, "nf")
                if is_conv1d or (weight.dim() == 2 and weight.shape[1] == 3 * self.dim):
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                extracted["q"] = (weight[:split_size, :], bias_data[:split_size] if bias_data is not None else None)
                extracted["k"] = (
                    weight[split_size : 2 * split_size, :],
                    bias_data[split_size : 2 * split_size] if bias_data is not None else None,
                )
                extracted["v"] = (
                    weight[2 * split_size :, :],
                    bias_data[2 * split_size :] if bias_data is not None else None,
                )
                self.is_decoder = True

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None and not self.is_nested_encoder:
            c_proj = getattr(old_attn.c_proj, "base_layer", old_attn.c_proj)
            if hasattr(c_proj, "weight") and c_proj.weight is not None:
                w_o = c_proj.weight.data
                is_conv1d = "Conv1D" in c_proj.__class__.__name__ or hasattr(c_proj, "nf")
                if is_conv1d or (w_o.dim() == 2 and w_o.shape[0] != self.dim and w_o.shape[1] == self.dim):
                    w_o = w_o.t().contiguous()
                b_o = getattr(c_proj, "bias", None)
                extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        named_map = {
            "q": ("q_proj", "q_lin", "query", "W_q"),
            "k": ("k_proj", "k_lin", "key", "W_k"),
            "v": ("v_proj", "v_lin", "value", "W_v", "in_proj"),
            "o": ("o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"),
        }
        nested_self = getattr(old_attn, "self", None)
        for proj, names in named_map.items():
            if proj == "o" and self.is_nested_encoder:
                continue
            if extracted[proj][0] is not None:
                continue
            for holder in (nested_self, old_attn):
                if holder is None:
                    continue
                for name in names:
                    mod = getattr(holder, name, None)
                    base = getattr(mod, "base_layer", mod) if mod is not None else None
                    if base is not None and hasattr(base, "weight") and base.weight is not None:
                        b = getattr(base, "bias", None)
                        extracted[proj] = (base.weight.data, b.data if b is not None else None)
                        break
                if extracted[proj][0] is not None:
                    break

        if extracted["o"][0] is None and not self.is_nested_encoder and hasattr(old_attn, "output"):
            out_mod = old_attn.output
            dense = getattr(out_mod, "dense", out_mod)
            dense = getattr(dense, "base_layer", dense)
            if hasattr(dense, "weight") and dense.weight is not None:
                b = getattr(dense, "bias", None)
                extracted["o"] = (dense.weight.data, b.data if b is not None else None)

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        def _align_kv(
            w: torch.Tensor,
            b: torch.Tensor | None,
            target_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if w.shape[0] == target_size:
                return w, b
            if w.shape[0] > target_size and w.shape[0] % target_size == 0:
                rep = w.shape[0] // target_size
                w_p = w.view(target_size, rep, -1).mean(dim=1).contiguous()
                b_p = b.view(target_size, rep).mean(dim=1).contiguous() if b is not None else None
                return w_p, b_p
            if target_size > w.shape[0] and target_size % w.shape[0] == 0:
                rep = target_size // w.shape[0]
                src_kv = max(1, w.shape[0] // max(1, self.head_dim))
                w_exp = (
                    w.view(src_kv, self.head_dim, -1)
                    .repeat_interleave(rep, dim=0)
                    .reshape(target_size, -1)
                    .contiguous()
                )
                b_exp = (
                    b.view(src_kv, self.head_dim).repeat_interleave(rep, dim=0).reshape(target_size).contiguous()
                    if b is not None
                    else None
                )
                return w_exp, b_exp
            return w[:target_size, :], (b[:target_size] if b is not None else None)

        with torch.no_grad():
            for proj_key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                if proj_key == "o" and self.is_nested_encoder:
                    self.out_proj = nn.Identity()
                    continue
                if not hasattr(target, "weight"):
                    continue

                w, b = extracted[proj_key]
                if w is None:
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                in_dim = self.all_head_size if proj_key == "o" else self.dim
                w = _orient(w, in_dim)

                if proj_key in ("k", "v"):
                    w, b = _align_kv(w, b, target.weight.size(0))

                d_out = min(target.weight.size(0), w.size(0))
                d_in = min(target.weight.size(1), w.size(1))
                target.weight.data[:d_out, :d_in].copy_(
                    w[:d_out, :d_in].to(device=target.weight.device, dtype=target.weight.dtype)
                )

                if b is not None and getattr(target, "bias", None) is not None:
                    b_data = b.data if isinstance(b, nn.Parameter) else b
                    mb = min(target.bias.numel(), b_data.numel())
                    target.bias.data[:mb].copy_(
                        b_data[:mb].to(device=target.bias.device, dtype=target.bias.dtype)
                    )
                elif getattr(target, "bias", None) is not None:
                    nn.init.zeros_(target.bias)

                if self.freeze_base and hasattr(target, "freeze_base_weights"):
                    target.freeze_base_weights()

    def _chunked_causal_scan(
        self,
        u: torch.Tensor,
        decay_t: torch.Tensor,
        init_state: torch.Tensor | None = None,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized O(L) chunked parallel scan for h_t = decay_t * h_{t-1} + u_t in FP32."""
        B, H, L, d = u.shape
        device = u.device
        h_prev = (
            init_state.float().view(B, H, 1, d)
            if init_state is not None
            else torch.zeros((B, H, 1, d), device=device, dtype=torch.float32)
        )

        log_decay = torch.log(decay_t.clamp(min=1e-4, max=1.0))
        C = min(chunk_size, max(1, L))
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            u_pad = F.pad(u, (0, 0, 0, pad_len))
            ld_pad = F.pad(log_decay, (0, 0, 0, pad_len))
        else:
            u_pad = u
            ld_pad = log_decay

        num_chunks = u_pad.shape[-2] // C
        tril_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool))

        chunks_out: list[torch.Tensor] = []
        for idx in range(num_chunks):
            start = idx * C
            end = start + C
            u_c = u_pad[:, :, start:end, :]
            ld_c = ld_pad[:, :, start:end, 0]

            p_cum = torch.cumsum(ld_c, dim=-1)
            diff = p_cum.unsqueeze(-1) - p_cum.unsqueeze(-2)
            diff_safe = torch.where(tril_mask, diff.clamp(min=-30.0, max=0.0), torch.zeros_like(diff))
            decay_mat = torch.where(tril_mask, torch.exp(diff_safe), torch.zeros_like(diff_safe))

            intra = torch.matmul(decay_mat, u_c)
            inter = torch.exp(p_cum.clamp(min=-30.0, max=0.0)).unsqueeze(-1) * h_prev
            h_c = inter + intra
            chunks_out.append(h_c)
            h_prev = h_c[:, :, -1:, :]

        h_all = torch.cat(chunks_out, dim=2)[:, :, :L, :]
        return h_all, h_prev

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
    ) -> tuple[torch.Tensor, Any | None]:
        if encoder_hidden_states is not None:
            return hidden_states, None

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        target_dtype = hidden_states.dtype
        device = hidden_states.device
        B, L, _ = hidden_states.shape
        H = self.num_heads
        d = self.head_dim

        is_dec = self.is_decoder or bool(kwargs.get("is_decoder", False)) or bool(kwargs.get("is_causal", False))

        proj_dtype = self.q_proj.weight.dtype if hasattr(self.q_proj, "weight") else target_dtype
        x_in = (
            hidden_states
            if (torch.is_autocast_enabled() or hidden_states.dtype == proj_dtype)
            else hidden_states.to(proj_dtype)
        )

        Q = self.q_proj(x_in).view(B, L, H, d).transpose(1, 2).contiguous()
        K = self.k_proj(x_in).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.v_proj(x_in).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()

        pos_emb = position_embeddings or kwargs.get("position_embeddings")
        pos_ids = kwargs.get("position_ids")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=pos_ids)
        elif self.rotary_emb is not None:
            try:
                cos, sin = self.rotary_emb(K, seq_len=L)
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=pos_ids)
            except Exception as e:
                logger.debug("SimpleSSM rotary_emb failed: %s", e)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        valid_m = _extract_valid_token_mask(attention_mask, B, L, device) if attention_mask is not None else None
        if valid_m is not None and valid_m.size(-1) != L:
            valid_m = valid_m[:, -L:] if valid_m.size(-1) > L else F.pad(valid_m, (0, L - valid_m.size(-1)), value=True)

        past_state_arg = kwargs.get("past_state", kwargs.get("state"))
        init_state = past_state_arg if isinstance(past_state_arg, torch.Tensor) else (
            past_key_value if isinstance(past_key_value, torch.Tensor) else None
        )

        # Execute recurrence and gating in FP32 to prevent FP16 underflow/overflow under AMP
        Q_f = Q.float()
        K_f = K.float()
        V_f = V.float()

        qk_dot = (Q_f * K_f).sum(dim=-1, keepdim=True) / math.sqrt(max(1, d))
        qk_mod = torch.tanh(qk_dot * 0.25)

        gate_scale_f = self.gate_scale.float().view(1, H, 1, 1)
        decay_base_f = self.decay_param.float().view(1, H, 1, 1)
        skip_f = torch.sigmoid(self.skip_logit.float()).view(1, H, 1, 1)

        next_state: torch.Tensor | None = None
        if is_dec:
            decay_t = torch.sigmoid(decay_base_f + gate_scale_f * qk_mod)
            u = (1.0 - decay_t) * V_f
            if valid_m is not None:
                vm = valid_m.view(B, 1, L, 1)
                u = torch.where(vm, u, torch.zeros_like(u))
                decay_t = torch.where(vm, decay_t, torch.ones_like(decay_t))
            h, final_state = self._chunked_causal_scan(u, decay_t, init_state=init_state, chunk_size=64)
            if kwargs.get("use_cache", False) or past_state_arg is not None or isinstance(past_key_value, torch.Tensor):
                next_state = final_state.to(target_dtype)
        else:
            decay = torch.sigmoid(decay_base_f)
            log_decay = torch.log(decay.clamp(min=1e-4, max=1.0 - 1e-4))
            key_gate = torch.sigmoid(gate_scale_f * qk_mod)

            if L <= 1024:
                pos = torch.arange(L, device=device, dtype=torch.float32)
                dist = torch.abs(pos.unsqueeze(-1) - pos.unsqueeze(-2)).unsqueeze(0).unsqueeze(0)
                w_mat = torch.exp((dist * log_decay).clamp(min=-30.0, max=0.0)) * key_gate.transpose(-2, -1)
                if valid_m is not None:
                    w_mat = w_mat * valid_m.view(B, 1, 1, L).float()
                denom = w_mat.sum(dim=-1, keepdim=True)
                has_valid = denom > 0.0
                w_norm = torch.where(has_valid, w_mat / denom.clamp(min=1e-6), torch.zeros_like(w_mat))
                h = torch.matmul(w_norm, V_f)
            else:
                decay_b = decay.expand(B, H, L, 1)
                u_fwd = (1.0 - decay_b) * key_gate * V_f
                w_fwd = (1.0 - decay_b) * key_gate
                if valid_m is not None:
                    vm = valid_m.view(B, 1, L, 1)
                    u_fwd = torch.where(vm, u_fwd, torch.zeros_like(u_fwd))
                    w_fwd = torch.where(vm, w_fwd, torch.zeros_like(w_fwd))
                    decay_b = torch.where(vm, decay_b, torch.ones_like(decay_b))
                h_fwd, _ = self._chunked_causal_scan(u_fwd, decay_b, chunk_size=64)
                n_fwd, _ = self._chunked_causal_scan(w_fwd, decay_b, chunk_size=64)

                u_bwd_in = torch.flip(u_fwd, dims=[2])
                w_bwd_in = torch.flip(w_fwd, dims=[2])
                d_bwd_in = torch.flip(decay_b, dims=[2])
                h_bwd, _ = self._chunked_causal_scan(u_bwd_in, d_bwd_in, chunk_size=64)
                n_bwd, _ = self._chunked_causal_scan(w_bwd_in, d_bwd_in, chunk_size=64)
                h_bwd = torch.flip(h_bwd, dims=[2])
                n_bwd = torch.flip(n_bwd, dims=[2])

                num = h_fwd + h_bwd - u_fwd
                den = n_fwd + n_bwd - w_fwd
                h = torch.where(den > 0.0, num / den.clamp(min=1e-6), torch.zeros_like(num))

        y_heads = skip_f * V_f + (1.0 - skip_f) * h
        y = y_heads.transpose(1, 2).contiguous().view(B, L, H * d)
        if y.shape[-1] != self.dim:
            if y.shape[-1] < self.dim:
                y = F.pad(y, (0, self.dim - y.shape[-1]))
            else:
                y = y[..., : self.dim]

        if self.is_nested_encoder or isinstance(self.out_proj, nn.Identity):
            out = y.to(target_dtype)
        else:
            out_w_dtype = self.out_proj.weight.dtype if hasattr(self.out_proj, "weight") else target_dtype
            y_in = y.to(target_dtype if torch.is_autocast_enabled() else out_w_dtype)
            out = self.out_proj(y_in).to(target_dtype)

        if target_dtype == torch.float16:
            out = out.clamp(min=-65000.0, max=65000.0)

        if valid_m is not None:
            out = out * valid_m.unsqueeze(-1).to(out.dtype)

        return out.to(target_dtype), next_state


def inject_simple_ssm(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    **kwargs: Any,
) -> Any | None:
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    if target is None:
        return None

    for attr in ("attention", "self_attn", "attn", "self"):
        if hasattr(target, attr):
            old = getattr(target, attr)
            new_block = SimpleSSM(config=cfg, old_attn=old, lora_rank=lora_rank, layer_idx=layer_idx, **kwargs)
            setattr(target, attr, new_block)
            return new_block

    if isinstance(target, nn.Module) and not hasattr(target, "forward"):
        new_block = SimpleSSM(config=cfg, old_attn=old_attn, lora_rank=lora_rank, layer_idx=layer_idx, **kwargs)
        return new_block

    return None
