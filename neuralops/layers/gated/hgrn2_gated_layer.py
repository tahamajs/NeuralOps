#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HGRN-2 Gated Recurrent Linear Attention Layer

Hierarchically gated recurrent neural network attention layer enabling linear computational complexity.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get, _env_float, _unpack_weight_bias
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

try:
    from neuralops.ops.triton.hgrn2_scan import hgrn2_scan_triton

    HAS_TRITON_HGRN2 = True
    logger.info("HGRN2: Triton scan available")
except Exception as _e:
    logger.warning("HGRN2: Triton scan unavailable: %s", _e)
    hgrn2_scan_triton = None
    HAS_TRITON_HGRN2 = False

try:
    from neuralops.ops.cuda.hgrn2_scan.hgrn2_autograd import hgrn2_scan as hgrn2_scan_cuda

    HAS_CUDA_HGRN2 = True
except Exception as _e:
    logger.debug("HGRN2: CUDA scan unavailable: %s", _e)
    hgrn2_scan_cuda = None
    HAS_CUDA_HGRN2 = False


class HGRN2CacheState:
    """Lightweight cache container for HGRN2Layer."""

    __slots__ = ("hgrn_state", "local_k", "local_v", "past_len")

    def __init__(
        self,
        hgrn_state: torch.Tensor | None = None,
        local_k: torch.Tensor | None = None,
        local_v: torch.Tensor | None = None,
        past_len: int = 0,
    ):
        self.hgrn_state = hgrn_state
        self.local_k = local_k
        self.local_v = local_v
        self.past_len = int(past_len)

    def __iter__(self):
        yield self.hgrn_state

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (self.hgrn_state,)[idx]


@ATTENTION_REGISTRY.register("hgrn2_gated_layer")
class HGRN2Layer(BaseAttention):
    """
    HGRN-2 hybrid with local sliding-window attention and full state caching.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | None = None,
        res_clip: float | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        freeze_base: bool = True,
        window_size: int = 64,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size")
            if d_model is None and isinstance(config, int):
                d_model = config
            if d_model is None and config is not None:
                d_model = _cfg_get(
                    config,
                    "hidden_size",
                    _cfg_get(
                        config,
                        "d_model",
                        _cfg_get(config, "n_embd", _cfg_get(config, "dim", 768)),
                    ),
                )
            if d_model is None:
                d_model = 768
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.d_model = int(d_model)
        self.dim = self.d_model
        self.hidden_size = self.d_model

        _res_clip_default = _env_float("HGRN2_RES_CLIP", 0.0)
        self.res_clip = res_clip if res_clip is not None else (_res_clip_default if _res_clip_default > 0 else None)

        self.freeze_base = bool(freeze_base)

        _mt = str(_cfg_get(config, "model_type", "") or "").lower()
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

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            cfg_is_dec = (
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "causal", False)
                or any(_mt.startswith(p) for p in _decoder_prefixes)
                or kwargs.get("is_decoder", False)
                or kwargs.get("is_causal", False)
            )
            old_attn_kw = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
            if old_attn_kw is not None:
                if hasattr(old_attn_kw, "c_attn") or "gpt" in old_attn_kw.__class__.__name__.lower():
                    cfg_is_dec = True
            if (config is None or getattr(config, "_is_synthetic", False)) and old_attn_kw is None:
                logger.info("HGRN2: defaulting to is_decoder=True for causal safety")
                cfg_is_dec = True
            self.is_decoder = bool(cfg_is_dec)

        if not self.is_decoder:
            logger.warning(
                "HGRN2: is_decoder=False — bidirectional mode. " "Future token leak risk for GPT-family models."
            )

        self.is_nested_encoder: bool = False
        self.lora_rank = int(lora_rank) if lora_rank is not None else 16
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (self.lora_alpha / max(1, self.lora_rank)) if self.lora_rank > 0 else 1.0

        self.num_heads = int(_cfg_get(config, "num_attention_heads", _cfg_get(config, "num_heads", 12)))
        while self.num_heads > 1 and (self.d_model % self.num_heads != 0):
            self.num_heads -= 1
        self.head_dim = self.d_model // max(1, self.num_heads)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj_attn = nn.Linear(self.d_model, self.d_model, bias=True)

        self.lora_forget_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.lora_input_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.lora_g_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        self.state_norm = nn.LayerNorm(self.d_model)
        with torch.no_grad():
            self.state_norm.weight.fill_(1.0)
            self.state_norm.bias.zero_()

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        _forget_bias = _env_float("HGRN2_FORGET_BIAS", -4.0)
        nn.init.normal_(self.lora_forget_gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.lora_forget_gate.bias, _forget_bias)

        nn.init.normal_(self.lora_input_gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.lora_input_gate.bias, 0.0)

        _g_bias = _env_float("HGRN2_G_BIAS", 0.0)
        nn.init.normal_(self.lora_g_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.lora_g_proj.bias, _g_bias)

        for mod in (self.v_proj, self.q_proj, self.k_proj, self.v_proj_attn, self.out_proj):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

        _out_scale = _env_float("HGRN2_OUTPUT_SCALE", 1.0)
        self.lora_output_scale = nn.Parameter(torch.full((1,), _out_scale))

        _fusion_init = _env_float("HGRN2_FUSION_GATE", -4.0)
        self.lora_fusion_gate = nn.Linear(self.d_model, 1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.lora_fusion_gate.weight)
            self.lora_fusion_gate.bias.fill_(_fusion_init)

        self.window_size = int(_cfg_get(config, "window_size", window_size))

        if self.lora_rank > 0:
            self.v_lora_A = nn.Parameter(torch.randn(self.lora_rank, self.d_model) * 0.01)
            self.v_lora_B = nn.Parameter(torch.randn(self.d_model, self.lora_rank) * 1e-4)
            self.out_lora_A = nn.Parameter(torch.randn(self.lora_rank, self.d_model) * 0.01)
            self.out_lora_B = nn.Parameter(torch.randn(self.d_model, self.lora_rank) * 1e-4)
        else:
            self.register_parameter("v_lora_A", None)
            self.register_parameter("v_lora_B", None)
            self.register_parameter("out_lora_A", None)
            self.register_parameter("out_lora_B", None)

        if self.freeze_base:
            for mod in (self.v_proj, self.out_proj, self.q_proj, self.k_proj, self.v_proj_attn):
                for p in mod.parameters():
                    p.requires_grad = False
            for mod in (
                self.lora_forget_gate,
                self.lora_input_gate,
                self.lora_g_proj,
                self.lora_fusion_gate,
            ):
                for p in mod.parameters():
                    p.requires_grad = True

        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
        if old_attn is not None:
            self.copy_weights(old_attn)
            target_dtype = kwargs.get("dtype")
            if target_dtype is None:
                p = next(old_attn.parameters(), None)
                if p is not None and isinstance(p.dtype, torch.dtype):
                    target_dtype = p.dtype
            if target_dtype is not None:
                self.to(target_dtype)

    @property
    def forget_gate(self) -> nn.Linear:
        return self.lora_forget_gate

    @property
    def input_gate(self) -> nn.Linear:
        return self.lora_input_gate

    @property
    def g_proj_alias(self) -> nn.Linear:
        return self.lora_g_proj

    @property
    def g_proj(self) -> nn.Linear:
        return self.lora_g_proj

    @property
    def fusion_gate(self) -> nn.Linear:
        return self.lora_fusion_gate

    @property
    def output_scale(self) -> nn.Parameter:
        return self.lora_output_scale

    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj_attn

    def copy_weights(self, old_attn: nn.Module) -> None:
        w_q = w_k = w_v = w_o = None
        b_q = b_k = b_v = b_o = None

        src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
        is_nested = (
            src_cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (hasattr(old_attn, "query") and not hasattr(old_attn, "dense") and not hasattr(old_attn, "out_proj"))
            or (hasattr(old_attn, "self") and hasattr(old_attn, "output"))
        )
        self.is_nested_encoder = bool(is_nested)

        if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
            src = old_attn.self
            if hasattr(old_attn.output, "dense"):
                w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
        else:
            src = old_attn

        if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
            c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
            if c_attn_w is not None:
                split_size = c_attn_w.shape[0] // 3
                w_q = c_attn_w[:split_size, :]
                w_k = c_attn_w[split_size : 2 * split_size, :]
                w_v = c_attn_w[2 * split_size :, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:split_size]
                    b_k = c_attn_b[split_size : 2 * split_size]
                    b_v = c_attn_b[2 * split_size :]
            w_o, b_o = _unpack_weight_bias(src.c_proj)

        elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
            w_q, b_q = _unpack_weight_bias(src.query)
            w_k, b_k = _unpack_weight_bias(src.key)
            w_v, b_v = _unpack_weight_bias(src.value)
            if w_o is None:
                for attr in ("dense", "out_proj", "o_proj"):
                    if hasattr(src, attr):
                        w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                        break

        elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
            w_q, b_q = _unpack_weight_bias(src.q_proj)
            w_k, b_k = _unpack_weight_bias(src.k_proj)
            w_v, b_v = _unpack_weight_bias(src.v_proj)
            if hasattr(src, "o_proj"):
                w_o, b_o = _unpack_weight_bias(src.o_proj)

        elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
            w_q, b_q = _unpack_weight_bias(src.q_lin)
            w_k, b_k = _unpack_weight_bias(src.k_lin)
            w_v, b_v = _unpack_weight_bias(src.v_lin)
            if hasattr(src, "out_lin"):
                w_o, b_o = _unpack_weight_bias(src.out_lin)

        if w_q is None or w_k is None or w_v is None:
            try:
                adapter = UniversalAdapter(old_attn)
                extracted = adapter.extract_qkv_out()
                if extracted:
                    w_q = extracted.get("q_weight") or w_q
                    w_k = extracted.get("k_weight") or w_k
                    w_v = extracted.get("v_weight") or w_v
                    if w_o is None:
                        w_o = extracted.get("out_weight")
            except Exception as ex:
                logger.debug("UniversalAdapter fallback failed: %s", ex)

        def _copy_param(target_linear: nn.Linear, w, b):
            if w is None:
                return
            with torch.no_grad():
                tw = target_linear.weight
                if w.shape == tw.shape:
                    tw.copy_(w.to(tw.dtype))
                else:
                    min_r = min(tw.shape[0], w.shape[0])
                    min_c = min(tw.shape[1], w.shape[1])
                    tw.data.zero_()
                    tw.data[:min_r, :min_c].copy_(w[:min_r, :min_c].to(tw.dtype))
                if b is not None and target_linear.bias is not None:
                    tb = target_linear.bias
                    if b.shape == tb.shape:
                        tb.copy_(b.to(tb.dtype))
                    else:
                        min_b = min(tb.numel(), b.numel())
                        tb.data.zero_()
                        tb.data[:min_b].copy_(b[:min_b].to(tb.dtype))

        _copy_param(self.q_proj, w_q, b_q)
        _copy_param(self.k_proj, w_k, b_k)
        _copy_param(self.v_proj_attn, w_v, b_v)
        _copy_param(self.v_proj, w_v, b_v)
        _copy_param(self.out_proj, w_o, b_o)

        _copy_param(self.lora_g_proj, w_q, b_q)
        _copy_param(self.lora_input_gate, w_k, b_k)

        fnorm = float(torch.linalg.norm(self.v_proj.weight.detach().float()))
        if fnorm < 1e-4:
            logger.warning("HGRN2Layer v_proj weight norm near zero: %g", fnorm)

        if self.freeze_base:
            for mod in (self.v_proj, self.out_proj, self.q_proj, self.k_proj, self.v_proj_attn):
                for p in mod.parameters():
                    p.requires_grad = False

    def _scan_direction(
        self,
        f: torch.Tensor,
        iv: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        chunk_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Chunked parallel scan for HGRN-2 recurrence.

        Args:
            f: (B, L, D) forget gates in (0, 1)
            iv: (B, L, D) input values
            initial_state: (B, D) or None — past hidden state h_{t-1}
            chunk_size: int

        Returns:
            h: (B, L, D) all hidden states
            h_last: (B, D) final state
        """
        # Triton _hgrn2_bwd_kernel requires power-of-2 hidden dims (e.g. D=896 in Qwen2.5-0.5B segfaults)
        D_dim = iv.shape[-1]
        if HAS_TRITON_HGRN2 and hgrn2_scan_triton is not None and f.is_cuda and D_dim > 0 and (D_dim & (D_dim - 1)) == 0:
            try:
                out = hgrn2_scan_triton(
                    f.contiguous(),
                    iv.contiguous(),
                    res_clip=self.res_clip,
                    initial_state=initial_state,
                    return_final_state=True,
                )
                if isinstance(out, tuple):
                    h, h_last = out
                    return h, h_last
                return out, out[:, -1]
            except Exception as e:
                logger.warning("HGRN2Layer: Triton scan failed (%s); falling back to chunked PyTorch.", e)

        B, L, D = iv.shape
        f_fp32 = f.float()
        iv_fp32 = iv.float()
        log_f = torch.log(f_fp32.clamp(min=1e-6))

        use_cp = torch.is_grad_enabled()
        eff_chunk = max(1, min(chunk_size, 16) if (use_cp and L > 64) else chunk_size)
        if use_cp:
            import torch.utils.checkpoint as cp

        def _hgrn2_chunk_step(
            h_in: torch.Tensor,
            f_c: torch.Tensor,
            iv_c: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            B_c, C_len, D_c = iv_c.shape
            A = torch.cumsum(f_c, dim=1)
            mask = torch.tril(torch.ones(C_len, C_len, device=iv_c.device, dtype=torch.bool))
            mask_4d = mask.unsqueeze(0).unsqueeze(-1)
            A_diff = (A.unsqueeze(2) - A.unsqueeze(1)).masked_fill(~mask_4d, -1e4)
            decay_weights = torch.exp(A_diff) * mask_4d.to(A_diff.dtype)

            dw_bd = decay_weights.permute(0, 3, 1, 2).reshape(B_c * D_c, C_len, C_len)
            iv_bd = iv_c.transpose(1, 2).reshape(B_c * D_c, C_len, 1)
            intra = torch.bmm(dw_bd, iv_bd).view(B_c, D_c, C_len).transpose(1, 2)
            inter = h_in.unsqueeze(1) * torch.exp(A)

            h_c = inter + intra
            if self.res_clip is not None and self.res_clip > 0:
                h_c = torch.clamp(h_c, -self.res_clip, self.res_clip)
            return h_c, h_c[:, -1]

        h_chunks = []
        if initial_state is not None:
            h_prev = initial_state.to(device=iv.device, dtype=torch.float32)
        else:
            h_prev = torch.zeros(B, D, device=iv.device, dtype=torch.float32)

        for start in range(0, L, eff_chunk):
            end = min(start + eff_chunk, L)
            if use_cp:
                h_c, h_prev = cp.checkpoint(
                    _hgrn2_chunk_step,
                    h_prev,
                    log_f[:, start:end],
                    iv_fp32[:, start:end],
                    use_reentrant=False,
                )
            else:
                h_c, h_prev = _hgrn2_chunk_step(
                    h_prev,
                    log_f[:, start:end],
                    iv_fp32[:, start:end],
                )
            h_chunks.append(h_c)

        h = torch.cat(h_chunks, dim=1).to(iv.dtype)
        return h, h_prev.to(iv.dtype)

    def _recurrent_scan(
        self,
        f: torch.Tensor,
        iv: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Forward + (optional) backward scan.

        Returns:
            h_fwd: (B, L, D)
            h_bwd: (B, L, D) or None
            h_last: (B, D) final forward state
        """
        if mask is not None:
            if mask.dim() == 4:
                mask = (mask > -1e4) if mask.dtype.is_floating_point else mask.bool()
                mask = mask.any(dim=-2).any(dim=1)
            elif mask.dim() == 3:
                mask = (mask > -1e4) if mask.dtype.is_floating_point else mask.bool()
                mask = mask.any(dim=1)
            mask_float = mask.float().unsqueeze(-1)
            iv = iv * mask_float
            f = torch.where(mask_float.bool(), f, torch.ones_like(f))

        if self.is_decoder:
            h_fwd, h_last = self._scan_direction(f, iv, initial_state=initial_state)
            return h_fwd, None, h_last

        # Non-causal bidirectional fast path: check Level 1 CUDA (inference only; autograd backward uses inplace PyTorch loop)
        if initial_state is None and f.is_cuda and not torch.is_grad_enabled():
            if HAS_CUDA_HGRN2 and hgrn2_scan_cuda is not None:
                try:
                    from neuralops.ops.cuda.hgrn2_scan.hgrn2_autograd import _get_module as _get_hgrn2_cuda_mod

                    if _get_hgrn2_cuda_mod() is not None:
                        f_4d = f.unsqueeze(1).contiguous()
                        iv_4d = iv.unsqueeze(1).contiguous()
                        fused_out = hgrn2_scan_cuda(iv_4d, f_4d).squeeze(1)
                        return fused_out, None, fused_out[:, -1]
                    logger.warning(
                        "HGRN2Layer: CUDA hgrn2_scan module unavailable; falling back to chunked scan."
                    )
                except Exception as exc:
                    logger.warning("HGRN2Layer: CUDA fused bidirectional scan failed (%s); falling back.", exc)

        h_fwd, h_last = self._scan_direction(f, iv, initial_state=initial_state)
        f_rev = torch.flip(f, dims=[1])
        iv_rev = torch.flip(iv, dims=[1])
        h_rev, _ = self._scan_direction(f_rev, iv_rev, initial_state=None)
        h_bwd = torch.flip(h_rev, dims=[1])
        return h_fwd, h_bwd, h_last

    def _local_attention(
        self,
        hidden_states: torch.Tensor,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        past_len: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute local sliding-window attention with rolling cache.

        Returns:
            out: (B, L, D)
            new_k: (B, H, L, d)  (this step's K, to append to cache)
            new_v: (B, H, L, d)
        """
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states.to(self.q_proj.weight.dtype)).view(B, L, H, d).transpose(1, 2)
        k_new = self.k_proj(hidden_states.to(self.k_proj.weight.dtype)).view(B, L, H, d).transpose(1, 2)
        v_new = self.v_proj_attn(hidden_states.to(self.v_proj_attn.weight.dtype)).view(B, L, H, d).transpose(1, 2)

        if past_k is not None and past_v is not None and past_len > 0:
            k = torch.cat([past_k.to(k_new.dtype), k_new], dim=2)
            v = torch.cat([past_v.to(v_new.dtype), v_new], dim=2)
        else:
            k, v = k_new, v_new

        N_kv = k.shape[2]

        out = None
        if past_len == 0:
            if self.window_size >= L:
                try:
                    drop_p = getattr(self.dropout, "p", 0.0) if (self.training and hasattr(self, "dropout")) else 0.0
                    out = F.scaled_dot_product_attention(
                        q, k, v, is_causal=self.is_decoder, dropout_p=drop_p
                    )
                    out = out.transpose(1, 2).contiguous().view(B, L, D)
                except Exception as exc:
                    logger.warning("HGRN2Layer: SDPA failed (%s); falling back to chunked local attention.", exc)
                    out = None
            elif q.is_cuda and d > 0 and (d & (d - 1)) == 0:
                try:
                    from neuralops.ops.triton.sliding_window import sliding_window_attention_kernel

                    out_win = sliding_window_attention_kernel(
                        q.contiguous(), k.contiguous(), v.contiguous(), window_size=self.window_size, is_causal=self.is_decoder
                    )
                    out = out_win.transpose(1, 2).contiguous().view(B, L, D)
                except Exception as exc:
                    logger.warning(
                        "HGRN2Layer: Triton sliding_window_attention_kernel failed (%s); falling back to PyTorch.",
                        exc,
                    )
                    out = None

        if out is None:
            scale = 1.0 / math.sqrt(max(int(d), 1))
            chunk_q = max(64, self.window_size)
            if L <= chunk_q:
                q_pos = (past_len + torch.arange(L, device=hidden_states.device)).unsqueeze(1)
                k_pos = torch.arange(N_kv, device=hidden_states.device).unsqueeze(0)
                scores = torch.matmul(q, k.transpose(-1, -2)) * scale
                dist = (q_pos - k_pos).abs()
                window_mask = dist <= self.window_size
                if self.is_decoder:
                    window_mask = window_mask & (k_pos <= q_pos)
                scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), -1e4)
                attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
                out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
            else:
                out_chunks = []
                k_pos_all = torch.arange(N_kv, device=hidden_states.device)
                for qs in range(0, L, chunk_q):
                    qe = min(qs + chunk_q, L)
                    q_c = q[:, :, qs:qe, :]
                    q_pos_c = (past_len + torch.arange(qs, qe, device=hidden_states.device)).unsqueeze(1)
                    k_start = max(0, past_len + qs - self.window_size)
                    k_end = N_kv if self.is_decoder and past_len + qe - 1 >= N_kv else min(N_kv, past_len + qe + self.window_size)
                    k_c = k[:, :, k_start:k_end, :]
                    v_c = v[:, :, k_start:k_end, :]
                    k_pos_c = k_pos_all[k_start:k_end].unsqueeze(0)

                    scores_c = torch.matmul(q_c, k_c.transpose(-1, -2)) * scale
                    w_mask = (q_pos_c - k_pos_c).abs() <= self.window_size
                    if self.is_decoder:
                        w_mask = w_mask & (k_pos_c <= q_pos_c)
                    scores_c = scores_c.masked_fill(~w_mask.unsqueeze(0).unsqueeze(0), -1e4)
                    attn_c = F.softmax(scores_c.float(), dim=-1).to(v_c.dtype)
                    out_chunks.append(torch.matmul(attn_c, v_c))
                out = torch.cat(out_chunks, dim=2).transpose(1, 2).contiguous().view(B, L, D)

        return out.to(hidden_states.dtype), k_new.to(hidden_states.dtype), v_new.to(hidden_states.dtype)

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
        is_causal = kwargs.get("is_causal", None)
        if is_causal is not None and self.is_decoder != is_causal:
            raise RuntimeError(
                f"HGRN2: is_decoder={self.is_decoder} but forward(is_causal={is_causal}). "
                f"Causal decoders MUST use is_decoder=True to prevent future-token leakage. "
                f"See docs/COMPILE_LIMITATIONS.md."
            )

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention not supported in HGRN2Layer")

        B, L, D = hidden_states.shape
        target_dtype = hidden_states.dtype

        past_hgrn_state = None
        past_local_k = None
        past_local_v = None
        past_len = 0

        if past_key_value is not None:
            if isinstance(past_key_value, HGRN2CacheState):
                past_hgrn_state = past_key_value.hgrn_state
                past_local_k = past_key_value.local_k
                past_local_v = past_key_value.local_v
                past_len = past_key_value.past_len
            elif isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) >= 1:
                    past_hgrn_state = past_key_value[0]
                if len(past_key_value) >= 2:
                    past_local_k = past_key_value[1]
                if len(past_key_value) >= 3:
                    past_local_v = past_key_value[2]
                if len(past_key_value) >= 4:
                    past_len = int(past_key_value[3])

        raw_log_decay = -F.softplus(self.lora_forget_gate(hidden_states.to(self.lora_forget_gate.weight.dtype)))
        bounded_log_decay = raw_log_decay.clamp(min=-3.0, max=-1e-4)
        f = torch.exp(bounded_log_decay).to(target_dtype)

        i = torch.sigmoid(self.lora_input_gate(hidden_states.to(self.lora_input_gate.weight.dtype))).to(target_dtype)
        input_scale = ((1.0 - f) * i).to(target_dtype)

        v = self.v_proj(hidden_states.to(self.v_proj.weight.dtype)).to(target_dtype)
        if self.lora_rank > 0 and self.v_lora_A is not None and self.v_lora_B is not None:
            v_lora = (
                torch.matmul(
                    torch.matmul(
                        hidden_states,
                        self.v_lora_A.t().contiguous().to(target_dtype),
                    ),
                    self.v_lora_B.t().contiguous().to(target_dtype),
                )
                * self.lora_scaling
            )
            v = v + v_lora

        z = self.lora_g_proj(hidden_states.to(self.lora_g_proj.weight.dtype)).to(target_dtype)
        iv = (input_scale * v).to(target_dtype)

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 3:
                m = (attention_mask > -1e4) if attention_mask.dtype.is_floating_point else attention_mask.bool()
                mask_bool = m.any(dim=1) if m.shape[1] > 1 else m[:, 0, :]
            elif attention_mask.dim() == 4:
                m = (attention_mask > -1e4) if attention_mask.dtype.is_floating_point else attention_mask.bool()
                mask_bool = m.any(dim=-2).any(dim=1)

        h_fwd, h_bwd, next_hgrn_state = self._recurrent_scan(f, iv, initial_state=past_hgrn_state, mask=mask_bool)

        if self.is_decoder or h_bwd is None:
            h_ctx = h_fwd.to(target_dtype)
        else:
            h_ctx = (h_fwd + h_bwd).to(target_dtype)

        sn_dtype = (
            self.state_norm.weight.dtype
            if hasattr(self.state_norm, "weight") and self.state_norm.weight is not None
            else target_dtype
        )
        h_norm = self.state_norm(h_ctx.to(sn_dtype)).to(target_dtype)
        hgrn_out = (h_norm * F.silu(z)).to(target_dtype)

        local_out, next_local_k, next_local_v = self._local_attention(
            hidden_states,
            past_k=past_local_k,
            past_v=past_local_v,
            past_len=past_len,
        )

        fusion_logits = self.lora_fusion_gate(hidden_states.to(self.lora_fusion_gate.weight.dtype)).to(target_dtype)
        gate = torch.sigmoid(fusion_logits)
        fused = (gate * hgrn_out + (1 - gate) * local_out.to(target_dtype)).to(target_dtype)

        out = self.out_proj(fused.to(self.out_proj.weight.dtype)).to(target_dtype)
        if self.lora_rank > 0 and self.out_lora_A is not None and self.out_lora_B is not None:
            out_lora = (
                torch.matmul(
                    torch.matmul(
                        fused,
                        self.out_lora_A.t().contiguous().to(target_dtype),
                    ),
                    self.out_lora_B.t().contiguous().to(target_dtype),
                )
                * self.lora_scaling
            )
            out = out + out_lora

        out = out * self.lora_output_scale

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., : hidden_states.shape[-1]]

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        new_past_key_value: Any = None
        use_cache = kwargs.get("use_cache", False) or (past_key_value is not None and past_key_value is not False)
        if use_cache or self.is_decoder:
            new_past_key_value = HGRN2CacheState(
                hgrn_state=next_hgrn_state,
                local_k=next_local_k,
                local_v=next_local_v,
                past_len=past_len + L,
            )

        return out.to(target_dtype), new_past_key_value


def inject_hgrn2_layer(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_hgrn2_layer: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and not hasattr(target, "forward"))
    ):
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer

        if layer_list is not None:
            if kwargs.get("freeze_base", True):
                for p in target.parameters():
                    p.requires_grad = False
            num_injected = 0
            tot = len(layer_list)
            sub_kwargs = {k: v for k, v in kwargs.items() if k not in ("lora_rank", "lora_alpha", "freeze_base")}
            for idx, lyr in enumerate(layer_list):
                res = inject_hgrn2_layer(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    _depth=_depth + 1,
                    **sub_kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    curr_attn = old_attn
    is_nested_target = False
    if curr_attn is None and target is not None:
        if hasattr(target, "attention") and hasattr(target.attention, "self"):
            curr_attn = target.attention
            is_nested_target = True
        elif hasattr(target, "attention"):
            curr_attn = target.attention
        elif hasattr(target, "self"):
            curr_attn = target.self
            is_nested_target = True
        elif hasattr(target, "attn"):
            curr_attn = target.attn
        elif hasattr(target, "self_attn"):
            curr_attn = target.self_attn

    if curr_attn is None:
        return None

    if isinstance(curr_attn, HGRN2Layer):
        return None
    if "HGRN2" in curr_attn.__class__.__name__:
        return None

    _mt = str(getattr(cfg, "model_type", "") or "").lower()
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
        bool(getattr(cfg, "is_decoder", False) if cfg else False)
        or bool(getattr(cfg, "causal", False) if cfg else False)
        or any(_mt.startswith(p) for p in _dec_pfx)
        or hasattr(curr_attn, "c_attn")
        or "gpt" in curr_attn.__class__.__name__.lower()
    )

    try:
        new_layer = HGRN2Layer(
            config=cfg,
            old_attn=curr_attn,
            is_decoder=is_dec,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            **{k: v for k, v in kwargs.items() if k not in ("lora_rank", "lora_alpha")},
        )
        if curr_attn is not None:
            p = next(curr_attn.parameters(), None)
            if p is not None:
                new_layer.to(device=p.device, dtype=p.dtype)
    except Exception as ex:
        logger.exception("Failed to instantiate HGRN2Layer: %s", ex)
        return None

    if target is not None:
        spliced = False
        if is_nested_target and hasattr(target, "attention") and hasattr(target.attention, "self"):
            target.attention.self = new_layer
            if hasattr(target.attention, "output") and hasattr(target.attention.output, "dense"):
                target.attention.output.dense = nn.Identity()
            spliced = True
        elif hasattr(target, "attention"):
            target.attention = new_layer
            spliced = True
        elif hasattr(target, "self"):
            target.self = new_layer
            if hasattr(target, "output") and hasattr(target.output, "dense"):
                target.output.dense = nn.Identity()
            spliced = True
        elif hasattr(target, "attn"):
            target.attn = new_layer
            spliced = True
        elif hasattr(target, "self_attn"):
            target.self_attn = new_layer
            spliced = True

        if not spliced:
            logger.error(
                "inject_hgrn2_layer: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

    return new_layer


__all__ = [
    "HGRN2Layer",
    "HGRN2CacheState",
    "inject_hgrn2_layer",
]
