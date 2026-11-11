#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GLA: Gated Linear Attention with Data-Dependent Decay

Gated linear attention introducing data-dependent decay gates to associative state updates.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

FLA_AVAILABLE = True

try:
    from fla.ops.gla import chunk_gla
except Exception:
    chunk_gla = None
    FLA_AVAILABLE = False
    logger.debug("chunk_gla unavailable — using PyTorch fallback")


@ATTENTION_REGISTRY.register(
    "gla_attention",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=False,
)
class GLAttention(BaseAttention):
    """
    Gated Linear Attention layer with FLA backend.

    Recurrence:
        S_t = exp(g_t) * S_{t-1} + (K_t / sqrt(d))^T @ V_t
        O_t = Q_t @ S_t

    where g_t = logsigmoid(gate_logits / tau) <= 0
    """

    def __init__(
        self,
        config=None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        gate_rank: int = 16,
        gate_tau: float = 16.0,
        use_fla: bool = True,
        compute_dtype: str = "bfloat16",
        old_attn: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(config, None)

        if config is not None:
            if hidden_size is None:
                hidden_size = getattr(
                    config, "hidden_size", getattr(config, "n_embd", getattr(config, "d_model", None))
                )
            if num_heads is None:
                num_heads = getattr(
                    config, "num_attention_heads", getattr(config, "num_heads", getattr(config, "n_head", None))
                )
            if head_dim is None and hasattr(config, "head_dim"):
                head_dim = config.head_dim

        if hidden_size is None:
            hidden_size = kwargs.get("dim", 768)
        if num_heads is None:
            num_heads = 12
        if head_dim is None:
            head_dim = hidden_size // num_heads

        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.all_head_size = num_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.gate_rank = gate_rank
        self.gate_tau = gate_tau
        self.use_fla = use_fla and FLA_AVAILABLE
        self.compute_dtype = getattr(torch, compute_dtype) if isinstance(compute_dtype, str) else compute_dtype

        model_type = str(getattr(config, "model_type", "") or "").lower() if config else ""
        dec_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo")
        is_dec = False
        if config is not None:
            is_dec = bool(getattr(config, "is_decoder", False) or getattr(config, "is_causal", False))
        if (
            any(model_type.startswith(p) for p in dec_prefixes)
            or kwargs.get("is_decoder", False)
            or kwargs.get("is_causal", False)
        ):
            is_dec = True
        if old_attn is not None and (hasattr(old_attn, "c_attn") or getattr(old_attn, "is_decoder", False)):
            is_dec = True
        self.is_decoder = is_dec
        self.is_nested_encoder = False

        self.q_proj = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.k_proj = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.v_proj = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.out_proj = nn.Linear(self.all_head_size, self.dim, bias=True)

        self.gate_W1 = nn.Linear(self.dim, gate_rank, bias=False)
        self.gate_W2 = nn.Linear(gate_rank, self.num_heads, bias=True)

        with torch.no_grad():
            self.gate_W2.bias.fill_(16.0)
            self.gate_W1.weight.normal_(0.0, 0.02)

        self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self._fla_kernel = chunk_gla if self.use_fla else None
        if self._fla_kernel is not None:
            logger.info("GLAttention: FLA chunk_gla loaded (Tensor Core)")
        else:
            logger.info("GLAttention: PyTorch fallback loaded")

        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, src: nn.Module) -> None:
        """Copies pretrained weights from standard attention projections."""
        if src is None:
            return

        if hasattr(src, "c_attn") and src.c_attn is not None:
            is_conv1d = "Conv1D" in type(src.c_attn).__name__
            c_w = src.c_attn.weight.data
            if is_conv1d or c_w.shape[0] < c_w.shape[1]:
                c_w = c_w.t()
            sz = c_w.size(0) // 3
            with torch.no_grad():
                self.q_proj.weight.copy_(c_w[:sz, :])
                self.k_proj.weight.copy_(c_w[sz : 2 * sz, :])
                self.v_proj.weight.copy_(c_w[2 * sz :, :])
                if src.c_attn.bias is not None:
                    c_b = src.c_attn.bias.data
                    self.q_proj.bias.copy_(c_b[:sz])
                    self.k_proj.bias.copy_(c_b[sz : 2 * sz])
                    self.v_proj.bias.copy_(c_b[2 * sz :])

            if hasattr(src, "c_proj") and src.c_proj is not None:
                is_conv1d_proj = "Conv1D" in type(src.c_proj).__name__
                o_w = src.c_proj.weight.data
                if is_conv1d_proj or o_w.shape[0] < o_w.shape[1]:
                    o_w = o_w.t()
                with torch.no_grad():
                    self.out_proj.weight.copy_(o_w)
                    if src.c_proj.bias is not None:
                        self.out_proj.bias.copy_(src.c_proj.bias.data)
        else:
            cls_name = getattr(src, "__class__", type(src)).__name__
            has_out = any(hasattr(src, a) for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))
            is_nested = False
            if (
                cls_name in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
                or (hasattr(src, "query") and not has_out)
                or (hasattr(src, "self") and hasattr(src, "output"))
            ):
                is_nested = True
            self.is_nested_encoder = bool(is_nested)

            aliases = {
                "q_proj": ["q_proj", "query", "q_lin"],
                "k_proj": ["k_proj", "key", "k_lin"],
                "v_proj": ["v_proj", "value", "v_lin"],
                "out_proj": ["out_proj", "o_proj", "out_lin", "dense"],
            }
            out_copied = False
            for tgt_name, names in aliases.items():
                for name in names:
                    src_mod = getattr(src, name, None)
                    if src_mod is not None and hasattr(src_mod, "weight"):
                        target = getattr(self, tgt_name)
                        with torch.no_grad():
                            src_w = src_mod.weight.data
                            if src_w.shape == target.weight.shape:
                                target.weight.copy_(src_w)
                                if tgt_name == "out_proj":
                                    out_copied = True
                            elif (
                                src_w.dim() == 2
                                and target.weight.dim() == 2
                                and src_w.shape[1] == target.weight.shape[1]
                                and target.weight.shape[0] % src_w.shape[0] == 0
                            ):
                                # GQA: expand num_kv_heads to num_q_heads via repeat_interleave
                                g = target.weight.shape[0] // src_w.shape[0]
                                target.weight.copy_(src_w.repeat_interleave(g, dim=0))
                                if tgt_name == "out_proj":
                                    out_copied = True

                            if hasattr(src_mod, "bias") and src_mod.bias is not None and target.bias is not None:
                                src_b = src_mod.bias.data
                                if src_b.shape == target.bias.shape:
                                    target.bias.copy_(src_b)
                                elif (
                                    src_b.dim() == 1
                                    and target.bias.dim() == 1
                                    and target.bias.shape[0] % src_b.shape[0] == 0
                                ):
                                    g = target.bias.shape[0] // src_b.shape[0]
                                    target.bias.copy_(src_b.repeat_interleave(g, dim=0))
                        break

            if self.is_nested_encoder and not out_copied:
                with torch.no_grad():
                    nn.init.eye_(self.out_proj.weight)
                    if self.out_proj.bias is not None:
                        self.out_proj.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        gate_logits = self.gate_W2(self.gate_W1(hidden_states))
        g = F.logsigmoid((gate_logits / self.gate_tau).float()).to(gate_logits.dtype)

        Q = self.q_proj(hidden_states).view(B, N, H, d)
        K = self.k_proj(hidden_states).view(B, N, H, d)
        V = self.v_proj(hidden_states).view(B, N, H, d)

        if not self.is_decoder:
            # Bidirectional mode with positive feature maps and symmetric state decay
            q_all = (F.elu(Q.float()) + 1.0).permute(0, 2, 1, 3)
            k_all = (F.elu(K.float()) + 1.0).permute(0, 2, 1, 3)
            v_all = V.float().permute(0, 2, 1, 3)

            diff_sym = -torch.abs(
                torch.arange(N, device=hidden_states.device).unsqueeze(0)
                - torch.arange(N, device=hidden_states.device).unsqueeze(1)
            ).float()
            decay_sym = torch.exp(0.01 * diff_sym).unsqueeze(0).unsqueeze(0)
            scores = torch.matmul(q_all, k_all.transpose(-1, -2)) * decay_sym
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask = (
                        (attention_mask > 0)
                        if not attention_mask.dtype.is_floating_point
                        else (attention_mask > -1e4)
                    ).unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(~mask, 0.0)
                elif attention_mask.dim() == 4:
                    mask = (
                        (attention_mask > -1e4)
                        if attention_mask.dtype.is_floating_point
                        else attention_mask.bool()
                    )
                    scores = scores.masked_fill(~mask, 0.0)
            den = scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            attn = scores / den
            out = torch.matmul(attn, v_all).permute(0, 2, 1, 3).contiguous().to(hidden_states.dtype)
        else:
            out = None
            if self._fla_kernel is not None and Q.is_cuda and Q.dtype == torch.bfloat16 and attention_mask is None:
                try:
                    Q_f = Q.contiguous()
                    K_f = K.contiguous()
                    V_f = V.contiguous()
                    g_f = g.unsqueeze(-1).expand(B, N, H, d).to(torch.bfloat16).contiguous()

                    out_raw, _ = self._fla_kernel(
                        Q_f,
                        K_f,
                        V_f,
                        g_f,
                        scale=self.scale,
                        output_final_state=False,
                    )
                    out = out_raw.float()
                except Exception as e:
                    from neuralops.utils.logging_config import warn_once

                    warn_once("gla_fla_fallback", "FLA failed with %s; switching to Triton/PyTorch fallback", e)

            if out is None:
                # Level 1: CUDA gla_scan fastpath
                if Q.is_cuda and Q.dtype in (torch.float16, torch.bfloat16) and d <= 128 and attention_mask is None:
                    try:
                        from neuralops.ops.cuda.gla_scan import gla_scan

                        q_t = Q.transpose(1, 2).contiguous()
                        k_t = K.transpose(1, 2).contiguous()
                        v_t = V.transpose(1, 2).contiguous()
                        alpha_t = torch.exp(g.transpose(1, 2).contiguous().float()).clamp(1e-4, 1.0 - 1e-4).to(Q.dtype)
                        out_t = gla_scan(q_t, k_t, v_t, alpha=alpha_t)
                        out = out_t.transpose(1, 2).contiguous().float()
                    except Exception:
                        out = None

            if out is None:
                # Numerically stable vectorized chunked GLAttention path in FP32
                out = self._pytorch_gla_stable(Q, K, V, g, attention_mask=attention_mask).float()

            out = self.o_norm(out.to(self.o_norm.weight.dtype))
            if hidden_states.dtype == torch.float16:
                fmax = torch.finfo(torch.float16).max * 0.5
                out = out.clamp(min=-fmax, max=fmax)
            out = out.to(hidden_states.dtype)

        out = out.contiguous().view(B, N, self.all_head_size)
        if self.is_nested_encoder:
            return out, None
        return self.out_proj(out), None

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for GLA."""
        from neuralops.ops.triton.state_update import fused_recurrent_gla_step

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

        gate_logits = self.gate_W2(self.gate_W1(x))
        g = F.logsigmoid((gate_logits / self.gate_tau).float()).to(gate_logits.dtype)

        Q = self.q_proj(x).view(B, 1, H, d).transpose(1, 2).squeeze(2)
        K = self.k_proj(x).view(B, 1, H, d).transpose(1, 2).squeeze(2)
        V = self.v_proj(x).view(B, 1, H, d).transpose(1, 2).squeeze(2)
        g_in = g.transpose(1, 2).squeeze(2)

        if "kv_state" not in state or state["kv_state"] is None:
            state["kv_state"] = torch.zeros((B, H, d, d), device=device, dtype=torch.float32)

        kv_state = state["kv_state"]

        out_t, kv_state = fused_recurrent_gla_step(Q, K, V, g_in, kv_state, scale=self.scale)
        state["kv_state"] = kv_state
        state["step_idx"] = step_idx + 1

        out_h = out_t.unsqueeze(1).view(B, 1, H, d)
        out_normed = self.o_norm(out_h.float()).to(dtype).contiguous().view(B, 1, self.all_head_size)

        v_flat = V.unsqueeze(1).contiguous().view(B, 1, self.all_head_size).float()
        out_f = out_normed.float()
        v_rms = torch.sqrt(torch.mean(v_flat**2, dim=-1, keepdim=True).clamp_min(1e-8))
        o_rms = torch.sqrt(torch.mean(out_f**2, dim=-1, keepdim=True).clamp_min(1e-8))
        norm_scale = (v_rms / o_rms).clamp(0.25, 4.0).to(dtype)
        out_scaled = out_normed * norm_scale

        out = self.out_proj(out_scaled)
        if is_2d:
            out = out.squeeze(1)

        return out, state

    def _chunked_gla_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_gate: torch.Tensor,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Chunked GLA forward — parallelizable, memory O(N*d).
        Mathematically identical to naive recurrent GLA.
        q, k, v: (B, H, N, D)
        log_gate: (B, H, N)
        Returns:
            output: (B, H, N, D)
            final_state: (B, H, D, D)
        """
        B, H, N, D = q.shape
        device = q.device
        dtype = q.dtype

        num_chunks = (N + chunk_size - 1) // chunk_size
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        state = torch.zeros(B, H, D, D, device=device, dtype=compute_dtype)
        outputs = []
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool))

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)
            L = end - start

            q_c = q[:, :, start:end].to(compute_dtype)
            k_c = k[:, :, start:end].to(compute_dtype)
            v_c = v[:, :, start:end].to(compute_dtype)
            g_c = log_gate[:, :, start:end].to(compute_dtype)

            g_cumsum = torch.cumsum(g_c, dim=-1)

            decay_inter = g_cumsum.exp()
            q_inter = q_c * decay_inter.unsqueeze(-1)
            o_inter = torch.einsum("bhld,bhde->bhle", q_inter, state)

            g_diff = g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)
            decay_intra = g_diff.exp() * causal_mask[:L, :L].float()
            qk = torch.einsum("bhld,bhsd->bhls", q_c, k_c)
            attn = qk * decay_intra
            o_intra = torch.einsum("bhls,bhsd->bhld", attn, v_c)

            outputs.append((o_inter + o_intra).to(dtype))

            g_last = g_cumsum[:, :, -1:]
            decay_state = g_last.exp().unsqueeze(-1)
            state = state * decay_state

            decay_kv = (g_last - g_cumsum).exp()
            v_weighted = v_c * decay_kv.unsqueeze(-1)
            kv_chunk = torch.einsum("bhld,bhle->bhde", k_c, v_weighted)
            state = state + kv_chunk

        out = torch.cat(outputs, dim=2)
        return out, state

    def _pytorch_gla_stable(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        g: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Vectorized chunked GLA fallback strictly in FP32 with consistent inter/intra normalization.
        Q, K, V: (B, N, H, d)
        g:       (B, N, H) <= 0
        """
        B, N, H, d = Q.shape
        device = Q.device
        chunk_size = 64
        pad_len = (chunk_size - (N % chunk_size)) % chunk_size

        q_f = (F.elu(Q.float()) + 1.0) * self.scale
        k_f = F.elu(K.float()) + 1.0
        v_f = V.float()
        g_f = g.float().clamp(max=0.0)

        if attention_mask is not None:
            from neuralops.core.attention_mask import get_token_valid_mask

            valid = get_token_valid_mask(attention_mask, B, N, device)
            if valid is not None:
                kv_mask = valid.view(B, N, 1, 1).float()
                k_f = k_f * kv_mask
                v_f = v_f * kv_mask
                g_f = g_f.masked_fill(~valid.view(B, N, 1), 0.0)

        q_h = q_f.transpose(1, 2)  # (B, H, N, d)
        k_h = k_f.transpose(1, 2)
        v_h = v_f.transpose(1, 2)
        g_h = g_f.transpose(1, 2)  # (B, H, N)

        if pad_len > 0:
            q_h = F.pad(q_h, (0, 0, 0, pad_len))
            k_h = F.pad(k_h, (0, 0, 0, pad_len))
            v_h = F.pad(v_h, (0, 0, 0, pad_len))
            g_h = F.pad(g_h, (0, pad_len), value=0.0)

        num_chunks = q_h.shape[2] // chunk_size
        q_blk = q_h.view(B, H, num_chunks, chunk_size, d)
        k_blk = k_h.view(B, H, num_chunks, chunk_size, d)
        v_blk = v_h.view(B, H, num_chunks, chunk_size, d)
        g_blk = g_h.view(B, H, num_chunks, chunk_size)

        b_log = torch.cumsum(g_blk, dim=-1)  # <= 0
        diff_log = (b_log.unsqueeze(-1) - b_log.unsqueeze(-2)).clamp(max=0.0)
        causal = torch.tril(torch.ones((chunk_size, chunk_size), device=device, dtype=torch.bool))
        decay = torch.exp(diff_log) * causal.float()

        scores_intra = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * decay
        num_intra = torch.matmul(scores_intra, v_blk)
        den_intra = scores_intra.sum(dim=-1, keepdim=True)

        q_b = q_blk * torch.exp(b_log).unsqueeze(-1)
        g_end = b_log[..., -1:]
        decay_total = torch.exp(g_end).unsqueeze(-1)  # (B, H, num_chunks, 1, 1)
        weights = torch.exp((g_end - b_log).clamp(max=0.0)).unsqueeze(-1)
        k_w = k_blk * weights
        kv_chunks = torch.matmul(k_w.transpose(-1, -2), v_blk)  # (B, H, num_chunks, d, d)
        z_chunks = k_w.sum(dim=-2)  # (B, H, num_chunks, d)

        S = torch.zeros((B, H, d, d), device=device, dtype=torch.float32)
        Z = torch.zeros((B, H, d), device=device, dtype=torch.float32)

        out_chunks = []
        for i in range(num_chunks):
            num_inter = torch.matmul(q_b[:, :, i], S)
            den_inter = (q_b[:, :, i] * Z.unsqueeze(-2)).sum(dim=-1, keepdim=True)
            den = (den_inter + den_intra[:, :, i]).clamp(min=1e-4)
            out_chunks.append((num_inter + num_intra[:, :, i]) / den)

            dec_i = decay_total[:, :, i]
            S = S * dec_i + kv_chunks[:, :, i]
            Z = Z * dec_i.squeeze(-1) + z_chunks[:, :, i]

        out = torch.cat(out_chunks, dim=2)[:, :, :N, :]
        return out.transpose(1, 2).contiguous()


def _chunked_gla_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_gate: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Module-level function for chunked GLA forward pass."""
    dummy = GLAttention(hidden_size=q.shape[-1] * q.shape[1], num_heads=q.shape[1]).to(q.device)
    return dummy._chunked_gla_forward(q, k, v, log_gate, chunk_size)
