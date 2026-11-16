#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Kernels, routing, and configuration for Aegis-Spectron-FP8 Hybrid Attention."""

import logging
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def fp8_cast_simulate(
    x: torch.Tensor, format: str = "e4m3", per_channel: bool = False, use_ste: bool | None = None
) -> torch.Tensor:
    if os.environ.get("AEGIS_DISABLE_FP8", "0") == "1":
        return x
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        return x
    if use_ste is None:
        use_ste = x.requires_grad and torch.is_grad_enabled()
    orig_dtype = x.dtype

    if x.is_cuda and hasattr(torch, "float8_e4m3fn") and format.lower() in ("e4m3", "float8_e4m3fn"):
        x_f = x.float()
        amax = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = 448.0 / amax
        x_scaled = (x_f * scale).clamp(min=-448.0, max=448.0)
        x_dequant = x_scaled.to(torch.float8_e4m3fn).to(orig_dtype) / scale.to(orig_dtype)
        if use_ste:
            return x + (x_dequant - x).detach()
        return x_dequant

    x_f = x.float()
    fmt = format.lower()
    if fmt in ("e4m3", "float8_e4m3fn"):
        max_val, mantissa_bits, torch_dtype = 448.0, 3, getattr(torch, "float8_e4m3fn", None)
    else:
        max_val, mantissa_bits, torch_dtype = 57344.0, 2, getattr(torch, "float8_e5m2", None)

    amax = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)

    scale = max_val / amax
    x_scaled = (x_f * scale).clamp(min=-max_val, max=max_val)
    x_dequant = None

    if torch_dtype is not None:
        try:
            x_dequant = (x_scaled.to(torch_dtype).float() / scale.float().clamp(min=1e-8)).to(orig_dtype)
        except Exception as exc:
            logger.debug("Native FP8 dequantization failed: %s; falling back to software simulation", exc)

    if x_dequant is None:
        abs_x = x_scaled.abs().clamp(min=1e-12)
        exp = torch.floor(torch.log2(abs_x))
        factor = 2.0 ** (exp - mantissa_bits)
        rounded = torch.round(abs_x / factor) * factor
        x_dequant = (
            (torch.sign(x_scaled) * rounded).clamp(min=-max_val, max=max_val) / scale.float().clamp(min=1e-8)
        ).to(orig_dtype)

    if use_ste:
        return x + (x_dequant - x).detach()
    return x_dequant


class AegisSpectronFP8Config:
    def __init__(self, **kwargs: Any):
        self.hidden_size = int(kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768))))
        self.num_heads = int(kwargs.get("num_heads", kwargs.get("num_attention_heads", kwargs.get("n_heads", 4))))
        self.num_key_value_heads = int(kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", self.num_heads)))
        self.head_dim = int(
            kwargs.get(
                "head_dim",
                kwargs.get("kv_channels", self.hidden_size // max(1, self.num_heads)),
            )
        )
        self.window_size = int(kwargs.get("window_size", kwargs.get("local_window", kwargs.get("block_size", 64))))
        self.svd_rank = int(kwargs.get("svd_rank", kwargs.pop("rank", min(128, self.hidden_size))))
        self.lora_rank = int(kwargs.get("lora_rank", kwargs.get("lora_r", 32)))
        self.temperature = float(kwargs.get("temperature", 1.0))
        self.lambda_cost = float(kwargs.get("lambda_cost", 0.1))
        self.c_loc = float(kwargs.get("c_loc", 0.5))
        self.c_glob = float(kwargs.get("c_glob", 0.8))
        self.c_lin = float(kwargs.get("c_lin", 1.0))
        self.dropout = float(kwargs.get("dropout", 0.1))
        env_bias = os.environ.get("AEGIS_ROUTER_BIAS", None)
        if env_bias is not None:
            init_bias = tuple(float(x.strip()) for x in env_bias.split(","))
        else:
            init_bias = tuple(kwargs.get("router_init_bias", (4.0, 0.0, 0.0)))
        self.router_init_bias = init_bias
        self.is_decoder = bool(kwargs.get("is_decoder", False))
        self.fp8_format = str(kwargs.get("fp8_format", "e4m3"))
        self.chunk_size = int(kwargs.get("chunk_size", 64))


class SpectronSpectralEntropyRouter(nn.Module):
    def __init__(self, config: AegisSpectronFP8Config):
        super().__init__()
        self.tau = float(config.temperature)
        self.lam = float(config.lambda_cost)
        self.register_buffer("costs", torch.tensor([config.c_loc, config.c_glob, config.c_lin]))
        self.h_proj = nn.Linear(config.hidden_size, 3)
        nn.init.zeros_(self.h_proj.weight)
        self.h_proj.bias.data.copy_(torch.tensor(config.router_init_bias, dtype=self.h_proj.bias.dtype))

    def forward(self, X: torch.Tensor, token_valid: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = X.shape

        with torch.no_grad():
            S_sq = X.float().pow(2)
            denom = S_sq.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            p = (S_sq / denom).clamp(min=1e-8, max=1.0)
            entropy = -(p * torch.log(p)).sum(dim=-1)

            if token_valid is not None:
                entropy = entropy * token_valid.float()

            entropy = entropy / math.log(D)

        h = self.h_proj(X.to(self.h_proj.weight.dtype))
        h = h * entropy.unsqueeze(-1).to(h.dtype)

        c = self.costs.to(h.dtype)
        logits = (h - self.lam * c) / max(self.tau, 1e-4)
        return F.softmax(logits.float(), dim=-1).to(X.dtype)


def _local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    token_valid: torch.Tensor | None,
    is_decoder: bool,
    past_len: int,
    dropout_p: float,
    training: bool,
    return_attn: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, h, N, d = Q.shape
    N_k = K.shape[2]
    radius = window // 2
    scale = 1.0 / math.sqrt(max(int(d), 1))

    if (
        Q.is_cuda
        and (token_valid is None or bool(token_valid.all()))
        and past_len == 0
        and N == N_k
        and d <= 256
        and (not training or dropout_p == 0.0)
    ):
        try:
            from neuralops.ops.cuda.hybrid_local_kernel import hybrid_local_attn

            return hybrid_local_attn(Q, K, V, radius=radius, is_causal=is_decoder), None
        except Exception as exc:
            logger.warning("CUDA hybrid_local_attn failed (%s); falling back to PyTorch implementation", exc)

    if os.environ.get("AEGIS_DISABLE_LOCAL", "0") == "1":
        abs_q = (past_len + torch.arange(N, device=Q.device)).unsqueeze(1)
        abs_k = torch.arange(N_k, device=Q.device).unsqueeze(0)
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
        if is_decoder:
            causal = abs_k <= abs_q
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)
        if token_valid is not None:
            scores = scores.masked_fill(~token_valid[:, None, None, :].to(torch.bool), -1e4)
        attn = F.softmax(scores.float(), dim=-1).to(Q.dtype)
        attn = F.dropout(attn, p=dropout_p, training=training)
        return torch.matmul(attn, V.to(attn.dtype)), attn

    if (
        Q.is_cuda
        and hasattr(F, "scaled_dot_product_attention")
        and not return_attn
        and (token_valid is None or bool(token_valid.all()))
    ):
        abs_q = (past_len + torch.arange(N, device=Q.device)).unsqueeze(1)
        abs_k = torch.arange(N_k, device=Q.device).unsqueeze(0)
        band = ((abs_q - abs_k).abs() <= radius).unsqueeze(0).unsqueeze(0)
        if is_decoder:
            band = band & (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)
        mask_val = -1e4
        attn_bias = torch.zeros(1, 1, N, N_k, device=Q.device, dtype=Q.dtype)
        attn_bias = attn_bias.masked_fill(~band, mask_val)
        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_bias,
            dropout_p=dropout_p if training else 0.0,
            is_causal=False,
        )
        return out, None

    if not return_attn:
        chunk_size = 64
        mask_val = -1e4
        outputs = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            q_abs_start = past_len + start
            q_abs_end = past_len + end
            k_start = max(0, q_abs_start - radius)
            k_end = min(N_k, q_abs_end if is_decoder else q_abs_end + radius)
            if k_end <= k_start:
                outputs.append(Q.new_zeros(B, h, end - start, d))
                continue

            Qc = Q[:, :, start:end]
            Kc = K[:, :, k_start:k_end]
            Vc = V[:, :, k_start:k_end]

            abs_q = torch.arange(q_abs_start, q_abs_end, device=Q.device).unsqueeze(1)
            abs_k = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
            band = ((abs_q - abs_k).abs() <= radius).unsqueeze(0).unsqueeze(0)
            if is_decoder:
                band = band & (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)

            scores = torch.matmul(Qc.float(), Kc.float().transpose(-2, -1)) * scale
            scores = scores.masked_fill(~band, mask_val)
            if token_valid is not None:
                kv_valid = token_valid[:, None, None, k_start:k_end].to(torch.bool)
                scores = scores.masked_fill(~kv_valid, mask_val)
                combined = band & kv_valid
            else:
                combined = band
            scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)
            any_valid = combined.any(dim=-1, keepdim=True).to(scores.dtype)
            attn_c = F.softmax(scores.float(), dim=-1).to(Q.dtype) * any_valid
            attn_c = F.dropout(attn_c, p=dropout_p, training=training)
            outputs.append(torch.matmul(attn_c, Vc.to(attn_c.dtype)))

        out = torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
        return out, None

    abs_q = (past_len + torch.arange(N, device=Q.device)).unsqueeze(1)
    abs_k = torch.arange(N_k, device=Q.device).unsqueeze(0)
    band = ((abs_q - abs_k).abs() <= radius).unsqueeze(0).unsqueeze(0)
    if is_decoder:
        band = band & (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
    mask_val = -1e4
    scores = scores.masked_fill(~band, mask_val)

    if token_valid is not None:
        kv_valid = token_valid[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(~kv_valid, mask_val)

    scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)

    combined = band if token_valid is None else (band & token_valid[:, None, None, :])
    any_valid = combined.any(dim=-1, keepdim=True).to(scores.dtype)

    attn = F.softmax(scores.float(), dim=-1).to(Q.dtype) * any_valid
    attn = F.dropout(attn, p=dropout_p, training=training)
    return torch.matmul(attn, V.to(attn.dtype)), attn


def _causal_global_linear_attention(
    Q_feat: torch.Tensor,
    K_feat: torch.Tensor,
    V_flat: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Chunked dual-form causal linear attention, memory O(B * chunk^2 + B * rank * h*d)."""
    B, N, rank = Q_feat.shape
    N_kv = K_feat.shape[1]
    past_len = max(0, N_kv - N)
    D = V_flat.shape[-1]
    dtype = Q_feat.dtype
    device = Q_feat.device

    state_kv = torch.zeros(B, rank, D, device=device, dtype=dtype)
    state_k = torch.zeros(B, rank, device=device, dtype=dtype)

    if past_len > 0:
        for p_start in range(0, past_len, chunk_size):
            p_end = min(p_start + chunk_size, past_len)
            Kp = K_feat[:, p_start:p_end]
            Vp = V_flat[:, p_start:p_end]
            state_kv = state_kv + torch.matmul(Kp.transpose(-2, -1), Vp)
            state_k = state_k + Kp.sum(dim=1)

    outs = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Qc = Q_feat[:, start:end]
        Kc = K_feat[:, past_len + start : past_len + end]
        Vc = V_flat[:, past_len + start : past_len + end]

        intra_scores = torch.matmul(Qc, Kc.transpose(-2, -1)).tril()
        if start == 0 and past_len == 0:
            num = torch.matmul(intra_scores, Vc)
            den = intra_scores.sum(dim=-1).unsqueeze(-1).clamp(min=1e-6)
        else:
            num = torch.matmul(Qc, state_kv) + torch.matmul(intra_scores, Vc)
            den = (torch.matmul(Qc, state_k.unsqueeze(-1)).squeeze(-1) + intra_scores.sum(dim=-1)).unsqueeze(-1).clamp(
                min=1e-6
            )
        outs.append(num / den)

        if end < N:
            state_kv = state_kv + torch.matmul(Kc.transpose(-2, -1), Vc)
            state_k = state_k + Kc.sum(dim=1)

    return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


def _noncausal_global_linear_attention(
    Q_feat: torch.Tensor,
    K_feat: torch.Tensor,
    V_flat: torch.Tensor,
) -> torch.Tensor:
    """Bidirectional linear attention via KV-first associativity."""
    KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V_flat)
    Z = K_feat.sum(dim=-2, keepdim=True)
    num = torch.matmul(Q_feat, KV)
    den = torch.matmul(Q_feat, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
    return num / den


def _state_recurrence_jit(u: torch.Tensor, gamma: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    if os.environ.get("AEGIS_DISABLE_RECURRENCE", "0") == "1":
        return u
    if reverse:
        u = torch.flip(u, dims=[1])
    B, N, D = u.shape

    if N <= 64:
        gamma_b = gamma.view(1, 1, -1).squeeze(1)
        state = torch.zeros(B, D, device=u.device, dtype=u.dtype)
        outs = []
        for t in range(N):
            state = gamma_b * state + u[:, t, :]
            outs.append(state)
        out = torch.stack(outs, dim=1)
        if reverse:
            out = torch.flip(out, dims=[1])
        return out

    chunk = min(32, N)
    pad_len = (chunk - (N % chunk)) % chunk
    if pad_len > 0:
        u_p = F.pad(u, (0, 0, 0, pad_len))
    else:
        u_p = u

    N_pad = u_p.shape[1]
    num_chunks = N_pad // chunk

    pos = torch.arange(chunk, device=u.device, dtype=torch.float32)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    tril = (diff >= 0).unsqueeze(-1)

    log_gamma = torch.log(gamma.float().clamp(min=1e-5, max=1.0 - 1e-5)).view(1, 1, D)
    decay_intra = torch.where(
        tril,
        torch.exp(diff.unsqueeze(-1) * log_gamma),
        torch.zeros(1, 1, 1, device=u.device),
    ).to(u.dtype)
    gamma_chunk = torch.exp(chunk * log_gamma).to(u.dtype).view(1, 1, D)
    decay_inter = torch.exp((pos + 1).view(1, chunk, 1) * log_gamma).to(u.dtype)
    decay_carry = torch.exp((chunk - 1 - pos).view(1, chunk, 1) * log_gamma).to(u.dtype)

    u_chunks = u_p.view(B, num_chunks, chunk, D)
    state = torch.zeros(B, 1, D, device=u.device, dtype=u.dtype)
    outs = []

    for c in range(num_chunks):
        uc = u_chunks[:, c]
        out_intra = torch.einsum("bcd,kcd->bkd", uc, decay_intra)
        out_inter = state * decay_inter
        yc = out_intra + out_inter
        outs.append(yc)
        state = state * gamma_chunk + (uc * decay_carry).sum(dim=1, keepdim=True)

    out = torch.cat(outs, dim=1)
    if pad_len > 0:
        out = out[:, :N, :]
    if reverse:
        out = torch.flip(out, dims=[1])
    return out

