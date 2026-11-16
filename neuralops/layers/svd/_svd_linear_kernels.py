#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Kernel execution helper routines and mixin for SVDLinearAttention.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.attention_utils import normalize_attention_mask as _normalize_mask

try:
    from neuralops.core.triton_linear_attn import fast_linear_attention
except Exception:
    fast_linear_attention = None

logger = logging.getLogger(__name__)


def compute_exponential_decay(decay_logits: torch.Tensor, activation: str = "softplus") -> torch.Tensor:
    logits_f = decay_logits.float()
    if activation == "softplus":
        decay = torch.exp(-F.softplus(logits_f))
        return torch.clamp(decay, min=1e-5, max=1.0)
    elif activation == "sigmoid":
        decay = torch.sigmoid(logits_f)
        return torch.clamp(decay, min=1e-5, max=1.0 - 1e-5)
    else:
        raise ValueError(f"Unsupported decay activation '{activation}'. Must be 'softplus' or 'sigmoid'.")


class SVDLinearKernelsMixin:
    """Mixin providing forward kernel implementations for SVDLinearAttention."""

    def _forward_decay_gated_linear_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        compute_dtype = torch.float32
        min_den = max(self.eps, 1e-6)

        phi_q = F.elu(q.to(compute_dtype)) + 1.0
        phi_k = F.elu(k.to(compute_dtype)) + 1.0
        v_f = v.to(compute_dtype)

        mask_expanded = _normalize_mask(mask, B, H, N, compute_dtype, q.device)
        if mask_expanded is not None:
            phi_k = phi_k * mask_expanded

        decay_logits = self.decay_gate(hidden_states).float()
        if self.decay_channel_wise:
            decay_logits = decay_logits.view(B, N, H, D_k).permute(0, 2, 1, 3)
            lambda_decay = (
                compute_exponential_decay(decay_logits, activation=self.decay_activation)
                .unsqueeze(-1)
                .to(compute_dtype)
            )
        else:
            decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            lambda_decay = compute_exponential_decay(decay_logits, activation=self.decay_activation).to(compute_dtype)

        if mask_expanded is not None:
            m_dec = mask_expanded
            while m_dec.dim() < lambda_decay.dim():
                m_dec = m_dec.unsqueeze(-1)
            lambda_decay = torch.where(m_dec > 0, lambda_decay, torch.ones_like(lambda_decay))

        if not self.decay_channel_wise:
            chunk_size = self.chunk_size
            num_chunks = (N + chunk_size - 1) // chunk_size
            if past_state is not None:
                S_chunk = past_state[0].to(device=q.device, dtype=compute_dtype)
                Z_chunk = past_state[1].to(device=q.device, dtype=compute_dtype)
            else:
                S_chunk = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
                Z_chunk = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)
            outputs = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                qc = phi_q[:, :, start:end, :]
                kc = phi_k[:, :, start:end, :]
                vc = v_f[:, :, start:end, :]
                lam_c = lambda_decay[:, :, start:end].squeeze(-1).squeeze(-1)
                Lc = end - start

                log_lam_c = torch.log(torch.clamp(lam_c, min=1e-6, max=1.0))
                log_gamma_c = torch.cumsum(log_lam_c, dim=-1)

                log_M_intra = log_gamma_c.unsqueeze(-1) - log_gamma_c.unsqueeze(-2)
                causal_bool = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool, device=qc.device))
                M_intra = torch.exp(torch.clamp(log_M_intra, max=0.0)) * causal_bool.unsqueeze(0).unsqueeze(0)

                aff_intra = torch.matmul(qc, kc.transpose(-1, -2).contiguous()) * M_intra
                num_intra = torch.matmul(aff_intra, vc)
                den_intra = aff_intra.sum(dim=-1, keepdim=True)

                decay_from_past = torch.exp(torch.clamp(log_gamma_c, max=0.0)).unsqueeze(-1)
                num_inter = torch.matmul(qc, S_chunk) * decay_from_past
                den_inter = torch.matmul(qc, Z_chunk) * decay_from_past

                den_total = (den_intra + den_inter + self.eps).clamp(min=min_den)
                out_c = (num_intra + num_inter) / den_total
                outputs.append(out_c)

                total_chunk_decay = torch.exp(torch.clamp(log_gamma_c[:, :, -1:], max=0.0)).unsqueeze(-1)
                log_w_to_end = log_gamma_c[:, :, -1:] - log_gamma_c
                w_to_end = torch.exp(torch.clamp(log_w_to_end, max=0.0)).unsqueeze(-1)
                kc_w = kc * w_to_end

                S_chunk = total_chunk_decay * S_chunk + torch.matmul(kc_w.transpose(-1, -2).contiguous(), vc)
                Z_chunk = total_chunk_decay * Z_chunk + kc_w.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()

            context = torch.cat(outputs, dim=2).to(q.dtype)
            return context, (S_chunk, Z_chunk)

        if past_state is not None:
            S, Z = past_state
            S = S.to(device=q.device, dtype=compute_dtype)
            Z = Z.to(device=q.device, dtype=compute_dtype)
        else:
            S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
            Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)

        outputs = []
        for t in range(N):
            qt = phi_q[:, :, t : t + 1, :]
            kt = phi_k[:, :, t : t + 1, :]
            vt = v_f[:, :, t : t + 1, :]
            lam_t = lambda_decay[:, :, t]

            kt_t = kt.transpose(-1, -2).contiguous()
            kv_t = torch.matmul(kt_t, vt)
            S = lam_t * S + kv_t
            Z = lam_t * Z + kt_t

            num_t = torch.matmul(qt, S)
            den_t = (torch.matmul(qt, Z) + self.eps).clamp(min=min_den)
            outputs.append(num_t / den_t)

        context = torch.cat(outputs, dim=2).to(q.dtype)
        return context, (S, Z)

    def _forward_linear_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None and q.device.type == "cuda" and fast_linear_attention is not None:
            try:
                return fast_linear_attention(
                    q,
                    k,
                    v,
                    is_causal=self.is_decoder,
                    eps=self.eps,
                )
            except Exception as e:
                logger.warning("Fast Triton kernel failed (%s). Falling back to PyTorch.", e)

        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        phi_q = F.elu(q.float()) + 1.0
        phi_k = F.elu(k.float()) + 1.0
        v_f = v.float()

        mask_expanded = _normalize_mask(mask, B, H, N, torch.float32, q.device)
        if mask_expanded is not None:
            phi_k = phi_k * mask_expanded

        if self.is_decoder and N > 1:
            chunk_size = self.chunk_size
            num_chunks = (N + chunk_size - 1) // chunk_size
            S_chunk = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)
            Z_chunk = torch.zeros(B, H, D_k, 1, device=q.device, dtype=torch.float32)
            outputs = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                qc = phi_q[:, :, start:end, :]
                kc = phi_k[:, :, start:end, :]
                vc = v_f[:, :, start:end, :]
                Lc = end - start

                aff_intra = torch.matmul(qc, kc.transpose(-1, -2).contiguous())
                causal_intra = (
                    torch.tril(torch.ones(Lc, Lc, device=q.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
                )
                aff_intra = aff_intra * causal_intra
                num_intra = torch.matmul(aff_intra, vc)
                den_intra = aff_intra.sum(dim=-1, keepdim=True)

                num_inter = torch.matmul(qc, S_chunk)
                den_inter = torch.matmul(qc, Z_chunk)

                out_c = (num_intra + num_inter) / (den_intra + den_inter + self.eps).clamp(min=max(self.eps, 1e-6))
                outputs.append(out_c)

                S_chunk = S_chunk + torch.matmul(kc.transpose(-1, -2).contiguous(), vc)
                Z_chunk = Z_chunk + kc.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()

            context = torch.cat(outputs, dim=2).to(q.dtype)
        else:
            kv_state = torch.matmul(phi_k.transpose(-1, -2).contiguous(), v_f)
            numerator = torch.matmul(phi_q, kv_state)
            k_sum = phi_k.sum(dim=-2, keepdim=True)
            denominator = (torch.matmul(phi_q, k_sum.transpose(-1, -2).contiguous()) + self.eps).clamp(
                min=max(self.eps, 1e-6)
            )
            context = (numerator / denominator).to(q.dtype)

        return context

    def _forward_exact_softmax(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        is_cross: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        is_causal_attn = self.is_decoder and not is_cross and q.shape[-2] > 1

        if not output_attentions:
            try:
                sdpa_mask = None
                can_use_sdpa = False
                if mask is None:
                    can_use_sdpa = True
                elif mask.dim() == 2 and not is_causal_attn:
                    sdpa_mask = mask.bool().view(mask.shape[0], 1, 1, mask.shape[1])
                    can_use_sdpa = True
                elif mask.dim() == 4 and not is_causal_attn:
                    sdpa_mask = mask.bool() if mask.dtype == torch.bool else mask.to(dtype=q.dtype)
                    can_use_sdpa = True

                if can_use_sdpa:
                    drop_p = self.dropout.p if self.training else 0.0
                    context = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=sdpa_mask,
                        dropout_p=drop_p,
                        is_causal=is_causal_attn if sdpa_mask is None else False,
                    )
                    return context, None
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass

        scale = 1.0 / math.sqrt(max(int(self.head_dim), 1))
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2).contiguous()) * scale

        if is_causal_attn:
            L, S = q.shape[-2], k.shape[-2]
            causal_mask = (
                torch.triu(
                    torch.full(
                        (L, S),
                        float("-inf"),
                        device=scores.device,
                        dtype=scores.dtype,
                    ),
                    diagonal=1 + (S - L),
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal_mask
        if mask is not None:
            scores = apply_attention_mask(scores, mask)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v.float()).to(q.dtype)
        return context, probs.to(q.dtype)
