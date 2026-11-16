# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Component layers and config for NEXUS Attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class NEXUSConfig:
    block_size: int = 16
    top_k_blocks: int = 8
    xattn_enabled: bool = True
    xattn_stride: int = 2
    seer_enabled: bool = True
    seer_gate_rank: int = 16
    seer_top_k_blocks: int = 4
    seer_temperature: float = 1.0
    dual_stage_enabled: bool = True
    intra_block_full: bool = True
    stripe_enabled: bool = True
    stripe_width: int = 4
    rectify_interval: int = 4
    rff_features: int = 128
    softpick_enabled: bool = False
    softpick_temperature: float = 1.0
    sparse_min_seq_len: int = 8
    enable_spectraformer: bool = False
    dropout: float = 0.1
    causal_chunk_size: int = 256
    causal_landmarks: int = 32


class XAttentionScorer:
    @staticmethod
    def score_blocks(Q: torch.Tensor, K: torch.Tensor, block_size: int, stride: int = 2) -> torch.Tensor:
        B, H, N, D = Q.shape
        n_blocks = (N + block_size - 1) // block_size

        with torch.amp.autocast(device_type=Q.device.type, enabled=False):
            Q_sample = Q[:, :, ::stride, :].float()
            K_sample = K[:, :, ::stride, :].float()
            N_s = Q_sample.size(2)

            scale = 1.0 / math.sqrt(max(int(D), 1))
            attn_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1).contiguous()) * scale

            b_s = max(1, block_size // stride)
            target = n_blocks * b_s
            if N_s < target:
                pad = target - N_s
                attn_sample = F.pad(attn_sample, (0, pad, 0, pad), value=0.0)
            attn_sample = attn_sample[:, :, :target, :target]

            attn_blocks = attn_sample.view(B, H, n_blocks, b_s, n_blocks, b_s)
            scores = attn_blocks.mean(dim=(3, 5))
            return scores


class SeerGating(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        gate_rank: int,
        top_k: int,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.temperature = temperature

        self.q_pool = nn.Sequential(
            nn.Linear(head_dim, gate_rank),
            nn.GELU(),
            nn.Linear(gate_rank, gate_rank),
        )
        self.k_pool = nn.Sequential(
            nn.Linear(head_dim, gate_rank),
            nn.GELU(),
            nn.Linear(gate_rank, gate_rank),
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor, n_blocks: int, training: bool = True) -> torch.Tensor:
        B, H, N, D = Q.shape
        gr = self.q_pool[0].out_features

        Q_pooled = Q.mean(dim=2)
        K_pooled = K.mean(dim=2)

        p_weight = self.q_pool[0].weight
        p_dtype = p_weight.dtype
        p_device = p_weight.device
        q_in = Q_pooled.to(device=p_device, dtype=p_dtype)
        k_in = K_pooled.to(device=p_device, dtype=p_dtype)
        q_gate = self.q_pool(q_in).float().view(B, H, 1, gr)
        k_gate = self.k_pool(k_in).float().view(B, H, gr, 1)

        with torch.amp.autocast(device_type=Q.device.type, enabled=False):
            q_gate = q_gate.to(Q.device).float()
            k_gate = k_gate.to(Q.device).float()
            gate_score = torch.sigmoid(torch.matmul(q_gate, k_gate).squeeze(-1).squeeze(-1) / self.temperature)

            k_float = gate_score * n_blocks

            idx = torch.arange(n_blocks, device=Q.device).view(1, 1, 1, n_blocks)
            k_float_exp = k_float.unsqueeze(-1).unsqueeze(-1)

            temperature = 0.1
            soft_mask = torch.sigmoid((k_float_exp - idx) / temperature)

            k_int = torch.clamp(k_float.round(), min=1, max=n_blocks).long().unsqueeze(-1).unsqueeze(-1)
            hard_mask = (idx < k_int).float()

            ste_mask = hard_mask + soft_mask - soft_mask.detach()

            mask = torch.zeros(B, H, n_blocks, n_blocks, device=Q.device, dtype=torch.float32)
            mask = mask + ste_mask

            diag_idx = torch.arange(n_blocks, device=Q.device)
            mask[:, :, diag_idx, diag_idx] = 1.0
            mask[:, :, :, 0] = 1.0

            return mask


class DualStageStripeAttention(nn.Module):
    def __init__(
        self,
        cfg: NEXUSConfig,
        head_dim: int,
        num_heads: int = 1,
        is_decoder: bool = False,
        is_nested_encoder: bool = False,
        seer_beta: nn.Parameter | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.is_decoder = is_decoder
        self.is_nested_encoder = is_nested_encoder
        self.seer_beta = seer_beta
        self.xattn_scorer = XAttentionScorer()

    def _inter_block_topk(self, scores: torch.Tensor, top_k: int) -> torch.Tensor:
        B, H, nb, nb2 = scores.shape
        tk = min(top_k, nb2)
        scores_inter = scores.clone()
        min_dim = min(nb, nb2)
        diag_idx = torch.arange(min_dim, device=scores.device)
        scores_inter[:, :, diag_idx, diag_idx] = float("-inf")
        _, idx = scores_inter.topk(tk, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idx, True)
        mask[:, :, diag_idx, diag_idx] = True
        return mask

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: torch.Tensor | None = None,
        seer_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_dtype = Q.dtype
        B, H, N, D = Q.shape
        bs = self.cfg.block_size
        nb = (N + bs - 1) // bs
        top_k = self.cfg.top_k_blocks
        scale = 1.0 / math.sqrt(max(int(D), 1))

        with torch.amp.autocast(device_type=Q.device.type, enabled=False):
            Q_f = Q.float()
            K_f = K.float()
            V_f = V.float()

            if self.cfg.xattn_enabled:
                block_scores = self.xattn_scorer.score_blocks(Q_f, K_f, bs, self.cfg.xattn_stride)
                inter_mask = self._inter_block_topk(block_scores, top_k)
            else:
                inter_mask = torch.ones(B, H, nb, nb, device=Q.device, dtype=torch.bool)

            inter_mask[:, :, :, 0] = True
            diag_idx = torch.arange(nb, device=Q.device)
            inter_mask[:, :, diag_idx, diag_idx] = True

            scores = torch.matmul(Q_f, K_f.transpose(-2, -1).contiguous()) * scale

            expanded_inter_mask = inter_mask.repeat_interleave(bs, dim=2).repeat_interleave(bs, dim=3)[:, :, :N, :N]
            scores = scores.masked_fill(~expanded_inter_mask, -10000.0)

            if seer_mask is not None:
                expanded_seer = seer_mask.float().repeat_interleave(bs, dim=2).repeat_interleave(bs, dim=3)[:, :, :N, :N]
                beta_factor = torch.tanh(self.seer_beta.float()).to(scores.dtype) if self.seer_beta is not None else 1.0
                scores = scores + beta_factor * (expanded_seer - 1.0) * 10000.0

            if self.cfg.stripe_enabled and N > 256:
                sw = self.cfg.stripe_width
                idx = torch.arange(N, device=Q.device)
                stripe_group = idx // sw
                q_stripe = stripe_group.unsqueeze(1)
                k_stripe = stripe_group.unsqueeze(0)
                stripe_mask = (q_stripe - k_stripe).abs() <= 1
                scores = scores.masked_fill(~stripe_mask.unsqueeze(0).unsqueeze(0), -10000.0)

            if token_valid is not None:
                scores = scores.masked_fill((~token_valid.bool()).unsqueeze(1).unsqueeze(2), -10000.0)

            if self.cfg.softpick_enabled:
                attn = softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature)
            else:
                scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
                attn = F.softmax(scores, dim=-1)

            context = torch.matmul(attn, V_f)
            return context.to(orig_dtype)


class SpectraformerRectify(nn.Module):
    def __init__(self, head_dim: int, rff_features: int = 128, **kwargs) -> None:
        super().__init__()
        self.rff_features = rff_features
        proj_init = torch.randn(head_dim, rff_features)
        if head_dim <= rff_features:
            proj_q, _ = torch.linalg.qr(proj_init.T)
            proj_init = proj_q.T
        else:
            proj_q, _ = torch.linalg.qr(proj_init)
            proj_init = proj_q
        self.proj = nn.Parameter(proj_init / math.sqrt(head_dim))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, N, D = Q.shape
        F_dim = self.rff_features

        orig_dtype = Q.dtype
        with torch.amp.autocast(device_type=Q.device.type, enabled=False):
            Q_f, K_f, V_f = Q.float(), K.float(), V.float()
            proj = self.proj.to(Q.device).float()

            Q_proj = torch.matmul(F.normalize(Q_f, p=2, dim=-1), proj).clamp(max=11.0)
            K_proj = torch.matmul(F.normalize(K_f, p=2, dim=-1), proj).clamp(max=11.0)

            Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
            K_feat = torch.exp(K_proj) / math.sqrt(F_dim)

            KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V_f)
            Z = K_feat.sum(dim=2)
            ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV)
            norm = torch.einsum("bhnf,bhf->bhn", Q_feat, Z).unsqueeze(-1)
            return (ctx / norm.clamp(min=1e-5)).to(orig_dtype)


__all__ = [
    "NEXUSConfig",
    "XAttentionScorer",
    "SeerGating",
    "DualStageStripeAttention",
    "SpectraformerRectify",
]
