"""
ASVD Attention: Activation‑Aware Singular Value Decomposition for Attention Compression
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("asvd")
class ASVDAttention(BaseAttention):
    """
    Replaces MultiheadAttention with low‑rank decomposed projections
    where the SVD is weighted by activation variance (activation‑aware).
    Stores U, S, Vh separately per projection.
    """

    def __init__(self, config, rank=64, activation_scales=None):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank

        self.query_u = nn.Parameter(torch.randn(self.all_head_size, self.rank) * 0.02)
        self.query_s = nn.Parameter(torch.ones(self.rank))
        self.query_vh = nn.Parameter(torch.randn(self.rank, config.hidden_size) * 0.02)

        self.key_u = nn.Parameter(torch.randn(self.all_head_size, self.rank) * 0.02)
        self.key_s = nn.Parameter(torch.ones(self.rank))
        self.key_vh = nn.Parameter(torch.randn(self.rank, config.hidden_size) * 0.02)

        self.value_u = nn.Parameter(torch.randn(self.all_head_size, self.rank) * 0.02)
        self.value_s = nn.Parameter(torch.ones(self.rank))
        self.value_vh = nn.Parameter(torch.randn(self.rank, config.hidden_size) * 0.02)

        self.out_proj_u = nn.Parameter(
            torch.randn(config.hidden_size, self.rank) * 0.02
        )
        self.out_proj_s = nn.Parameter(torch.ones(self.rank))
        self.out_proj_vh = nn.Parameter(
            torch.randn(self.rank, self.all_head_size) * 0.02
        )

        self.dropout = nn.Dropout(config.dropout)

        if activation_scales is not None:
            self.apply_activation_scales(activation_scales)

    def apply_activation_scales(self, scales):
        """Apply activation scales to the right singular vectors for ASVD."""
        if scales is not None:
            if not isinstance(scales, torch.Tensor):
                scales = torch.tensor(
                    scales, device=self.query_vh.device, dtype=self.query_vh.dtype
                )
            if scales.dim() == 1:
                scales = scales.unsqueeze(0)
            self.query_vh.data.mul_(scales)
            self.key_vh.data.mul_(scales)
            self.value_vh.data.mul_(scales)
            self.out_proj_vh.data.mul_(scales)

    def _reshape(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        W_q = (self.query_u * self.query_s) @ self.query_vh
        W_k = (self.key_u * self.key_s) @ self.key_vh
        W_v = (self.value_u * self.value_s) @ self.value_vh
        W_o = (self.out_proj_u * self.out_proj_s) @ self.out_proj_vh

        Q = (
            F.linear(hidden_states, W_q, None)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K = (
            F.linear(hidden_states, W_k, None)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V = (
            F.linear(hidden_states, W_v, None)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        scale = 1.0 / (self.head_dim**0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )

        out = F.linear(context, W_o)
        if kwargs.get("output_attentions"):
            return (out, probs)
        return (out, None)

    @staticmethod
    def from_module(module, rank, activation_scales=None):
        """Create ASVDAttention by SVD of the original module's weights."""
        attn = ASVDAttention(module, rank, activation_scales)
        w_qkv = module.in_proj_weight.data.cpu().numpy()
        w_o = module.out_proj.weight.data.cpu().numpy()
        all_head = module.embed_dim * module.num_heads
        W_q = w_qkv[:all_head, :]
        W_k = w_qkv[all_head : 2 * all_head, :]
        W_v = w_qkv[2 * all_head :, :]

        def decomposed_weights(W, rank):
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            return (
                torch.tensor(U_r, dtype=torch.float32),
                torch.tensor(S_r, dtype=torch.float32),
                torch.tensor(Vh_r, dtype=torch.float32),
            )

        qu, qs, qvh = decomposed_weights(W_q, rank)
        ku, ks, kvh = decomposed_weights(W_k, rank)
        vu, vs, vvh = decomposed_weights(W_v, rank)
        ou, os, ovh = decomposed_weights(w_o, rank)

        attn.query_u.data = qu
        attn.query_s.data = qs
        attn.query_vh.data = qvh
        attn.key_u.data = ku
        attn.key_s.data = ks
        attn.key_vh.data = kvh
        attn.value_u.data = vu
        attn.value_s.data = vs
        attn.value_vh.data = vvh
        attn.out_proj_u.data = ou
        attn.out_proj_s.data = os
        attn.out_proj_vh.data = ovh
        return attn
