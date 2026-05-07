"""
SVD‑LLM V2 Attention: Truncation‑aware SVD for LLM Compression (improved version)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("svd_llm_v2")
class SVDLLMV2Attention(BaseAttention):
    """
    SVD factorised attention with dynamic rank based on energy threshold.
    """
    def __init__(self, config, rank=64, energy_threshold=0.95, pretrained_attn=None):
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank
        self.energy_threshold = energy_threshold

        # Create parameter buffers
        self.query_u = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.query_s = nn.Parameter(torch.empty(self.rank))
        self.query_vh = nn.Parameter(torch.empty(self.rank, config.hidden_size))
        self.key_u = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.key_s = nn.Parameter(torch.empty(self.rank))
        self.key_vh = nn.Parameter(torch.empty(self.rank, config.hidden_size))
        self.value_u = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.value_s = nn.Parameter(torch.empty(self.rank))
        self.value_vh = nn.Parameter(torch.empty(self.rank, config.hidden_size))
        self.out_proj_u = nn.Parameter(torch.empty(config.hidden_size, self.rank))
        self.out_proj_s = nn.Parameter(torch.empty(self.rank))
        self.out_proj_vh = nn.Parameter(torch.empty(self.rank, self.all_head_size))

        # Initialise from pretrained if provided
        if pretrained_attn is not None:
            self._init_from_pretrained(pretrained_attn)
        else:
            self._init_random()

        self.dropout = nn.Dropout(config.dropout)

    def _init_random(self):
        nn.init.orthogonal_(self.query_u)
        nn.init.ones_(self.query_s)
        nn.init.orthogonal_(self.query_vh)
        nn.init.orthogonal_(self.key_u)
        nn.init.ones_(self.key_s)
        nn.init.orthogonal_(self.key_vh)
        nn.init.orthogonal_(self.value_u)
        nn.init.ones_(self.value_s)
        nn.init.orthogonal_(self.value_vh)
        nn.init.orthogonal_(self.out_proj_u)
        nn.init.ones_(self.out_proj_s)
        nn.init.orthogonal_(self.out_proj_vh)

    def _init_from_pretrained(self, pretrained_attn):
        """Initialise SVD factors from pretrained linear layers."""
        with torch.no_grad():
            # Q projection
            W_q = pretrained_attn.q_lin.weight.data.float()
            U, S, Vt = torch.linalg.svd(W_q, full_matrices=False)
            r = min(self.rank, len(S))
            self.query_u.data = (U[:, :r] * S[:r]).to(dtype=torch.float32)
            self.query_s.data = torch.ones(r)
            self.query_vh.data = Vt[:r, :].to(dtype=torch.float32)
            if r < self.rank:
                self._pad_factors(self.query_u, self.query_s, self.query_vh, self.rank)

            # K projection
            W_k = pretrained_attn.k_lin.weight.data.float()
            U, S, Vt = torch.linalg.svd(W_k, full_matrices=False)
            r = min(self.rank, len(S))
            self.key_u.data = (U[:, :r] * S[:r]).to(dtype=torch.float32)
            self.key_s.data = torch.ones(r)
            self.key_vh.data = Vt[:r, :].to(dtype=torch.float32)
            if r < self.rank:
                self._pad_factors(self.key_u, self.key_s, self.key_vh, self.rank)

            # V projection
            W_v = pretrained_attn.v_lin.weight.data.float()
            U, S, Vt = torch.linalg.svd(W_v, full_matrices=False)
            r = min(self.rank, len(S))
            self.value_u.data = (U[:, :r] * S[:r]).to(dtype=torch.float32)
            self.value_s.data = torch.ones(r)
            self.value_vh.data = Vt[:r, :].to(dtype=torch.float32)
            if r < self.rank:
                self._pad_factors(self.value_u, self.value_s, self.value_vh, self.rank)

            # Output projection
            W_o = pretrained_attn.out_lin.weight.data.float()
            U, S, Vt = torch.linalg.svd(W_o, full_matrices=False)
            r = min(self.rank, len(S))
            self.out_proj_u.data = (U[:, :r] * S[:r]).to(dtype=torch.float32)
            self.out_proj_s.data = torch.ones(r)
            self.out_proj_vh.data = Vt[:r, :].to(dtype=torch.float32)
            if r < self.rank:
                self._pad_factors(self.out_proj_u, self.out_proj_s, self.out_proj_vh, self.rank)

    def _pad_factors(self, u, s, vh, target_rank):
        """Pad factors to target rank with small random values."""
        current_rank = u.shape[1]
        if current_rank >= target_rank:
            return
        pad_u = torch.randn(u.shape[0], target_rank - current_rank) * 0.01
        u.data = torch.cat([u, pad_u], dim=1)
        pad_s = torch.ones(target_rank - current_rank) * 0.01
        s.data = torch.cat([s, pad_s], dim=0)
        pad_vh = torch.randn(target_rank - current_rank, vh.shape[1]) * 0.01
        vh.data = torch.cat([vh, pad_vh], dim=0)

    def _reshape(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        W_q = (self.query_u * self.query_s) @ self.query_vh
        W_k = (self.key_u * self.key_s) @ self.key_vh
        W_v = (self.value_u * self.value_s) @ self.value_vh
        W_o = (self.out_proj_u * self.out_proj_s) @ self.out_proj_vh

        Q = F.linear(hidden_states, W_q).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = F.linear(hidden_states, W_k).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = F.linear(hidden_states, W_v).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V)
        context = context.permute(0,2,1,3).contiguous().view(batch_size, seq_len, self.all_head_size)
        out = F.linear(context, W_o)
        return out, None


@ATTENTION_REGISTRY.register("svd_llm_v2")
def inject_svd_llm_v2(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """Injection function for DistilBERT."""
    rank = getattr(comp_config, 'rank', 64)
    energy = getattr(comp_config, 'svd_energy', 0.95)
    new_attn = SVDLLMV2Attention(model_config, rank=rank, energy_threshold=energy, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn