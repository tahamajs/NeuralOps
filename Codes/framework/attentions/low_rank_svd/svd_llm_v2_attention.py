"""
SVD‑LLM V2 Attention: Truncation‑aware SVD for LLM Compression (improved version)
Paper: SVD-LLM V2 (further optimizations on top of SVD-LLM)
"""
import torch
import torch.nn as nn
import numpy as np
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("svd_llm_v2")
class SVDLLMV2Attention(BaseAttention):
    """
    Similar to ASVD but uses a different truncation strategy: dynamic rank based on
    singular value cumulative energy threshold, and optionally applies weight tying.
    This implementation stores the truncated matrices as parameters.
    """
    def __init__(self, config, rank=64, energy_threshold=0.95):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank
        self.energy_threshold = energy_threshold

        # Decomposed matrices (similar to ASVD)
        self.query_u  = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.query_s  = nn.Parameter(torch.empty(self.rank))
        self.query_vh = nn.Parameter(torch.empty(self.rank, config.hidden_size))

        self.key_u    = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.key_s    = nn.Parameter(torch.empty(self.rank))
        self.key_vh   = nn.Parameter(torch.empty(self.rank, config.hidden_size))

        self.value_u  = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.value_s  = nn.Parameter(torch.empty(self.rank))
        self.value_vh = nn.Parameter(torch.empty(self.rank, config.hidden_size))

        self.out_proj_u = nn.Parameter(torch.empty(config.hidden_size, self.rank))
        self.out_proj_s = nn.Parameter(torch.empty(self.rank))
        self.out_proj_vh = nn.Parameter(torch.empty(self.rank, self.all_head_size))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _reshape(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        # Reconstruct weight matrices
        W_q = (self.query_u * self.query_s) @ self.query_vh
        W_k = (self.key_u   * self.key_s)   @ self.key_vh
        W_v = (self.value_u * self.value_s) @ self.value_vh
        W_o = (self.out_proj_u * self.out_proj_s) @ self.out_proj_vh

        Q = F.linear(hidden_states, W_q).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = F.linear(hidden_states, W_k).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = F.linear(hidden_states, W_v).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V)
        context = context.permute(0,2,1,3).contiguous().view(batch_size, seq_len, self.all_head_size)
        out = F.linear(context, W_o)
        return (out, None) if kwargs.get('output_attentions') else (out,)

    @staticmethod
    def from_module(module, energy_threshold=0.95, min_rank=None):
        """
        Create SVDLLMV2Attention by performing SVD on the original module's weights,
        automatically choosing ranks based on energy threshold.
        """
        w_qkv = module.in_proj_weight.data.cpu().numpy()
        w_o   = module.out_proj.weight.data.cpu().numpy()
        all_head = module.embed_dim * module.num_heads

        W_q = w_qkv[:all_head, :]
        W_k = w_qkv[all_head:2*all_head, :]
        W_v = w_qkv[2*all_head:, :]

        def truncate_svd(W):
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            # Determine rank by cumulative energy
            cum_energy = np.cumsum(S) / np.sum(S)
            rank = np.searchsorted(cum_energy, energy_threshold) + 1
            if min_rank:
                rank = max(rank, min_rank)
            rank = min(rank, len(S))
            return (torch.tensor(U[:, :rank], dtype=torch.float32),
                    torch.tensor(S[:rank], dtype=torch.float32),
                    torch.tensor(Vh[:rank, :], dtype=torch.float32)), rank

        (qu, qs, qvh), rq = truncate_svd(W_q)
        (ku, ks, kvh), rk = truncate_svd(W_k)
        (vu, vs, vvh), rv = truncate_svd(W_v)
        (ou, os, ovh), ro = truncate_svd(w_o)

        # Use max rank to create the attention layer (all same size for simplicity)
        max_r = max(rq, rk, rv, ro)
        # Pad or recreate tensors to max_r
        def pad_to(t, target_r):
            if t.shape[-1] < target_r:
                pad = torch.zeros(*t.shape[:-1], target_r - t.shape[-1], dtype=t.dtype)
                return torch.cat([t, pad], dim=-1)
            return t

        attn = SVDLLMV2Attention(module, rank=max_r, energy_threshold=energy_threshold)
        attn.query_u.data = pad_to(qu, max_r)
        attn.query_s.data = pad_to(qs, max_r)
        attn.query_vh.data = pad_to(qvh, max_r)
        attn.key_u.data   = pad_to(ku, max_r)
        attn.key_s.data   = pad_to(ks, max_r)
        attn.key_vh.data  = pad_to(kvh, max_r)
        attn.value_u.data = pad_to(vu, max_r)
        attn.value_s.data = pad_to(vs, max_r)
        attn.value_vh.data = pad_to(vvh, max_r)
        attn.out_proj_u.data = pad_to(ou, max_r)
        attn.out_proj_s.data = pad_to(os, max_r)
        attn.out_proj_vh.data = pad_to(ovh, max_r)
        return attn