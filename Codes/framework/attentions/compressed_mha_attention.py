"""
Compressed MHA Attention – SVD‑compressed Multi‑head Attention.
The combined QKV projection and the output projection are each compressed
by low‑rank SVD factors.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class CompressedAttentionLayer(nn.Module):
    """
    Compressed Multi‑head Attention layer from the GLUE framework.
    Q, K, V are computed from a single projected tensor via SVD factorisation.
    """
    def __init__(self, model_config, rank_qkv: int = 32, rank_out: int = 32,
                 dropout: float = 0.1, pretrained_attn: Optional[nn.Module] = None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot initialisation")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.dropout = dropout

        # Original combined QKV weight (out_features = 3 * hidden_size, in_features = hidden_size)
        w_qkv = pretrained_attn.q_lin.weight.data.cpu().numpy()  # shape (3*H, H)
        w_out = pretrained_attn.out_lin.weight.data.cpu().numpy()  # shape (H, 3*H)

        # SVD factorisation
        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        U_out, S_out, Vh_out = np.linalg.svd(w_out, full_matrices=False)

        rank_qkv = min(rank_qkv, w_qkv.shape[1])
        rank_out = min(rank_out, w_out.shape[1])

        # Store factors as buffers
        self.register_buffer('U_qkv', torch.from_numpy(U_qkv[:, :rank_qkv]).float())
        self.register_buffer('S_qkv_sqrt', torch.from_numpy(np.sqrt(S_qkv[:rank_qkv])).float())
        self.register_buffer('Vh_qkv', torch.from_numpy(Vh_qkv[:rank_qkv, :]).float())

        self.register_buffer('U_out', torch.from_numpy(U_out[:, :rank_out]).float())
        self.register_buffer('S_out_sqrt', torch.from_numpy(np.sqrt(S_out[:rank_out])).float())
        self.register_buffer('Vh_out', torch.from_numpy(Vh_out[:rank_out, :]).float())

        # Bias handling
        if pretrained_attn.q_lin.bias is not None:
            self.bias_qkv = nn.Parameter(pretrained_attn.q_lin.bias.data.clone())
        else:
            self.bias_qkv = None

        if pretrained_attn.out_lin.bias is not None:
            self.bias_out = nn.Parameter(pretrained_attn.out_lin.bias.data.clone())
        else:
            self.bias_out = None

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape

        # Step 1: compressed projection to QKV space
        proj = torch.matmul(hidden_states, self.Vh_qkv.t())
        qkv_comp = torch.matmul(proj, self.U_qkv.t() * self.S_qkv_sqrt.view(1, -1, 1))
        if self.bias_qkv is not None:
            qkv_comp += self.bias_qkv

        # Split into Q, K, V
        dim = self.hidden_size
        q = qkv_comp[:, :, :dim]
        k = qkv_comp[:, :, dim:2*dim]
        v = qkv_comp[:, :, 2*dim:]

        # Reshape to multi‑head
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, N, dim)

        # Step 2: compressed output projection
        context_fused = torch.matmul(context, self.Vh_out.t() * self.S_out_sqrt.view(1, -1, 1))
        output = torch.matmul(context_fused, self.U_out.t())
        if self.bias_out is not None:
            output += self.bias_out

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("compressed_mha")
def inject_compressed_mha(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank_qkv = getattr(comp_config, 'svd_rank_qkv', 32)
    rank_out = getattr(comp_config, 'svd_rank_out', 32)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = CompressedAttentionLayer(model_config, rank_qkv, rank_out, dropout, old_attn)
    layer.attention = new_attn
    return new_attn