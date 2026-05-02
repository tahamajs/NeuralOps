"""
Optimized-Hybrid-Attention (OHA): sliding window local + low‑rank global,
with gated fusion.  Drop‑in for DistilBERT.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class OptimizedHybridAttentionConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_probs_dropout_prob: float = 0.1
    window_size: int = 128
    compression_rank: int = 64
    use_gate: bool = True
    gate_bias: float = 2.0
    layer_norm_eps: float = 1e-12

class OptimizedHybridAttention(nn.Module):
    def __init__(self, config: OptimizedHybridAttentionConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.window_size = config.window_size
        self.compression_rank = config.compression_rank
        self.use_gate = config.use_gate

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.compress_k = nn.Linear(config.hidden_size, self.compression_rank, bias=False)
        self.compress_v = nn.Linear(config.hidden_size, self.compression_rank, bias=False)

        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if self.use_gate:
            self.gate = nn.Linear(config.hidden_size, self.num_heads)
            nn.init.constant_(self.gate.bias, config.gate_bias)
            nn.init.normal_(self.gate.weight, std=0.02)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.compress_k.weight, std=0.02)
        nn.init.normal_(self.compress_v.weight, std=0.02)

    def _transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _sliding_window_mask(self, seq_len, device):
        idx = torch.arange(seq_len, device=device)
        mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape

        Q = self._transpose_for_scores(self.query(hidden_states))
        K = self._transpose_for_scores(self.key(hidden_states))
        V = self._transpose_for_scores(self.value(hidden_states))

        # --- mask handling (same as original) ---
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask_3d = attention_mask[:, None, :].to(hidden_states.dtype)
                local_pad_mask = (1.0 - pad_mask_3d.unsqueeze(1)) * -10000.0
                comp_mask = (1.0 - pad_mask_3d.permute(0, 2, 1)) * -10000.0
                global_q_mask = (1.0 - pad_mask_3d.unsqueeze(-1)) * -10000.0
            elif attention_mask.dim() == 4:
                mask_4d = attention_mask.to(hidden_states.dtype)
                local_pad_mask = mask_4d
                comp_mask = mask_4d.squeeze(1).squeeze(1).unsqueeze(-1)
                global_q_mask = mask_4d.permute(0, 1, 3, 2)
            else:
                raise ValueError(f"Unsupported mask shape: {attention_mask.shape}")
        else:
            local_pad_mask = comp_mask = global_q_mask = None

        # --- local attention ---
        scale = 1.0 / math.sqrt(self.head_dim)
        Q_s = Q * scale
        local_scores = torch.matmul(Q_s, K.transpose(-1, -2))
        win_mask = self._sliding_window_mask(N, Q.device)
        local_scores = local_scores.masked_fill(~win_mask, float("-inf"))
        if local_pad_mask is not None:
            local_scores = local_scores + local_pad_mask
        local_probs = F.softmax(local_scores, dim=-1)
        local_probs = self.dropout(local_probs)
        local_ctx = torch.matmul(local_probs, V)

        # --- global attention (low‑rank) ---
        comp_k = self.compress_k(hidden_states)
        comp_v = self.compress_v(hidden_states)
        if comp_mask is not None:
            comp_k = comp_k + comp_mask
            comp_v = comp_v + comp_mask
        comp_k = F.softmax(comp_k, dim=1)
        comp_v = F.softmax(comp_v, dim=1)

        full_k = self.key(hidden_states).view(B, N, self.num_heads, self.head_dim)
        full_v = self.value(hidden_states).view(B, N, self.num_heads, self.head_dim)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask[:, :, None].to(full_k.dtype)
            else:
                mask_bool = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) > -1000.0).to(full_k.dtype)
            full_k = full_k * mask_bool
            full_v = full_v * mask_bool

        ck = torch.matmul(full_k.permute(0,1,3,2), comp_k.unsqueeze(1)).permute(0,1,3,2)  # (B,H,R,D)
        cv = torch.matmul(full_v.permute(0,1,3,2), comp_v.unsqueeze(1)).permute(0,1,3,2)

        global_scores = torch.matmul(Q_s, ck.transpose(-1,-2))
        if global_q_mask is not None:
            global_scores = global_scores + global_q_mask
        global_probs = F.softmax(global_scores, dim=-1)
        global_probs = self.dropout(global_probs)
        global_ctx = torch.matmul(global_probs, cv)

        # --- fusion ---
        if self.use_gate:
            gate = torch.sigmoid(self.gate(hidden_states)).permute(0,2,1).unsqueeze(-1)  # (B,H,N,1)
            context = gate * local_ctx + (1 - gate) * global_ctx
        else:
            context = local_ctx + global_ctx

        context = context.permute(0,2,1,3).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(context)
        return (output, local_probs if output_attentions else None) if output_attentions else (output, None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.weight.device)

    @classmethod
    def from_distilbert_pretrained(cls, distilbert_attention_layer, config):
        oha = cls(config)
        src = distilbert_attention_layer
        oha.query.weight.data.copy_(src.q_lin.weight.data)
        oha.query.bias.data.copy_(src.q_lin.bias.data)
        oha.key.weight.data.copy_(src.k_lin.weight.data)
        oha.key.bias.data.copy_(src.k_lin.bias.data)
        oha.value.weight.data.copy_(src.v_lin.weight.data)
        oha.value.bias.data.copy_(src.v_lin.bias.data)
        oha.out_proj.weight.data.copy_(src.out_lin.weight.data)
        oha.out_proj.bias.data.copy_(src.out_lin.bias.data)
        return oha


# ------------------------------------------------------------
# Framework injection / registration
# ------------------------------------------------------------
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("oha")
def inject_oha(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    oha_cfg = OptimizedHybridAttentionConfig(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        window_size=getattr(comp_config, 'block_size', 128),
        compression_rank=getattr(comp_config, 'compression_rank', 64),
        use_gate=getattr(comp_config, 'use_gate', True),
        gate_bias=getattr(comp_config, 'gate_bias', 2.0),
    )
    new_attn = OptimizedHybridAttention.from_distilbert_pretrained(old_attn, oha_cfg)
    layer.attention = new_attn
    return new_attn
