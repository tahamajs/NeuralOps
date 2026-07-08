"""
Optimized-Hybrid-Attention (OHA): sliding window local + low‑rank global,
with gated fusion.  Drop‑in for DistilBERT.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def transpose_for_scores(x, num_attention_heads, head_dim):
    # x shape: (batch_size, seq_len, num_heads * head_dim)
    new_x_shape = x.size()[:-1] + (num_attention_heads, head_dim)
    x = x.view(*new_x_shape)  # (B, N, num_heads, head_dim)
    return x.permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)


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


@ATTENTION_REGISTRY.register("oha")
class OptimizedHybridAttention(BaseAttention):
    def __init__(self, config: OptimizedHybridAttentionConfig):
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.window_size = config.window_size
        self.compression_rank = config.compression_rank
        self.use_gate = config.use_gate

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.compress_k = nn.Linear(
            config.hidden_size, self.compression_rank, bias=False
        )
        self.compress_v = nn.Linear(
            config.hidden_size, self.compression_rank, bias=False
        )
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

    def _sliding_window_mask(self, seq_len, device):
        idx = torch.arange(seq_len, device=device)
        mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        B, N, _ = hidden_states.shape
        Q = transpose_for_scores(
            self.query(hidden_states), self.num_heads, self.head_dim
        )
        K = transpose_for_scores(self.key(hidden_states), self.num_heads, self.head_dim)
        V = transpose_for_scores(
            self.value(hidden_states), self.num_heads, self.head_dim
        )

        # Convert attention_mask to additive shape
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_2d = attention_mask.to(dtype=hidden_states.dtype)
                local_add = (1.0 - mask_2d[:, None, None, :]) * -10000.0
                comp_add = (1.0 - mask_2d[:, :, None]) * -10000.0
            elif attention_mask.dim() == 4:
                local_add = attention_mask
                comp_add = attention_mask.squeeze(1).squeeze(1).unsqueeze(-1)
            else:
                raise ValueError(f"Unsupported mask shape: {attention_mask.shape}")
        else:
            local_add = None
            comp_add = None

        # Local sliding-window attention
        scale = 1.0 / math.sqrt(self.head_dim)
        Q_s = Q * scale
        local_scores = torch.matmul(Q_s, K.transpose(-2, -1))
        win_mask = self._sliding_window_mask(N, Q.device)
        local_scores = local_scores.masked_fill(~win_mask, float("-inf"))
        if local_add is not None:
            local_scores = local_scores + local_add
        local_probs = F.softmax(local_scores, dim=-1)
        local_probs = self.dropout(local_probs)
        local_ctx = torch.matmul(local_probs, V)

        # Global low‑rank attention
        comp_k_raw = self.compress_k(hidden_states)
        comp_v_raw = self.compress_v(hidden_states)
        if comp_add is not None:
            comp_k_raw = comp_k_raw + comp_add
            comp_v_raw = comp_v_raw + comp_add
        comp_k_weights = F.softmax(comp_k_raw, dim=1)
        comp_v_weights = F.softmax(comp_v_raw, dim=1)

        full_k = K.permute(0, 1, 3, 2)
        full_v = V.permute(0, 1, 3, 2)
        ck = torch.matmul(full_k, comp_k_weights.unsqueeze(1)).permute(0, 1, 3, 2)
        cv = torch.matmul(full_v, comp_v_weights.unsqueeze(1)).permute(0, 1, 3, 2)
        global_scores = torch.matmul(Q_s, ck.transpose(-2, -1))
        global_probs = F.softmax(global_scores, dim=-1)
        global_probs = self.dropout(global_probs)
        global_ctx = torch.matmul(global_probs, cv)

        # Gated fusion
        if self.use_gate:
            gate = torch.sigmoid(self.gate(hidden_states))
            gate = gate.permute(0, 2, 1).unsqueeze(-1)
            context = gate * local_ctx + (1.0 - gate) * global_ctx
        else:
            context = local_ctx + global_ctx

        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        output = self.out_proj(context)
        return output, (local_probs if output_attentions else None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.weight.device)

    @classmethod
    def from_distilbert_pretrained(
        cls, distilbert_attention_layer, config: OptimizedHybridAttentionConfig
    ):
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


@ATTENTION_REGISTRY.register("oha")
def inject_oha(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    oha_cfg = OptimizedHybridAttentionConfig(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        window_size=getattr(comp_config, "block_size", 128),
        compression_rank=getattr(comp_config, "compression_rank", 64),
        use_gate=getattr(comp_config, "use_gate", True),
        gate_bias=getattr(comp_config, "gate_bias", 2.0),
    )
    new_attn = OptimizedHybridAttention.from_distilbert_pretrained(old_attn, oha_cfg)
    layer.attention = new_attn
    return new_attn
