"""
Native Hybrid Attention (NHA) – Real attention module combining local sliding-window
attention with global linear attention via Random Fourier Features.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("nha_compressor")
class NativeHybridAttention(BaseAttention):
    def __init__(
        self,
        config: Any,
        rank: int = 64,
        window_size: int = 64,
        rff_features: int = 128,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.window_size = window_size
        self.rff_features = rff_features

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)

        if pretrained_attn is not None:
            with torch.no_grad():
                self.q_proj.weight.data.copy_(pretrained_attn.q_lin.weight.data)
                self.q_proj.bias.data.copy_(pretrained_attn.q_lin.bias.data)
                self.k_proj.weight.data.copy_(pretrained_attn.k_lin.weight.data)
                self.k_proj.bias.data.copy_(pretrained_attn.k_lin.bias.data)
                self.v_proj.weight.data.copy_(pretrained_attn.v_lin.weight.data)
                self.v_proj.bias.data.copy_(pretrained_attn.v_lin.bias.data)
                self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
                self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

        # RFF projection for global branch
        self.register_buffer("rff_proj", torch.randn(self.head_dim, rff_features) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # Learnable gate between local and global
        self.gate = nn.Linear(config.hidden_size, 1)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _local_window_attention(self, Q, K, V, valid_mask):
        """Local sliding window attention with padding mask."""
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Window mask
        idx = torch.arange(N, device=Q.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        window_mask = dist > self.window_size
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Padding mask
        pad_mask = ~valid_mask.unsqueeze(1).unsqueeze(2)   # [B,1,1,N] True for padding keys
        scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)

    def _global_rff_attention(self, Q, K, V, valid_mask):
        """Global linear attention with Random Fourier Features."""
        B, H, N, d = Q.shape

        Qf = torch.matmul(Q, self.rff_proj)   # (B,H,N,F)
        Kf = torch.matmul(K, self.rff_proj)

        norm = 1.0 / math.sqrt(self.rff_features)
        phi_Q = torch.cat([torch.cos(Qf), torch.sin(Qf)], dim=-1) * norm
        phi_K = torch.cat([torch.cos(Kf), torch.sin(Kf)], dim=-1) * norm

        # Apply padding mask to K and V
        pad = (~valid_mask).unsqueeze(1).unsqueeze(-1).to(Q.dtype)   # [B,1,N,1] True for padding
        phi_K = phi_K.masked_fill(pad, 0.0)
        V_masked = V.masked_fill(pad, 0.0)

        KV = torch.matmul(phi_K.transpose(-2, -1), V_masked)   # (B,H,2F,d)
        Z = phi_K.sum(dim=-2, keepdim=True)                    # (B,H,1,2F)
        num = torch.matmul(phi_Q, KV)
        den = torch.matmul(phi_Q, Z.transpose(-2, -1)) + 1e-5
        return num / den

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape

        # Convert mask to boolean valid [B,N]
        if attention_mask is None:
            valid_mask = torch.ones(B, N, device=hidden_states.device, dtype=torch.bool)
        elif attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                valid_mask = attention_mask
            else:
                valid_mask = attention_mask > 0
        elif attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                valid_mask = attention_mask[:, 0, 0, :]
            else:
                valid_mask = attention_mask[:, 0, 0, :] > -10000.0 if attention_mask.min() < -1000 else attention_mask[:, 0, 0, :] > 0
        else:
            valid_mask = torch.ones(B, N, device=hidden_states.device, dtype=torch.bool)

        Q = self._reshape(self.q_proj(hidden_states))
        K = self._reshape(self.k_proj(hidden_states))
        V = self._reshape(self.v_proj(hidden_states))

        local_out = self._local_window_attention(Q, K, V, valid_mask)
        global_out = self._global_rff_attention(Q, K, V, valid_mask)

        # Dynamic gate per sequence (mean pooling)
        seq_summary = hidden_states.mean(dim=1)   # (B,D)
        gate = torch.sigmoid(self.gate(seq_summary)).view(B, 1, 1, 1)   # (B,1,1,1)

        out = gate * local_out + (1 - gate) * global_out
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.out_proj(out), None


@ATTENTION_REGISTRY.register("nha_compressor_inject")
def inject_nha_compressor(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    window_size = getattr(comp_config, "window_size", 64)
    rff_features = getattr(comp_config, "rff_features", 128)
    new_attn = NativeHybridAttention(
        model_config,
        window_size=window_size,
        rff_features=rff_features,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn