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
from framework.attentions.core.base import BaseAttention
from framework.attentions.core.registry import ATTENTION_REGISTRY


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
    """
    OHA attention: local sliding‑window + low‑rank global context, fused by a
    learned gate.

    The global branch compresses the sequence into a fixed‑length low‑rank
    representation (via softmax aggregation) and then performs attention
    against this compact K/V tensor.
    """

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

        # Standard Q/K/V projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Low‑rank compression (per‑token → R)
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

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, all_head_size) → (B, num_heads, N, head_dim)"""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Boolean mask: True for positions within the window."""
        idx = torch.arange(seq_len, device=device)
        mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape

        Q = self._transpose_for_scores(self.query(hidden_states))  # (B, H, N, D)
        K = self._transpose_for_scores(self.key(hidden_states))
        V = self._transpose_for_scores(self.value(hidden_states))

        # -----------------------------
        # Convert attention_mask to additive shape
        # -----------------------------
        if attention_mask is not None:
            if attention_mask.dim() == 2:                 # (B, N) – 1=keep, 0=pad
                mask_2d = attention_mask.to(dtype=hidden_states.dtype)
                pad_mask_3d = mask_2d[:, None, :]        # (B, 1, N)
                # Local: (B, 1, 1, N) broadcast
                local_add = (1.0 - pad_mask_3d.unsqueeze(1)) * -10000.0
                # For global compression: softmax over tokens, need per‑token weight
                comp_add = (1.0 - pad_mask_3d.permute(0, 2, 1)) * -10000.0   # (B, N, 1)
                # Global query mask: (B, 1, 1, R) for broadcasting over low‑rank dim
                global_q_add = (1.0 - mask_2d[:, None, :, None]) * -10000.0   # (B, 1, N, 1)
            elif attention_mask.dim() == 4:               # (B, 1, 1, N)
                mask_val = attention_mask.to(dtype=hidden_states.dtype)
                local_add = mask_val
                # For compression, we want a (B, N, 1) mask
                comp_add = mask_val.squeeze(1).squeeze(1).unsqueeze(-1)  # (B, N, 1)
                global_q_add = mask_val.permute(0, 1, 3, 2)               # (B, 1, R, N) – adjust later
            else:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")
        else:
            local_add = comp_add = global_q_add = None

        # -----------------------------
        # Local sliding‑window attention
        # -----------------------------
        scale = 1.0 / math.sqrt(self.head_dim)
        Q_s = Q * scale
        local_scores = torch.matmul(Q_s, K.transpose(-1, -2))          # (B, H, N, N)
        win_mask = self._sliding_window_mask(N, Q.device)
        local_scores = local_scores.masked_fill(~win_mask, float("-inf"))
        if local_add is not None:
            local_scores = local_scores + local_add
        local_probs = F.softmax(local_scores, dim=-1)
        local_probs = self.dropout(local_probs)
        local_ctx = torch.matmul(local_probs, V)                       # (B, H, N, D)

        # -----------------------------
        # Global low‑rank attention
        # -----------------------------
        # 1) Compress the sequence into R tokens via softmax aggregation
        comp_k_raw = self.compress_k(hidden_states)                    # (B, N, R)
        comp_v_raw = self.compress_v(hidden_states)                    # (B, N, R)
        if comp_add is not None:
            comp_k_raw = comp_k_raw + comp_add                         # mask out padded tokens before softmax
            comp_v_raw = comp_v_raw + comp_add
        comp_k_weights = F.softmax(comp_k_raw, dim=1)                  # (B, N, R) – softmax over N per R
        comp_v_weights = F.softmax(comp_v_raw, dim=1)

        # 2) Gather full K/V but apply masking to ignore pads
        full_k = K.permute(0, 2, 3, 1)                                  # (B, H, D, N) – for matmul with (B, N, R)
        full_v = V.permute(0, 2, 3, 1)

        if attention_mask is not None:
            # Boolean mask: 1=keep, 0=drop
            if attention_mask.dim() == 2:
                bool_mask = attention_mask[:, :, None, None].to(full_k.dtype)   # (B, N, 1, 1)
            else:  # 4D
                bool_mask = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) > -1000.0).to(full_k.dtype)
            # Apply to K/V before mixing
            full_k = full_k * bool_mask.permute(0, 2, 3, 1)   # (B, H, D, N)
            full_v = full_v * bool_mask.permute(0, 2, 3, 1)

        # 3) Create compressed K/V: (B, H, R, D)
        ck = torch.matmul(full_k, comp_k_weights.unsqueeze(1))   # (B, H, D, R) → need to get (B, H, R, D)
        # Actually: full_k: (B, H, D, N), comp_k_weights: (B, N, R) → matmul gives (B, H, D, R), then transpose
        ck = torch.matmul(full_k, comp_k_weights.unsqueeze(1)).permute(0, 1, 3, 2)  # (B, H, R, D)
        cv = torch.matmul(full_v, comp_v_weights.unsqueeze(1)).permute(0, 1, 3, 2)

        # 4) Global attention: Q is (B, H, N, D), ck is (B, H, R, D)
        global_scores = torch.matmul(Q_s, ck.transpose(-1, -2))           # (B, H, N, R)
        if global_q_add is not None:
            # global_q_add needs to be broadcastable (B, 1, N, R) or appropriate
            global_scores = global_scores + global_q_add
        global_probs = F.softmax(global_scores, dim=-1)
        global_probs = self.dropout(global_probs)
        global_ctx = torch.matmul(global_probs, cv)                       # (B, H, N, D)

        # -----------------------------
        # Gated fusion
        # -----------------------------
        if self.use_gate:
            gate = torch.sigmoid(self.gate(hidden_states))                # (B, N, H)
            gate = gate.permute(0, 2, 1).unsqueeze(-1)                    # (B, H, N, 1)
            context = gate * local_ctx + (1.0 - gate) * global_ctx
        else:
            context = local_ctx + global_ctx

        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(context)

        attn_weights = local_probs if output_attentions else None
        return (output, attn_weights) if output_attentions else (output,)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.weight.device)

    @classmethod
    def from_distilbert_pretrained(cls, distilbert_attention_layer, config: OptimizedHybridAttentionConfig):
        """Initialise OHA by copying weights from an existing DistilBERT attention layer."""
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
    """Create OHA from compression config and replace the layer's attention."""
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