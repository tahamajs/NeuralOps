import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("qcqa")
class QCQAAttention(nn.Module):
    """
    Quality & Capacity‑aware Grouped Query Attention.
    Dynamically mixes KV heads per query head using a learned router.
    No fixed grouping – each query head soft‑selects which KV heads to attend to.
    """
    def __init__(self, config, num_kv_heads=4, router_temp=1.0):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = config.hidden_size // self.num_heads

        # Projections: Q full heads, K/V compressed to num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)

        # Routing: for each query head, compute a score over KV heads
        self.router = nn.Linear(self.head_dim, self.num_kv_heads, bias=False)
        self.router_temp = router_temp

        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)  # (B, L, H_q, D)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)  # (B, L, H_kv, D)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Compute routing weights from queries (including a learned quality metric)
        # q: (B, L, H_q, D) -> route_logits: (B, L, H_q, H_kv)
        route_logits = self.router(q) / self.router_temp
        route_weights = F.softmax(route_logits, dim=-1)          # (B, L, H_q, H_kv)

        # Mix K and V per query head: (B, L, H_q, H_kv) x (B, L, H_kv, D) -> (B, L, H_q, D)
        # Using einsum for clarity
        k_mixed = torch.einsum('blhn,bmnd->bhlmd', route_weights, k)   # (B, H_q, L, H_kv, D) -> sum over H_kv later
        # Better: reshape and matmul
        # route_weights: (B, L, H_q, H_kv) -> (B*L, H_q, H_kv)
        # k: (B, L, H_kv, D) -> (B*L, H_kv, D)
        # mixed_k = (B*L, H_q, H_kv) @ (B*L, H_kv, D) = (B*L, H_q, D)
        B, L, H_q, H_kv = route_weights.shape
        k_flat = k.reshape(B*L, H_kv, self.head_dim)                     # (B*L, H_kv, D)
        w_flat = route_weights.reshape(B*L, H_q, H_kv)                   # (B*L, H_q, H_kv)
        k_mixed_flat = torch.bmm(w_flat, k_flat)                         # (B*L, H_q, D)
        k_mixed = k_mixed_flat.reshape(B, L, H_q, self.head_dim)         # (B, L, H_q, D)

        v_flat = v.reshape(B*L, H_kv, self.head_dim)
        v_mixed_flat = torch.bmm(w_flat, v_flat)                         # (B*L, H_q, D)
        v_mixed = v_mixed_flat.reshape(B, L, H_q, self.head_dim)

        # Now standard multi‑head attention with the mixed K/V
        q = q.transpose(1, 2)         # (B, H_q, L, D)
        k_mixed = k_mixed.transpose(1, 2)
        v_mixed = v_mixed.transpose(1, 2)

        scores = torch.matmul(q, k_mixed.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)

        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v_mixed)                # (B, H_q, L, D)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(context)

        return out, attn_probs