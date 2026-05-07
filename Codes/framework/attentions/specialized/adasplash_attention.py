# framework/attentions/specialized/adasplash_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


def entmax15(
    scores: torch.Tensor,
    dim: int = -1,
    n_iter: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute α‑entmax for α = 1.5 using a simple fixed‑point iteration.
    Derived from the original entmax algorithm with Halley’s method.
    """
    p = F.softmax(scores, dim=dim)
    for _ in range(n_iter):
        q = p.sqrt()
        tau = torch.sum(q * p, dim=dim, keepdim=True)
        p_1 = q / tau
        if torch.max(torch.abs(p_1 - p)) < eps:
            break
        p = p_1
    return p


@ATTENTION_REGISTRY.register("adasplash")
class AdaSplashAttention(BaseAttention):
    """
    Adaptive sparse attention using α‑entmax (AdaSplash).
    Paper: https://arxiv.org/abs/2502.12082
    """

    def __init__(self, config, rank: int = None, alpha: float = 1.5, n_iter: int = 10):
        super().__init__(config, rank)
        self.alpha = alpha
        self.n_iter = n_iter
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, D_head = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, D_head).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, D_head).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, D_head).transpose(1, 2)

        # Compute attention logits
        logits = torch.matmul(q, k.transpose(-2, -1)) / (D_head ** 0.5)

        # Apply mask (same broadcasting as in BERT)
        if attention_mask is not None:
            # attention_mask: (B, 1, 1, L) for key masking
            logits = logits + (attention_mask.float() - 1.0) * -1e9

        # α‑entmax activation
        if self.alpha == 1.5:
            attn = entmax15(logits, dim=-1, n_iter=self.n_iter)
        else:
            raise NotImplementedError(f"α={self.alpha} not yet implemented")
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), attn if kwargs.get("output_attentions") else None