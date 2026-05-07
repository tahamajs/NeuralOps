"""
SOA – Shared Orthogonal Attention
Project hidden states to a low‑rank latent space, then perform attention
in that space, and project back. Zero‑shot initialisation from pretrained
DistilBERT attention via SVD factorisation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.attentions.core.registry import ATTENTION_REGISTRY


class SOAAttention(nn.Module):
    """
    Shared Orthogonal Attention (SOA) – operates on a low‑rank latent space.
    For DistilBERT, we first compress hidden_states to rank `k_rank` using
    a learned linear map (initialised from pretrained QKV projections).
    Then we apply the original SOA logic.
    """
    def __init__(self, model_config, k_rank: int, dropout: float = 0.1,
                 pretrained_attn=None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.hidden_size = model_config.hidden_size
        self.k_rank = k_rank

        # ----- Down‑projection from hidden_size to latent space -----
        # We will initialise this from the original Q/K/V weights using SVD
        # to approximate the mapping: hidden -> latent.
        self.input_proj = nn.Linear(self.hidden_size, k_rank, bias=False)

        # ----- SOA core: latent -> Q/K/V -> attention -> output -----
        self.q_proj = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.dropout = nn.Dropout(dropout)

        # Initialise from pretrained attention
        self._init_from_pretrained(pretrained_attn)

    def _init_from_pretrained(self, pretrained_attn):
        """
        Zero‑shot initialisation: we want the composition
            (Q/K/V projection in SOA) ∘ (input_proj)
        to approximate the original Q/K/V projection.
        That is: W_orig ≈ W_soa @ W_input.
        We can achieve this by SVD: factorise W_orig into U Σ V^T,
        then set W_input = V^T (rank k_rank) and W_soa = U Σ.
        """
        with torch.no_grad():
            # Original Q weight (out=hidden_size, in=hidden_size)
            W_q = pretrained_attn.q_lin.weight.data.float()
            W_k = pretrained_attn.k_lin.weight.data.float()
            W_v = pretrained_attn.v_lin.weight.data.float()
            W_o = pretrained_attn.out_lin.weight.data.float()

            # For each, factorise and assign
            for W, proj, soa_proj in zip([W_q, W_k, W_v],
                                         [self.input_proj, self.input_proj, self.input_proj],
                                         [self.q_proj, self.k_proj, self.v_proj]):
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                k = min(self.k_rank, U.size(1), Vt.size(0))
                # Set input_proj to Vt[:k, :] (maps hidden -> latent)
                proj.weight.data[:k, :] = Vt[:k, :].to(proj.weight.dtype)
                # Set soa_proj to U[:, :k] @ diag(S[:k])
                soa_proj.weight.data[:, :k] = (U[:, :k] * S[:k]).to(soa_proj.weight.dtype)
                # Leave the rest of the factors zero

            # Output projection: similarly W_out ≈ W_out_soa @ W_input
            # but here input_proj is already used; we can directly factor W_o
            # and set out_proj.weight = U Σ, and we keep same input_proj (already set)
            U, S, Vt = torch.linalg.svd(W_o, full_matrices=False)
            k = min(self.k_rank, U.size(1), Vt.size(0))
            self.out_proj.weight.data[:, :k] = (U[:, :k] * S[:k]).to(self.out_proj.weight.dtype)
            # Bias is copied (if exists)
            if pretrained_attn.out_lin.bias is not None:
                self.out_proj.bias.data = pretrained_attn.out_lin.bias.data.clone()

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape

        # 1. Compress hidden states to latent space
        x_latent = self.input_proj(hidden_states)          # (B, N, k_rank)

        # 2. SOA core: Q/K/V projections
        q = self.q_proj(x_latent)      # (B, N, H)
        k = self.k_proj(x_latent)
        v = self.v_proj(x_latent)

        # 3. Multi‑head reshape
        H = self.num_heads
        D = self.head_dim
        q = q.view(B, N, H, D).transpose(1, 2)   # (B, H, N, D)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        # 4. Scaled dot‑product attention
        scale = D ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 5. Mask handling
        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]      # (B,1,1,N)
            scores = scores + (1.0 - mask) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. Apply attention to values
        context = torch.matmul(attn_weights, v)       # (B, H, N, D)
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)

        # 7. Final output projection
        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("soa_attn")
def inject_soa_attn(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """Injection function for DistilBERT."""
    k_rank = getattr(comp_config, 'soa_rank', 32)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = SOAAttention(model_config, k_rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn