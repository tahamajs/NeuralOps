"""
SOA Fused Attention – low‑rank latent space attention with zero‑shot init.
Project hidden states → latent space → Q/K/V → attention → output.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class SOAFusedAttention(nn.Module):
    """
    Attention that operates on a low‑rank latent representation.
    For DistilBERT, we first compress hidden states to rank `k_rank`,
    then run the original SOA logic.
    """
    def __init__(self, model_config, k_rank: int = 32, dropout: float = 0.1,
                 pretrained_attn=None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.hidden_size = model_config.hidden_size
        self.k_rank = k_rank

        # Down‑projection from hidden_size to latent space
        self.input_proj = nn.Linear(self.hidden_size, k_rank, bias=False)

        # SOA core: latent -> Q/K/V (full hidden_size)
        self.q_proj = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(k_rank, self.hidden_size, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.dropout = nn.Dropout(dropout)

        # Zero‑shot initialisation from pretrained weights
        self._init_from_pretrained(pretrained_attn)

    def _init_from_pretrained(self, pretrained_attn):
        """Use SVD to initialise so that W_orig ≈ (Q_proj) ∘ (input_proj)."""
        with torch.no_grad():
            # For Q, K, V we want: W_orig ≈ self.q_proj.weight @ self.input_proj.weight
            # Factor W_orig = U Σ V^T, then set input_proj = V^T (rank k_rank)
            # and q_proj = U Σ (first k_rank columns).
            for proj_name, orig_linear in [
                ('q_proj', pretrained_attn.q_lin),
                ('k_proj', pretrained_attn.k_lin),
                ('v_proj', pretrained_attn.v_lin),
            ]:
                W = orig_linear.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(self.k_rank, U.size(1), Vt.size(0))
                # input_proj is shared; we set it once using average of the three factors?
                # Simpler: set input_proj from the first factor (or average).
                # We'll set it from the first one (q) and copy for others.
                if proj_name == 'q_proj':
                    self.input_proj.weight.data[:r, :] = Vt[:r, :].to(self.input_proj.weight.dtype)
                proj = getattr(self, proj_name)
                proj.weight.data[:, :r] = (U[:, :r] * S[:r]).to(proj.weight.dtype)

            # Output projection: W_out ≈ out_proj.weight @ input_proj.weight
            W_out = pretrained_attn.out_lin.weight.data.float()
            U, S, Vt = torch.linalg.svd(W_out, full_matrices=False)
            r = min(self.k_rank, U.size(1), Vt.size(0))
            self.out_proj.weight.data[:, :r] = (U[:, :r] * S[:r]).to(self.out_proj.weight.dtype)
            if pretrained_attn.out_lin.bias is not None:
                self.out_proj.bias.data = pretrained_attn.out_lin.bias.data.clone()

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape

        # Compress to latent space
        x_latent = self.input_proj(hidden_states)          # (B, N, k_rank)

        # Project to Q, K, V (full hidden_size)
        q = self.q_proj(x_latent).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_latent).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_latent).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Mask handling
        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]            # (B,1,1,N)
            scores = scores + (1.0 - mask) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)          # (B, H, N, D)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)

        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("soa_fused_attn")
def inject_soa_fused_attn(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    k_rank = getattr(comp_config, 'soa_rank', 32)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = SOAFusedAttention(model_config, k_rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn

