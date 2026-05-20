# framework/attentions/minimal_svd.py
"""
Minimal SVD‑attention – zero‑shot drop‑in replacement for DistilBERT attention.
Factorises Q/K/V/Out projections with SVD (energy threshold) but keeps full attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class FactorisedLinear(nn.Module):
    """SVD‑factorised linear layer: W ≈ U·diag(S)·Vt → (A)(B) with A = U·diag(S), B = Vt."""
    def __init__(self, pretrained_weight: torch.Tensor, pretrained_bias: Optional[torch.Tensor],
                 energy: float = 0.95):
        """
        Args:
            pretrained_weight: Original weight matrix (out_dim, in_dim)
            pretrained_bias: Original bias (out_dim,), may be None
            energy: Fraction of singular value energy to retain (0..1)
        """
        super().__init__()
        out_dim, in_dim = pretrained_weight.shape

        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
            total_energy = (S ** 2).sum()
            cum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
            rank = int(torch.searchsorted(cum_energy, energy).item()) + 1
            rank = max(1, min(rank, min(out_dim, in_dim), 64))
            self.rank = rank

            # A = U[:,:rank] * S[:rank], shape (out_dim, rank)
            A = U[:, :rank] * S[:rank].unsqueeze(0)
            # B = Vt[:rank, :], shape (rank, in_dim)
            B = Vt[:rank, :]

        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up   = nn.Linear(rank, out_dim, bias=(pretrained_bias is not None))

        self.down.weight.data = B.float()
        self.up.weight.data   = A.float()

        if pretrained_bias is not None:
            self.up.bias.data = pretrained_bias.float()
        else:
            # If bias was present but is None, still ensure zero bias
            # But if bias=False in original, the up layer has bias=False anyway.
            # This line handles the case where bias is None but we created layer with bias=True? Not possible.
            # Safer: only set if bias exists.
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class MinimalSVDAttention(nn.Module):
    """
    DistilBERT attention replacement using SVD‑factorised projections.
    Attention remains full (quadratic) but linear projections are compressed.
    Zero‑shot initialisation from a pre‑trained attention module.
    """
    def __init__(self, model_config, energy: float = 0.95, dropout: float = 0.1,
                 pretrained_attn: Optional[nn.Module] = None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot initialisation")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size

        # Factorise each projection
        self.q_proj = FactorisedLinear(
            pretrained_attn.q_lin.weight, pretrained_attn.q_lin.bias, energy
        )
        self.k_proj = FactorisedLinear(
            pretrained_attn.k_lin.weight, pretrained_attn.k_lin.bias, energy
        )
        self.v_proj = FactorisedLinear(
            pretrained_attn.v_lin.weight, pretrained_attn.v_lin.bias, energy
        )
        self.out_proj = FactorisedLinear(
            pretrained_attn.out_lin.weight, pretrained_attn.out_lin.bias, energy
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        # Linear projections
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scale = head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Mask handling (DistilBERT provides 2D attention_mask: (batch, seq_len) with 1 for real tokens)
        if attention_mask is not None:
            # Expand mask to (batch, 1, 1, seq_len) for broadcasting
            if attention_mask.dim() == 2:
                extended_mask = attention_mask[:, None, None, :]
            else:
                extended_mask = attention_mask
            # Convert 1/0 to 0/-inf for addition
            extended_mask = (1.0 - extended_mask) * -1e9
            scores = scores + extended_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)

        # Reshape back to (batch, seq_len, hidden_size)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)

        # Always return a 2‑tuple: (output, attention_weights)
        # If attention weights not requested, return None as second element.
        if output_attentions:
            return output, attn_weights
        else:
            return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


# ----------------------------------------------------------------------
# Framework injection
# ----------------------------------------------------------------------
@ATTENTION_REGISTRY.register("minimal_svd")
def inject_minimal_svd(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """Factory function to replace a DistilBERT attention layer with MinimalSVDAttention."""
    energy = getattr(comp_config, 'svd_energy', 0.95)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = MinimalSVDAttention(
        model_config=model_config,
        energy=energy,
        dropout=dropout,
        pretrained_attn=old_attn
    )
    layer.attention = new_attn
    return new_attn
