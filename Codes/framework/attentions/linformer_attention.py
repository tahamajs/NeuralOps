"""
Linformer Attention – linear complexity self‑attention via low‑rank sequence projections.
Paper: https://arxiv.org/abs/2006.04768
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("linformer")
class LinformerAttention(nn.Module):
    r"""Linformer: Linear Complexity Multi-Head Self-Attention (Wang et al., 2020).

    Approximates standard $\mathcal{O}(N^2)$ attention by projecting Key and Value sequence dimensions
    from $N$ to a low-rank dimension $k \ll N$ using learnable linear projection matrices $E, F \in \mathbb{R}^{k \times N_{\text{max}}}$:
    $$K_{\text{proj}} = E K \in \mathbb{R}^{B \times H \times k \times d_k}, \quad V_{\text{proj}} = F V \in \mathbb{R}^{B \times H \times k \times d_k}$$
    $$\text{Attention}_{\text{Linformer}}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{proj}}^\top}{\sqrt{d_k}}\right) V_{\text{proj}}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot N \cdot k \cdot d_k)$
        - Memory: $\mathcal{O}(B \cdot H \cdot (N \cdot k + k \cdot d_k))$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, k])$.
    """

    def __init__(
        self,
        config: Any,
        k: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        self.k_proj_dim: int = getattr(
            config,
            "linformer_k",
            k if k is not None else getattr(config, "k", 64),
        )
        self.max_seq_len: int = getattr(
            config,
            "max_position_embeddings",
            max_seq_len if max_seq_len is not None else 512,
        )

        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)

        # Aliases
        # Sequence projection matrices E and F: (k, max_seq_len)
        self.E: nn.Parameter = nn.Parameter(
            torch.empty(self.k_proj_dim, self.max_seq_len)
        )
        self.F: nn.Parameter = nn.Parameter(
            torch.empty(self.k_proj_dim, self.max_seq_len)
        )
        nn.init.normal_(self.E, std=1.0 / math.sqrt(self.k_proj_dim))
        nn.init.normal_(self.F, std=1.0 / math.sqrt(self.k_proj_dim))

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for linear Linformer attention."""
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q: torch.Tensor = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        K: torch.Tensor = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
        V: torch.Tensor = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        # Zero out padding tokens before projection
        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # (B, N)
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
        K = K * mask_expanded
        V = V * mask_expanded

        # Slice or pad E, F to active sequence length N
        if N <= self.max_seq_len:
            E_act = self.E[:, :N]
            F_act = self.F[:, :N]
        else:
            E_act = F.pad(self.E, (0, N - self.max_seq_len))
            F_act = F.pad(self.F, (0, N - self.max_seq_len))

        # Project sequence dimension: E @ K -> (B, H, k, d)
        # K is (B, H, N, d), E_act is (k, N) -> torch.matmul(E_act, K)
        K_proj: torch.Tensor = torch.matmul(E_act, K)  # (B, H, k, d)
        V_proj: torch.Tensor = torch.matmul(F_act, V)  # (B, H, k, d)

        scale: float = d**-0.5
        scores: torch.Tensor = torch.matmul(Q, K_proj.transpose(-2, -1)) * scale  # (B, H, N, k)

        attn: torch.Tensor = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context: torch.Tensor = torch.matmul(attn, V_proj)  # (B, H, N, d)

        # Mask padding queries
        context = context * mask_expanded
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out: torch.Tensor = self.out_proj(context)

        return out, attn if output_attentions else None
