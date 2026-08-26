r"""
Random Projection Compressed (RPC) Attention.

Employs randomized linear subspace sketching via fixed head-wise Gaussian random projection buffers
to compress query-key similarity evaluations into lower-dimensional manifolds.

Mathematical Mechanism:
    Given queries $Q$, keys $K$, values $V \in \mathbb{R}^{B \times H \times L \times d}$ and random projection operators $P \in \mathbb{R}^{H \times d \times d_p}$:
    1. **Random Subspace Sketching:**
       $$\widetilde{Q} = Q P \in \mathbb{R}^{B \times H \times L \times d_p}, \quad \widetilde{K} = K P \in \mathbb{R}^{B \times H \times L \times d_p}$$
    2. **Sketched Attention Kernel:**
       $$A = \operatorname{Softmax}\left(\frac{\widetilde{Q} \widetilde{K}^T}{\sqrt{d_p}} + M\right) \in \mathbb{R}^{B \times H \times L \times L}$$
       $$O = (A V) W_O \in \mathbb{R}^{B \times L \times D}$$

Computational Complexity:
    - Projection sketching: $\mathcal{O}(B \cdot H \cdot L \cdot d \cdot d_p)$.
    - Attention matrix product: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_p)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot H \cdot L \cdot d_p \cdot (d + L))$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("rpc")
class RPCAttention(BaseAttention):
    r"""Random Projection Compressed Attention module.

    Args:
        config: Transformer configuration specifying `hidden_size`, `num_attention_heads`, and optional `dropout`.
        rank: Compressed sketch projection rank ($d_p$).
        pretrained_attn: Optional source module for weight initialization.
    """

    def __init__(
        self,
        config: Any,
        rank: int = 64,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.rank: int = rank if rank is not None else 64
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.dim: int = config.hidden_size
        dropout_p: float = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )

        self.W_q = nn.Linear(self.dim, self.dim)
        self.W_k = nn.Linear(self.dim, self.dim)
        self.W_v = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(dropout_p)

        self.register_buffer(
            "proj", torch.randn(self.num_heads, self.head_dim, self.rank) * (self.head_dim**-0.5)
        )

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        r"""Copy weights from pretrained transformer attention module."""
        with torch.no_grad():
            if hasattr(src, "q_lin"):
                self.W_q.weight.copy_(src.q_lin.weight)
                if src.q_lin.bias is not None:
                    self.W_q.bias.copy_(src.q_lin.bias)
                self.W_k.weight.copy_(src.k_lin.weight)
                if src.k_lin.bias is not None:
                    self.W_k.bias.copy_(src.k_lin.bias)
                self.W_v.weight.copy_(src.v_lin.weight)
                if src.v_lin.bias is not None:
                    self.W_v.bias.copy_(src.v_lin.bias)
                self.out_proj.weight.copy_(src.out_lin.weight)
                if src.out_lin.bias is not None:
                    self.out_proj.bias.copy_(src.out_lin.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute random projection compressed attention.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head masking tensor.
            output_attentions: Return explicit attention probabilities if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected representation `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax probability matrix or None.
        """
        x = hidden_states
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        proj = self.proj.to(x.device)
        k_feat = torch.einsum("bhld,hde->bhle", k, proj)
        q_feat = torch.einsum("bhld,hde->bhle", q, proj)

        scale = self.rank**-0.5
        scores = torch.matmul(q_feat, k_feat.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask) * -10000.0
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1)
            else:
                scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        output = self.out_proj(out)

        if output_attentions:
            return output, attn
        return output, None
