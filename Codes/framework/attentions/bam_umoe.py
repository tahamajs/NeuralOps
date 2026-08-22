r"""
BAM-UMoE: Unified Mixture of Experts Attention Block.

This module implements a Unified Mixture of Experts (UMoE) transformer layer with top-$k$ routing
applied jointly to multi-head projections and feed-forward transformation networks.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class UnifiedMoE(nn.Module):
    r"""
    Vectorized Mixture of Experts (MoE) layer with top-$k$ gating.

    Mathematical Formulation:
        Given an input token representation $x \in \mathbb{R}^{B \times L \times D}$:
        1. Routing Logits:
           $$R = x W_{\text{gate}} \in \mathbb{R}^{B \times L \times E}$$
           where $W_{\text{gate}} \in \mathbb{R}^{D \times E}$ and $E$ is the number of experts.
        2. Top-$k$ Expert Selection:
           $$\mathcal{I}_{\text{top-}k}, S_{\text{top-}k} = \text{Top-}k(R, k)$$
           $$p = \text{Softmax}(S_{\text{top-}k}) \in \mathbb{R}^{B \times L \times k}$$
        3. Expert Computation:
           $$\text{Expert}_e(x) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(x)))$$
        4. Gated Aggregation:
           $$y = \sum_{j=1}^k p_j \cdot \text{Expert}_{\mathcal{I}_j}(x)$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot (D \cdot E + E \cdot D \cdot d_{\text{out}}))$.
        - Memory: $\mathcal{O}(B \cdot L \cdot E \cdot d_{\text{out}})$ parallel activation buffer.

    Tensor Dimensions:
        - Input: $[B, L, D]$
        - Router Logits: $[B, L, E]$
        - Expert Stack: $[B, L, E, d_{\text{out}}]$
        - Output: $[B, L, d_{\text{out}}]$
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        out_dim = output_dim or dim
        self.out_dim: int = out_dim

        self.router = nn.Linear(dim, num_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, out_dim * 2),
                    nn.GELU(),
                    nn.Linear(out_dim * 2, out_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward execution pass for top-$k$ Unified MoE."""
        B, L, D = x.shape

        logits = self.router(x)  # (B, L, E)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # (B, L, k)

        scores = F.softmax(topk_vals, dim=-1)  # (B, L, k)

        # Compute all expert outputs in parallel
        experts_out = torch.stack([exp(x) for exp in self.experts], dim=2)  # (B, L, E, out_dim)

        # Gather expert outputs for top-k selections
        topk_out = torch.gather(
            experts_out,
            2,
            topk_idx.unsqueeze(-1).expand(B, L, self.top_k, self.out_dim),
        )  # (B, L, k, out_dim)

        # Weighted sum of top-k expert outputs
        out = torch.sum(topk_out * scores.unsqueeze(-1), dim=2)
        return out


@ATTENTION_REGISTRY.register("bam_umoe")
class BAMUMoEBlock(BaseAttention):
    r"""
    BAM Unified Mixture of Experts Attention Block.

    Replaces standard linear attention projections and feed-forward networks with
    parameter-efficient top-$k$ conditional computation blocks.

    Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot (D \cdot E + E \cdot D \cdot d_{\text{ffn}}))$.
        - Space: $\mathcal{O}(B \cdot L \cdot D)$ token representations.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, L, D]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        config: Any,
        rank: int = 64,
        num_experts: int = 8,
        top_k: int = 2,
        layer_id: Optional[int] = None,
        num_layers: Optional[int] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        self.dim: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.dim // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        self.qkv_moe = UnifiedMoE(
            self.dim, num_experts, top_k, output_dim=3 * self.all_head_size
        )
        self.ffn_moe = UnifiedMoE(self.dim, num_experts, top_k, output_dim=self.dim)

        self.output_proj = nn.Linear(self.all_head_size, self.dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(config.dropout if hasattr(config, "dropout") else 0.1)

        self.layer_id: Optional[int] = layer_id
        self.num_layers: Optional[int] = num_layers
        self.old_attn: Optional[nn.Module] = old_attn
        self.comp_config: Optional[Any] = comp_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for BAMUMoEBlock.

        Args:
            hidden_states (torch.Tensor): Input activations of shape $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Binary or float mask tensor.
            output_attentions (bool): Flag indicating whether to return attention maps.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed hidden representation of shape $[B, L, D]$.
                - None.
        """
        B, L, D = hidden_states.shape
        qkv = self.qkv_moe(hidden_states)
        _q, _k, v = qkv.chunk(3, dim=-1)
        proj = self.output_proj(v)
        h1 = self.norm1(hidden_states + self.dropout(proj))
        ffn = self.ffn_moe(h1)
        out = self.norm2(h1 + self.dropout(ffn))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, L) -> (B, L, 1)
                mask = attention_mask.unsqueeze(-1).to(out.dtype)
            elif attention_mask.dim() == 4:
                # Extracts [B, L] from [B, 1, 1, L] or [B, 1, L, L] -> (B, L, 1)
                mask = (attention_mask[:, 0, 0, :] > -10000.0).float().unsqueeze(-1).to(out.dtype)
            else:
                mask = attention_mask.unsqueeze(-1).to(out.dtype) if attention_mask.dim() == 1 else attention_mask.to(out.dtype)
            out = out * mask

        return out, None
