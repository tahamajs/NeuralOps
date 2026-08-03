from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.mamba import Mamba


@ATTENTION_REGISTRY.register("expert_choice_moe")
class ExpertChoiceMoE(nn.Module):
    r"""
    Top-$k$ Token-Choice Mixture of Experts (MoE) Layer.

    Mathematical Mechanism:
        Routes tokens dynamically to top-$k$ expert feed-forward networks:
        $$\begin{aligned}
        S &= X W_r \in \mathbb{R}^{B \times L \times E} \\
        W_{\text{top}}, \mathcal{I} &= \text{topk}(S, k, \text{dim}=-1) \\
        W_{\text{gate}} &= \text{softmax}(W_{\text{top}}, \text{dim}=-1) \\
        \text{MoE}(X) &= \sum_{e=1}^{E} \mathbb{I}(e \in \mathcal{I}) \cdot W_{\text{gate}, e} \cdot \text{FFN}_e(X)
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot (D \cdot E + k \cdot D \cdot D_{\text{ff}}))$
        - Space: $\mathcal{O}(B \cdot L \cdot E)$ routing logits.

    Tensor Dimensions:
        - Input $x$: `[B, L, D]`
        - Output: `[B, L, D]`
    """

    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.n_experts: int = n_experts
        self.top_k: int = top_k
        self.experts: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                )
                for _ in range(n_experts)
            ]
        )
        self.router: nn.Linear = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass through the MoE layer.

        Args:
            x: Input tensor of shape `[B, L, D]`.

        Returns:
            MoE transformed tensor of shape `[B, L, D]`.
        """
        B, L, D = x.shape
        scores = self.router(x)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (indices == e).any(dim=-1)
            if mask.any():
                selected_weights = (weights * (indices == e).float()).sum(
                    dim=-1, keepdim=True
                )
                out += selected_weights * self.experts[e](x * mask.unsqueeze(-1))
        return out


@ATTENTION_REGISTRY.register("jamba2")
class Jamba2Block(BaseAttention):
    r"""
    Jamba-2 Hybrid Transformer-Mamba-MoE Block.

    Mathematical Mechanism:
        Sequentially chains Mamba SSM, Multi-Head Attention, and Sparse Mixture of Experts
        with Pre-LayerNorm residual connections:
        $$\begin{aligned}
        X_1 &= \text{LayerNorm}(X + \text{Mamba}(X)) \\
        X_2 &= \text{LayerNorm}(X_1 + \text{MHA}(X_1, X_1, X_1)) \\
        X_3 &= \text{LayerNorm}(X_2 + \text{MoE}(X_2))
        \end{aligned}$$

    Computational Complexity:
        - Mamba Step: $\mathcal{O}(B \cdot L \cdot D^2)$
        - Attention Step: $\mathcal{O}(B \cdot L^2 \cdot D)$
        - MoE Step: $\mathcal{O}(B \cdot L \cdot k \cdot D \cdot D_{\text{ff}})$
        - Total Time: $\mathcal{O}(B \cdot L^2 \cdot D + B \cdot L \cdot D \cdot (D + k D_{\text{ff}}))$

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, attn_weights)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int = 8,
        top_k: int = 2,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.mamba: Mamba = Mamba(d_model)
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.moe: ExpertChoiceMoE = ExpertChoiceMoE(d_model, n_experts, top_k)
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm3: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Sequential execution through Mamba, MHA, and MoE.

        Args:
            hidden_states: Input tensor `[B, L, D]`.
            attention_mask: Optional mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Boolean flag to return attention map.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, attn_weights)` with output of shape `[B, L, D]`.
        """
        mamba_out, _ = self.mamba(hidden_states)
        x = self.norm1(hidden_states + mamba_out)
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm2(x + attn_out)
        x = self.norm3(x + self.moe(x))
        return x, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("jamba2")
def inject_jamba2(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> Jamba2Block:
    r"""
    Injection hook for Jamba2Block into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module.

    Returns:
        Instantiated `Jamba2Block` assigned to `layer.attention`.
    """
    d_model: int = model_config.hidden_size
    n_heads: int = getattr(model_config, "num_attention_heads", 12)
    n_experts: int = getattr(comp_config, "n_experts", 8)
    top_k: int = getattr(comp_config, "top_k", 2)
    new_block = Jamba2Block(
        d_model=d_model,
        n_heads=n_heads,
        n_experts=n_experts,
        top_k=top_k,
        config=model_config,
    )
    layer.attention = new_block
    return new_block
