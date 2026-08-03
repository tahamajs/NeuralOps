import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.simple_ssm import SimpleSSM


class InfoMambaConceptBottleneck(nn.Module):
    r"""
    Minimal-Bandwidth Concept Bottleneck MLP.

    Mathematical Mechanism:
        Compresses input representations into a low-dimensional concept subspace:
        $$\tilde{X} = \text{GELU}(X W_{\text{down}}) W_{\text{up}}$$
        where $W_{\text{down}} \in \mathbb{R}^{D \times d_{\text{bottleneck}}}$ and $W_{\text{up}} \in \mathbb{R}^{d_{\text{bottleneck}} \times D}$.

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{bottleneck}})$
        - Space: $\mathcal{O}(B \cdot L \cdot d_{\text{bottleneck}})$

    Tensor Dimensions:
        - Input: `[B, L, D]`
        - Output: `[B, L, D]`
    """

    def __init__(self, dim: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.dim: int = dim
        self.bottleneck_dim: int = bottleneck_dim
        self.W_down: nn.Linear = nn.Linear(dim, bottleneck_dim, bias=False)
        self.W_up: nn.Linear = nn.Linear(bottleneck_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Passes input through low-rank concept bottleneck.

        Args:
            x: Input tensor of shape `[B, L, D]`.

        Returns:
            Projected output of shape `[B, L, D]`.
        """
        return self.W_up(F.gelu(self.W_down(x)))


@ATTENTION_REGISTRY.register("jamba_info")
class JambaInfoBlock(BaseAttention):
    r"""
    Information-Bottlenecked Jamba Architecture with Dual Operational Modes.

    Mathematical Mechanism:
        - **Retrieval Mode**: Employs head-sparse Differential Attention:
          $$\begin{aligned}
          A_1 &= \text{softmax}\left(\frac{Q_1 K_1^\top}{\sqrt{d_k}}\right), \quad A_2 = \text{softmax}\left(\frac{Q_2 K_2^\top}{\sqrt{d_k}}\right) \\
          A_{\text{diff}} &= A_1 - 0.5 A_2 \\
          A_{\text{diff}}[h \ge H_{\text{active}}] &= 0 \\
          O &= (A_{\text{diff}} V) W_O
          \end{aligned}$$
        - **SSM Mode**: Routes through the concept bottleneck and SimpleSSM:
          $$O = W_O\left(X + \text{SimpleSSM}(\text{Bottleneck}(X))\right)$$

    Computational Complexity:
        - Retrieval Mode: $\mathcal{O}(B \cdot H_{\text{active}} \cdot L^2 \cdot d_k)$ time, $\mathcal{O}(B \cdot H \cdot L^2)$ space.
        - SSM Mode: $\mathcal{O}(B \cdot L \cdot (D \cdot d_{\text{bottleneck}} + D \cdot d_{\text{state}}))$ time — linear in sequence length.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, diff_attn)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 64,
        mode: str = "retrieval",
        sparsity: float = 0.15,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = getattr(config, "hidden_size", 768)
        self.mode: str = mode
        self.sparsity: float = sparsity

        if mode == "retrieval":
            self.num_heads: int = getattr(config, "num_attention_heads", 12)
            self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64
            self.active_heads: int = max(1, int(self.num_heads * sparsity))
            self.W_q1: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
            self.W_k1: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
            self.W_q2: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
            self.W_k2: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
            self.W_v: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
            self.out_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        else:
            self.bottleneck: InfoMambaConceptBottleneck = InfoMambaConceptBottleneck(self.dim)
            self.ssm: SimpleSSM = SimpleSSM(self.dim)
            self.out_proj: nn.Linear = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation for JambaInfoBlock.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional attention mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Boolean flag to return attention map.
            mask: Optional alias for attention mask.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, diff_attn)` where `output` is `[B, L, D]`.
        """
        if attention_mask is None and mask is not None:
            attention_mask = mask
        x = hidden_states
        B, L, D = x.shape
        diff_attn: Optional[torch.Tensor] = None

        if self.mode == "retrieval":
            q1 = self.W_q1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k1 = self.W_k1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            q2 = self.W_q2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k2 = self.W_k2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

            scale = 1.0 / math.sqrt(self.head_dim)
            attn1 = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) * scale, dim=-1)
            attn2 = F.softmax(torch.matmul(q2, k2.transpose(-2, -1)) * scale, dim=-1)
            diff_attn = attn1 - 0.5 * attn2
            diff_attn[:, self.active_heads :] = 0.0
            out = torch.matmul(diff_attn, v).transpose(1, 2).contiguous().view(B, L, D)
        else:
            out = x + self.ssm(self.bottleneck(x), attention_mask=attention_mask)

        out = self.out_proj(out)
        return out, (diff_attn if output_attentions else None)


@ATTENTION_REGISTRY.register("jamba_info")
def inject_jamba_info(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> JambaInfoBlock:
    r"""
    Injection hook for JambaInfoBlock into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module.

    Returns:
        Instantiated `JambaInfoBlock` assigned to `layer.attention`.
    """
    mode: str = getattr(comp_config, "mode", "retrieval")
    sparsity: float = getattr(comp_config, "sparsity", 0.15)
    new_block = JambaInfoBlock(
        config=model_config,
        rank=getattr(comp_config, "rank", 64),
        mode=mode,
        sparsity=sparsity,
    )
    layer.attention = new_block
    return new_block
