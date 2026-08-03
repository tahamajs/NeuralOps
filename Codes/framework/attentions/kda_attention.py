"""Kernelized Decay Attention (KDA) / Recurrent Linear Attention.

Linear recurrence with learned exponential gating decay.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("kda")
class KDAAttention(nn.Module):
    r"""Kernelized Decay Attention (KDA) Mechanism.

    Implements causal linear recurrent state tracking with a learned gating decay parameter $\beta$:
    $$S_t = \sigma(\beta) \odot S_{t-1} + k_t v_t^\top \in \mathbb{R}^{d_k \times d_v}$$
    $$o_t = S_t q_t \in \mathbb{R}^{d_v}$$
    where keys are normalized with an $\ell_2$ unit hypersphere projection $\tilde{k}_t = k_t / \|k_t\|_2$.

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot N \cdot d_k \cdot d_v)$
        - Memory: $\mathcal{O}(B \cdot H \cdot d_k \cdot d_v)$ recurrent state complexity.

    Tensor Dimensions:
        - hidden_states (or x): $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(self, config: Any, rank: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.dim: int = config.hidden_size
        self.q_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.k_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.v_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.out_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.beta: nn.Parameter = nn.Parameter(torch.ones(self.num_heads, 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for Kernelized Decay Attention.

        Args:
            hidden_states (torch.Tensor): Input activation tensor `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Attention mask tensor.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Contextualized representation `[Batch, SeqLen, HiddenDim]` and `None`.
        """
        x: torch.Tensor = hidden_states
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim
        q: torch.Tensor = self.q_proj(x).view(B, L, H, d).transpose(1, 2)
        k: torch.Tensor = F.normalize(
            self.k_proj(x).view(B, L, H, d).transpose(1, 2), dim=-1
        )
        v: torch.Tensor = self.v_proj(x).view(B, L, H, d).transpose(1, 2)
        h: torch.Tensor = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        outputs = []
        decay: torch.Tensor = torch.sigmoid(self.beta).view(1, H, 1, 1)
        for t in range(L):
            qt = q[:, :, t, :]
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]
            h = decay * h + (kt.unsqueeze(-1) @ vt.unsqueeze(-2))
            out_t = (h @ qt.unsqueeze(-1)).squeeze(-1)
            outputs.append(out_t)
        out: torch.Tensor = torch.stack(outputs, dim=2)  # (B, H, L, d)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), None
