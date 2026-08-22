r"""
Factorised Matrix State Attention Module.

Implements an associative recurrent matrix memory update rule combined with low-rank factorized
bilinear output projection matrices $W = L R^T$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("factorised_matrix_state")
class FactorisedMatrixState(nn.Module):
    r"""
    Factorised Matrix State Recurrent Attention.

    Mathematical Formulation:
        1. Recurrent Associative Matrix State:
           Given input activations $x_t \in \mathbb{R}^D$ at timestep $t$:
           $$S_t = S_{t-1} + x_t x_t^T \in \mathbb{R}^{D \times D}, \quad S_0 = \mathbf{0}$$
           $$u_t = S_t x_t = \sum_{\tau=1}^t x_\tau (x_\tau^T x_t) \in \mathbb{R}^D$$

        2. Low-Rank Factorized Output Projection:
           Transforms associative outputs $U = [u_1, \dots, u_N] \in \mathbb{R}^{B \times N \times D}$ via factorized weights $L, R \in \mathbb{R}^{D \times r}$:
           $$\text{Output} = (U L) R^T \in \mathbb{R}^{B \times N \times D}$$

    Complexity:
        - Time Complexity: $\mathcal{O}(B \cdot N \cdot D^2 + B \cdot N \cdot D \cdot r)$.
        - Memory Complexity: $\mathcal{O}(B \cdot D^2)$ for recurrent state storage.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, N, D]$
        - Recurrent State Matrix: $[B, D, D]$
        - Factorized Output: $[B, N, D]$
    """

    def __init__(self, config: Any, rank: int = 32) -> None:
        super().__init__()
        self.d_model: int = config.hidden_size
        self.rank: int = rank if rank is not None else 32
        self.L: nn.Parameter = nn.Parameter(torch.randn(self.d_model, self.rank) * 0.01)
        self.R: nn.Parameter = nn.Parameter(torch.randn(self.d_model, self.rank) * 0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for FactorisedMatrixState.

        Args:
            hidden_states (torch.Tensor): Input token activations $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor $[B, N]$ or $[B, 1, 1, N]$.
            output_attentions (bool): Whether to return intermediate states.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed activations $[B, N, D]$.
                - None.
        """
        B, N, D = hidden_states.shape

        mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            elif attention_mask.dim() == 4:
                mask = (attention_mask[:, 0, 0, :] > -10000.0).float().unsqueeze(-1).to(hidden_states.dtype)
            else:
                mask = attention_mask.to(hidden_states.dtype)
            hidden_states = hidden_states * mask

        outputs = []
        state = torch.zeros(
            B, D, D, device=hidden_states.device, dtype=hidden_states.dtype
        )
        for t in range(N):
            xt = hidden_states[:, t, :]  # (B, D)
            state = state + (xt.unsqueeze(-1) @ xt.unsqueeze(1))
            out_t = (state @ xt.unsqueeze(-1)).squeeze(-1)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1)  # (B, N, D)
        # Factorized low-rank matrix multiplication: (out @ L) @ R.T
        out = (out @ self.L) @ self.R.t()
        if mask is not None:
            out = out * mask
        return out, None
