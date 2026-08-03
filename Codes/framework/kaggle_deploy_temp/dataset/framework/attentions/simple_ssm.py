r"""
Simple State Space Model (SimpleSSM).

Lightweight selective state space recurrence block featuring causal depthwise 1D convolution,
input-dependent discretization timescales ($\Delta t$), diagonal state matrix transitions ($A$), and multiplicative gating.

Mathematical Mechanism:
    Given input sequence $X \in \mathbb{R}^{B \times L \times D}$:
    1. **Input Projection & Causal Convolution:**
       $$[X_{\text{ssm}}, Z] = X W_{\text{in}} \in \mathbb{R}^{B \times L \times 2 D_{\text{inner}}}$$
       $$\widetilde{X}_{\text{ssm}} = \operatorname{Conv1D}_{\text{causal}}(X_{\text{ssm}}) \in \mathbb{R}^{B \times L \times D_{\text{inner}}}$$
    2. **SSM Discretization & State Recurrence:**
       $$\Delta t = \operatorname{softplus}(\widetilde{X}_{\text{ssm}} W_{\Delta t}) \in \mathbb{R}^{B \times L \times D}$$
       $$\bar{A} = \exp(\Delta t \odot A) \in \mathbb{R}^{B \times L \times D \times d_{\text{state}}}, \quad \bar{B} = \Delta t \odot \widetilde{X}_{\text{ssm}}[:, :, :D]$$
       $$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \otimes \mathbf{1}_{d_{\text{state}}}$$
       $$y_t = \sum_{s=1}^{d_{\text{state}}} (h_{t, :, s} \cdot D_s)$$
    3. **Multiplicative Gating & Projection:**
       $$O = (y \odot \operatorname{SiLU}(Z[:, :, :D])) W_{\text{out}} \in \mathbb{R}^{B \times L \times D}$$

Computational Complexity:
    - 1D Causal Convolution: $\mathcal{O}(B \cdot L \cdot D_{\text{inner}} \cdot d_{\text{conv}})$.
    - State Space Recurrence: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$.
    - Total per-layer complexity: $\mathcal{O}(L \cdot D \cdot (d_{\text{conv}} + d_{\text{state}}))$, strictly linear in sequence length $L$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("simple_ssm")
class SimpleSSM(nn.Module):
    r"""Simplified selective State Space Model layer with multiplicative gating.

    Args:
        d_model: Input and output model dimension $D$.
        d_state: State space expansion dimension $d_{\text{state}}$.
        d_conv: Causal convolution filter kernel size.
        expand: Hidden dimension expansion multiplier.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.d_state: int = d_state
        self.d_conv: int = d_conv
        inner_dim: int = d_model * expand

        self.in_proj = nn.Linear(d_model, inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=d_conv,
            groups=inner_dim,
            padding=d_conv - 1,
        )
        self.dt_proj = nn.Linear(inner_dim, d_model, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute SSM recurrence over sequence tokens.

        Args:
            hidden_states: Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask `[Batch, SeqLen]` or broadcastable.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Layer output tensor of shape `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: None.
        """
        x = hidden_states
        batch, seq_len, d_model = x.shape

        x_proj = self.in_proj(x)
        x_ssm, z = x_proj.chunk(2, dim=-1)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(x.device)
                valid_expanded = valid.unsqueeze(-1).to(x.dtype)
                x_ssm = x_ssm * valid_expanded
                z = z * valid_expanded

        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[..., :seq_len]
        x_ssm = x_ssm.transpose(1, 2)

        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log)

        h = torch.zeros(batch, d_model, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            xt = x_ssm[:, t, :]
            xt_slice = xt[:, :d_model]
            h = h * torch.exp(dt[:, t].unsqueeze(-1) * A) + (
                xt_slice.unsqueeze(-1) * dt[:, t].unsqueeze(-1)
            )
            out_t = (h * self.D.unsqueeze(-1)).sum(dim=-1)
            outputs.append(out_t)
        out_ssm = torch.stack(outputs, dim=1)

        gate = self.act(z[:, :, :d_model])
        out = out_ssm * gate
        out = self.out_proj(out)
        return out, None
