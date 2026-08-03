r"""
UPI-Stuffed Mamba: Uncertainty-Penalized Interpolated Mamba SSM Block.

Extends the standard Mamba selective state-space recurrence with a learnable
exponential decay penalty $\gamma$ that interpolates between full associative memory
and fading memory, enabling uncertainty-calibrated context forgetting.

Mathematical Mechanism:
    Given input $X \in \mathbb{R}^{B \times L \times D}$:
    1. **Input Projection & Causal Convolution:**
       $$[X_{\text{ssm}}, Z] = X W_{\text{in}} \in \mathbb{R}^{B \times L \times 2 D_{\text{inner}}}$$
       $$\widetilde{X}_{\text{ssm}} = \text{CausalConv1D}(X_{\text{ssm}}) \in \mathbb{R}^{B \times L \times D_{\text{inner}}}$$
    2. **Uncertainty-Penalized SSM Recurrence:**
       $$\Delta t = \text{softplus}(\widetilde{X}_{\text{ssm}} W_{\Delta t}), \quad \bar{A} = \exp(\Delta t \odot A)$$
       $$h_t = ((1 - \gamma) \bar{A} \odot h_{t-1}) + \widetilde{x}_t \otimes \Delta t_t \in \mathbb{R}^{B \times D \times d_{\text{state}}}$$
       $$y_t = (h_t \odot D)\mathbf{1}_{d_{\text{state}}} \in \mathbb{R}^{B \times D}$$
       where $\gamma \in \mathbb{R}$ is a learnable decay penalty (initialized to 0).
    3. **Multiplicative Gating & Output:**
       $$O = (y \odot \text{SiLU}(Z)) W_{\text{out}} \in \mathbb{R}^{B \times L \times D}$$

Computational Complexity:
    - 1D Convolution: $\mathcal{O}(B \cdot L \cdot D_{\text{inner}} \cdot d_{\text{conv}})$.
    - SSM Recurrence: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$.
    - Total per-layer: $\mathcal{O}(L \cdot D \cdot (d_{\text{conv}} + d_{\text{state}}))$, linear $\mathcal{O}(L)$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


class UPIMambaBlock(nn.Module):
    r"""Uncertainty-Penalized Interpolated Mamba block with learnable decay.

    Args:
        dim: Input/output hidden dimension $D$.
        d_state: SSM state expansion factor $d_{\text{state}}$.
        d_conv: Causal depthwise convolution kernel width.
        expand: Inner dimension expansion multiplier.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.d_state: int = d_state
        self.expand: int = expand
        inner_dim: int = dim * expand
        self.in_proj: nn.Linear = nn.Linear(dim, inner_dim * 2)
        self.conv1d: nn.Conv1d = nn.Conv1d(
            inner_dim, inner_dim, d_conv, groups=inner_dim, padding=d_conv - 1
        )
        self.A_log: nn.Parameter = nn.Parameter(
            torch.log(torch.rand(dim, d_state) * 0.1)
        )
        self.D: nn.Parameter = nn.Parameter(torch.ones(dim))
        self.dt_proj: nn.Linear = nn.Linear(inner_dim, dim)
        self.gamma: nn.Parameter = nn.Parameter(torch.zeros(1))
        self.out_proj: nn.Linear = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute uncertainty-penalized SSM recurrence.

        Args:
            x: Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask (unused in SSM).
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of:
                - output: SSM output tensor `[Batch, SeqLen, HiddenDim]`.
                - None: No explicit attention weights.
        """
        B, L, D = x.shape
        x_proj = self.in_proj(x)
        x_ssm, z = x_proj.chunk(2, dim=-1)
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[..., :L].transpose(1, 2)
        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log)
        x_ssm_trunc = x_ssm[:, :, : self.dim]
        h = torch.zeros(B, self.dim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            xt = x_ssm_trunc[:, t]
            h = h * torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0)) + (
                xt.unsqueeze(-1) * dt[:, t].unsqueeze(-1)
            )
            h = h - self.gamma * h  # uncertainty-penalized decay
            out_t = (h * self.D.unsqueeze(-1)).sum(dim=-1)
            outputs.append(out_t)
        out_ssm = torch.stack(outputs, dim=1)
        out_ssm_full = F.pad(out_ssm, (0, x_ssm.size(-1) - self.dim))
        out = out_ssm_full * F.silu(z)
        return self.out_proj(out), None


@ATTENTION_REGISTRY.register("upi_stuffed_mamba")
def inject_upi_mamba(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> UPIMambaBlock:
    r"""Bridge helper injecting UPIMambaBlock into a transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration with `hidden_size`.
        comp_config: Compression configuration.
        layer_idx: Current layer index.
        total_layers: Total number of layers.
        old_attn: Existing attention module (unused).

    Returns:
        Instantiated :class:`UPIMambaBlock`.
    """
    dim: int = model_config.hidden_size
    new_block = UPIMambaBlock(dim=dim)
    layer.attention = new_block
    return new_block
