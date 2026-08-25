"""Simple State Space Model (SimpleSSM).

Lightweight bidirectional selective state space recurrence block featuring symmetric depthwise 1D convolution,
input-dependent discretization timescales (Delta t), diagonal state transitions (A), multiplicative gating,
and pretrained weight transfer.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("simple_ssm")
class SimpleSSM(BaseAttention):
    r"""Bidirectional Simple State Space Model (SimpleSSM) Module.

    Mathematical Mechanism:
        Given input sequence $X \in \mathbb{R}^{B \times L \times D}$:
        1. **Input Projection & Symmetric Convolution:**
           $$[X_{\text{ssm}}, Z] = X W_{\text{in}} \in \mathbb{R}^{B \times L \times 2 D}$$
           $$\widetilde{X}_{\text{ssm}} = \text{Conv1D}(X_{\text{ssm}}) \in \mathbb{R}^{B \times L \times D}$$
        2. **SSM Discretization & Bidirectional State Recurrence:**
           $$\Delta t = \text{softplus}(\widetilde{X}_{\text{ssm}} W_{\Delta t} + b_{\Delta t}) \in \mathbb{R}^{B \times L \times D}$$
           $$\bar{A} = \exp(-\exp(A_{\text{log}}) \odot \Delta t) \in (0, 1)^{B \times L \times D \times d_{\text{state}}}$$
           $$\vec{h}_t = \bar{A}_t \odot \vec{h}_{t-1} + (\Delta t_t \odot \widetilde{x}_t) \otimes \mathbf{1}_{d_{\text{state}}}$$
           $$\overleftarrow{h}_t = \bar{A}_t \odot \overleftarrow{h}_{t+1} + (\Delta t_t \odot \widetilde{x}_t) \otimes \mathbf{1}_{d_{\text{state}}}$$
           $$y_t = \sum_{s=1}^{d_{\text{state}}} \left( \frac{1}{2}(\vec{h}_{t, :, s} + \overleftarrow{h}_{t, :, s}) \cdot D_s \right)$$
        3. **Multiplicative SiLU Gating & Output:**
           $$\text{Output} = (y \odot \text{SiLU}(Z)) W_{\text{out}}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$ linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot D \cdot d_{\text{state}})$ recurrent memory footprint.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.d_model: int = int(d_model)
        self.d_state: int = d_state
        self.d_conv: int = d_conv
        self.expand: int = expand
        self.inner_dim: int = self.d_model * expand

        self.in_proj = nn.Linear(self.d_model, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
        )
        self.dt_proj = nn.Linear(self.inner_dim, self.d_model, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_model, 1)))
        self.D = nn.Parameter(torch.ones(self.d_model))
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        nn.init.constant_(self.dt_proj.bias, 0.1)

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights."""
        try:
            adapter = UniversalAdapter()
            w_q, b_q = adapter.get_weight_bias(old_attn, "q")
            w_k, b_k = adapter.get_weight_bias(old_attn, "k")
            w_v, b_v = adapter.get_weight_bias(old_attn, "v")
            w_o, b_o = adapter.get_weight_bias(old_attn, "o")

            with torch.no_grad():
                if self.in_proj.weight.shape == torch.Size([self.inner_dim * 2, self.d_model]):
                    self.in_proj.weight[: self.inner_dim].copy_(w_v[: self.inner_dim])
                    self.in_proj.weight[self.inner_dim :].copy_(w_q[: self.inner_dim])

                self.out_proj.weight.copy_(w_o)
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass

    def _ssm_scan(
        self,
        x_ssm: torch.Tensor,
        dt: torch.Tensor,
        A_log: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Bidirectional diagonal state scan."""
        B, L, D = x_ssm.shape
        d_state = self.d_state
        A = torch.exp(A_log)  # [D, d_state]

        h = torch.zeros(B, D, d_state, device=x_ssm.device, dtype=x_ssm.dtype)
        outputs = []

        step_range = range(L - 1, -1, -1) if reverse else range(L)
        for t in step_range:
            xt = x_ssm[:, t, :]                                  # [B, D]
            dt_t = dt[:, t, :].unsqueeze(-1)                     # [B, D, 1]
            dA = torch.exp(-dt_t * A.unsqueeze(0))                # [B, D, d_state]
            dB_x = (dt_t * xt.unsqueeze(-1)).repeat(1, 1, d_state) # [B, D, d_state]

            h_next = dA * h + dB_x

            if mask is not None:
                m_t = mask[:, t].view(B, 1, 1)
                h = torch.where(m_t, h_next, h)
            else:
                h = h_next

            out_t = (h * self.D.unsqueeze(-1)).sum(dim=-1)
            outputs.append(out_t)

        if reverse:
            outputs.reverse()

        return torch.stack(outputs, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute SSM recurrence over sequence tokens."""
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape

        x_proj = self.in_proj(hidden_states)
        x_ssm, z = x_proj.chunk(2, dim=-1)

        # 1D conv with exact sequence length retention
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[..., :L].transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        dt = F.softplus(self.dt_proj(x_ssm))

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()

        y_fwd = self._ssm_scan(x_ssm, dt, self.A_log, mask=mask_bool, reverse=False)
        y_bwd = self._ssm_scan(x_ssm, dt, self.A_log, mask=mask_bool, reverse=True)
        y = 0.5 * (y_fwd + y_bwd)

        out = self.out_proj(y * F.silu(z))

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


@ATTENTION_REGISTRY.register("simple_ssm_inject")
def inject_simple_ssm(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> SimpleSSM:
    r"""Injection hook for SimpleSSM."""
    d_model = 768
    if model_config is not None:
        d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
    elif hasattr(layer, "config"):
        d_model = getattr(layer.config, "hidden_size", 768)

    new_block = SimpleSSM(config=model_config, d_model=d_model, **kwargs)
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_block, "copy_weights"):
        new_block.copy_weights(src_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_block
    elif hasattr(layer, "self"):
        layer.self = new_block

    return new_block
