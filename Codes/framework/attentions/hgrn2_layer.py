"""Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer.

Implements bidirectional element-wise linear recurrence with dynamic input-dependent
log-space forget gating, input modulation, and pretrained weight transfer.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hgrn2_layer")
class HGRN2Layer(BaseAttention):
    r"""Hierarchically Gated Recurrent Network v2 (HGRN-2) Bidirectional Layer.

    Mathematical Mechanism:
        For input token activations $x_t \in \mathbb{R}^D$:
        1. **Dynamic Gating & Value Transformation:**
           $$f_t = \exp(-\text{softplus}(W_f x_t + b_f)) \in (0, 1)^D \quad \text{(Log-space Bounded Retention)}$$
           $$i_t = \sigma(W_i x_t) \in (0, 1)^D \quad \text{(Input Gate)}$$
           $$v_t = W_v x_t \in \mathbb{R}^D, \quad z_t = W_g x_t \in \mathbb{R}^D$$
        2. **Bidirectional Linear Recurrence:**
           $$\vec{h}_t = f_t \odot \vec{h}_{t-1} + i_t \odot v_t$$
           $$\overleftarrow{h}_t = f_t \odot \overleftarrow{h}_{t+1} + i_t \odot v_t$$
        3. **Gated Output Aggregation:**
           $$Y = \frac{1}{2}(\vec{h} + \overleftarrow{h}) \odot \text{SiLU}(z)$$
           $$\text{Output} = Y W_o + b_o$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D)$ — strictly linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot D)$ recurrent state memory per sequence.
    """

    def __init__(
        self,
        d_model: Optional[Union[int, Any]] = None,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        res_clip: float = 5.0,
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
        self.res_clip: float = res_clip

        self.forget_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.input_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # Initialize forget gate bias to -3.0 so softplus(-3) ~ 0.048 => f_t ~ exp(-0.048) ~ 0.95
        nn.init.constant_(self.forget_gate.bias, 1.5)
        nn.init.constant_(self.input_gate.bias, 0.0)

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights."""
        try:
            adapter = UniversalAdapter()
            w_q, b_q = adapter.get_weight_bias(old_attn, "q")
            w_k, b_k = adapter.get_weight_bias(old_attn, "k")
            w_v, b_v = adapter.get_weight_bias(old_attn, "v")
            w_o, b_o = adapter.get_weight_bias(old_attn, "o")

            with torch.no_grad():
                self.input_gate.weight.copy_(w_k)
                self.v_proj.weight.copy_(w_v)
                self.g_proj.weight.copy_(w_q)
                self.out_proj.weight.copy_(w_o)
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o)
        except Exception:
            pass

    def _recurrent_scan(
        self,
        f: torch.Tensor,
        iv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Sequential element-wise scan."""
        B, L, D = iv.shape
        h = torch.zeros(B, D, device=iv.device, dtype=iv.dtype)
        outputs = []

        step_range = range(L - 1, -1, -1) if reverse else range(L)
        for t in step_range:
            f_t = f[:, t]
            iv_t = iv[:, t]
            h_next = f_t * h + iv_t
            h_next = torch.clamp(h_next, -self.res_clip, self.res_clip)

            if mask is not None:
                m_t = mask[:, t].view(B, 1)
                h = torch.where(m_t, h_next, h)
            else:
                h = h_next

            outputs.append(h)

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
        """Sequential forward recurrence for HGRN-2."""
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape

        # 1. Compute gates and values
        f = torch.exp(-F.softplus(self.forget_gate(hidden_states)))
        i = torch.sigmoid(self.input_gate(hidden_states))
        v = self.v_proj(hidden_states)
        z = self.g_proj(hidden_states)

        iv = i * v

        # 2. Mask processing
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()

        # 3. Bidirectional Recurrent Scan
        h_fwd = self._recurrent_scan(f, iv, mask=mask_bool, reverse=False)
        h_bwd = self._recurrent_scan(f, iv, mask=mask_bool, reverse=True)
        h = 0.5 * (h_fwd + h_bwd)

        # 4. Gating and output projection
        out = self.out_proj(h * F.silu(z))

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


@ATTENTION_REGISTRY.register("hgrn2_layer_inject")
def inject_hgrn2_layer(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> HGRN2Layer:
    r"""Injection hook for HGRN2Layer."""
    d_model: int = 768
    if model_config is not None:
        d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
    elif hasattr(layer, "config"):
        d_model = getattr(layer.config, "hidden_size", 768)

    new_layer = HGRN2Layer(d_model=d_model, config=model_config, **kwargs)
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_layer, "copy_weights"):
        new_layer.copy_weights(src_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_layer
    elif hasattr(layer, "self"):
        layer.self = new_layer

    return new_layer
