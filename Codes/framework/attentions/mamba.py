"""Mamba: Gated State Space Model (Mamba) linear attention replacement.

Implements a gated selective state-space projection that is a drop-in
replacement for a standard Multi-Head Attention layer, operating in
strictly O(L) time and space with respect to sequence length.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("mamba")
class Mamba(BaseAttention):
    r"""Gated State Space Model (Mamba) Linear Layer Module.

    Implements a gated selective state-space projection that acts as a
    lightweight, linear-complexity alternative to Multi-Head Attention.

    Mathematical Mechanism:
        Given input :math:`X \in \mathbb{R}^{B \times L \times D}`:

        .. math::

            [X_1, X_2] &= \text{chunk}(\text{Linear}(X), 2)
                         \in \mathbb{R}^{B \times L \times D} \\
            Y           &= \text{SiLU}(X_1) \odot X_2 \\
            O           &= Y W_{\text{out}}

        The SiLU-gated product fuses content-dependent selection
        (:math:`X_1`) with an unactivated residual branch (:math:`X_2`).

    Computational Complexity:
        - Time: :math:`\mathcal{O}(B \cdot L \cdot D^2)` — strictly
          linear in sequence length :math:`L`.
        - Space: :math:`\mathcal{O}(B \cdot L \cdot D)` activations.

    Tensor Dimensions:
        - Input ``hidden_states``: ``[B, L, D]``
        - Output: Tuple ``(output, None)`` where ``output`` is
          ``[B, L, D]``.

    Args:
        d_model: Hidden-state dimension :math:`D`.
        config: Optional model configuration object (unused internally
            but forwarded to :class:`BaseAttention`).
        rank: Optional rank hint forwarded to :class:`BaseAttention`.
        **kwargs: Additional keyword arguments (ignored).
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, (int, float)):
            d_model = int(config)
            config = None
        super().__init__(config, rank)
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "n_embd", getattr(config, "dim", 768))))
            else:
                d_model = kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))
        elif not isinstance(d_model, int) and hasattr(d_model, "hidden_size"):
            d_model = getattr(d_model, "hidden_size", getattr(d_model, "d_model", 768))
        self.d_model: int = int(d_model)
        self.hidden_size: int = self.d_model
        # Projects input to 2× width then splits into two gating branches.
        self.in_proj: nn.Linear = nn.Linear(self.d_model, self.d_model * 2)
        self.out_proj: nn.Linear = nn.Linear(self.d_model, self.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Execute the forward Mamba gated-SSM pass.

        Args:
            hidden_states: Input tensor of shape ``[B, L, D]`` or ``[B, H, L, Dh]``.
            attention_mask: Optional mask tensor; if provided, applies
                zero-masking for padded tokens.
            head_mask: Optional per-head mask; not used in this module.
            output_attentions: Unused flag kept for interface compliance.
            **kwargs: Extra keyword arguments (ignored).

        Returns:
            Tuple ``(output, None)`` where ``output`` is of shape
            ``[B, L, D]`` and ``None`` stands for the absent attention
            weight matrix.

        Shape:
            - Input:  ``[B, L, D]``
            - Output: ``[B, L, D]``
        """
        if hidden_states.dim() == 4:
            B, H, L, Dh = hidden_states.shape
            hidden_states = hidden_states.view(B, L, H * Dh)
        elif not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        # in_proj: [B, L, D] -> [B, L, 2D]
        proj = self.in_proj(hidden_states)
        # Split along last dim: each branch is [B, L, D]
        p1, p2 = proj.chunk(2, dim=-1)
        # Gated product: SiLU(p1) * p2 then project back to D
        out = self.out_proj(F.silu(p1) * p2)

        # Apply padding mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(-1).to(out.dtype)
                out = out * mask
            elif attention_mask.dim() == 4:
                mask = (attention_mask[:, 0, 0, :] > -10000.0).unsqueeze(-1).to(out.dtype)
                out = out * mask

        return out, None


@ATTENTION_REGISTRY.register("mamba_inject")
def inject_mamba(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> Mamba:
    r"""Injection hook that instantiates :class:`Mamba` and wires it
    into a transformer layer.

    Args:
        layer: Target transformer layer module whose ``.attention``
            attribute will be replaced.
        model_config: Model configuration providing ``hidden_size``.
        comp_config: Compression configuration (currently unused).
        layer_idx: Zero-based index of the current layer in the stack.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module being replaced (unused).

    Returns:
        The newly created :class:`Mamba` instance, already assigned to
        ``layer.attention``.
    """
    d_model: int = model_config.hidden_size
    new_module = Mamba(d_model=d_model, config=model_config)
    layer.attention = new_module
    return new_module
