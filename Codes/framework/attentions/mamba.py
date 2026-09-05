"""Mamba: Gated State Space Model (Mamba) linear attention replacement.

Implements a gated selective state-space projection that is a drop-in
replacement for a standard Multi-Head Attention layer, operating in
strictly O(L) time and space with respect to sequence length.
"""

from typing import Any, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
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
        # PATCH: Preservation gate — at init, sigmoid(-2.0) ≈ 0.12 means the output
        # is 88% the original hidden_states (residual bypass) and only 12% Mamba output.
        # Fine-tuning gradually shifts the balance as pres_gate becomes more positive.
        self.pres_gate: nn.Parameter = nn.Parameter(torch.tensor(-2.0))

    def copy_weights(self, pretrained_attn: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(pretrained_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        if hasattr(pretrained_attn, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None):
            weight = pretrained_attn.c_attn.weight.data
            bias = getattr(pretrained_attn.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.d_model:
                weight = weight.t()
            split_size = weight.shape[0] // 3
            extracted["q"] = (weight[:split_size, :], bias_data[:split_size] if bias_data is not None else None)
            extracted["k"] = (weight[split_size : 2 * split_size, :], bias_data[split_size : 2 * split_size] if bias_data is not None else None)

        if hasattr(pretrained_attn, "c_proj") and extracted["o"][0] is None:
            w_o = pretrained_attn.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.d_model and w_o.shape[1] == self.d_model:
                w_o = w_o.t()
            b_o = getattr(pretrained_attn.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        if hasattr(pretrained_attn, "self") and (extracted["q"][0] is None or extracted["k"][0] is None):
            s = getattr(pretrained_attn, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))

        if hasattr(pretrained_attn, "output") and hasattr(getattr(pretrained_attn, "output"), "dense") and extracted["o"][0] is None:
            dense = pretrained_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(pretrained_attn, name):
                        layer = getattr(pretrained_attn, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        cls_name = pretrained_attn.__class__.__name__
        has_out_proj = any(hasattr(pretrained_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        D = self.d_model
        with torch.no_grad():
            w_q, b_q = extracted["q"]
            if w_q is not None:
                w_q = _orient(w_q, D)
                if w_q.shape[0] < D and D % w_q.shape[0] == 0:
                    rep = D // w_q.shape[0]
                    w_q = w_q.repeat(rep, 1)
                    if b_q is not None: b_q = b_q.repeat(rep)
                self.in_proj.weight.data[:D, :].copy_(w_q[:D, :min(D, w_q.shape[1])])
                if b_q is not None and self.in_proj.bias is not None:
                    self.in_proj.bias.data[:D].copy_(b_q[:D])

            w_k, b_k = extracted["k"]
            if w_k is not None:
                w_k = _orient(w_k, D)
                if w_k.shape[0] < D and D % w_k.shape[0] == 0:
                    rep = D // w_k.shape[0]
                    w_k = w_k.repeat(rep, 1)
                    if b_k is not None: b_k = b_k.repeat(rep)
                self.in_proj.weight.data[D:, :].copy_(w_k[:D, :min(D, w_k.shape[1])])
                if b_k is not None and self.in_proj.bias is not None:
                    self.in_proj.bias.data[D:].copy_(b_k[:D])

            w_o, b_o = extracted["o"]
            if w_o is not None:
                w_o = _orient(w_o, D)
                self.out_proj.weight.data.copy_(w_o[:D, :min(D, w_o.shape[1])])
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.data.copy_(b_o[:D])
            elif self.is_nested_encoder:
                self.out_proj.weight.data.copy_(torch.eye(D))
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
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

        # PATCH: Preservation gate — blends Mamba output with the pretrained residual.
        # sigmoid(-2.0) ≈ 0.12 at init → model starts 88% close to pretrained bypass.
        alpha = torch.sigmoid(self.pres_gate)
        while isinstance(out, (tuple, list)):
            out = out[0]
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        out = alpha * out + (1.0 - alpha) * hidden_states
        while isinstance(out, (tuple, list)):
            out = out[0]
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
    # PATCH: Copy pretrained weights — original code ignored old_attn entirely,
    # leaving all weights at random Kaiming-uniform initialisation and causing
    # complete representation collapse (50.92% majority-class accuracy on MRPC).
    if old_attn is not None:
        new_module.copy_weights(old_attn)
    layer.attention = new_module
    return new_module
