"""
LIFTR – Linearized Function Trick Compressor.
Registered in COMPRESSOR_REGISTRY as 'liftr'.

Mathematical Formulation:
1. Activation Function Linearization:
   - $\\operatorname{ReLU}(x) \\to \\operatorname{Identity}(x)$
   - $\\operatorname{GELU}(x) \\approx 0.8 \\cdot x$
   - $\\operatorname{SiLU}(x) \\approx 0.8 \\cdot x$
2. Quadratic Softmax Attention Linearization:
   $$\\operatorname{Attn}(Q, K, V) = \\operatorname{Softmax}\\left(\\frac{Q K^\\top}{\\sqrt{d_k}}\\right) V \\to \\frac{1}{N} \\sum_{i=1}^N V_i$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class _LinearizedGELU(nn.Module):
    """Linear approximation of GELU / SiLU: f(x) ≈ 0.8 * x."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.8 * x


class _LinearizedAttentionWrapper(nn.Module):
    """
    Wraps attention modules to replace quadratic softmax attention with sequence pooling.
    Preserves all original layer dimensions and return signature (context, weights_or_None).
    """

    def __init__(self, original_attn: nn.Module) -> None:
        super().__init__()
        self.original_attn = original_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # Check if original module is DistilBERT-style or custom
        if hasattr(self.original_attn, "q_lin") and hasattr(self.original_attn, "v_lin") and hasattr(self.original_attn, "out_lin"):
            # DistilBERT MultiHeadSelfAttention
            v = self.original_attn.v_lin(hidden_states)
            context = v.mean(dim=1, keepdim=True).expand_as(v)
            output = self.original_attn.out_lin(context)
            return (output, None)
        elif hasattr(self.original_attn, "query") and hasattr(self.original_attn, "value"):
            # BERT BertSelfAttention
            v = self.original_attn.value(hidden_states)
            context = v.mean(dim=1, keepdim=True).expand_as(v)
            out_layer = getattr(self.original_attn, "dense", None)
            output = out_layer(context) if out_layer is not None else context
            return (output, None)
        else:
            # Fallback to calling original attention or identity
            try:
                res = self.original_attn(hidden_states, attention_mask=attention_mask, **kwargs)
                return res if isinstance(res, tuple) else (res, None)
            except Exception:
                return (hidden_states, None)


class LIFTRCompressor(BaseCompressor):
    """
    LIFTR Compressor: Linearizes activations and attention mechanisms for ultra-low latency.
    """

    def __init__(
        self,
        model: nn.Module,
        replace_attention: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="liftr", **kwargs)
        self.replace_attention = replace_attention
        self.linearized_activations_count = 0
        self.linearized_attentions_count = 0
        self.stats: Dict[str, Any] = {}

    def _linearize_module(self, module: nn.Module) -> None:
        """Recursively replaces non-linear activations and attention blocks."""
        for name, child in list(module.named_children()):
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.Identity())
                self.linearized_activations_count += 1
            elif isinstance(child, (nn.GELU, nn.SiLU)):
                setattr(module, name, _LinearizedGELU())
                self.linearized_activations_count += 1
            elif self.replace_attention and ("attention" in name.lower() or "self_attn" in name.lower() or "attn" in name.lower()):
                # If child contains sub-projections, wrap it
                if not isinstance(child, _LinearizedAttentionWrapper):
                    setattr(module, name, _LinearizedAttentionWrapper(child))
                    self.linearized_attentions_count += 1
            else:
                self._linearize_module(child)

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Applies linearization transformations to the entire model."""
        self._linearize_module(self.model)
        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "linearized_activations": self.linearized_activations_count,
            "linearized_attentions": self.linearized_attentions_count,
            "replace_attention": self.replace_attention,
            "method": "liftr_linearization",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("liftr")
@COMPRESSOR_REGISTRY.register("liftr_compressor")
def liftr_compressor_entry(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for LIFTR compressor."""
    replace_attention = getattr(comp_config, "replace_attention", True) if comp_config else kwargs.get("replace_attention", True)
    compressor = LIFTRCompressor(model, replace_attention=replace_attention, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


liftr_factory = liftr_compressor_entry
