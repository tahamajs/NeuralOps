"""
LIFTR – Linearized Function Trick Compressor.
Replaces non‑linear activations and softmax attention with linear approximations,
reducing the entire model to a single linear function for extreme inference speed.
"""

from typing import Any, Dict, Tuple

import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class _LinearizedGELU(nn.Module):
    """Linear approximation of GELU: 0.5 * x + 0.5 * sqrt(2/π) * x ≈ 0.8 * x"""

    def forward(self, x) -> Any:
        """Forward pass."""
        return 0.8 * x


class _LinearizedAttention(nn.Module):
    """
    Linearized attention: replaces softmax with average over sequence dimension.
    Preserves input/output shapes for DistilBERT compatibility.
    """

    def __init__(self, original_attn):
        super().__init__()
        # Keep references to the original projections (they are linear)
        self.q_lin = original_attn.q_lin
        self.k_lin = original_attn.k_lin
        self.v_lin = original_attn.v_lin
        self.out_lin = original_attn.out_lin

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Compute Q, K, V (linear projections)
        """Forward pass."""
        self.q_lin(hidden_states)
        self.k_lin(hidden_states)
        v = self.v_lin(hidden_states)

        # Linearized attention: average over sequence length
        # shape: (B, N, D) -> mean over N -> (B, 1, D) -> expand to (B, N, D)
        context = v.mean(dim=1, keepdim=True).expand_as(v)

        # Output projection
        output = self.out_lin(context)
        return output, None  # no attention weights


class LIFTRCompressor(BaseCompressor):
    """
    Linearizes the entire model: ReLU → Identity, GELU → 0.8*x,
    softmax attention → average pooling. This makes the model extremely fast
    at the cost of accuracy. Ideal for quick baselines or edge deployment.

    Args:
        model: PyTorch model (DistilBERT or BERT).
        replace_attention: if True, also linearize attention (default True).
    """

    def __init__(self, model: nn.Module, replace_attention: bool = True, **kwargs):
        super().__init__(model, name="liftr", **kwargs)
        self.replace_attention = replace_attention

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply linearization to the model and return compressed model + stats."""
        self._linearize_module(self.model)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _linearize_module(self, module: nn.Module) -> None:
        """Recursively replace non‑linear layers and attention modules."""
        for name, child in list(module.named_children()):
            # Replace activation functions
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.Identity())
            elif isinstance(child, nn.GELU):
                setattr(module, name, _LinearizedGELU())
            elif isinstance(child, nn.SiLU):
                setattr(module, name, _LinearizedGELU())
            # Replace attention: DistilBERT attention has attribute 'attention'
            elif self.replace_attention and name == "attention":
                # For DistilBERT, the child is the attention module
                # We replace it with our linearized wrapper
                setattr(module, name, _LinearizedAttention(child))
            # Recurse into children
            else:
                self._linearize_module(child)


# ----------------------------------------------------------------------
# Registry entry point – returns the compressed model (nn.Module)
# ----------------------------------------------------------------------
@COMPRESSOR_REGISTRY.register("liftr")
def liftr_compressor_entry(model: nn.Module, comp_config) -> nn.Module:
    """
    Factory function called by the framework.
    Expects comp_config to have attribute `replace_attention` (default True).
    """
    replace_attention = getattr(comp_config, "replace_attention", True)
    compressor = LIFTRCompressor(model, replace_attention=replace_attention)
    compressed_model, _ = compressor.compress()
    return compressed_model
