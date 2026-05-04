"""
LIFTR – Linearized Function Trick Compressor.
Replaces non‑linear activations and softmax attention with linear approximations,
reducing the entire model to a single linear function for extreme inference speed.
"""

import torch
import torch.nn as nn
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY


class _LinearizedGELU(nn.Module):
    """Linear approximation of GELU: 0.5 * x + 0.5 * sqrt(2/π) * x ≈ 0.8 * x"""
    def forward(self, x):
        return 0.8 * x


class _LinearizedSoftmaxAttention(nn.Module):
    """
    Replaces softmax attention with a simple linear projection.
    Actually just returns the mean of the values.
    """
    def forward(self, q, k, v):
        # shape: q (B,*,L,D), k, v similar
        return v.mean(dim=-2, keepdim=True).expand_as(v)


@COMPRESSOR_REGISTRY.register("liftr")
class LIFTRCompressor(BaseCompressor):
    """
    Linearizes the entire model: ReLU → Identity, GELU → 0.8*x,
    softmax attention → average pooling. This makes the model extremely fast
    at the cost of accuracy. Ideal for quick baselines or edge deployment.

    Args:
        model: PyTorch model.
        replace_attention: if True, also linearize attention (default True).
    """

    def __init__(self, model: nn.Module, replace_attention: bool = True, **kwargs):
        super().__init__(model, name="liftr", **kwargs)
        self.replace_attention = replace_attention

    def compress(self):
        self._linearize_module(self.model)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _linearize_module(self, module: nn.Module):
        for name, child in module.named_children():
            # Replace activations
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.Identity())
            elif isinstance(child, nn.GELU):
                setattr(module, name, _LinearizedGELU())
            elif isinstance(child, nn.SiLU):
                setattr(module, name, _LinearizedGELU())  # similar
            # Replace attention
            elif self.replace_attention and isinstance(child, nn.MultiheadAttention):
                # We replace the forward method by wrapping the module
                child.forward = self._linear_attention_forward
            # Recursively process children
            else:
                self._linearize_module(child)

    def _linear_attention_forward(self, query, key, value, **kwargs):
        """
        A purely linear 'attention' – just averages the values.
        Keeps the output shape consistent.
        """
        # query, key, value: (L, B, D) or (B, L, D) depending on batch_first
        # We'll just average over the sequence dimension.
        if query.dim() == 3:
            # batch_first=False: (L, B, D) -> average over L
            out = value.mean(dim=0, keepdim=True).expand_as(value)
        elif query.dim() == 4:
            # batch_first=True: (B, L, D) -> average over L
            out = value.mean(dim=1, keepdim=True).expand_as(value)
        else:
            out = value
        return out, None  # attention weights