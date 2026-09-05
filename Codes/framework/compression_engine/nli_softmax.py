"""
NLI Softmax approximation – a fast, quantisation‑friendly softmax.
Uses standard forward softmax, simplified sigmoid‑based backward.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class NLISoftmaxFunction(torch.autograd.Function):
    """Custom autograd function for NLI Softmax approximation.
    
    This function implements a fast, quantization-friendly softmax
    that uses the standard forward softmax but a simplified sigmoid-based backward.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Forward pass for NLI Softmax.
        
        Args:
            ctx: The autograd context used to save variables for backward.
            x: Input tensor.
            dim: Dimension along which softmax is applied.
            
        Returns:
            The standard softmax of the input tensor.
        """
        y = F.softmax(x, dim=dim)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        """Backward pass for NLI Softmax.
        
        Args:
            ctx: The autograd context containing saved variables.
            grad_output: The gradient of the loss with respect to the output.
            
        Returns:
            The gradient of the loss with respect to the input tensor.
        """
        (x,) = ctx.saved_tensors
        # Simplified gradient: sigmoid(x) * (1 - sigmoid(x)), numerically stable across FP16 and FP32
        sig = torch.sigmoid(x.float()).to(dtype=grad_output.dtype)
        grad_input = grad_output * sig * (1.0 - sig)
        return grad_input, None


def nli_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply NLI softmax approximation (standard forward, simplified backward).
    
    Args:
        x: Input tensor to apply softmax on.
        dim: Dimension along which to apply softmax.
        
    Returns:
        Tensor after applying the NLI softmax approximation.
    """
    return NLISoftmaxFunction.apply(x, dim)


class NLISoftmax(nn.Module):
    """Module wrapper for NLI Softmax approximation."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nli_softmax(x, dim=self.dim)


from framework.core.model_utils import replace_module


class NLISoftmaxCompressor(BaseCompressor):
    """
    Compressor that replaces standard softmax layers with quantization-friendly NLI softmax.
    """

    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        super().__init__(model, name="nli_softmax", **kwargs)
        self.replaced_layers = []

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace nn.Softmax modules with NLISoftmax."""
        self.replaced_layers = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Softmax):
                dim = getattr(module, "dim", -1)
                new_mod = NLISoftmax(dim=dim)
                replace_module(self.model, name, new_mod)
                self.replaced_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["softmax_modules_replaced"] = len(self.replaced_layers)
        stats["replaced_layers"] = self.replaced_layers
        return self.model, stats


@COMPRESSOR_REGISTRY.register("nli_softmax")
@COMPRESSOR_REGISTRY.register("nli_softmax_compressor")
def nli_softmax_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """NLI Softmax compressor factory function."""
    compressor = NLISoftmaxCompressor(model, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model

