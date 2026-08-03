"""
NLI Softmax approximation – a fast, quantisation‑friendly softmax.
Uses standard forward softmax, simplified sigmoid‑based backward.
"""

import torch
import torch.nn.functional as F
from typing import Any


class NLISoftmaxFunction(torch.autograd.Function):
    """Custom autograd function for NLI Softmax approximation.
    
    This function implements a fast, quantization-friendly softmax
    that uses the standard forward softmax but a simplified sigmoid-based backward.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for NLI Softmax.
        
        Args:
            ctx: The autograd context used to save variables for backward.
            x: Input tensor.
            
        Returns:
            The standard softmax of the input tensor.
        """
        # Standard softmax forward (exact)
        y = F.softmax(x, dim=-1)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass for NLI Softmax.
        
        Args:
            ctx: The autograd context containing saved variables.
            grad_output: The gradient of the loss with respect to the output.
            
        Returns:
            The gradient of the loss with respect to the input tensor.
        """
        (x,) = ctx.saved_tensors
        # Simplified gradient: sigmoid(x) * (1 - sigmoid(x))
        sig = torch.sigmoid(x)
        grad_input = grad_output * sig * (1 - sig)
        return grad_input


def nli_softmax(x: torch.Tensor) -> torch.Tensor:
    """Apply NLI softmax approximation (standard forward, simplified backward).
    
    Args:
        x: Input tensor to apply softmax on.
        
    Returns:
        Tensor after applying the NLI softmax approximation.
    """
    return NLISoftmaxFunction.apply(x)
