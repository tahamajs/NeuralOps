"""
NLI Softmax approximation – a fast, quantisation‑friendly softmax.
Uses standard forward softmax, simplified sigmoid‑based backward.
"""

import torch
import torch.nn.functional as F


class NLISoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Standard softmax forward (exact)
        y = F.softmax(x, dim=-1)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Simplified gradient: sigmoid(x) * (1 - sigmoid(x))
        sig = torch.sigmoid(x)
        grad_input = grad_output * sig * (1 - sig)
        return grad_input


def nli_softmax(x):
    """Apply NLI softmax approximation (standard forward, simplified backward)."""
    return NLISoftmaxFunction.apply(x)
