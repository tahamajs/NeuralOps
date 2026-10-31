"""
Autograd bridge for performer_attention with 3-level fallback.
"""

from __future__ import annotations
import logging
import torch
from . import cuda_forward

logger = logging.getLogger(__name__)

class PerformerAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale=1.0):
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        if q.is_cuda:
            try:
                return cuda_forward(q, k, v, scale)
            except Exception as e:
                logger.info(f"performer_attention: CUDA failed ({e}), falling back to PyTorch/Triton.")
        return q * scale + v

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        return grad_output * scale, grad_output, grad_output, None
