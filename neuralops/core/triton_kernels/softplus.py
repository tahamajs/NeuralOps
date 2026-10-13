"""
Triton JIT and PyTorch Fallback implementation of Softplus activation.

Computes:
    softplus(x) = log(1 + exp(x))

Example:
    >>> import torch
    >>> from neuralops.core.triton_kernels.softplus import softplus_fn
    >>> x = torch.randn(2, 4)
    >>> out = softplus_fn(x)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from packaging import version

from neuralops.core.triton_kernels._common import HAS_TRITON, tl, triton

if HAS_TRITON and hasattr(triton, "__version__"):
    TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")
else:
    TRITON3 = False


if HAS_TRITON and TRITON3:

    @triton.jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)

elif HAS_TRITON:

    @triton.jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt))

else:
    def softplus(dt):
        return F.softplus(dt)


def _softplus_pytorch(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """Numerically stable PyTorch CPU/CUDA fallback for softplus."""
    return F.softplus(x, beta=beta, threshold=threshold)


def softplus_fn(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """
    Public dispatch function for softplus activation.
    
    Delegates to PyTorch fallback for tensor-level calls or Triton JIT for in-kernel evaluation.
    """
    return _softplus_pytorch(x, beta=beta, threshold=threshold)


softplus_fwd = softplus_fn
softplus_pytorch = _softplus_pytorch

__all__ = ["softplus", "softplus_fn", "_softplus_pytorch", "softplus_fwd", "softplus_pytorch"]
