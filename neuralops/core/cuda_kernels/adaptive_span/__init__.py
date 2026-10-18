"""
Adaptive Span Attention Hardware Operators.

Author: Taha Majlesi
"""

from neuralops.core.cuda_kernels.adaptive_span.adaptive_span_autograd import (
    adaptive_span,
    fused_adaptive_span_penalty_pytorch,
)

__all__ = ["adaptive_span", "fused_adaptive_span_penalty_pytorch"]
