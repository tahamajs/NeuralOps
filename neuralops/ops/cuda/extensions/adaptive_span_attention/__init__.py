"""
CUDA native extension package for adaptive_span_attention.
"""

from __future__ import annotations
from neuralops.core.errors import IntegrationNotImplementedError

try:
    from ._C import forward as cuda_forward
except ImportError as _err:
    _underlying_err = str(_err)
    def cuda_forward(*args, **kwargs):
        raise IntegrationNotImplementedError(
            f"adaptive_span_attention CUDA extension not compiled. Run `python setup.py build_ext --inplace` "
            f"in neuralops/ops/cuda/extensions/adaptive_span_attention/. Underlying error: {_underlying_err}"
        )

__all__ = ["cuda_forward"]
