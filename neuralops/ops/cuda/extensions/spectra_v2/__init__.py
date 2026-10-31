"""
CUDA native extension package for spectra_v2.
"""

from __future__ import annotations
from neuralops.core.errors import IntegrationNotImplementedError

try:
    from ._C import forward as cuda_forward
except ImportError as _err:
    _underlying_err = str(_err)
    def cuda_forward(*args, **kwargs):
        raise IntegrationNotImplementedError(
            f"spectra_v2 CUDA extension not compiled. Run `python setup.py build_ext --inplace` "
            f"in neuralops/ops/cuda/extensions/spectra_v2/. Underlying error: {_underlying_err}"
        )

__all__ = ["cuda_forward"]
