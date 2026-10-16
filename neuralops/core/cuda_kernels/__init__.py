"""NeuralOps CUDA C++ Kernel Package.

High-performance native CUDA C++ kernels augmenting and accelerating hot-paths
on modern NVIDIA architectures (Volta SM 7.0 through Blackwell SM 10.0).
Provides transparent 3-level fallback: CUDA C++ -> Triton JIT -> PyTorch Reference.
"""

from __future__ import annotations

from neuralops.core.cuda_kernels._bindings import load_cuda_module, get_cuda_arch_flags
from neuralops.core.cuda_kernels._fallback import dispatch
from neuralops.core.cuda_kernels._utils import (
    is_cuda_available,
    get_device_capability,
    check_cuda_error,
)

__all__ = [
    "load_cuda_module",
    "get_cuda_arch_flags",
    "dispatch",
    "is_cuda_available",
    "get_device_capability",
    "check_cuda_error",
]
