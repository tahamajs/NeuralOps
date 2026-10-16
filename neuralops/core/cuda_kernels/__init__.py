"""NeuralOps CUDA C++ Kernel Package.

High-performance native CUDA C++ kernels augmenting and accelerating hot-paths
on modern NVIDIA architectures (Volta SM 7.0 through Blackwell SM 10.0).
Provides transparent 3-level fallback: CUDA C++ -> Triton JIT -> PyTorch Reference.
"""

from __future__ import annotations

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch_3level
from neuralops.core.cuda_kernels._types import DataType, KernelLaunchConfig
from neuralops.core.cuda_kernels._utils import (
    check_cuda_compatibility,
    get_cuda_device_capability,
    is_cuda_cpp_available,
)

__all__ = [
    "load_cuda_module",
    "dispatch_3level",
    "DataType",
    "KernelLaunchConfig",
    "check_cuda_compatibility",
    "get_cuda_device_capability",
    "is_cuda_cpp_available",
]
