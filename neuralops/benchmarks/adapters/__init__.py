#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Benchmarks Adapters Subpackage.
"""

from neuralops.benchmarks.adapters.classification import ClassificationAdapter
from neuralops.benchmarks.adapters.compressor import CompressorModuleAdapter
from neuralops.benchmarks.adapters.cuda_adapter import (
    CANONICAL_CUDA_MODULES,
    CUDAKernelAdapter,
    CUDA_KERNEL_ALIASES,
)
from neuralops.benchmarks.adapters.model import (
    BaseModelAdapter,
    HuggingFaceModelAdapter,
    SurgerizedModelAdapter,
    UniversalModelAdapter,
)
from neuralops.benchmarks.adapters.perplexity import PerplexityAdapter
from neuralops.benchmarks.adapters.ruler import RulerAdapter
from neuralops.benchmarks.adapters.triton_adapter import (
    CANONICAL_TRITON_MODULES,
    EXTENSION_TRITON_MODULES,
    TritonKernelAdapter,
)
from neuralops.benchmarks.adapters.zeroshot import ZeroShotAdapter

__all__ = [
    "BaseModelAdapter",
    "HuggingFaceModelAdapter",
    "SurgerizedModelAdapter",
    "UniversalModelAdapter",
    "ClassificationAdapter",
    "PerplexityAdapter",
    "ZeroShotAdapter",
    "RulerAdapter",
    "CompressorModuleAdapter",
    "TritonKernelAdapter",
    "CANONICAL_TRITON_MODULES",
    "EXTENSION_TRITON_MODULES",
    "CUDAKernelAdapter",
    "CANONICAL_CUDA_MODULES",
    "CUDA_KERNEL_ALIASES",
]

from neuralops.benchmarks.adapters import classification
from neuralops.benchmarks.adapters import perplexity
from neuralops.benchmarks.adapters import ruler
from neuralops.benchmarks.adapters import zeroshot
