#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Cosformer Linear Attention Hardware Operators.
"""

from neuralops.ops.cuda.cosformer.cosformer_autograd import (
    cosformer_linear_attention,
    cosformer_linear_attention_pytorch_ref,
)

__all__ = [
    "cosformer_linear_attention",
    "cosformer_linear_attention_pytorch_ref",
]
