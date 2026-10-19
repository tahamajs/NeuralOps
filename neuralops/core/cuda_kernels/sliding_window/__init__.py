# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Sliding Window Attention Hardware Operators.
"""

from neuralops.core.cuda_kernels.sliding_window.sliding_autograd import (
    sliding_window_attn,
    sliding_window_pytorch_ref,
)

__all__ = ["sliding_window_attn", "sliding_window_pytorch_ref"]
