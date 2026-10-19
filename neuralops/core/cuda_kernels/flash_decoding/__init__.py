#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Flash-Decoding Attention Hardware Operators.
"""

from neuralops.core.cuda_kernels.flash_decoding.flash_dec_autograd import (
    flash_decoding_attention,
    flash_decoding_attention_pytorch_ref,
)

__all__ = [
    "flash_decoding_attention",
    "flash_decoding_attention_pytorch_ref",
]
