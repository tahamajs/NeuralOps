#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Flash Attention / Flash Decoding Operators.

Forwarding interface for split-KV flash decoding and attention kernels.
"""

from __future__ import annotations

from neuralops.ops.triton.flash_decoding import (
    flash_decode_kernel,
    flash_decoding_attention,
    flash_decoding_pytorch,
)

flash_attention = flash_decoding_attention
flash_attention_pytorch = flash_decoding_pytorch

__all__ = [
    "flash_attention",
    "flash_attention_pytorch",
    "flash_decoding_attention",
    "flash_decoding_pytorch",
    "flash_decode_kernel",
]
