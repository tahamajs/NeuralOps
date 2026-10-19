# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Mask Utilities

High-performance tensor masking helpers for causal, block-diagonal, and padded sequence shapes.
"""

from __future__ import annotations

import torch

__all__ = ["safe_mask_value", "_safe_mask_value"]


def safe_mask_value(dtype: torch.dtype) -> float:
    """Return safe additive mask value for the given dtype to avoid underflow/NaN."""
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


_safe_mask_value = safe_mask_value
