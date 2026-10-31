#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FP8 Compressor CUDA Kernel Operator.
"""

from __future__ import annotations

import torch
from neuralops.ops.cuda.fp8_compressor.autograd import FP8Function


def fp8_compressor_compress(
    W: torch.Tensor,
    scale_type: str = "per_tensor",
    fp8_format: str = "e4m3",
) -> torch.Tensor:
    """FP8 compression using CUDA C++ kernel."""
    max_val = 448.0 if fp8_format == "e4m3" else 57344.0
    w_float = W.contiguous().to(torch.float32)

    max_abs = w_float.abs().amax().clamp(min=1e-8).item()
    scale = float(max_abs / max_val)

    out = FP8Function.apply(w_float, scale)
    return out.reshape(W.shape).to(dtype=W.dtype)


__all__ = ["fp8_compressor_compress", "FP8Function"]
