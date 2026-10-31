#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
KMeans Quantize CUDA Kernel Operator.
"""

from __future__ import annotations

import torch
from neuralops.compression.quantization.kmeans_quantize import fast_kmeans_1d
from neuralops.ops.cuda.kmeans_quantize.autograd import KMeansQuantizeFunction


def kmeans_quantize_compress(
    W: torch.Tensor,
    num_clusters: int = 256,
) -> torch.Tensor:
    """K-Means codebook quantization using CUDA C++ kernel."""
    centroids, _ = fast_kmeans_1d(W, n_clusters=num_clusters)
    centroids = centroids.contiguous().to(device=W.device, dtype=torch.float32)
    W_fp32 = W.contiguous().to(torch.float32)

    out = KMeansQuantizeFunction.apply(W_fp32, centroids)
    return out.reshape(W.shape).to(dtype=W.dtype)


__all__ = ["kmeans_quantize_compress", "KMeansQuantizeFunction"]
