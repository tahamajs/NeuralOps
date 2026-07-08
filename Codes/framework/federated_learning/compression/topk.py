"""Top‑K sparsification – keep only the largest absolute values."""

import numpy as np

from .base import COMPRESSION_REGISTRY, BaseCompressor


@COMPRESSION_REGISTRY.register("topk")
class TopKCompressor(BaseCompressor):
    def __init__(self, config):
        super().__init__(config)
        self.fraction = config.compression_topk_fraction

    def compress(self, update):
        compressed = {}
        for k, v in update.items():
            v_flat = v.flatten()
            kk = max(1, int(len(v_flat) * self.fraction))
            top_indices = np.argpartition(np.abs(v_flat), -kk)[-kk:]
            sparse_mask = np.zeros_like(v_flat, dtype=bool)
            sparse_mask[top_indices] = True
            compressed[k] = (v_flat[top_indices].copy(), top_indices.copy(), v.shape)
        return compressed

    def decompress(self, compressed):
        decompressed = {}
        for k, (values, indices, shape) in compressed.items():
            arr = np.zeros(np.prod(shape), dtype=values.dtype)
            arr[indices] = values
            decompressed[k] = arr.reshape(shape)
        return decompressed
