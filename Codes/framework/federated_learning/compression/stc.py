"""STC – Sparse Ternary Compression (Sattler et al. 2019)."""

import numpy as np

from .base import COMPRESSION_REGISTRY, BaseCompressor


@COMPRESSION_REGISTRY.register("stc")
class STCCompressor(BaseCompressor):
    def __init__(self, config):
        super().__init__(config)
        self.fraction = config.compression_topk_fraction

    def compress(self, update):
        compressed = {}
        for k, v in update.items():
            v_flat = v.flatten()
            kk = max(1, int(len(v_flat) * self.fraction))
            top_indices = np.argpartition(np.abs(v_flat), -kk)[-kk:]
            values = np.sign(v_flat[top_indices])  # ternary: +1, 0, -1
            compressed[k] = (values, top_indices.copy(), v.shape)
        return compressed

    def decompress(self, compressed):
        decompressed = {}
        for k, (values, indices, shape) in compressed.items():
            arr = np.zeros(np.prod(shape), dtype=np.float32)
            arr[indices] = values
            decompressed[k] = arr.reshape(shape)
        return decompressed
