"""Random sparsification – keep a random fraction of elements."""

import numpy as np

from .base import COMPRESSION_REGISTRY, BaseCompressor


@COMPRESSION_REGISTRY.register("random_sparsification")
class RandomSparsifier(BaseCompressor):
    def __init__(self, config):
        super().__init__(config)
        self.fraction = config.compression_topk_fraction

    def compress(self, update):
        compressed = {}
        for k, v in update.items():
            v_flat = v.flatten()
            mask = np.random.rand(len(v_flat)) < self.fraction
            compressed[k] = (v_flat[mask].copy(), mask, v.shape)
        return compressed

    def decompress(self, compressed):
        decompressed = {}
        for k, (values, mask, shape) in compressed.items():
            arr = np.zeros(np.prod(shape), dtype=values.dtype)
            arr[mask] = values
            decompressed[k] = arr.reshape(shape)
        return decompressed
