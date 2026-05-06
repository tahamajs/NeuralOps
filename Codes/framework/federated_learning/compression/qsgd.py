"""QSGD – Quantized Stochastic Gradient Descent (Alistarh et al. 2017)."""

import numpy as np
from .base import BaseCompressor, COMPRESSION_REGISTRY

@COMPRESSION_REGISTRY.register("qsgd")
class QSGDCompressor(BaseCompressor):
    def __init__(self, config):
        super().__init__(config)
        self.num_bits = config.compression_quantization_bits
        self.levels = 2 ** self.num_bits - 1

    def compress(self, update):
        compressed = {}
        for k, v in update.items():
            norm = np.linalg.norm(v)
            if norm == 0:
                compressed[k] = (np.zeros(self.levels), norm, v.shape)
            else:
                scaled = v / norm
                quantized = self._stochastic_quantize(scaled)
                compressed[k] = (quantized, norm, v.shape)
        return compressed

    def decompress(self, compressed):
        decompressed = {}
        for k, (quantized, norm, shape) in compressed.items():
            if isinstance(quantized, np.ndarray) and quantized.size > 1:
                decompressed[k] = (norm.reshape(shape) * quantized.reshape(shape)).reshape(shape)
            else:
                decompressed[k] = np.zeros(shape)
        return decompressed

    def _stochastic_quantize(self, arr):
        """Stochastic quantization to {0, 1/(levels-1), ..., 1}."""
        arr = np.abs(arr) / np.max(np.abs(arr)) if np.max(np.abs(arr)) > 0 else arr
        bins = np.linspace(0, 1, self.levels)
        indices = np.digitize(arr, bins) - 1
        indices = np.clip(indices, 0, self.levels - 1)
        return indices.astype(np.int8)