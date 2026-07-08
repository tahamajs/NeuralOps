"""signSGD – compress gradients to just their signs (Bernstein et al. 2018)."""

import numpy as np

from .base import COMPRESSION_REGISTRY, BaseCompressor


@COMPRESSION_REGISTRY.register("sign_sgd")
class SignSGDCompressor(BaseCompressor):
    def compress(self, update):
        return {k: np.sign(v) for k, v in update.items()}

    def decompress(self, compressed):
        return compressed  # server uses sign directly
