"""No compression – identity."""

from .base import COMPRESSION_REGISTRY, BaseCompressor


@COMPRESSION_REGISTRY.register("none")
class NoCompression(BaseCompressor):
    def compress(self, update):
        return update

    def decompress(self, compressed):
        return compressed
