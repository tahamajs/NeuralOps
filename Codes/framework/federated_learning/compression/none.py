"""No compression – identity."""

from .base import BaseCompressor, COMPRESSION_REGISTRY

@COMPRESSION_REGISTRY.register("none")
class NoCompression(BaseCompressor):
    def compress(self, update): return update
    def decompress(self, compressed): return compressed