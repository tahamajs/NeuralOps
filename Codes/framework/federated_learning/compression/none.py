"""No compression – identity."""
from typing import Any

from .base import COMPRESSION_REGISTRY, BaseCompressor


@COMPRESSION_REGISTRY.register("none")
class NoCompression(BaseCompressor):
    def compress(self, update) -> Any:
        """Compress the target module or tensor."""
        return update

    def decompress(self, compressed) -> Any:
        """Decompress the target module or tensor."""
        return compressed
