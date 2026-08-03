"""Base class for communication compression strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from framework.core.registry import Registry

COMPRESSION_REGISTRY = Registry("fl_compression")


class BaseCompressor(ABC):
    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def compress(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Compress the update before sending to server."""
        pass

    @abstractmethod
    def decompress(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress the update on server side."""
        pass
