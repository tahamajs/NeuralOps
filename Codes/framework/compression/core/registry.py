"""
Registry for compression methods.
Allows plug‑and‑play loading of compressors via a simple key.
"""

from typing import Dict, Optional, Type
from .base import BaseCompressor

from framework.attentions.core.registry import ATTENTION_REGISTRY

class CompressorRegistry:
    """
    A registry for compressor classes.

    Usage::

        @COMPRESSOR_REGISTRY.register("my_method")
        class MyCompressor(BaseCompressor):
            ...

        compressor_cls = COMPRESSOR_REGISTRY.get("my_method")
        compressor = compressor_cls(model, ...)
    """

    def __init__(self, name: str = "compressor_registry"):
        self.name = name
        self._registry: Dict[str, Type[BaseCompressor]] = {}

    def register(self, key: Optional[str] = None):
        """
        Decorator to register a compressor class.

        Args:
            key: Optional string identifier. If not given, uses the class name.
        """
        def decorator(obj: Type[BaseCompressor]) -> Type[BaseCompressor]:
            nonlocal key
            if not issubclass(obj, BaseCompressor):
                raise TypeError(
                    f"Registered class {obj.__name__} must inherit from BaseCompressor"
                )
            if key is None:
                key = obj.__name__
            self._registry[key] = obj
            return obj
        return decorator

    def get(self, key: str) -> Optional[Type[BaseCompressor]]:
        """Retrieve a compressor class by key."""
        return self._registry.get(key)

    def keys(self) -> list:
        """Return all registered compressor keys."""
        return list(self._registry.keys())

    def registered(self, key: str) -> bool:
        """Check if a key is registered."""
        return key in self._registry

    def list(self) -> None:
        """Print all registered compressors."""
        print(f"Registered compressors ({len(self._registry)}):")
        for key, cls in self._registry.items():
            print(f"  - {key}: {cls.__name__}")

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        return f"<CompressorRegistry: {len(self._registry)} methods>"


# Singleton instance used across the framework
COMPRESSOR_REGISTRY = CompressorRegistry()

__all__ = ["ATTENTION_REGISTRY", "COMPRESSOR_REGISTRY"]
