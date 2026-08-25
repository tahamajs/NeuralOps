"""
Central registry system for the Efficient Transformer Toolkit.
All framework components (attentions, compressors, trainers, visualizers, etc.)
are stored in registries, enabling dynamic discovery and plug‑and‑play integration.
"""

from typing import Callable, Dict, List, Optional


class Registry:
    """A named dictionary that maps string keys to classes or factory functions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._dict: Dict[str, Callable] = {}

    def register(self, key: Optional[str] = None) -> Callable:
        """
        Decorator to register a class/factory under the given key.

        Usage:
            @ATTENTION_REGISTRY.register("griffin")
            class GriffinAttention(BaseAttention): ...
        """

        def decorator(obj: Callable) -> Callable:
            """Decorator."""
            nonlocal key
            if key is None:
                key = obj.__name__
            self._dict[key.lower()] = obj
            return obj

        return decorator

    def _try_lazy_load(self, key: str) -> None:
        """Attempt to lazily load package modules when a key is queried."""
        k = key.lower()
        if k in self._dict:
            return
        if self.name == "compressors":
            try:
                import framework.compression_engine as _ce
                # Try accessing key from compression_engine
                _ = getattr(_ce, key, None)
                if k not in self._dict:
                    # Try importing module by key directly
                    import importlib
                    importlib.import_module(f"framework.compression_engine.{k}")
            except Exception:
                pass
        elif self.name == "attentions":
            try:
                import framework.attentions as _att
                _ = getattr(_att, key, None)
                if k not in self._dict:
                    import importlib
                    importlib.import_module(f"framework.attentions.{k}")
            except Exception:
                pass

    def get(self, key: str) -> Optional[Callable]:
        """Safely retrieve a registered component, or None if not found."""
        self._try_lazy_load(key)
        return self._dict.get(key.lower())

    def resolve(self, key: str) -> Callable:
        """Retrieve a registered component and raise a descriptive error if missing."""
        self._try_lazy_load(key)
        k = key.lower()
        if k not in self._dict:
            raise KeyError(
                f"Unknown {self.name.rstrip('s')} '{key}'. "
                f"Available: {list(self._dict.keys())}"
            )
        return self._dict[k]

    def keys(self) -> List[str]:
        """Return a list of all registered keys (in original casing)."""
        return list(self._dict.keys())

    def list_available(self) -> List[str]:
        """Return a list of all registered keys."""
        return list(self._dict.keys())

    def items(self):
        """Return registered key-value pairs."""
        return self._dict.items()

    def values(self):
        """Return registered values."""
        return self._dict.values()

    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, key: str) -> bool:
        self._try_lazy_load(key)
        return key.lower() in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, {len(self)} items)"


# ----------------------------------------------------------------------
# Global registries – populated by their respective packages
# ----------------------------------------------------------------------
ATTENTION_REGISTRY = Registry("attentions")  # attention modules (Nexus, LEKA, ...)
COMPRESSOR_REGISTRY = Registry("compressors")  # compression strategies (SVD, TT, ...)
METHOD_REGISTRY = Registry("methods")  # training methods (LoRA, R-Drop, ...)
AUX_REGISTRY = Registry(
    "auxiliary"
)  # auxiliary modules (RoPE, SSM, token pruning, ...)

# Optional registries (if needed)
MODEL_REGISTRY = Registry("models")  # different base models
DATASET_REGISTRY = Registry("datasets")  # GLUE, SuperGLUE, etc.
LOSS_REGISTRY = Registry("losses")  # custom loss functions
TRAINER_REGISTRY = Registry("trainers")  # trainer classes
VISUALIZER_REGISTRY = Registry("visualizers")  # visualization utilities


def list_all_components() -> dict:
    """Convenience function that returns all global registry keys."""
    return {
        "attentions": ATTENTION_REGISTRY.keys(),
        "compressors": COMPRESSOR_REGISTRY.keys(),
        "methods": METHOD_REGISTRY.keys(),
        "auxiliary": AUX_REGISTRY.keys(),
        "models": MODEL_REGISTRY.keys(),
        "datasets": DATASET_REGISTRY.keys(),
        "losses": LOSS_REGISTRY.keys(),
        "trainers": TRAINER_REGISTRY.keys(),
        "visualizers": VISUALIZER_REGISTRY.keys(),
    }
