"""
Central registries for the Efficient Transformer Toolkit.

All framework components (attentions, compressors, trainers, etc.)
are stored in registries, enabling dynamic discovery and plug‑and‑play integration.
"""

from typing import Dict, Callable, Optional, List


class Registry:
    """A named dictionary that maps string keys to classes or factory functions."""

    def __init__(self, name: str):
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
            nonlocal key
            if key is None:
                key = obj.__name__
            self._dict[key.lower()] = obj
            return obj
        return decorator

    def get(self, key: str) -> Optional[Callable]:
        """Safely retrieve a registered component, or None if not found."""
        return self._dict.get(key.lower())

    def resolve(self, key: str) -> Callable:
        """Retrieve a registered component and raise a descriptive error if missing."""
        if key.lower() not in self._dict:
            raise KeyError(
                f"Unknown {self.name.rstrip('s')} '{key}'. "
                f"Available: {list(self._dict.keys())}"
            )
        return self._dict[key.lower()]

    def keys(self) -> List[str]:
        return list(self._dict.keys())

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, {len(self)} items)"


# ----------------------------------------------------------------------
# Global registries – populated by their respective packages
# ----------------------------------------------------------------------
ATTENTION_REGISTRY = Registry("attentions")
COMPRESSOR_REGISTRY = Registry("compressors")   # legacy, still used by some compressors
METHOD_REGISTRY = Registry("methods")
AUX_REGISTRY = Registry("auxiliary")

# New: Unified component registry for all model parts
COMPONENT_REGISTRY = Registry("components")     # keys like "attention:linformer", "ffn:svd", ...

MODEL_REGISTRY = Registry("models")
DATASET_REGISTRY = Registry("datasets")
LOSS_REGISTRY = Registry("losses")
TRAINER_REGISTRY = Registry("trainers")
VISUALIZER_REGISTRY = Registry("visualizers")


def list_all_components() -> dict:
    return {
        "attentions": ATTENTION_REGISTRY.keys(),
        "compressors": COMPRESSOR_REGISTRY.keys(),
        "methods": METHOD_REGISTRY.keys(),
        "auxiliary": AUX_REGISTRY.keys(),
        "components": COMPONENT_REGISTRY.keys(),
        "models": MODEL_REGISTRY.keys(),
        "datasets": DATASET_REGISTRY.keys(),
        "losses": LOSS_REGISTRY.keys(),
        "trainers": TRAINER_REGISTRY.keys(),
        "visualizers": VISUALIZER_REGISTRY.keys(),
    }