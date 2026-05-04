# framework/core/registry.py
from typing import Dict, Callable, Optional, List

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._dict: Dict[str, Callable] = {}

    def register(self, key: Optional[str] = None) -> Callable:
        def decorator(obj: Callable) -> Callable:
            nonlocal key
            if key is None:
                key = obj.__name__
            self._dict[key.lower()] = obj
            return obj
        return decorator

    def get(self, key: str) -> Optional[Callable]:
        return self._dict.get(key.lower())

    def keys(self) -> List[str]:
        return list(self._dict.keys())

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._dict

    def __len__(self) -> int:
        return len(self._dict)

# Global registries
ATTENTION_REGISTRY = Registry("attentions")
COMPRESSOR_REGISTRY = Registry("compressors")
TRAINER_REGISTRY = Registry("trainers")
VISUALIZER_REGISTRY = Registry("visualizers")

def list_all_components() -> dict:
    return {
        "attentions": ATTENTION_REGISTRY.keys(),
        "compressors": COMPRESSOR_REGISTRY.keys(),
        "trainers": TRAINER_REGISTRY.keys(),
        "visualizers": VISUALIZER_REGISTRY.keys(),
    }