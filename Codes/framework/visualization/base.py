import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._dict: Dict[str, Any] = {}

    def register(self, key: Optional[str] = None):
        def decorator(obj):
            nonlocal key
            if key is None:
                key = obj.__name__
            self._dict[key] = obj
            return obj
        return decorator

    def get(self, key: str):
        return self._dict.get(key.lower())

    def resolve(self, key: str):
        if key.lower() not in self._dict:
            raise KeyError(f"Unknown {self.name} '{key}'. Available: {list(self._dict.keys())}")
        return self._dict[key.lower()]

    def keys(self):
        return list(self._dict.keys())

    def __contains__(self, key: str):
        return key.lower() in self._dict

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f"Registry({self.name!r}, {len(self)} items)"

VISUALIZER_REGISTRY = Registry("visualizers")

class BaseVisualizer(ABC):
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def _save_figure(self, fig, filename: str, dpi: int = 150, bbox_inches: str = "tight") -> str:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        return path