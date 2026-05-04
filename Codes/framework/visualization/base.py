# framework/visualization/base.py
import os
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
        return self._dict.get(key)

    def keys(self):
        return list(self._dict.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._dict

VISUALIZER_REGISTRY = Registry("visualizers")

class BaseVisualizer(ABC):
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def _save_figure(self, fig, filename: str, dpi: int = 150, bbox_inches: str = "tight") -> str:
        import matplotlib.pyplot as plt
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        return path