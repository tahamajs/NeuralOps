"""
Base classes and registry for visualizers.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Registry:
    """Simple registry for visualizers."""
    def __init__(self, name):
        self.name = name
        self._dict = {}

    def register(self, key=None):
        def decorator(obj):
            nonlocal key
            if key is None:
                key = obj.__name__
            self._dict[key] = obj
            return obj
        return decorator

    def get(self, key):
        return self._dict.get(key)

    def keys(self):
        return list(self._dict.keys())

    def __contains__(self, key):
        return key in self._dict


# Global registry for visualizers
VISUALIZER_REGISTRY = Registry("visualizers")


class BaseVisualizer(ABC):
    """
    Base class for all visualizers.

    Visualizers generate plots and visualizations from model outputs.
    """

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement generate()")

    def _save_figure(self, fig, filename: str, dpi: int = 150, bbox_inches: str = 'tight'):
        import matplotlib.pyplot as plt
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        return path