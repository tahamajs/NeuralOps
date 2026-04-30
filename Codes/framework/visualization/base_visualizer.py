# framework/visualization/base_visualizer.py
"""
Base classes for visualizers with registry integration.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..core.registry import VISUALIZER_REGISTRY


class BaseVisualizer(ABC):
    """
    Base class for all visualizers.
    
    Visualizers generate plots and visualizations from model outputs.
    
    Example:
        >>> class MyVisualizer(BaseVisualizer):
        ...     def generate(self, model, dataloader, **kwargs):
        ...         # generate visualizations
        ...         return figure_path
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        Generate visualizations.
        
        Args:
            *args: Variable arguments
            **kwargs: Keyword arguments
            
        Returns:
            Path or list of paths to generated visualizations
        """
        pass
    
    def _save_figure(self, fig, filename: str, dpi: int = 150, bbox_inches: str = 'tight'):
        """Helper to save a matplotlib figure."""
        import matplotlib.pyplot as plt
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        return path


# Register base visualizer (not directly usable, but for inheritance)
@VISUALIZER_REGISTRY.register("base")
class BaseRegisteredVisualizer(BaseVisualizer):
    """Registered base visualizer (abstract)."""
    
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement generate()")