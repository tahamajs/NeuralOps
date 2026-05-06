# framework/visualization/plotter.py
from abc import ABC, abstractmethod
from .base import Registry

PLOTTER_REGISTRY = Registry("plotters")

class BasePlotter(ABC):
    """Interface that every plotting backend must implement."""
    @abstractmethod
    def figure(self, *args, **kwargs):
        """Return a figure object suitable for the backend."""
        pass

    @abstractmethod
    def save(self, fig, path: str, **kwargs):
        """Save the figure to disk."""
        pass

    @abstractmethod
    def line_plot(self, x, y, **kwargs):
        """Add a line trace / series to the figure."""
        pass

    def finalise(self, fig, title: str, xlabel: str, ylabel: str, **kwargs):
        """Add labels and return the finished figure."""
        raise NotImplementedError