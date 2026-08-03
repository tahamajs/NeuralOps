# framework/visualization/plotter.py
from abc import ABC, abstractmethod

from .base import Registry

PLOTTER_REGISTRY = Registry("plotters")


class BasePlotter(ABC):
    """Interface that every plotting backend must implement."""

    @abstractmethod
    def figure(self, *args, **kwargs) -> None:
        """Return a figure object suitable for the backend."""
        pass

    @abstractmethod
    def save(self, fig, path: str, **kwargs) -> None:
        """Save the figure to disk."""
        pass

    @abstractmethod
    def line_plot(self, x, y, **kwargs) -> None:
        """Add a line trace / series to the figure."""
        pass

    def finalise(self, fig, title: str, xlabel: str, ylabel: str, **kwargs) -> None:
        """Add labels and return the finished figure."""
        raise NotImplementedError
