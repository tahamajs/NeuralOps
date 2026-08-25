"""
Plotting abstractions and registry for visualization backends.
Provides BasePlotter interface, Matplotlib/headless default implementation,
and high-level visualization utilities.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import Registry

logger = logging.getLogger(__name__)

PLOTTER_REGISTRY = Registry("plotters")


class BasePlotter(ABC):
    """Interface that every plotting backend must implement."""

    @abstractmethod
    def figure(self, *args: Any, **kwargs: Any) -> Any:
        """Return a figure object suitable for the backend."""
        pass

    @abstractmethod
    def save(self, fig: Any, path: str, **kwargs: Any) -> None:
        """Save the figure to disk."""
        pass

    @abstractmethod
    def line_plot(self, ax_or_fig: Any, x: Any, y: Any, **kwargs: Any) -> Any:
        """Add a line trace / series to the figure."""
        pass

    def scatter_plot(self, ax_or_fig: Any, x: Any, y: Any, **kwargs: Any) -> Any:
        """Add a scatter plot series to the figure."""
        if hasattr(ax_or_fig, "scatter"):
            return ax_or_fig.scatter(x, y, **kwargs)
        return None

    def bar_plot(self, ax_or_fig: Any, categories: Any, values: Any, **kwargs: Any) -> Any:
        """Add a bar chart to the figure."""
        if hasattr(ax_or_fig, "bar"):
            return ax_or_fig.bar(categories, values, **kwargs)
        return None

    def finalise(
        self,
        fig: Any,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        **kwargs: Any,
    ) -> Any:
        """Add labels, title, and return the finished figure."""
        if isinstance(fig, tuple) and len(fig) == 2:
            real_fig, ax = fig
            if hasattr(ax, "set_title") and title:
                ax.set_title(title)
            if hasattr(ax, "set_xlabel") and xlabel:
                ax.set_xlabel(xlabel)
            if hasattr(ax, "set_ylabel") and ylabel:
                ax.set_ylabel(ylabel)
            if hasattr(ax, "grid"):
                ax.grid(True, alpha=kwargs.get("alpha", 0.3))
            return real_fig
        elif hasattr(fig, "suptitle") and title:
            fig.suptitle(title)
        return fig


@PLOTTER_REGISTRY.register("default")
@PLOTTER_REGISTRY.register("matplotlib")
class DefaultPlotter(BasePlotter):
    """Production Matplotlib-based plotting engine with headless rendering safety."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid", dpi: int = 150) -> None:
        self.dpi = dpi
        self.style = style

    def figure(
        self,
        figsize: Tuple[int, int] = (10, 6),
        nrows: int = 1,
        ncols: int = 1,
        **kwargs: Any,
    ) -> Any:
        """Create and return a new Matplotlib (fig, ax) handle."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
        except Exception as e:
            logger.warning("Matplotlib figure creation failed (%s), returning headless placeholder", e)
            return (None, None)

    def save(self, fig: Any, path: str, dpi: Optional[int] = None, **kwargs: Any) -> None:
        """Save figure to destination path."""
        try:
            import matplotlib.pyplot as plt

            actual_fig = fig[0] if isinstance(fig, tuple) else fig
            if actual_fig is not None and hasattr(actual_fig, "savefig"):
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                actual_fig.savefig(path, dpi=dpi or self.dpi, bbox_inches="tight", **kwargs)
                plt.close(actual_fig)
        except Exception as e:
            logger.warning("Failed to save figure to %s: %s", path, e)

    def line_plot(self, ax_or_fig: Any, x: Any, y: Any, **kwargs: Any) -> Any:
        """Add line trace to the active axes."""
        ax = ax_or_fig[1] if isinstance(ax_or_fig, tuple) else ax_or_fig
        if ax is not None and hasattr(ax, "plot"):
            return ax.plot(x, y, **kwargs)
        return None

    def heatmap(
        self,
        ax_or_fig: Any,
        data: Any,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> Any:
        """Add 2D heatmap / attention matrix plot."""
        ax = ax_or_fig[1] if isinstance(ax_or_fig, tuple) else ax_or_fig
        if ax is not None and hasattr(ax, "imshow"):
            return ax.imshow(data, cmap=cmap, aspect="auto", **kwargs)
        return None


def get_plotter(backend: str = "matplotlib") -> BasePlotter:
    """Retrieve plotter instance from registry with default fallback."""
    plotter_cls = PLOTTER_REGISTRY.get(backend)
    if plotter_cls is None:
        plotter_cls = DefaultPlotter
    return plotter_cls()

