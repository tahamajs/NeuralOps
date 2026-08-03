# framework/visualization/plotters/matplotlib_plotter.py
from typing import Any
import matplotlib.pyplot as plt

from ..plotter import PLOTTER_REGISTRY, BasePlotter


@PLOTTER_REGISTRY.register("matplotlib")
class MatplotlibPlotter(BasePlotter):
    def figure(self, figsize=(10, 6)) -> Any:
        """Figure."""
        return plt.subplots(figsize=figsize)

    def save(self, fig, path, dpi=150, **kwargs) -> None:
        """Save."""
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def line_plot(self, ax, x, y, **kwargs) -> None:
        """Line plot."""
        ax.plot(x, y, **kwargs)

    def finalise(self, figure_axes, title, xlabel, ylabel) -> Any:
        """Finalise."""
        fig, ax = figure_axes
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return fig
