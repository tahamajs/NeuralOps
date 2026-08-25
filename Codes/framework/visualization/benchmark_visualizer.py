"""
Publication-grade benchmark visualization module for SVD Linear Attention and Transformer architectures.
Provides high-quality Matplotlib/Seaborn visualization routines for attention analysis,
spectral decay, KV cache memory footprint, Pareto efficiency frontiers, task radar charts,
and layerwise rank/sparsity metrics.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from .utils import save_figure, set_plot_style

DEFAULT_STYLE = "seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default"


def _to_numpy(data: Any) -> np.ndarray:
    """Helper to convert Tensor, list, or sequence to numpy ndarray."""
    if data is None:
        return np.array([])
    if hasattr(data, "detach"):
        return data.detach().cpu().float().numpy()
    if hasattr(data, "cpu"):
        return data.cpu().numpy()
    return np.asarray(data)


def plot_attention_heatmap(
    attn_weights: Any,
    tokens: Optional[Sequence[str]] = None,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None,
    cmap: str = "viridis",
    show_annotations: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300,
    **kwargs: Any,
) -> Figure:
    """
    Plot publication-quality 2D attention weight heatmap.

    Args:
        attn_weights: 2D array-like attention matrix (seq_len x seq_len) or (heads, seq_len, seq_len)
        tokens: Optional list of token strings for tick labels
        title: Title of the heatmap
        save_path: Destination file path for saving figure
        cmap: Colormap name
        show_annotations: Whether to annotate numerical values in cells
        figsize: Figure dimensions (width, height)
        dpi: Dots per inch for rasterization
        **kwargs: Additional keyword arguments passed to sns.heatmap

    Returns:
        matplotlib Figure object
    """
    matrix = _to_numpy(attn_weights)
    
    # If 3D (heads, seq, seq), average across heads or take first head
    if matrix.ndim == 3:
        matrix = np.mean(matrix, axis=0)
    elif matrix.ndim != 2:
        raise ValueError(f"Expected 2D or 3D attention tensor, got ndim={matrix.ndim}")

    fig: Optional[Figure] = None
    try:
        # FIX: Wrap matplotlib calls in with plt.style.context(): to prevent global style contamination
        with plt.style.context(DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            seq_len = matrix.shape[0]
            xticklabels = tokens if tokens and len(tokens) == seq_len else "auto"
            yticklabels = tokens if tokens and len(tokens) == seq_len else "auto"

            annot = show_annotations if (seq_len <= 32 and show_annotations) else False

            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                annot=annot,
                fmt=".2f" if annot else "",
                cbar_kws={"label": "Attention Weight", "shrink": 0.8},
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                linewidths=0.2 if seq_len <= 32 else 0.0,
                square=True,
                **kwargs,
            )

            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Key Token Index", fontsize=11, labelpad=8)
            ax.set_ylabel("Query Token Index", fontsize=11, labelpad=8)

            if tokens and len(tokens) == seq_len:
                ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
                ax.set_yticklabels(tokens, rotation=0, fontsize=9)

            fig.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig
    finally:
        # FIX: Ensure figure cleanup, fig.clf() and plt.close(fig) in finally block, followed by gc.collect()
        if save_path and fig is not None:
            try:
                fig.clf()
                plt.close(fig)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass
        gc.collect()


def plot_singular_value_decay(
    singular_values_dict: Dict[str, Any],
    title: str = "Singular Value Spectrum Decay",
    save_path: Optional[str] = None,
    log_scale: bool = True,
    normalize: bool = True,
    top_k: Optional[int] = 128,
    figsize: Tuple[int, int] = (8, 5),
    dpi: int = 300,
    **kwargs: Any,
) -> Figure:
    """
    Plot singular value decay curves across multiple models/layers to visualize low-rank compressibility.

    Args:
        singular_values_dict: Dict mapping label (e.g. 'Standard Attn Layer 4', 'SVD Linear') to array of singular values
        title: Plot title
        save_path: Filepath to save image
        log_scale: Whether to use semi-log y-axis (log10 singular values)
        normalize: Normalize singular values by max singular value (sigma_i / sigma_1)
        top_k: Maximum rank index to plot
        figsize: Figure dimensions
        dpi: Output resolution
    """
    fig: Optional[Figure] = None
    try:
        # FIX: Wrap matplotlib calls in with plt.style.context():
        with plt.style.context(DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            palette = sns.color_palette("tab10", n_colors=max(len(singular_values_dict), 1))

            for idx, (label, s_vals) in enumerate(singular_values_dict.items()):
                vals = _to_numpy(s_vals).flatten()
                vals = np.sort(vals)[::-1]  # descending
                if top_k is not None:
                    vals = vals[:top_k]
                
                if normalize and len(vals) > 0 and vals[0] > 0:
                    vals = vals / vals[0]

                ranks = np.arange(1, len(vals) + 1)
                ax.plot(
                    ranks,
                    vals,
                    label=label,
                    color=palette[idx % len(palette)],
                    linewidth=2.0,
                    marker="o" if len(vals) <= 20 else None,
                    markersize=4,
                    alpha=0.9,
                )

            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Singular Value Rank Index ($k$)", fontsize=11, labelpad=8)
            ax.set_ylabel(
                "Normalized Magnitude ($\\sigma_k / \\sigma_1$)" if normalize else "Singular Value ($\\sigma_k$)",
                fontsize=11,
                labelpad=8,
            )

            if log_scale:
                ax.set_yscale("log")
                ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            else:
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="upper right")
            fig.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig
    finally:
        # FIX: Ensure fig.clf() and plt.close(fig) are executed in finally block followed by gc.collect()
        if save_path and fig is not None:
            try:
                fig.clf()
                plt.close(fig)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass
        gc.collect()


def plot_kv_cache_scaling(
    seq_lens: Sequence[int],
    memory_mb_dict: Dict[str, Sequence[float]],
    title: str = "KV Cache Memory Footprint vs. Sequence Length",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
    dpi: int = 300,
    **kwargs: Any,
) -> Figure:
    """
    Plot KV cache memory scaling comparing quadratic Full Attention O(N) cache vs linear SVD O(1) state.

    Args:
        seq_lens: List/Array of sequence lengths (e.g. [512, 1024, 2048, 4096, 8192, 16384])
        memory_mb_dict: Mapping from model/variant name to sequence of memory values (MB)
        title: Plot title
        save_path: Output path
        figsize: Figure dimensions
        dpi: Dots per inch
    """
    fig: Optional[Figure] = None
    try:
        # FIX: Wrap matplotlib calls in with plt.style.context():
        with plt.style.context(DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            lens = np.asarray(seq_lens)
            palette = sns.color_palette("Set2", n_colors=max(len(memory_mb_dict), 1))

            for idx, (label, mem_vals) in enumerate(memory_mb_dict.items()):
                mem = np.asarray(mem_vals)
                ax.plot(
                    lens,
                    mem,
                    label=label,
                    color=palette[idx % len(palette)],
                    linewidth=2.2,
                    marker="s",
                    markersize=5,
                )
                ax.fill_between(lens, mem, alpha=0.1, color=palette[idx % len(palette)])

            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Context Length (Tokens)", fontsize=11, labelpad=8)
            ax.set_ylabel("KV Cache Memory (MB)", fontsize=11, labelpad=8)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend(frameon=True, fontsize=10, loc="upper left")
            fig.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig
    finally:
        # FIX: Ensure fig.clf() and plt.close(fig) are executed in finally block followed by gc.collect()
        if save_path and fig is not None:
            try:
                fig.clf()
                plt.close(fig)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass
        gc.collect()


def plot_pareto_frontier(
    ppl_list: Sequence[float],
    latency_list: Sequence[float],
    labels: Sequence[str],
    title: str = "Pareto Efficiency Frontier (Perplexity vs. Latency)",
    save_path: Optional[str] = None,
    latency_unit: str = "ms",
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300,
    **kwargs: Any,
) -> Figure:
    """
    Plot Pareto frontier comparing perplexity (lower is better) vs latency (lower is better).
    Computes and highlights the non-dominated Pareto optimal front.

    Args:
        ppl_list: Validation perplexity values
        latency_list: Latency / runtime values
        labels: Label for each data point
        title: Plot title
        save_path: Destination path
        latency_unit: Latency unit string ('ms', 's', 'tokens/sec')
        figsize: Figure dimensions
        dpi: Resolution
    """
    fig: Optional[Figure] = None
    try:
        # FIX: Wrap matplotlib calls in with plt.style.context():
        with plt.style.context(DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            x = np.asarray(latency_list, dtype=float)
            y = np.asarray(ppl_list, dtype=float)

            # Determine Pareto non-dominated points (minimize both x and y)
            sorted_indices = np.argsort(x)
            pareto_indices = []
            min_y = float("inf")
            for idx in sorted_indices:
                if y[idx] < min_y:
                    pareto_indices.append(idx)
                    min_y = y[idx]

            # Scatter all points
            scatter = ax.scatter(
                x,
                y,
                c=y,
                cmap="coolwarm_r",
                s=80,
                edgecolor="black",
                linewidth=0.8,
                zorder=4,
            )
            fig.colorbar(scatter, ax=ax, label="Perplexity (PPL)", shrink=0.8)

            # Plot pareto frontier curve
            pareto_x = x[pareto_indices]
            pareto_y = y[pareto_indices]
            ax.plot(
                pareto_x,
                pareto_y,
                color="#e74c3c",
                linestyle="--",
                linewidth=2.0,
                label="Pareto Optimal Frontier",
                zorder=3,
            )

            # Label points
            for i, label in enumerate(labels):
                is_pareto = i in pareto_indices
                fontweight = "bold" if is_pareto else "normal"
                color = "#c0392b" if is_pareto else "#2c3e50"
                ax.annotate(
                    label,
                    (x[i], y[i]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight=fontweight,
                    color=color,
                    alpha=0.95,
                )

            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel(f"Inference Latency ({latency_unit}) [lower is better]", fontsize=11, labelpad=8)
            ax.set_ylabel("Perplexity (PPL) [lower is better]", fontsize=11, labelpad=8)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.legend(frameon=True, loc="upper right", fontsize=9)
            fig.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig
    finally:
        # FIX: Ensure fig.clf() and plt.close(fig) are executed in finally block followed by gc.collect()
        if save_path and fig is not None:
            try:
                fig.clf()
                plt.close(fig)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass
        gc.collect()


def plot_dataset_radar_chart(
    dataset_scores_dict: Dict[str, Dict[str, float]],
    title: str = "Benchmark Performance Across Tasks",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 300,
    **kwargs: Any,
) -> Figure:
    """
    Plot publication-grade multi-model radar (spider) chart across evaluation datasets/tasks.

    Args:
        dataset_scores_dict: Dict of model_name -> {task_name: score}
        title: Plot title
        save_path: Filepath to save output
        figsize: Figure dimensions
        dpi: Output resolution
    """
    first_model = next(iter(dataset_scores_dict.keys()))
    categories = list(dataset_scores_dict[first_model].keys())
    num_vars = len(categories)

    if num_vars < 3:
        raise ValueError("Radar chart requires at least 3 distinct tasks/categories.")

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig: Optional[Figure] = None
    try:
        # FIX: Wrap matplotlib calls in with plt.style.context():
        with plt.style.context(DEFAULT_STYLE):
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), dpi=dpi)
            palette = sns.color_palette("muted", n_colors=max(len(dataset_scores_dict), 1))

            for idx, (model_name, scores) in enumerate(dataset_scores_dict.items()):
                values = [scores.get(cat, 0.0) for cat in categories]
                values += values[:1]
                
                ax.plot(
                    angles,
                    values,
                    color=palette[idx % len(palette)],
                    linewidth=2,
                    label=model_name,
                )
                ax.fill(angles, values, color=palette[idx % len(palette)], alpha=0.15)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=9.5, fontweight="semibold")
            ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9, frameon=True)
            fig.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig
    finally:
        # FIX: Ensure fig.clf() and plt.close(fig) are executed in finally block followed by gc.collect()
        if save_path and fig is not None:
            try:
                fig.clf()
                plt.close(fig)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass
        gc.collect()


def plot_layerwise_sparsity_rank(
    layer_ranks: Sequence[Union[int, float]],
    layer_sparsities: Sequence[float],
    title: str = "Layerwise Effective Rank & Sparsity Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 5),
    dpi: int = 300,
    **kwargs: Any,
) -> Figure:
    """
    Dual-axis plot showing layer-by-layer effective rank and activation/weight sparsity percentage.

    Args:
        layer_ranks: Sequence of rank values per layer
        layer_sparsities: Sequence of sparsity ratios (0.0 - 1.0 or 0 - 100%) per layer
        title: Plot title
        save_path: Destination path
        figsize: Figure dimensions
        dpi: Dots per inch
    """
    fig: Optional[Figure] = None
    try:
        # FIX: Wrap matplotlib calls in with plt.style.context():
        with plt.style.context(DEFAULT_STYLE):
            fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

            layers = np.arange(1, len(layer_ranks) + 1)
            ranks = np.asarray(layer_ranks)
            sparsities = np.asarray(layer_sparsities)
            
            if len(sparsities) > 0 and np.max(sparsities) <= 1.0:
                sparsities = sparsities * 100.0

            color_rank = "#2980b9"
            color_sparsity = "#e67e22"

            # Primary axis: Effective Rank
            bars = ax1.bar(
                layers - 0.15,
                ranks,
                width=0.3,
                color=color_rank,
                alpha=0.85,
                label="Effective Rank ($r$)",
            )
            ax1.set_xlabel("Transformer Layer Index", fontsize=11, labelpad=8)
            ax1.set_ylabel("Effective Rank", color=color_rank, fontsize=11, labelpad=8)
            ax1.tick_params(axis="y", labelcolor=color_rank)
            ax1.set_xticks(layers)

            # Secondary axis: Sparsity %
            ax2 = ax1.twinx()
            lines = ax2.plot(
                layers + 0.15,
                sparsities,
                color=color_sparsity,
                linewidth=2.5,
                marker="o",
                label="Sparsity (%)",
            )
            ax2.set_ylabel("Sparsity Ratio (%)", color=color_sparsity, fontsize=11, labelpad=8)
            ax2.tick_params(axis="y", labelcolor=color_sparsity)
            ax2.set_ylim([0, 100])

            ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

            handles = [bars, lines[0]]
            labels = ["Effective Rank", "Sparsity (%)"]
            ax1.legend(handles, labels, loc="upper left", frameon=True, fontsize=9.5)

            fig.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig
    finally:
        # FIX: Ensure fig.clf() and plt.close(fig) are executed in finally block followed by gc.collect()
        if save_path and fig is not None:
            try:
                fig.clf()
                plt.close(fig)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass
        gc.collect()
