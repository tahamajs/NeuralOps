"""
Rank comparison and model comparison plots – used by the ResultsSaver
when users select "rank_allocation" or "pareto_front" in their save_config.
"""
import os, json, logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def plot_rank_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    task_name: str = "",
    save_path: Optional[str] = None,
    figsize=(10, 6),
    dpi=150,
    **kwargs
) -> Optional[str]:
    """
    Draw accuracy vs. rank for a single task across multiple ranks.

    Args:
        results_dict: { "rank": {"accuracy": 0.85, ...}, ... }
        task_name: used in the title
        save_path: where to write the PNG
    """
    # Filter out the full‑attention entry and convert keys to int
    ranks = []
    accuracies = []
    for k, v in results_dict.items():
        if k.lower() == "none":
            full_acc = v.get("accuracy")
        try:
            r = int(k)
            if "accuracy" in v:
                ranks.append(r)
                accuracies.append(v["accuracy"])
        except ValueError:
            continue

    if not ranks:
        logger.info("No rank data to plot – skipping rank comparison")
        return None

    ranks = np.array(ranks)
    accuracies = np.array(accuracies)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ranks, accuracies, "o-", color="#440154", markersize=8, linewidth=2)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{task_name} – Performance vs Compression Rank")
    ax.grid(True, alpha=0.3)

    if "None" in results_dict and "accuracy" in results_dict["None"]:
        ax.axhline(y=results_dict["None"]["accuracy"], color="red",
                   linestyle="--", label="Full Attention")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return None


def plot_model_comparison(
    all_model_results: Dict[str, Dict[str, Dict[str, Any]]],
    task_name: str = "",
    save_path: Optional[str] = None,
    figsize=(12, 6),
    dpi=150,
    **kwargs
) -> Optional[str]:
    """
    Draw accuracy vs. rank curves for several models on the same task.

    Args:
        all_model_results: { model_name: { "rank": {"accuracy": ...}, ... } }
    """
    colors = ["#440154", "#3B528B", "#21908C", "#5DC963", "#FDE725"]
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=figsize)

    for i, (model_name, results) in enumerate(all_model_results.items()):
        ranks, accuracies = [], []
        for k, v in results.items():
            if k.lower() == "none":
                continue
            try:
                r = int(k)
                if "accuracy" in v:
                    ranks.append(r)
                    accuracies.append(v["accuracy"])
            except ValueError:
                continue
        if ranks:
            ranks = np.array(ranks)
            accuracies = np.array(accuracies)
            ax.plot(ranks, accuracies, marker=markers[i % len(markers)],
                    color=colors[i % len(colors)], linewidth=2, markersize=8,
                    label=model_name)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{task_name} – Model Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return None