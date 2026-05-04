"""
Visualization tools for SVD‑based compression analysis.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import subspace_angles
from .base import BaseVisualizer, VISUALIZER_REGISTRY
from .utils import create_custom_colormap, set_plot_style


@VISUALIZER_REGISTRY.register("svd_analysis")
class SVDAttentionVisualizer(BaseVisualizer):
    """
    Visualizer for SVD‑compressed attention matrices.
    Provides singular value spectra, cumulative energy, subspace alignment,
    and comparative plots across ranks and models.
    """

    def __init__(self, output_dir="visualizations"):
        super().__init__(output_dir)
        self.cmap = create_custom_colormap()

    def plot_singular_values(self, attention_matrix, rank, title_prefix="", save=True):
        """Plot singular value spectrum and cumulative energy."""
        U, s, Vt = np.linalg.svd(attention_matrix, full_matrices=False)
        cum_energy = np.cumsum(s**2) / np.sum(s**2)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(s, "o-", color="#3B528B", markersize=4, linewidth=1.5)
        axes[0].axvline(x=rank, color="red", linestyle="--", label=f"Rank {rank}")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Singular Value")
        axes[0].set_title("Singular Value Spectrum")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(cum_energy, "o-", color="#21908C", markersize=4, linewidth=1.5)
        axes[1].axhline(y=0.95, color="green", linestyle="--", label="95% Energy")
        axes[1].axvline(x=rank, color="red", linestyle="--", label=f"Rank {rank}")
        axes[1].set_xlabel("Number of SV")
        axes[1].set_ylabel("Cumulative Energy")
        axes[1].set_title("Cumulative Energy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(title_prefix)
        plt.tight_layout()
        if save:
            return self._save_figure(fig, f"svd_spectrum_{title_prefix}.png")
        return fig

    def subspace_alignment(self, full_attn, compressed_attn):
        """Compute principal angles between subspaces."""
        Uf, _, _ = np.linalg.svd(full_attn, full_matrices=False)
        Uc, _, _ = np.linalg.svd(compressed_attn, full_matrices=False)
        min_dim = min(Uf.shape[1], Uc.shape[1])
        angles = subspace_angles(Uf[:, :min_dim], Uc[:, :min_dim])
        return np.degrees(angles)

    def plot_subspace_alignment(self, angles, rank, title_prefix="", save=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(angles, "o-", color="#5DC963", markersize=6, linewidth=1.5)
        ax.set_xlabel("Principal Angle Index")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(f"{title_prefix} - Subspace Alignment (Rank {rank})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            return self._save_figure(fig, f"subspace_alignment_{title_prefix}_rank{rank}.png")
        return fig

    def plot_rank_comparison(self, results_dict, task_name, save=True):
        """Plot accuracy vs rank from results dict."""
        ranks = []
        accs = []
        for r, v in results_dict.items():
            if r != "None" and "accuracy" in v:
                ranks.append(int(r))
                accs.append(v["accuracy"])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ranks, accs, "o-", color="#440154", markersize=8, linewidth=2)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task_name} - Performance vs Rank")
        ax.grid(True, alpha=0.3)
        if "None" in results_dict and "accuracy" in results_dict["None"]:
            ax.axhline(y=results_dict["None"]["accuracy"], color="red", linestyle="--", label="Full Attention")
            ax.legend()
        plt.tight_layout()
        if save:
            return self._save_figure(fig, f"rank_comparison_{task_name}.png")
        return fig

    def plot_model_comparison(self, all_model_results, task_name, save=True):
        """Compare multiple models on one task."""
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#440154", "#3B528B", "#21908C", "#5DC963", "#FDE725"]
        markers = ["o", "s", "^", "D", "v"]
        for i, (model, results) in enumerate(all_model_results.items()):
            ranks = [int(r) for r in results.keys() if r != "None"]
            accs = [results[str(r)]["accuracy"] for r in ranks if str(r) in results]
            ax.plot(ranks, accs, marker=markers[i % len(markers)], color=colors[i % len(colors)],
                    linewidth=2, markersize=8, label=model)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task_name} - Model Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            return self._save_figure(fig, f"model_comparison_{task_name}.png")
        return fig

    def generate_training_plots(self, training_logs, model_name, task_name, rank, save=True):
        """Plot training curves."""
        epochs = [log["epoch"] for log in training_logs]
        losses = [log["loss"] for log in training_logs]
        eval_epochs = [log["epoch"] for log in training_logs if "eval_score" in log]
        eval_scores = [log["eval_score"] for log in training_logs if "eval_score" in log]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, losses, "o-", color="#3B528B", markersize=4, linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss Curve")
        axes[0].grid(True, alpha=0.3)
        if eval_scores:
            axes[1].plot(eval_epochs, eval_scores, "o-", color="#21908C", markersize=6, linewidth=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Eval Score")
            axes[1].set_title("Validation Performance")
            axes[1].grid(True, alpha=0.3)
        fig.suptitle(f"{model_name} - {task_name} (Rank {rank})")
        plt.tight_layout()
        if save:
            return self._save_figure(fig, f"training_curves_{model_name}_{task_name}_rank{rank}.png")
        return fig