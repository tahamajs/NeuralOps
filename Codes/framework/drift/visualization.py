"""Drift visualisation – overlapping histograms + dashboard summary."""

import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .report import DriftReport


def generate_drift_plots(
    report: DriftReport,
    reference_data: Dict[str, np.ndarray] = None,
    current_data: Dict[str, np.ndarray] = None,
    output_dir: str = "./drift_reports",
) -> list:
    """
    Generate publication‑quality drift visualisations.

    Returns list of saved file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Summary bar chart: feature vs drift score (first method)
    if report.features:
        fig, ax = plt.subplots(figsize=(12, 6))
        names = [f.feature for f in report.features]
        # Use the statistic from the first method as representative
        scores = [list(f.methods.values())[0]["statistic"] for f in report.features]
        colors = [
            "#ef4444" if list(f.methods.values())[0]["drifted"] else "#10b981"
            for f in report.features
        ]
        ax.barh(names, scores, color=colors, edgecolor="white")
        ax.axvline(
            x=report.features[0].methods[list(report.features[0].methods.keys())[0]][
                "threshold"
            ],
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Threshold",
        )
        ax.set_xlabel("Drift Score")
        ax.set_title("Feature Drift Summary")
        ax.legend()
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "drift_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # 2. Overlapping histograms for drifted features (top 6)
    if reference_data and current_data:
        drifted_features = [
            f for f in report.features if any(m["drifted"] for m in f.methods.values())
        ]
        top = drifted_features[:6] if drifted_features else report.features[:3]
        n = len(top)
        if n > 0:
            cols = min(3, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(
                rows, cols, figsize=(cols * 5, rows * 4), squeeze=False
            )
            for idx, feat in enumerate(top):
                ax = axes[idx // cols][idx % cols]
                ref = reference_data.get(feat.feature, [])
                cur = current_data.get(feat.feature, [])
                if len(ref) > 0 and len(cur) > 0:
                    ax.hist(ref, bins=30, alpha=0.6, label="Reference", color="#3B528B")
                    ax.hist(cur, bins=30, alpha=0.6, label="Current", color="#FDE725")
                ax.set_title(feat.feature, fontsize=9)
                ax.legend(fontsize=7)
            for idx in range(n, rows * cols):
                axes[idx // cols][idx % cols].set_visible(False)
            fig.suptitle("Drifted Features – Distribution Comparison", fontsize=12)
            plt.tight_layout()
            path = os.path.join(output_dir, "drift_histograms.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    return saved
