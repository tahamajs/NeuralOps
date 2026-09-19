#!/usr/bin/env python3
"""
Pareto Frontier Analysis and Plotting for Attention Compression.

Generates 4 publication-quality 300-DPI Pareto frontier charts:
  1. Accuracy vs Compression Ratio (Speedup / Parameter reduction)
  2. Accuracy vs Peak VRAM (Memory footprint)
  3. Accuracy vs Throughput (Tokens per second)
  4. Perplexity vs Compression (for decoder runs)

Also identifies Pareto-optimal points across multi-objective spaces and
outputs formatted markdown/LaTeX tables.

Usage:
    python plot_pareto.py --input_csv results/unified_sweep/distilbert_mnli_full/summary.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pareto_front(df: pd.DataFrame, objectives: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Identifies Pareto-optimal points across arbitrary multi-objective dimensions.

    Args:
        df: Input DataFrame containing evaluated methods.
        objectives: List of tuples (column_name, 'max' or 'min').
                    'max': higher is better (e.g. accuracy, throughput)
                    'min': lower is better (e.g. VRAM, latency, perplexity)

    Returns:
        Subset of df representing the non-dominated Pareto frontier.
    """
    if df.empty:
        return df

    # Normalize criteria so that larger values are always strictly better
    norm_vals = []
    for col, direction in objectives:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        if direction.lower() == "min":
            vals = -vals
        norm_vals.append(vals)

    if not norm_vals:
        return df

    points = np.stack(norm_vals, axis=1)
    is_pareto = np.ones(points.shape[0], dtype=bool)

    for i, c in enumerate(points):
        if is_pareto[i]:
            # Point c is dominated if another point p is >= in all dims and > in at least one
            is_pareto[is_pareto] = ~np.all(points[is_pareto] <= c, axis=1) | np.all(points[is_pareto] == c, axis=1)
            is_pareto[i] = True

    return df.iloc[is_pareto].copy()


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_accuracy_vs_compression(df: pd.DataFrame, out_path: Path):
    """Figure 1: Accuracy (%) vs Compression Rate (X)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x_col = "compression_rate"
    y_col = "accuracy_pct"

    df_clean = df.dropna(subset=[x_col, y_col]).copy()
    df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors="coerce")
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[x_col, y_col])

    # Compute 2D Pareto front
    p_front = pareto_front(df_clean, [(x_col, "max"), (y_col, "max")])
    p_front = p_front.sort_values(by=x_col)

    # Plot all models
    ax.scatter(df_clean[x_col], df_clean[y_col], c="#4A90E2", alpha=0.6, s=60, edgecolors="none", label="Evaluated Modules")

    # Plot Pareto line and points
    ax.plot(p_front[x_col], p_front[y_col], color="#D0021B", linestyle="--", linewidth=1.5, label="Pareto Frontier")
    ax.scatter(p_front[x_col], p_front[y_col], c="#D0021B", s=90, edgecolors="black", linewidths=1.0, zorder=5)

    # Annotate Pareto points
    for _, row in p_front.iterrows():
        ax.annotate(
            row["module_name"],
            (row[x_col], row[y_col]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="semibold",
        )

    ax.set_xlabel("Compression Rate (x)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy vs Compression Rate")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path}")


def plot_accuracy_vs_vram(df: pd.DataFrame, out_path: Path):
    """Figure 2: Accuracy (%) vs Peak VRAM (MB)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x_col = "peak_vram_mb"
    y_col = "accuracy_pct"

    df_clean = df.dropna(subset=[x_col, y_col]).copy()
    df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors="coerce")
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
    df_clean = df_clean[df_clean[x_col] > 0.0]

    p_front = pareto_front(df_clean, [(x_col, "min"), (y_col, "max")])
    p_front = p_front.sort_values(by=x_col)

    ax.scatter(df_clean[x_col], df_clean[y_col], c="#50E3C2", alpha=0.7, s=60, edgecolors="#1B806A", label="Evaluated Modules")
    ax.plot(p_front[x_col], p_front[y_col], color="#BD10E0", linestyle="--", linewidth=1.5, label="Pareto Frontier")
    ax.scatter(p_front[x_col], p_front[y_col], c="#BD10E0", s=90, edgecolors="black", linewidths=1.0, zorder=5)

    for _, row in p_front.iterrows():
        ax.annotate(
            row["module_name"],
            (row[x_col], row[y_col]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="semibold",
        )

    ax.set_xlabel("Peak VRAM (MB)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy vs Peak Memory Footprint")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path}")


def plot_accuracy_vs_throughput(df: pd.DataFrame, out_path: Path):
    """Figure 3: Accuracy (%) vs Throughput (Tokens / sec)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x_col = "tokens_per_sec"
    y_col = "accuracy_pct"

    df_clean = df.dropna(subset=[x_col, y_col]).copy()
    df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors="coerce")
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
    df_clean = df_clean[df_clean[x_col] > 0.0]

    p_front = pareto_front(df_clean, [(x_col, "max"), (y_col, "max")])
    p_front = p_front.sort_values(by=x_col)

    ax.scatter(df_clean[x_col], df_clean[y_col], c="#F5A623", alpha=0.7, s=60, edgecolors="#A86E10", label="Evaluated Modules")
    ax.plot(p_front[x_col], p_front[y_col], color="#4A90E2", linestyle="--", linewidth=1.5, label="Pareto Frontier")
    ax.scatter(p_front[x_col], p_front[y_col], c="#4A90E2", s=90, edgecolors="black", linewidths=1.0, zorder=5)

    for _, row in p_front.iterrows():
        ax.annotate(
            row["module_name"],
            (row[x_col], row[y_col]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="semibold",
        )

    ax.set_xlabel("Throughput (Tokens / sec)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy vs Inference Throughput")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path}")


def plot_ppl_vs_compression(df: pd.DataFrame, out_path: Path):
    """Figure 4: Perplexity vs Compression (Lower PPL is better)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x_col = "compression_rate"
    y_col = "perplexity"

    if y_col not in df.columns or df[y_col].dropna().empty:
        # Generate placeholder chart if perplexity data not yet generated
        ax.text(0.5, 0.5, "Perplexity vs Compression Frontier\n(Awaiting Decoder Sweep Execution)",
                horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,
                fontsize=12, color="gray")
        ax.set_xlabel("Compression Rate (x)")
        ax.set_ylabel("Perplexity (Lower is Better)")
        ax.set_title("GPT-2 Language Modeling Pareto Curve")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[PLOT] Saved placeholder: {out_path}")
        return

    df_clean = df.dropna(subset=[x_col, y_col]).copy()
    df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors="coerce")
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
    df_clean = df_clean[(df_clean[y_col] > 0.0) & (df_clean[y_col] < 1000.0)]

    p_front = pareto_front(df_clean, [(x_col, "max"), (y_col, "min")])
    p_front = p_front.sort_values(by=x_col)

    ax.scatter(df_clean[x_col], df_clean[y_col], c="#7ED321", alpha=0.7, s=60, edgecolors="#417505", label="Evaluated Modules")
    ax.plot(p_front[x_col], p_front[y_col], color="#E60000", linestyle="--", linewidth=1.5, label="Pareto Frontier")
    ax.scatter(p_front[x_col], p_front[y_col], c="#E60000", s=90, edgecolors="black", linewidths=1.0, zorder=5)

    for _, row in p_front.iterrows():
        ax.annotate(
            row["module_name"],
            (row[x_col], row[y_col]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="semibold",
        )

    ax.set_xlabel("Compression Rate (x)")
    ax.set_ylabel("Perplexity (WikiText-2)")
    ax.set_title("Perplexity vs Compression Rate (GPT-2)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path}")


def plot_all(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    plot_accuracy_vs_compression(df, out_dir / "pareto_accuracy_vs_compression.png")
    plot_accuracy_vs_vram(df, out_dir / "pareto_accuracy_vs_vram.png")
    plot_accuracy_vs_throughput(df, out_dir / "pareto_accuracy_vs_throughput.png")
    plot_ppl_vs_compression(df, out_dir / "pareto_ppl_vs_compression.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot Pareto frontiers for attention compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_csv", type=str, default="results/unified_sweep/distilbert_mnli_full/summary.csv",
                        help="Path to summary.csv benchmark results")
    parser.add_argument("--output_dir", type=str, default="paper/iclr2027-submission/figures",
                        help="Directory to save generated 300 DPI figures")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        print(f"[ERROR] Input CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df_pass = df[df["status"] == "PASS"].copy()
    print(f"[INFO] Loaded {len(df_pass)} PASS modules from {csv_path}")

    out_dir = Path(args.output_dir)
    plot_all(df_pass, out_dir)

    # Compute 4-objective Pareto optimal set: Accuracy (max), Compression (max), VRAM (min), Throughput (max)
    objectives = [
        ("accuracy_pct", "max"),
        ("compression_rate", "max"),
        ("peak_vram_mb", "min"),
        ("tokens_per_sec", "max"),
    ]
    p_optimal = pareto_front(df_pass, objectives)
    print("\n[TABLE 8: PARETO OPTIMAL MODULES]")
    cols_show = ["module_name", "accuracy_pct", "compression_rate", "peak_vram_mb", "tokens_per_sec"]
    print(p_optimal[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
