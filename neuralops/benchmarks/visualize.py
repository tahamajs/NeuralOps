#!/usr/bin/env python3
"""
Benchmark Results Visualizer and Performance Report Chart Plotter

Modeled after FLA benchmark reporting toolchain.

Author: Taha Majlesi
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("neuralops.benchmarks.visualize")


def plot_operator_latencies(
    benchmark_data: List[Dict[str, Any]],
    output_path: str = "operator_latencies.png",
    title: str = "NeuralOps & FLA Operator Execution Latency",
) -> None:
    """
    Plots a bar chart comparing latency across different operators.
    """
    valid_data = [d for d in benchmark_data if "latency_ms" in d and "error" not in d]
    if not valid_data:
        logger.warning("No valid benchmark records to plot.")
        return

    ops = [f"{d['op']} ({d.get('mode', 'fwd')})" for d in valid_data]
    latencies = [d["latency_ms"] for d in valid_data]
    min_latencies = [d.get("min_ms", d["latency_ms"]) for d in valid_data]
    max_latencies = [d.get("max_ms", d["latency_ms"]) for d in valid_data]

    yerr = [
        [lat - m for lat, m in zip(latencies, min_latencies)],
        [m - lat for lat, m in zip(latencies, max_latencies)],
    ]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.bar(ops, latencies, yerr=yerr, capsize=5, color="#2b5c8f", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=25, ha="right", fontsize=10)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p)
    plt.close(fig)
    logger.info(f"Saved operator latency plot to {output_path}")


def plot_positional_perplexity(
    ppl_data: Dict[str, Any],
    output_path: str = "positional_perplexity.png",
    title: Optional[str] = None,
) -> None:
    """
    Plots positional NLL buckets across the context window.
    """
    bucket_nlls = ppl_data.get("bucket_nlls", [])
    bucket_size = ppl_data.get("bucket_size", 64)
    model = ppl_data.get("model", "Model")
    module = ppl_data.get("module_name", "baseline")

    if not bucket_nlls:
        logger.warning("No bucket NLLs found in data.")
        return

    positions = [(i + 0.5) * bucket_size for i in range(len(bucket_nlls))]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(positions, bucket_nlls, marker="o", color="#d95f02", linewidth=2, label=f"{model} ({module})")

    ax.set_xlabel("Token Position in Context Window", fontsize=12, fontweight="bold")
    ax.set_ylabel("Negative Log-Likelihood (NLL)", fontsize=12, fontweight="bold")
    ax.set_title(title or f"Positional NLL Across Context Window - {model} ({module})", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(frameon=True)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p)
    plt.close(fig)
    logger.info(f"Saved positional perplexity plot to {output_path}")


def plot_scaling_comparison(
    records: List[Dict[str, Any]],
    x_metric: str = "seq_len",
    y_metric: str = "throughput_tokens_per_sec",
    group_by: str = "module_name",
    output_path: str = "scaling_comparison.png",
    title: str = "Throughput Scaling Comparison",
) -> None:
    """
    Plots throughput or latency scaling curves grouped by module name.
    """
    df = pd.DataFrame(records)
    if df.empty or x_metric not in df or y_metric not in df:
        logger.warning("Insufficient data columns for scaling plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    groups = df.groupby(group_by) if group_by in df else [(None, df)]

    for name, group in groups:
        sorted_group = group.sort_values(by=x_metric)
        label = str(name) if name is not None else "Run"
        ax.plot(sorted_group[x_metric], sorted_group[y_metric], marker="s", linewidth=2.5, label=label)

    ax.set_xlabel(x_metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_ylabel(y_metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(frameon=True)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p)
    plt.close(fig)
    logger.info(f"Saved scaling comparison plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralOps Benchmark Visualizer")
    parser.add_argument("--input", type=str, required=True, help="Path to benchmark JSON results")
    parser.add_argument("--output", type=str, default="benchmark_plot.png", help="Output plot destination")
    parser.add_argument("--type", type=str, default="ops", choices=["ops", "ppl", "scaling"], help="Plot type")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.type == "ops":
        plot_operator_latencies(data, output_path=args.output)
    elif args.type == "ppl":
        plot_positional_perplexity(data, output_path=args.output)
    elif args.type == "scaling":
        plot_scaling_comparison(data, output_path=args.output)


if __name__ == "__main__":
    main()
