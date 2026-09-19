#!/usr/bin/env python3
"""
Sequence-Length Complexity Scaling Plotter.

Generates log-log scaling curves illustrating memory footprint (Peak VRAM) and
latency as sequence length scales from 128 to 4096 tokens, contrasting O(N) linear
attention vs O(N^2) quadratic softmax baselines.

Usage:
    python plot_seqlen_scaling.py --input_csv results/seqlen_scaling/summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_scaling(df: pd.DataFrame, out_path: Path):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    modules = df["module"].unique()
    colors = ["#000000", "#D0021B", "#4A90E2", "#7ED321", "#9013FE", "#F5A623"]

    for idx, mod in enumerate(modules):
        mod_df = df[(df["module"] == mod) & (df["status"] == "PASS")].sort_values(by="seq_len")
        if mod_df.empty:
            continue
        c = colors[idx % len(colors)]
        style = "--" if mod == "dense" else "-"
        marker = "s" if mod == "dense" else "o"
        lw = 2.0 if mod in ("dense", "a3", "svd-mamba") else 1.2

        # 1. Peak VRAM vs Sequence Length (Log-Log)
        ax1.plot(mod_df["seq_len"], mod_df["peak_vram_mb"], label=mod, color=c,
                 linestyle=style, marker=marker, linewidth=lw)

        # 2. Latency vs Sequence Length (Log-Log)
        ax2.plot(mod_df["seq_len"], mod_df["latency_ms"], label=mod, color=c,
                 linestyle=style, marker=marker, linewidth=lw)

    # Reference theoretical lines
    seqlens = [128, 512, 1024, 2048, 4096]
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Sequence Length $N$ (Tokens)")
    ax1.set_ylabel("Peak VRAM (MB)")
    ax1.set_title("Memory Complexity Scaling: $O(N)$ vs $O(N^2)$")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Sequence Length $N$ (Tokens)")
    ax2.set_ylabel("Latency per Batch (ms)")
    ax2.set_title("Runtime Latency Scaling")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Scaling plot saved to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot sequence length scaling analysis")
    parser.add_argument("--input_csv", type=str, default="results/seqlen_scaling/summary.csv",
                        help="Path to scaling summary.csv")
    parser.add_argument("--output_file", type=str, default="paper/iclr2027-submission/figures/seqlen_scaling.png",
                        help="Destination PNG file path")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        print(f"[NOTE] {csv_path} not found. Generating template scaling placeholder chart.")
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "Sequence-Length Scaling Curve (Log-Log)\n(Awaiting Benchmark Sweep Execution)",
                horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,
                fontsize=12, color="gray")
        ax.set_xlabel("Sequence Length (Tokens)")
        ax.set_ylabel("Peak VRAM / Latency")
        ax.set_title("Theoretical Complexity Scaling")
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_file)
        plt.close(fig)
        print(f"[PLOT] Placeholder chart saved to: {args.output_file}")
        return

    df = pd.read_csv(csv_path)
    plot_scaling(df, Path(args.output_file))


if __name__ == "__main__":
    main()
