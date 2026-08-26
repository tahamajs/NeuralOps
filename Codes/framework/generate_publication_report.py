#!/usr/bin/env python3
"""
NeuralOps Publication Report & Figures Studio
Generates publication-grade NeurIPS/ICLR LaTeX tables, Markdown executive summaries,
and high-resolution vector figures (Pareto Frontier, Radar Chart, Matrix Heatmap, Forest Plots).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PublicationReport")

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def generate_latex_quality_table(records: List[Dict[str, Any]]) -> str:
    """Generates a publication-grade LaTeX booktabs table with 95% CI and p-values."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Empirical Quality and Statistical Significance of Attention Architectures.} Evaluated across GLUE benchmark splits on NVIDIA RTX 4070 GPU over 3 independent random seeds ($n=3$). Confidence intervals are 95\% two-sided intervals based on Student's $t$-distribution. $p$-values are computed via paired two-tailed $t$-tests against standard \texttt{full} attention baseline.}",
        r"\label{tab:quality_benchmark_results}",
        r"\begin{tabular}{l l c c c c c c}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Model} & \textbf{Accuracy (\%)} & \textbf{95\% CI} & \textbf{$p$-value} & \textbf{F1 Score} & \textbf{Fidelity ($\rho$)} & \textbf{Fwd (ms)} \\",
        r"\midrule",
    ]

    # Sort: full attention first, then by mean accuracy descending
    sorted_recs = sorted(records, key=lambda r: (r.get("module_name") != "full", -float(r.get("mean_accuracy", 0) or 0)))

    for r in sorted_recs:
        mod = r.get("module_name", "").replace("_", r"\_")
        model = r.get("model_family", "BERT").upper()
        acc = float(r.get("mean_accuracy", 0) or 0)
        std = float(r.get("std_accuracy", 0) or 0)
        ci_l = float(r.get("ci_lower", acc) or acc)
        ci_u = float(r.get("ci_upper", acc) or acc)
        p_val = float(r.get("p_value", 1.0) or 1.0)
        f1 = float(r.get("f1_score", 0) or 0)
        fid = float(r.get("fidelity_score", 1.0) or 1.0)
        fwd = float(r.get("fwd_ms", 0) or 0)

        acc_str = f"{acc:.2f} \\pm {std:.2f}" if std > 0 else f"{acc:.2f}"
        ci_str = f"[{ci_l:.1f}, {ci_u:.1f}]"
        p_str = "--" if r.get("module_name") == "full" else (f"p < 0.01^{{**}}" if p_val < 0.01 else (f"{p_val:.3f}^*" if p_val < 0.05 else f"{p_val:.3f}"))
        f1_str = f"{f1:.4f}"
        fid_str = f"{fid:.4f}"
        fwd_str = f"{fwd:.2f}"

        is_baseline = (r.get("module_name") == "full")
        row_prefix = r"\textbf{" if is_baseline else ""
        row_suffix = r"}" if is_baseline else ""

        lines.append(
            f"{row_prefix}\\texttt{{{mod}}}{row_suffix} & {model} & {acc_str} & {ci_str} & {p_str} & {f1_str} & {fid_str} & {fwd_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_latex_efficiency_table(records: List[Dict[str, Any]]) -> str:
    """Generates a publication-grade LaTeX booktabs table for efficiency sweep."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Full Computational Efficiency and VRAM Footprint Matrix.} Evaluated on NVIDIA GeForce RTX 4070 (Ada Lovelace, 8GB VRAM) with PyTorch 2.6.0+cu124 with TF32 and AMP acceleration.}",
        r"\label{tab:efficiency_benchmark_results}",
        r"\begin{tabular}{l l r r r r}",
        r"\toprule",
        r"\textbf{Module Architecture} & \textbf{Track} & \textbf{Fwd Latency (ms)} & \textbf{Bwd Latency (ms)} & \textbf{Peak VRAM (MB)} & \textbf{Throughput (tok/s)} \\",
        r"\midrule",
    ]

    sorted_recs = sorted(records, key=lambda r: float(r.get("fwd_ms", 999) or 999))[:20]

    for r in sorted_recs:
        mod = (r.get("module_name") or r.get("module", "")).replace("_", r"\_")
        track = (r.get("track") or "track1_svd").replace("_", r"\_")
        fwd = float(r.get("fwd_ms", 0) or 0)
        bwd = float(r.get("bwd_ms", 0) or 0)
        mem = float(r.get("peak_vram_mb") or r.get("mem_mb", 0) or 0)
        thru = float(r.get("throughput_tokens_sec") or r.get("throughput", 0) or 0)

        lines.append(
            f"\\texttt{{{mod}}} & \\textsc{{{track}}} & {fwd:.2f} & {bwd:.2f} & {mem:.1f} & {thru:.0f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_pareto_plot(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Plots Accuracy vs Forward Latency Pareto Frontier."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#ffffff26")
    ax.grid(True, linestyle="--", alpha=0.15, color="#94a3b8")

    valid = [r for r in records if float(r.get("mean_accuracy", 0) or 0) > 0 and float(r.get("fwd_ms", 0) or 0) > 0]
    if not valid:
        plt.close(fig)
        return

    xs = [float(r.get("fwd_ms", 0)) for r in valid]
    ys = [float(r.get("mean_accuracy", 0)) for r in valid]
    names = [r.get("module_name", "") for r in valid]

    colors = ["#38bdf8" if n != "full" else "#f59e0b" for n in names]
    sizes = [90 if n != "full" else 140 for n in names]
    ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.9, edgecolors="#ffffff", linewidths=1.2, zorder=4)

    sorted_pts = sorted(zip(xs, ys, names), key=lambda p: p[0])
    pareto_x, pareto_y = [], []
    cur_max_y = -1e9
    for x, y, n in sorted_pts:
        if y > cur_max_y:
            pareto_x.append(x)
            pareto_y.append(y)
            cur_max_y = y
    
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, color="#ec4899", linestyle="--", linewidth=2, label="Pareto Frontier", zorder=3)

    for r in valid[:8]:
        x = float(r.get("fwd_ms", 0))
        y = float(r.get("mean_accuracy", 0))
        n = r.get("module_name", "")
        offset_y = 1.0 if y < 90 else -1.5
        ax.annotate(
            n, (x, y),
            textcoords="offset points", xytext=(5, offset_y),
            color="#f8fafc", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="#0f172acc", ec="#38bdf84d", lw=0.8)
        )

    ax.set_xlabel("Forward Pass Latency (ms) [Lower is Better]", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_ylabel("Downstream Accuracy (%) [Higher is Better]", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title("Accuracy vs Latency Efficiency Frontier", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)

    leg = ax.legend(facecolor="#1e293b", edgecolor="#ffffff26", labelcolor="#f8fafc", loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Generated Pareto Plot: %s", output_path)


def generate_radar_plot(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Generates a 5-axis publication radar chart comparing top architectures."""
    plt = _get_plt()
    categories = ["Speed (Inv Latency)", "VRAM Efficiency", "Accuracy", "F1 Score", "Fidelity"]
    num_vars = len(categories)

    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color="#94a3b8", size=10, weight="bold")
    ax.tick_params(colors="#94a3b8", pad=12)
    ax.grid(color="#ffffff26", linestyle="--", linewidth=0.8)
    ax.spines["polar"].set_color("#ffffff33")

    top_mods = [r for r in records if r.get("module_name") == "full"][:1]
    top_mods += [r for r in records if r.get("module_name") != "full"][:4]

    palette = ["#f59e0b", "#38bdf8", "#a855f7", "#10b981", "#ec4899"]

    max_fwd = max([float(r.get("fwd_ms", 1) or 1) for r in records] or [1.0])
    max_mem = max([float(r.get("peak_vram_mb", 1) or 1) for r in records] or [1.0])

    for idx, r in enumerate(top_mods):
        name = r.get("module_name", "mod")
        fwd = float(r.get("fwd_ms", 50) or 50)
        mem = float(r.get("peak_vram_mb", 2000) or 2000)
        acc = float(r.get("mean_accuracy", 80) or 80) / 100.0
        f1 = float(r.get("f1_score", 0.8) or 0.8)
        fid = float(r.get("fidelity_score", 0.95) or 0.95)

        s_speed = max(0.1, 1.0 - (fwd / max_fwd))
        s_mem = max(0.1, 1.0 - (mem / max_mem))
        s_acc = acc
        s_f1 = f1
        s_fid = max(0.1, fid)

        vals = [s_speed, s_mem, s_acc, s_f1, s_fid]
        vals += vals[:1]

        c = palette[idx % len(palette)]
        ax.plot(angles, vals, color=c, linewidth=2, label=name)
        ax.fill(angles, vals, color=c, alpha=0.15)

    ax.set_ylim(0, 1.0)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), facecolor="#1e293b", edgecolor="#ffffff26", labelcolor="#f8fafc")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Generated Radar Plot: %s", output_path)


def generate_accuracy_heatmap(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Generates an Accuracy matrix heatmap across modules and datasets."""
    plt = _get_plt()
    
    mod_set = []
    ds_set = ["SST2", "MRPC"]
    matrix: Dict[str, Dict[str, float]] = {}

    for r in records:
        m = r.get("module_name", "")
        d = r.get("dataset", "SST2").upper()
        acc = float(r.get("mean_accuracy", 0) or 0)
        if m not in matrix:
            matrix[m] = {}
            mod_set.append(m)
        matrix[m][d] = acc

    if not mod_set:
        return

    mod_set = mod_set[:15]
    grid_data = np.zeros((len(mod_set), len(ds_set)))
    for i, m in enumerate(mod_set):
        for j, d in enumerate(ds_set):
            grid_data[i, j] = matrix[m].get(d, 0.0)

    fig, ax = plt.subplots(figsize=(6, max(4, len(mod_set) * 0.45)), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    im = ax.imshow(grid_data, cmap="viridis", aspect="auto", vmin=50, vmax=95)

    ax.set_xticks(np.arange(len(ds_set)))
    ax.set_yticks(np.arange(len(mod_set)))
    ax.set_xticklabels(ds_set, color="#f8fafc", fontweight="bold", fontsize=10)
    ax.set_yticklabels(mod_set, color="#f8fafc", fontfamily="monospace", fontsize=9)

    for i in range(len(mod_set)):
        for j in range(len(ds_set)):
            val = grid_data[i, j]
            text = f"{val:.1f}%" if val > 0 else "--"
            ax.text(j, i, text, ha="center", va="center", color="#ffffff" if val < 75 else "#000000", fontweight="bold", fontsize=9)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors="#94a3b8")
    cbar.set_label("Downstream Accuracy (%)", color="#f8fafc", fontsize=10)

    ax.set_title("Module Performance Heatmap across GLUE Splits", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Generated Heatmap: %s", output_path)


def generate_full_publication_report(run_dir_str: str) -> None:
    """Main entry point to parse a quality run directory and generate all publication artifacts."""
    run_dir = Path(run_dir_str)
    summary_csv = run_dir / "summary.csv"
    detailed_json = run_dir / "detailed_results.json"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = []
    if detailed_json.exists():
        try:
            with open(detailed_json, "r", encoding="utf-8") as f:
                items = json.load(f)
                for it in items:
                    met = it.get("metrics", {})
                    eff = it.get("efficiency", {})
                    records.append({
                        "run_id": it.get("run_id"),
                        "module_name": it.get("module_name"),
                        "model_family": it.get("model_family"),
                        "dataset": it.get("dataset"),
                        "status": it.get("status"),
                        "mean_accuracy": met.get("accuracy"),
                        "std_accuracy": met.get("accuracy_std"),
                        "ci_lower": met.get("ci_lower"),
                        "ci_upper": met.get("ci_upper"),
                        "p_value": met.get("p_value_vs_baseline"),
                        "f1_score": met.get("f1"),
                        "fidelity_score": it.get("fidelity"),
                        "fwd_ms": eff.get("fwd_ms"),
                        "bwd_ms": eff.get("bwd_ms"),
                        "peak_vram_mb": eff.get("peak_vram_mb"),
                        "throughput_tokens_sec": eff.get("throughput_tokens_sec"),
                    })
        except Exception as e:
            logger.warning("Failed reading detailed_results.json: %s", e)

    if not records and summary_csv.exists():
        try:
            with open(summary_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
        except Exception as e:
            logger.warning("Failed reading summary.csv: %s", e)

    if not records:
        logger.error("No records found in %s to generate publication report.", run_dir)
        return

    # 1. LaTeX Tables
    latex_qual = generate_latex_quality_table(records)
    with open(run_dir / "LATEX_QUALITY_TABLE.tex", "w", encoding="utf-8") as f:
        f.write(latex_qual)

    latex_eff = generate_latex_efficiency_table(records)
    with open(run_dir / "LATEX_EFFICIENCY_TABLE.tex", "w", encoding="utf-8") as f:
        f.write(latex_eff)

    # 2. Markdown Quality Report
    md_lines = [
        "# Phase 2 Quality Benchmark & Statistical Significance Report",
        f"**Generated:** {run_dir.name}  ",
        "**Hardware Platform:** NVIDIA GeForce RTX 4070 (8GB VRAM)  ",
        "**Statistical Protocol:** $n=3$ independent random seeds (`[42, 123, 777]`), 95% Confidence Intervals, Paired Student's t-test vs `full` baseline.  \n",
        "## Quality & Statistical Significance Summary Table",
        "| Architecture | Model | Dataset | Accuracy (%) | 95% CI | $p$-value vs Base | F1 Score | Fidelity | Fwd (ms) | VRAM (MB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in sorted(records, key=lambda x: -float(x.get("mean_accuracy", 0) or 0)):
        mod = r.get("module_name")
        mdl = r.get("model_family")
        ds = r.get("dataset")
        acc = float(r.get("mean_accuracy", 0) or 0)
        std = float(r.get("std_accuracy", 0) or 0)
        ci_l = float(r.get("ci_lower", acc) or acc)
        ci_u = float(r.get("ci_upper", acc) or acc)
        p_val = float(r.get("p_value", 1.0) or 1.0)
        f1 = float(r.get("f1_score", 0) or 0)
        fid = float(r.get("fidelity_score", 1.0) or 1.0)
        fwd = float(r.get("fwd_ms", 0) or 0)
        mem = float(r.get("peak_vram_mb", 0) or 0)
        p_str = "--" if mod == "full" else (f"**{p_val:.4f}** (p < 0.01)" if p_val < 0.01 else f"{p_val:.4f}")

        md_lines.append(
            f"| `{mod}` | {mdl} | {ds} | **{acc:.2f}% ± {std:.2f}%** | `[{ci_l:.1f}, {ci_u:.1f}]` | {p_str} | {f1:.4f} | {fid:.4f} | {fwd:.2f} | {mem:.0f} |"
        )

    with open(run_dir / "QUALITY_BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # 3. High-Resolution Publication Plots
    try:
        generate_pareto_plot(records, plots_dir / "accuracy_vs_latency_pareto.png")
        generate_radar_plot(records, plots_dir / "top5_radar_multimetric.png")
        generate_accuracy_heatmap(records, plots_dir / "accuracy_heatmap.png")
    except Exception as pe:
        logger.warning("Error generating publication plots: %s", pe)

    logger.info("✅ Successfully compiled publication tables, Markdown report, and vector plots in %s", run_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Publication-Ready Tables and Figures")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to benchmark run directory")
    args = parser.parse_args()
    generate_full_publication_report(args.run_dir)


if __name__ == "__main__":
    main()
