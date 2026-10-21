#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Sweep Publication Artifacts Generator

Pipeline compiling sweep results into publication-ready tables, figures, and statistical summaries.
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_sweep_artifacts")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

FAMILY_COLORS = {
    "Full Attention Baseline": "#2b5c8f",
    "State-Space Model (SSM)": "#d95f02",
    "Linear / Kernel Attention": "#7570b3",
    "Low-Rank / SVD Attention": "#e7298a",
    "Sparse / Chunked Attention": "#66a61e",
    "Gated / Recurrent Attention": "#e6ab02",
    "SVD Factorization Compressor": "#1b9e77",
    "Structured Pruning Compressor": "#a6761d",
    "Quantization / Rotational Compressor": "#666666",
    "Tensor Decomposition Compressor": "#1f78b4",
    "Other": "#999999",
}


def classify_module(name: str, path: str = "") -> tuple[str, str]:
    """Classify module name into primary type and fine-grained architectural sub-family."""
    name_lower = name.lower()

    if name_lower == "full":
        return "Attention", "Full Attention Baseline"

    is_compressor = any(
        kw in name_lower
        for kw in [
            "compress",
            "prun",
            "shed",
            "slicegpt",
            "quik",
            "laser",
            "whiten",
            "tensor_train",
            "kronecker_linear",
        ]
    ) or (
        name_lower
        in {
            "aasvd",
            "asvd",
            "a3",
            "hybrid_svd",
            "hybridkv",
            "liftr",
            "replaceme",
            "svd_linear",
        }
    )

    if is_compressor:
        if any(k in name_lower for k in ["prun", "shed", "replaceme"]):
            return "Compressor", "Structured Pruning Compressor"
        elif any(k in name_lower for k in ["quant", "slicegpt", "int4", "int8", "rotat"]):
            return "Compressor", "Quantization / Rotational Compressor"
        elif any(k in name_lower for k in ["kronecker", "tensor", "tt_"]):
            return "Compressor", "Tensor Decomposition Compressor"
        else:
            return "Compressor", "SVD Factorization Compressor"

    if any(k in name_lower for k in ["mamba", "ssm", "s4", "s5", "h3", "rwkv", "retnet", "hyena"]):
        return "Attention", "State-Space Model (SSM)"
    elif any(
        k in name_lower for k in ["linear", "kernel", "cosformer", "performer", "reformer", "favor"]
    ):
        return "Attention", "Linear / Kernel Attention"
    elif any(k in name_lower for k in ["svd", "lora", "low_rank", "rank", "aegis", "lola"]):
        return "Attention", "Low-Rank / SVD Attention"
    elif any(
        k in name_lower
        for k in [
            "sparse",
            "chunk",
            "block",
            "local",
            "sliding",
            "longformer",
            "bigbird",
        ]
    ):
        return "Attention", "Sparse / Chunked Attention"
    elif any(k in name_lower for k in ["gate", "recurrent", "delta", "gla", "hgrn"]):
        return "Attention", "Gated / Recurrent Attention"
    else:
        return "Attention", "Other"


def load_and_enrich_summary(summary_csv: Path) -> pd.DataFrame:
    """Read summary.csv and enrich with classifications and derived metrics."""
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if df.empty:
        return df

    df.columns = [c.strip() for c in df.columns]
    required = {"module_name", "status", "accuracy_pct"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"Summary file is missing required result columns: {', '.join(missing)}"
        )
    for col in ("error", "path"):
        if col not in df.columns:
            df[col] = ""
    if "elapsed_s" not in df.columns:
        df["elapsed_s"] = np.nan

    df["accuracy_pct"] = pd.to_numeric(df["accuracy_pct"], errors="coerce")
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")

    classifications = [
        classify_module(row["module_name"], str(row.get("path", ""))) for _, row in df.iterrows()
    ]
    df["module_type"] = [c[0] for c in classifications]
    df["subfamily"] = [c[1] for c in classifications]

    full_baseline = df[df["module_name"] == "full"]
    baseline_acc = (
        float(full_baseline["accuracy_pct"].iloc[0])
        if not full_baseline.empty
        else np.nan
    )
    df["baseline_acc"] = baseline_acc
    df["delta_acc"] = df["accuracy_pct"] - baseline_acc
    df["retention_pct"] = (
        (df["accuracy_pct"] / baseline_acc) * 100.0
        if np.isfinite(baseline_acc) and baseline_acc != 0
        else np.nan
    )

    return df


def plot_fig1_top_modules(df: pd.DataFrame, out_dir: Path, top_n: int = 25):
    """Plot top performing modules vs full baseline."""
    passing = df[df["status"] == "PASS"].sort_values("accuracy_pct", ascending=True)
    if passing.empty:
        logger.warning("No passing modules found for Fig 1.")
        return

    subset = passing.tail(top_n)
    baseline_acc = subset["baseline_acc"].iloc[0] if "baseline_acc" in subset else np.nan

    fig, ax = plt.subplots(figsize=(10, max(6, len(subset) * 0.32)), tight_layout=True)

    colors = [FAMILY_COLORS.get(sub, "#2b5c8f") for sub in subset["subfamily"]]
    bars = ax.barh(
        subset["module_name"],
        subset["accuracy_pct"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        height=0.68,
    )

    if pd.notna(baseline_acc):
        ax.axvline(
            baseline_acc,
            color="#d95f02",
            linestyle="--",
            linewidth=1.5,
            label=f"Full Attention Baseline ({baseline_acc:.2f}%)",
        )

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlim(0, 100)
    dataset_name = (
        str(df["dataset"].dropna().iloc[0])
        if "dataset" in df and df["dataset"].notna().any()
        else "evaluation dataset"
    )
    ax.set_xlabel(f"Validation Accuracy — {dataset_name} (%)", fontweight="bold")
    ax.set_title(f"Top {len(subset)} Passing Evaluated Modules", fontweight="bold", pad=12)

    handles, labels = ax.get_legend_handles_labels()
    present_families = subset["subfamily"].unique()
    for fam in present_families:
        if fam in FAMILY_COLORS:
            patch = plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=FAMILY_COLORS[fam],
                edgecolor="black",
                linewidth=0.5,
            )
            handles.append(patch)
            labels.append(fam)
    ax.legend(handles, labels, loc="lower right", framealpha=0.9)

    out_png = out_dir / "fig1_top_modules_ranked.png"
    out_pdf = out_dir / "fig1_top_modules_ranked.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    logger.info("Saved Fig 1 to %s and %s", out_png.name, out_pdf.name)


def plot_fig2_accuracy_vs_runtime(df: pd.DataFrame, out_dir: Path):
    """Plot Accuracy vs Runtime / Latency Pareto Frontier."""
    passing = df[df["status"] == "PASS"].copy()
    if passing.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6), tight_layout=True)

    attns = passing[passing["module_type"] == "Attention"]
    comps = passing[passing["module_type"] == "Compressor"]

    if not attns.empty:
        ax.scatter(
            attns["elapsed_s"],
            attns["accuracy_pct"],
            color="#2b5c8f",
            s=55,
            alpha=0.85,
            edgecolors="k",
            linewidth=0.5,
            label="Attention Replacements",
        )
    if not comps.empty:
        ax.scatter(
            comps["elapsed_s"],
            comps["accuracy_pct"],
            color="#1b9e77",
            s=65,
            marker="^",
            alpha=0.85,
            edgecolors="k",
            linewidth=0.5,
            label="Compressors",
        )

    top_performers = passing.nlargest(6, "accuracy_pct")
    for _, row in top_performers.iterrows():
        ax.annotate(
            row["module_name"],
            (row["elapsed_s"], row["accuracy_pct"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="yellow",
                alpha=0.4,
                edgecolor="none",
            ),
        )

    sorted_df = passing.sort_values(by=["elapsed_s", "accuracy_pct"], ascending=[True, False])
    pareto_points = []
    max_acc = -1.0
    for _, row in sorted_df.iterrows():
        if row["accuracy_pct"] > max_acc:
            pareto_points.append((row["elapsed_s"], row["accuracy_pct"]))
            max_acc = row["accuracy_pct"]

    if len(pareto_points) > 1:
        px, py = zip(*pareto_points)
        ax.step(
            px,
            py,
            where="post",
            color="#e7298a",
            linestyle="-.",
            linewidth=1.5,
            label="Pareto Frontier",
        )

    ax.set_xlabel("Elapsed Time (s)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(
        "Accuracy vs. Evaluation & Recovery Time (Pareto Efficiency)",
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", framealpha=0.9)

    out_png = out_dir / "fig2_accuracy_vs_runtime.png"
    out_pdf = out_dir / "fig2_accuracy_vs_runtime.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    logger.info("Saved Fig 2 to %s and %s", out_png.name, out_pdf.name)


def plot_fig3_status_breakdown(df: pd.DataFrame, out_dir: Path):
    """Plot Status Breakdown (PASS, INCOMPATIBLE, SKIP, FAIL) by Module Type."""
    if df.empty:
        return

    def simplify_status(st: str) -> str:
        st_upper = str(st).upper()
        if st_upper == "PASS":
            return "PASS"
        elif "INCOMPATIBLE" in st_upper:
            return "INCOMPATIBLE"
        elif "OOM" in st_upper:
            return "OOM_SKIP"
        elif "SKIP" in st_upper:
            return "OTHER_SKIP"
        else:
            return "FAIL"

    df_copy = df.copy()
    df_copy["status_group"] = df_copy["status"].apply(simplify_status)

    ct = pd.crosstab(df_copy["module_type"], df_copy["status_group"])

    colors = {
        "PASS": "#2ca02c",
        "INCOMPATIBLE": "#ff7f0e",
        "OOM_SKIP": "#d62728",
        "OTHER_SKIP": "#9467bd",
        "FAIL": "#e377c2",
    }
    col_order = [
        c for c in ["PASS", "INCOMPATIBLE", "OOM_SKIP", "OTHER_SKIP", "FAIL"] if c in ct.columns
    ]
    ct = ct[col_order]

    fig, ax = plt.subplots(figsize=(7, 4.5), tight_layout=True)
    ct.plot(
        kind="bar",
        stacked=True,
        color=[colors.get(c, "#333333") for c in col_order],
        ax=ax,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_ylabel("Number of Modules", fontweight="bold")
    ax.set_xlabel("Architecture Category", fontweight="bold")
    ax.set_title(
        "Sweep Outcome Breakdown: Attention Mechanisms vs. Compressors",
        fontweight="bold",
        pad=12,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontweight="bold")
    ax.legend(title="Outcome", framealpha=0.9)

    out_png = out_dir / "fig3_status_breakdown.png"
    out_pdf = out_dir / "fig3_status_breakdown.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    logger.info("Saved Fig 3 to %s and %s", out_png.name, out_pdf.name)


def plot_fig4_category_performance(df: pd.DataFrame, out_dir: Path):
    """Plot Mean Accuracy and Variance across Architectural Sub-families."""
    passing = df[df["status"] == "PASS"]
    if passing.empty or len(passing["subfamily"].unique()) < 2:
        return

    stats = (
        passing.groupby("subfamily")["accuracy_pct"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(stats) * 0.4)), tight_layout=True)
    bars = ax.barh(
        stats.index,
        stats["mean"],
        xerr=stats["std"].fillna(0),
        color="#1b9e77",
        edgecolor="black",
        linewidth=0.6,
        height=0.6,
        capsize=4,
    )

    for bar, count in zip(bars, stats["count"]):
        w = bar.get_width()
        ax.text(
            w + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}% (n={count})",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Mean Accuracy (%) ± Std Dev", fontweight="bold")
    ax.set_title("Performance Aggregated by Architectural Sub-Family", fontweight="bold", pad=12)

    out_png = out_dir / "fig4_category_performance.png"
    out_pdf = out_dir / "fig4_category_performance.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    logger.info("Saved Fig 4 to %s and %s", out_png.name, out_pdf.name)


def generate_latex_tables(df: pd.DataFrame, out_file: Path):
    """Generate publication-ready LaTeX tables using booktabs."""
    passing = df[df["status"] == "PASS"].sort_values("accuracy_pct", ascending=False)

    latex_lines = [
        "% ==============================================================================",
        "% Auto-generated Publication Tables for Qwen3.5 Attention & Compressor Sweep",
        f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "% ==============================================================================\n",
    ]

    top_attns = passing[passing["module_type"] == "Attention"].head(10)
    if not top_attns.empty:
        latex_lines.extend(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                r"\caption{Top Performing Attention Mechanisms in the Supplied Evaluation}",
                r"\label{tab:top_attentions}",
                r"\begin{tabular}{llccc}",
                r"\toprule",
                r"\textbf{Rank} & \textbf{Module} & \textbf{Sub-Family} & \textbf{Accuracy (\%)} & \textbf{Latency (s)} \\",
                r"\midrule",
            ]
        )
        for idx, (_, row) in enumerate(top_attns.iterrows(), start=1):
            name_esc = str(row["module_name"]).replace("_", r"\_")
            sub_esc = str(row["subfamily"]).replace("_", r"\_")
            latex_lines.append(
                f"{idx} & {name_esc} & {sub_esc} & {row['accuracy_pct']:.2f} & {row['elapsed_s']:.2f} \\\\"
            )
        latex_lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}\n",
            ]
        )

    top_comps = passing[passing["module_type"] == "Compressor"].head(10)
    if not top_comps.empty:
        latex_lines.extend(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                r"\caption{Top Performing Compression Engines in the Supplied Evaluation}",
                r"\label{tab:top_compressors}",
                r"\begin{tabular}{llccc}",
                r"\toprule",
                r"\textbf{Rank} & \textbf{Module} & \textbf{Sub-Family} & \textbf{Accuracy (\%)} & \textbf{Latency (s)} \\",
                r"\midrule",
            ]
        )
        for idx, (_, row) in enumerate(top_comps.iterrows(), start=1):
            name_esc = str(row["module_name"]).replace("_", r"\_")
            sub_esc = str(row["subfamily"]).replace("_", r"\_")
            latex_lines.append(
                f"{idx} & {name_esc} & {sub_esc} & {row['accuracy_pct']:.2f} & {row['elapsed_s']:.2f} \\\\"
            )
        latex_lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}\n",
            ]
        )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    logger.info("Saved LaTeX tables to %s", out_file.name)


def generate_markdown_report(df: pd.DataFrame, out_file: Path, output_dir: Path):
    """Generate extensive publication-ready markdown report."""
    total_eval = len(df)
    passing = df[df["status"] == "PASS"]
    num_pass = len(passing)
    num_incompat = len(df[df["status"].str.contains("INCOMPATIBLE", case=False, na=False)])
    num_oom = len(df[df["status"].str.contains("OOM", case=False, na=False)])
    num_fail = len(df[df["status"] == "FAIL"])

    full_baseline = df[df["module_name"] == "full"]
    baseline_acc = (
        float(full_baseline["accuracy_pct"].iloc[0])
        if not full_baseline.empty
        else np.nan
    )

    top_attns = (
        passing[passing["module_type"] == "Attention"]
        .sort_values("accuracy_pct", ascending=False)
        .head(10)
    )
    top_comps = (
        passing[passing["module_type"] == "Compressor"]
        .sort_values("accuracy_pct", ascending=False)
        .head(10)
    )

    lines = [
        "# Attention & Compressor Evaluation Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Output Directory:** `{output_dir.resolve()}`  ",
        f"**Model Backbone:** {df['model_name'].dropna().iloc[0] if 'model_name' in df and df['model_name'].notna().any() else 'Not recorded in input'}  ",
        f"**Dataset:** {df['dataset'].dropna().iloc[0] if 'dataset' in df and df['dataset'].notna().any() else 'Not recorded in input'}  \n",
        "---",
        "## 1. Executive Summary",
        f"- **Total Modules Evaluated:** {total_eval}",
        f"- **Pass Rate:** {num_pass}/{total_eval} ({(num_pass / max(1, total_eval)) * 100:.1f}%)",
        f"- **Reported Architectural Incompatibilities:** {num_incompat}",
        f"- **CUDA Out-of-Memory (OOM):** {num_oom}",
        f"- **Execution / Dtype Failures:** {num_fail}",
        f"- **Full Attention Baseline Accuracy:** **{baseline_acc:.2f}%**\n" if np.isfinite(baseline_acc) else "- **Full Attention Baseline Accuracy:** Not present in the input records.\n",
        "---",
        "## 2. Top-10 Attention Replacements",
        "| Rank | Module | Sub-Family | Accuracy (%) | Delta vs Baseline | Latency (s) |",
        "|:---:|:---|:---|:---:|:---:|:---:|",
    ]

    for idx, (_, row) in enumerate(top_attns.iterrows(), start=1):
        lines.append(
            f"| {idx} | `{row['module_name']}` | {row['subfamily']} | **{row['accuracy_pct']:.2f}%** | {row['delta_acc']:+.2f}% | {row['elapsed_s']:.2f}s |"
        )

    lines.extend(
        [
            "\n---",
            "## 3. Top-10 Compression Engines",
            "| Rank | Module | Sub-Family | Accuracy (%) | Retention (%) | Latency (s) |",
            "|:---:|:---|:---|:---:|:---:|:---:|",
        ]
    )

    for idx, (_, row) in enumerate(top_comps.iterrows(), start=1):
        lines.append(
            f"| {idx} | `{row['module_name']}` | {row['subfamily']} | **{row['accuracy_pct']:.2f}%** | {row['retention_pct']:.1f}% | {row['elapsed_s']:.2f}s |"
        )

    lines.extend(
        [
            "\n---",
            "## 4. Complete Module Registry Results",
            "| Module Name | Type | Status | Accuracy (%) | Elapsed (s) | Error Diagnostics |",
            "|:---|:---:|:---:|:---:|:---:|:---|",
        ]
    )

    for _, row in df.sort_values(["status", "accuracy_pct"], ascending=[True, False]).iterrows():
        err_snippet = str(row.get("error", ""))[:80].replace("|", "\\|")
        lines.append(
            f"| `{row['module_name']}` | {row['module_type']} | `{row['status']}` | {row['accuracy_pct']:.2f}% | {row['elapsed_s']:.2f}s | {err_snippet} |"
        )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved Markdown report to %s", out_file.name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-grade artifacts from sweep results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/final_paper_run",
        help="Path to sweep directory",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="If > 0, continuously watch and regenerate every N seconds",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def run_once():
        summary_csv = results_dir / "summary.csv"
        if not summary_csv.exists():
            logger.warning("summary.csv does not exist yet in %s", results_dir)
            return False

        df = load_and_enrich_summary(summary_csv)
        if df.empty:
            logger.warning("summary.csv is empty")
            return False

        logger.info("Loaded %d records from %s", len(df), summary_csv)

        json_path = results_dir / "comprehensive_results.json"
        df.to_json(json_path, orient="records", indent=2)

        plot_fig1_top_modules(df, figures_dir)
        plot_fig2_accuracy_vs_runtime(df, figures_dir)
        plot_fig3_status_breakdown(df, figures_dir)
        plot_fig4_category_performance(df, figures_dir)

        tex_path = results_dir / "paper_tables.tex"
        generate_latex_tables(df, tex_path)

        md_path = results_dir / "comprehensive_report.md"
        generate_markdown_report(df, md_path, results_dir)

        return True

    if args.watch > 0:
        logger.info("Starting watch mode (interval: %d seconds)...", args.watch)
        try:
            while True:
                run_once()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            logger.info("Watch mode interrupted by user.")
    else:
        run_once()


if __name__ == "__main__":
    main()
