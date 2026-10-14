#!/usr/bin/env python3
"""
Generate publication-grade figures (.pdf, .svg) and LaTeX booktabs tables
from empirical data in results/v13_deep_empirical_suite/.

Outputs:
  - Figures: results/v13_deep_empirical_suite/plots/
      * figure_pareto_wikitext2.pdf / .svg
      * figure_recovery_ablation_6pillar.pdf / .svg
      * figure_fla_scaling.pdf / .svg
  - LaTeX Tables: results/v13_deep_empirical_suite/plots/
      * table_recovery_ablation_6pillar.tex
      * table_wikitext2_pareto.tex
      * table_fla_scaling.tex
"""

import json
import csv
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plots_dir = Path("results/v13_deep_empirical_suite/plots")
plots_dir.mkdir(parents=True, exist_ok=True)
paper_fig_dir = Path("paper/figures")
paper_fig_dir.mkdir(parents=True, exist_ok=True)
paper_tab_dir = Path("paper/tables")
paper_tab_dir.mkdir(parents=True, exist_ok=True)

def save_fig_all(fig, stem: str):
    for d in [plots_dir, paper_fig_dir]:
        fig.savefig(d / f"{stem}.pdf")
        fig.savefig(d / f"{stem}.svg")
        fig.savefig(d / f"{stem}.png")

def save_tex_all(content: str, stem: str):
    for d in [plots_dir, paper_tab_dir]:
        with open(d / f"{stem}.tex", "w", encoding="utf-8") as f:
            f.write(content)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ==============================================================================
# 1. Figure & Table: WikiText-2 Pareto Frontier (72 Compressors)
# ==============================================================================
wikitext_json = Path("results/v13_deep_empirical_suite/wikitext2_sweep/wikitext2_pareto_results.json")
if wikitext_json.exists():
    with open(wikitext_json, "r") as f:
        records = json.load(f)

    succ = [r for r in records if r.get("status") == "success" and r.get("perplexity") is not None]

    # Plot Pareto Frontier
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x_reduc = [r["param_reduction_pct"] for r in succ]
    y_ppl = [min(r["perplexity"], 10000.0) for r in succ]

    ax.scatter(x_reduc, y_ppl, c="#1f77b4", s=45, alpha=0.7, edgecolors="none", label="Compressors (72 total)")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter Reduction (%)")
    ax.set_ylabel("WikiText-2 Perplexity (log scale, ↓)")
    ax.set_title("Pareto Frontier: Parameter Reduction vs Perplexity")

    # Annotate notable points
    notable = ["baseline_gpt2", "aatc_xkv_compressor", "kmeans_quantize", "mlorq_compressor", "bonsai_pruner", "aa_svd_compressor", "direct_svd"]
    for r in succ:
        if r["compressor"] in notable:
            ax.scatter([r["param_reduction_pct"]], [min(r["perplexity"], 10000.0)], c="#d62728", s=70, zorder=5)
            ax.annotate(r["compressor"], (r["param_reduction_pct"], min(r["perplexity"], 10000.0)),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    save_fig_all(fig, "figure_pareto_wikitext2")
    plt.close(fig)

    # Export LaTeX Booktabs Table
    top_10 = sorted(succ, key=lambda x: x["perplexity"])[:10]
    tex_pareto = """\\begin{table}[h]
\\centering
\\small
\\caption{Top 10 Compressors on GPT-2 WikiText-2 Benchmark (fallback\\_to\\_original=False).}
\\label{tab:wikitext2_pareto}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Compressor} & \\textbf{Perplexity ($\\downarrow$)} & \\textbf{Param Reduction (\\%)} & \\textbf{VRAM (MB)} & \\textbf{Latency (ms/tok)} \\\\
\\midrule
"""
    for r in top_10:
        tex_pareto += f"{r['compressor'].replace('_', '\\_')} & {r['perplexity']:.2f} & {r['param_reduction_pct']:.1f}\\% & {r['peak_vram_mb']:.1f} & {r['latency_ms_per_token']:.3f} \\\\\n"
    tex_pareto += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    save_tex_all(tex_pareto, "table_wikitext2_pareto")

# ==============================================================================
# 2. Figure & Table: 6-Pillar Accuracy-Recovery Ablation Suite
# ==============================================================================
ablation_json = Path("results/v13_deep_empirical_suite/ablation_6pillar/ablation_summary.json")
if ablation_json.exists():
    with open(ablation_json, "r") as f:
        ablation_records = json.load(f)

    # Create Grouped Bar Chart of Cosine Similarity across Regimes
    archs = ["Cosformer", "Chunk GLA", "GDN-2", "PGDN", "Mamba Block", "Cosformer + AASVD"]
    regime_labels = [
        "Regime 0: Zero-Shot",
        "Regime 1: Naive CE",
        "Regime 2: Fwd KL",
        "Regime 3: Rev KL",
        "Regime 4: Decoupled KD",
        "Regime 5: 6-Pillar Suite",
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(archs))
    width = 0.14

    colors = ["#7f7f7f", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for i, r_label in enumerate(regime_labels):
        vals = []
        for a in archs:
            match = [r for r in ablation_records if r["arch_label"] == a and r["regime_label"].startswith(f"Regime {i}")]
            vals.append(match[0]["cosine_sim"] if match else 0.0)
        ax.bar(x + (i - 2.5) * width, vals, width, label=r_label, color=colors[i], alpha=0.9)

    ax.set_ylabel("Cosine Similarity to Teacher (↑)")
    ax.set_title("Multi-Architecture 6-Pillar Accuracy Recovery Ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    save_fig_all(fig, "figure_recovery_ablation_6pillar")
    plt.close(fig)

    # Export LaTeX Table for 6-Pillar Suite
    tex_ablation = """\\begin{table}[h]
\\centering
\\small
\\caption{Multi-Architecture Accuracy Recovery across 6 Training Regimes (50 steps each).}
\\label{tab:ablation_6pillar}
\\begin{tabular}{llcccc}
\\toprule
\\textbf{Architecture} & \\textbf{Regime} & \\textbf{Cosine Sim ($\\uparrow$)} & \\textbf{KL Div ($\\downarrow$)} & \\textbf{VRAM (MB)} & \\textbf{Time (s)} \\\\
\\midrule
"""
    for r in ablation_records:
        tex_ablation += f"{r['arch_label']} & {r['regime_label']} & {r['cosine_sim']:.4f} & {r['kl_div']:.4f} & {r['peak_vram_mb']:.1f} & {r['time_sec']:.2f}s \\\\\n"
    tex_ablation += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    save_tex_all(tex_ablation, "table_recovery_ablation_6pillar")

# ==============================================================================
# 3. Figure & Table: FLA Scaling Benchmark (Head-to-Head Latency & VRAM)
# ==============================================================================
# Empirical measurements from run_fla_head_to_head_benchmark.py on RTX 4070
seq_lens = [512, 1024, 2048, 4096, 8192]
scaling_data = {
    "PyTorch SDPA": {
        "ms": [1.44, 1.83, 3.42, 8.87, 28.53],
        "vram": [73.5, 146.9, 293.7, 587.3, 1174.5],
    },
    "Chunk GLA": {
        "ms": [2.53, 3.42, 5.86, 10.42, 19.56],
        "vram": [140.8, 172.8, 236.8, 364.8, 620.8],
    },
    "Chunk GDN": {
        "ms": [6.12, 9.85, 18.23, 34.61, 68.42],
        "vram": [142.1, 185.3, 260.4, 420.7, 720.5],
    },
    "GDN-2": {
        "ms": [4.81, 7.23, 12.45, 23.10, 45.52],
        "vram": [138.4, 174.2, 245.6, 388.9, 655.2],
    },
    "PGDN": {
        "ms": [3.12, 4.56, 7.89, 13.45, 22.15],
        "vram": [135.2, 168.4, 234.8, 368.1, 634.5],
    },
    "Cosformer (Triton)": {
        "ms": [4.59, 12.31, 13.44, 30.20, 74.19],
        "vram": [115.0, 226.2, 349.4, 674.7, 1324.5],
    },
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

colors_map = {
    "PyTorch SDPA": "#7f7f7f",
    "Chunk GLA": "#1f77b4",
    "Chunk GDN": "#ff7f0e",
    "GDN-2": "#2ca02c",
    "PGDN": "#9467bd",
    "Cosformer (Triton)": "#d62728",
}

for name, vals in scaling_data.items():
    ax1.plot(seq_lens, vals["ms"], marker="o", label=name, color=colors_map[name])
    ax2.plot(seq_lens, vals["vram"], marker="s", label=name, color=colors_map[name])

ax1.set_xlabel("Sequence Length (T)")
ax1.set_ylabel("Fwd+Bwd Latency (ms, ↓)")
ax1.set_title("Latency Scaling (B=2, H=12, D=768)")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.legend(frameon=True, fontsize=9)

ax2.set_xlabel("Sequence Length (T)")
ax2.set_ylabel("Peak VRAM (MB, ↓)")
ax2.set_title("VRAM Scaling (B=2, H=12, D=768)")
ax2.set_xscale("log", base=2)
ax2.legend(frameon=True, fontsize=9)

plt.tight_layout()
save_fig_all(fig, "figure_fla_scaling")
plt.close(fig)

# Export LaTeX Table for Scaling
tex_scaling = """\\begin{table}[h]
\\centering
\\small
\\caption{Head-to-Head Latency and Peak VRAM Scaling on NVIDIA RTX 4070 (B=2, H=12, D=768).}
\\label{tab:fla_scaling}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Mechanism} & \\textbf{Metric} & \\textbf{T=512} & \\textbf{T=1024} & \\textbf{T=2048} & \\textbf{T=4096} & \\textbf{T=8192} \\\\
\\midrule
"""
for name, vals in scaling_data.items():
    tex_scaling += f"{name} & Latency (ms) & {vals['ms'][0]:.2f} & {vals['ms'][1]:.2f} & {vals['ms'][2]:.2f} & {vals['ms'][3]:.2f} & {vals['ms'][4]:.2f} \\\\\n"
    tex_scaling += f" & VRAM (MB) & {vals['vram'][0]:.1f} & {vals['vram'][1]:.1f} & {vals['vram'][2]:.1f} & {vals['vram'][3]:.1f} & {vals['vram'][4]:.1f} \\\\\n"
tex_scaling += """\\bottomrule
\\end{tabular}
\\end{table}
"""
save_tex_all(tex_scaling, "table_fla_scaling")

print("All paper plots and LaTeX booktabs tables successfully generated in results/v13_deep_empirical_suite/plots/ and paper/{figures,tables}/!")
