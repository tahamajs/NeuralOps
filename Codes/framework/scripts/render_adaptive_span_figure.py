"""
Standalone High-Fidelity Scientific Illustrator for Adaptive Span Attention (NeurIPS / ICLR standard).
Produces publication-grade 3-panel architecture, heatmap dynamics, and scaling profiles.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Scientific styling parameters
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIRS = [
    Path("paper/neuralops_paper_final/figures/catalog_visuals"),
    Path("paper/neuralops_paper_iclr/figures/catalog_visuals"),
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def create_adaptive_span_figure():
    fig = plt.figure(figsize=(16, 5.0), facecolor="white")
    gs = GridSpec(1, 3, width_ratios=[1.35, 0.95, 0.85], figure=fig, wspace=0.28)

    # =========================================================================
    # PANEL (a): COMPUTATIONAL ARCHITECTURE DIAGRAM
    # =========================================================================
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0, 100)
    ax0.set_ylim(0, 100)
    ax0.axis("off")
    ax0.set_title(r"$\mathbf{(a)}$ Computational Architecture & Dataflow", weight="bold", pad=12, loc="left")

    def draw_box(ax, x, y, w, h, title, subtitle=None, tag=None, bg="#F8FAFC", ec="#94A3B8", lw=1.2, r=1.5):
        # Subtle drop shadow
        shadow = patches.FancyBboxPatch((x+0.5, y-0.6), w, h, boxstyle=f"round,pad=0.2,rounding_size={r}",
                                        facecolor="#E2E8F0", edgecolor="none", zorder=1)
        ax.add_patch(shadow)
        # Main box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={r}",
                                      facecolor=bg, edgecolor=ec, linewidth=lw, zorder=2)
        ax.add_patch(rect)
        
        # Text
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=7.8, weight="bold", color="#0F172A", zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=6.8, color="#334155", zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=7.8, weight="bold", color="#0F172A", zorder=3)
            
        if tag:
            tag_box = patches.FancyBboxPatch((x + w/2 - 4.5, y + h - 1.2), 9, 3.2, boxstyle="round,pad=0.1,rounding_size=0.8",
                                            facecolor="#0EA5E9", edgecolor="none", zorder=4)
            ax.add_patch(tag_box)
            ax.text(x + w/2, y + h + 0.4, tag, ha="center", va="center", fontsize=5.8, weight="bold", color="white", zorder=5)

    def draw_arrow(ax, x1, y1, x2, y2, label=None, color="#475569", lw=1.1, rad=0.0):
        con = patches.ConnectionStyle(f"arc3,rad={rad}")
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, connectionstyle=con, mutation_scale=9),
                    zorder=2)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.6, label, fontsize=6.2, color="#0F172A", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#CBD5E1", lw=0.6, alpha=0.9), zorder=6)

    # 1. Input Tensor
    draw_box(ax0, 1.5, 38, 11, 24, "Input $X$", subtitle=r"$[B, N, D]$", bg="#F1F5F9", ec="#64748B")

    # 2. Three Linear Projections (Q, K, V)
    draw_box(ax0, 17.5, 68, 15, 18, "Linear $W_Q$", subtitle=r"$[B, H, N, d]$", tag="Query", bg="#E0F2FE", ec="#38BDF8")
    draw_box(ax0, 17.5, 41, 15, 18, "Linear $W_K$", subtitle=r"$[B, H, N, d]$", tag="Key", bg="#E0F2FE", ec="#38BDF8")
    draw_box(ax0, 17.5, 14, 15, 18, "Linear $W_V$", subtitle=r"$[B, H, N, d]$", tag="Value", bg="#E0F2FE", ec="#38BDF8")

    draw_arrow(ax0, 12.5, 52, 17.5, 77, rad=0.1)
    draw_arrow(ax0, 12.5, 50, 17.5, 50)
    draw_arrow(ax0, 12.5, 48, 17.5, 23, rad=-0.1)

    # 3. Scaled MatMul (Q, K) -> Raw Scores S
    draw_box(ax0, 37.5, 55, 15, 20, "Scaled Dot-Prod", subtitle=r"$S = \frac{QK^\top}{\sqrt{d}}$", tag="$[B, H, N, N]$", bg="#FEF3C7", ec="#FBBF24")
    draw_arrow(ax0, 32.5, 77, 37.5, 68, label="$Q$")
    draw_arrow(ax0, 32.5, 50, 37.5, 62, label="$K$")

    # 4. Learnable Dynamic Span Branch
    draw_box(ax0, 37.5, 16, 15, 26, "Learnable Span",
             subtitle=r"$\text{span}_h = \sigma(\theta_h) S_{\max}$" + "\n" + r"$\text{dist}_{ij} = |i - j|$",
             tag="Param $\\theta$", bg="#FFEDD5", ec="#FB923C")

    # 5. Mask Application
    draw_box(ax0, 58, 44, 16, 28, "Dynamic Span Mask",
             subtitle=r"$M_{ij}^{(h)} = 0 \text{ if } \text{dist} \leq \text{span}_h$" + "\n" + r"$-\infty \text{ otherwise}$" + "\n" + r"$A = \operatorname{Softmax}(S + M)$",
             tag="Banded Mask", bg="#EDE9FE", ec="#A78BFA")
    draw_arrow(ax0, 52.5, 65, 58, 60, label="Scores $S$")
    draw_arrow(ax0, 52.5, 29, 58, 48, label=r"$\text{span}_h$", rad=0.15)

    # 6. Context Aggregation & Out Projection
    draw_box(ax0, 78.5, 34, 11, 26, "MatMul $V$", subtitle=r"$C = A V$" + "\n" + r"$\text{Reshape } [B,N,D]$", bg="#DCFCE7", ec="#4ADE80")
    draw_arrow(ax0, 74, 58, 78.5, 50, label="Sparse $A$")
    draw_arrow(ax0, 32.5, 23, 78.5, 40, label="$V$", rad=-0.22, color="#059669")

    # 7. Output Linear Projection
    draw_box(ax0, 93, 38, 6.5, 18, "Out $W_O$", subtitle="$[B,N,D]$", bg="#F1F5F9", ec="#64748B")
    draw_arrow(ax0, 89.5, 47, 93, 47)


    # =========================================================================
    # PANEL (b): STRUCTURAL PATTERN AND DYNAMICS (2x2 HEAD HEATMAP GRID)
    # =========================================================================
    ax1 = fig.add_subplot(gs[1])
    ax1.set_title(r"$\mathbf{(b)}$ Multi-Head Learned Span Dynamics", weight="bold", pad=12, loc="left")

    N_vis = 32
    pos = np.arange(N_vis)
    dist = np.abs(pos[:, None] - pos[None, :])

    # 4 Representative Heads with different effective spans
    heads_info = [
        ("Head 1: Ultra-Local", 4, r"$S_1 = 16 \text{ tokens}$", "#1E40AF"),
        ("Head 2: Local Window", 8, r"$S_2 = 32 \text{ tokens}$", "#2563EB"),
        ("Head 3: Mid-Range Context", 16, r"$S_3 = 64 \text{ tokens}$", "#0284C7"),
        ("Head 4: Global Attention", 28, r"$S_4 = 112 \text{ tokens}$", "#0D9488"),
    ]

    for idx, (h_title, span_val, span_lbl, text_col) in enumerate(heads_info):
        # Compute smooth decay within span
        mask = (dist <= span_val).astype(float)
        fade = np.clip(1.0 - (dist / (span_val + 1.0)) * 0.7, 0.25, 1.0) * mask

        r_offset = 18 if idx < 2 else 0
        c_offset = 0 if (idx % 2 == 0) else 18

        # Draw Heatmap
        ax1.imshow(fade, cmap="Blues", extent=[c_offset, c_offset+14, r_offset, r_offset+14], vmin=0, vmax=1, zorder=2)
        # Bounding box
        rect = patches.Rectangle((c_offset, r_offset), 14, 14, fill=False, edgecolor="#64748B", lw=0.9, zorder=3)
        ax1.add_patch(rect)

        # Labels
        ax1.text(c_offset + 7, r_offset + 14.8, h_title, ha="center", va="bottom", fontsize=7.2, weight="bold", color=text_col, zorder=4)
        ax1.text(c_offset + 7, r_offset - 1.2, span_lbl, ha="center", va="top", fontsize=6.8, color="#334155",
                 bbox=dict(boxstyle="round,pad=0.15", fc="#F8FAFC", ec="#E2E8F0", lw=0.6), zorder=4)

    ax1.set_xlim(-1.5, 33.5)
    ax1.set_ylim(-3.0, 35.0)
    ax1.axis("off")


    # =========================================================================
    # PANEL (c): COMPLEXITY AND SCALING PROFILE
    # =========================================================================
    ax2 = fig.add_subplot(gs[2])
    ax2.set_title(r"$\mathbf{(c)}$ Complexity & Scaling Profile", weight="bold", pad=12, loc="left")

    seq_lens = np.logspace(6, 15, num=120, base=2)
    D = 768
    S_avg = 64

    dense_flops = 2 * (seq_lens**2) * D
    dense_impl_span = 2 * (seq_lens**2) * D + seq_lens * D * 0.05
    ideal_banded_flops = 2 * seq_lens * S_avg * D

    ax2.loglog(seq_lens, dense_flops, label=r"Dense Full Attention $\mathcal{O}(N^2 D)$", color="#DC2626", lw=1.8, linestyle="--")
    ax2.loglog(seq_lens, dense_impl_span, label=r"Adaptive Span (Dense Impl) $\mathcal{O}(N^2 D)$", color="#EA580C", lw=2.0)
    ax2.loglog(seq_lens, ideal_banded_flops, label=r"Ideal Block-Sparse $\mathcal{O}(N \bar{S} D)$", color="#059669", lw=2.2)

    # Crossover annotations & Speedup tags
    ax2.axvline(2048, color="#64748B", linestyle=":", lw=1.0, alpha=0.7)
    ax2.text(2048, 2e7, r"$N=2\text{k}$" + "\n" + r"$\mathbf{32\times}$ Sparsity", fontsize=7.2, color="#0F172A", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.2", fc="#ECFDF5", ec="#A7F3D0", lw=0.8))

    ax2.axvline(8192, color="#64748B", linestyle=":", lw=1.0, alpha=0.7)
    ax2.text(8192, 1e8, r"$N=8\text{k}$" + "\n" + r"$\mathbf{128\times}$ Theoretical Speedup", fontsize=7.2, color="#0F172A", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.2", fc="#EFF6FF", ec="#BFDBFE", lw=0.8))

    ax2.set_xlabel(r"Sequence Length $N$ (tokens)", fontsize=8.8)
    ax2.set_ylabel(r"Attention FLOPs (log-scale)", fontsize=8.8)
    ax2.grid(True, which="both", ls=":", alpha=0.45, color="#94A3B8")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95, fontsize=6.8)

    # Save to all required filenames and formats
    filenames = [
        "adaptive_span_attention_arch",
        "module_adaptive_span",
        "module_adaptive_span_attention",
    ]

    for d in OUT_DIRS:
        for fname in filenames:
            fig.savefig(d / f"{fname}.pdf", bbox_inches="tight")
            fig.savefig(d / f"{fname}.png", bbox_inches="tight", dpi=300)

    plt.close(fig)
    print("Adaptive Span Attention figure generated at pristine publication standard!")


if __name__ == "__main__":
    create_adaptive_span_figure()
