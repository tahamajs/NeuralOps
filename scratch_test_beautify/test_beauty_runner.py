import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
import matplotlib.colors as mcolors
from matplotlib.path import Path
import numpy as np
import fitz

OUTPUT_DIR = "/Users/tahamajs/Documents/uni/Project/scratch_test_beautify"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modern Academic Dark-Glass Theme Palette
THEME = {
    "bg": "#0B0F19",          # Deep midnight slate
    "card_bg": "#111827",     # Dark glass card
    "card_border": "#1F2937", # Subtle border
    "text_main": "#F9FAFB",   # Pure bright white
    "text_muted": "#9CA3AF",  # Muted silver
    "text_dim": "#6B7280",    # Dim label
    "grid": "#1F2937",
    "accents": {
        "linear": {"primary": "#38BDF8", "secondary": "#0284C7", "glow": "#0ea5e9"},   # Electric Cyan
        "ssm": {"primary": "#10B981", "secondary": "#059669", "glow": "#34d399"},      # Emerald Neon
        "moe": {"primary": "#F59E0B", "secondary": "#D97706", "glow": "#fbbf24"},      # Amber Gold
        "sparse": {"primary": "#EF4444", "secondary": "#DC2626", "glow": "#f87171"},   # Coral Crimson
        "kernel": {"primary": "#A855F7", "secondary": "#7E22CE", "glow": "#c084fc"},   # Royal Purple
        "compression": {"primary": "#06B6D4", "secondary": "#0891B2", "glow": "#22d3ee"}# Neon Teal
    }
}

def render_ultra_beautified_figure(meta, output_pdf, output_png):
    fig = plt.figure(figsize=(14.0, 4.8), facecolor=THEME["bg"], dpi=300)
    
    # 3 Grid Subplots with clean margins
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.97, top=0.86, bottom=0.08, wspace=0.18)
    
    mod_name = meta.get("module_name", "module")
    class_name = meta.get("class_name", mod_name)
    family = meta.get("family", "linear")
    acc = THEME["accents"].get(family, THEME["accents"]["linear"])
    
    # Super Title Header Banner
    fig.text(0.03, 0.94, f"{class_name}", color=THEME["text_main"], fontsize=14, fontweight="bold", fontfamily="sans-serif")
    fig.text(0.03, 0.89, f"Registration ID: {mod_name}  •  Track: {family.upper()}", color=acc["primary"], fontsize=9, fontweight="medium")
    fig.text(0.97, 0.92, "NeuralOps Operator Benchmark Catalog", color=THEME["text_muted"], fontsize=9, ha="right", fontfamily="sans-serif")
    
    # =========================================================================
    # PANEL A: Computational Architecture Dataflow (PlotNeuralNet Isometric Style)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(THEME["card_bg"])
    for spine in ax1.spines.values():
        spine.set_edgecolor(THEME["card_border"])
        spine.set_linewidth(1.2)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("(a) Computational Architecture & Dataflow", color=THEME["text_main"], fontsize=10, fontweight="bold", pad=8, loc="left")
    
    # Add subtle background grid lines for engineering look
    for x in np.linspace(0, 10, 11):
        ax1.axvline(x, color=THEME["grid"], lw=0.5, alpha=0.4, ls=":")
    for y in np.linspace(0, 10, 11):
        ax1.axhline(y, color=THEME["grid"], lw=0.5, alpha=0.4, ls=":")

    # Draw layered architecture blocks with 3D shadow & rounded borders
    def draw_layer_block(ax, x, y, w, h, label, sublabel, fill_col, border_col, is_core=False):
        # Shadow
        shadow = FancyBboxPatch((x+0.08, y-0.08), w, h, boxstyle="round,pad=0.08,rounding_size=0.2",
                                facecolor="#000000", edgecolor="none", alpha=0.45, zorder=2)
        ax.add_patch(shadow)
        # Main card
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.2",
                              facecolor=fill_col, edgecolor=border_col, linewidth=1.5 if is_core else 1.0, zorder=3)
        ax.add_patch(card)
        # Labels
        ax.text(x + w/2, y + h*0.62, label, color=THEME["text_main"], fontsize=8.5, fontweight="bold" if is_core else "medium",
                ha="center", va="center", zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h*0.28, sublabel, color=acc["primary"] if is_core else THEME["text_muted"],
                    fontsize=7.0, ha="center", va="center", zorder=4)

    # Blocks sequence
    draw_layer_block(ax1, 1.0, 8.0, 8.0, 1.2, "Input Sequence Tensor $X$", "$[B, L, D_{\\text{model}}]$", "#1E293B", "#334155")
    draw_layer_block(ax1, 1.0, 5.8, 8.0, 1.3, "Q, K, V Projections & Normalization", "$W_q, W_k, W_v \\in \\mathbb{R}^{D \\times d} \\mid \\text{RMSNorm}$", "#1E293B", "#334155")
    
    # Core Specialized Operator Block (Vibrant Glowing Card)
    if family == "ssm":
        core_title = "Selective State-Space Scan Core"
        core_sub = "$h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t \\mid y_t = C_t h_t$"
    elif family == "moe":
        core_title = "Top-$k$ Router & Parallel Experts"
        core_sub = "$\\operatorname{TopK}(\\operatorname{Softmax}(x W_g)) \\cdot \\sum E_i(x)$"
    elif family == "sparse":
        core_title = "Structured Sparsity & Window Masking"
        core_sub = "$M = \\mathbb{I}(|W| \\ge \\tau) \\odot \\operatorname{Band}(w)$"
    elif family == "kernel":
        core_title = "Kernel Feature Map Contraction"
        core_sub = "$\\phi(Q) \\cdot (\\phi(K)^\\top V) \\mid \\text{FAVOR+}$"
    elif family == "compression":
        core_title = "Optimal Codebook Quantization Core"
        core_sub = "$\\min \\|(W - \\hat{W}) H\\|_F^2 \\mid \\text{INT4/INT8}$"
    else:
        core_title = "Truncated SVD Low-Rank Contraction"
        core_sub = "$K_r = U_r \\Sigma_r V_r^\\top \\mid S = V_r^\\top V \\in \\mathbb{R}^{r \\times d_v}$"

    draw_layer_block(ax1, 0.8, 3.2, 8.4, 1.6, core_title, core_sub, "#172554" if family=="linear" else "#064E3B", acc["primary"], is_core=True)
    draw_layer_block(ax1, 1.0, 0.8, 8.0, 1.3, "Output Linear Projection & Residual", "$Y = \\operatorname{OutProj}(\\cdot) + X \\mid [B, L, D]$", "#1E293B", "#334155")

    # Connective Glowing Dataflow Arrows
    def draw_arrow(ax, y_start, y_end, label):
        arrow = FancyArrowPatch((5.0, y_start), (5.0, y_end), arrowstyle="-|>",
                                mutation_scale=12, color=acc["primary"], lw=1.5, zorder=5)
        ax.add_patch(arrow)
        if label:
            ax.text(5.2, (y_start + y_end)/2, label, color=THEME["text_muted"], fontsize=6.8, va="center", zorder=6)

    draw_arrow(ax1, 8.0, 7.1, "$[B, L, D]$")
    draw_arrow(ax1, 5.8, 4.8, "$Q, K, V$")
    draw_arrow(ax1, 3.2, 2.1, "$S_t / Y_t$")

    # =========================================================================
    # PANEL B: Structural Dynamics & Transformation Heatmap / Spectrum
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(THEME["card_bg"])
    for spine in ax2.spines.values():
        spine.set_edgecolor(THEME["card_border"])
        spine.set_linewidth(1.2)
    ax2.set_title("(b) Structural Pattern & Matrix Dynamics", color=THEME["text_main"], fontsize=10, fontweight="bold", pad=8, loc="left")

    np.random.seed(abs(hash(mod_name)) % (2**32))
    
    if family == "ssm":
        # Continuous state decay kernel
        decay = np.exp(-np.linspace(0, 4, 32)[:, None] * np.linspace(0, 1, 32)[None, :])
        im = ax2.imshow(decay, cmap="viridis", aspect="auto", extent=[0, 32, 32, 0])
        ax2.set_xlabel("Time Step Sequence ($t$)", color=THEME["text_muted"], fontsize=8)
        ax2.set_ylabel("State Memory Horizon ($s$)", color=THEME["text_muted"], fontsize=8)
        ax2.tick_params(colors=THEME["text_muted"], labelsize=7)
        cb = plt.colorbar(im, ax=ax2, pad=0.03, fraction=0.046)
        cb.set_label("Decay $\\bar{A} = \\exp(\\Delta A)$", color=THEME["text_muted"], fontsize=7.5)
        cb.ax.tick_params(colors=THEME["text_muted"], labelsize=6.5)
        
    elif family == "moe":
        # Dynamic routing tokens vs experts
        tokens, experts = 24, 8
        routes = np.zeros((tokens, experts))
        for t in range(tokens):
            top_e = np.random.choice(experts, size=2, replace=False)
            routes[t, top_e] = np.random.uniform(0.6, 1.0, size=2)
        im = ax2.imshow(routes, cmap="magma", aspect="auto", extent=[0, experts, tokens, 0])
        ax2.set_xlabel("Expert Index ($E_i$)", color=THEME["text_muted"], fontsize=8)
        ax2.set_ylabel("Input Token Index ($t$)", color=THEME["text_muted"], fontsize=8)
        ax2.tick_params(colors=THEME["text_muted"], labelsize=7)
        cb = plt.colorbar(im, ax=ax2, pad=0.03, fraction=0.046)
        cb.set_label("Router Gating Weight", color=THEME["text_muted"], fontsize=7.5)
        cb.ax.tick_params(colors=THEME["text_muted"], labelsize=6.5)

    elif family == "sparse":
        # Banded sliding window + random global tokens
        N = 32
        mask = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if abs(i - j) <= 2 or i == 0 or j == 0 or (i % 8 == 0):
                    mask[i, j] = 1.0
        im = ax2.imshow(mask, cmap="Reds", aspect="auto", extent=[0, N, N, 0])
        ax2.set_xlabel("Key Position ($j$)", color=THEME["text_muted"], fontsize=8)
        ax2.set_ylabel("Query Position ($i$)", color=THEME["text_muted"], fontsize=8)
        ax2.tick_params(colors=THEME["text_muted"], labelsize=7)
        cb = plt.colorbar(im, ax=ax2, pad=0.03, fraction=0.046)
        cb.set_label("Sparsity Mask $M_{ij}$", color=THEME["text_muted"], fontsize=7.5)
        cb.ax.tick_params(colors=THEME["text_muted"], labelsize=6.5)

    elif family == "compression":
        # Quantization distribution & bins
        weights = np.random.normal(0, 1, 1000)
        n, bins, patches_list = ax2.hist(weights, bins=24, color=acc["primary"], edgecolor=THEME["bg"], alpha=0.85, density=True)
        ax2.axvline(-1.8, color="#EF4444", ls="--", lw=1.2, label="Negative Outliers")
        ax2.axvline(1.8, color="#EF4444", ls="--", lw=1.2, label="Positive Outliers")
        ax2.set_xlabel("Weight Magnitude Distribution", color=THEME["text_muted"], fontsize=8)
        ax2.set_ylabel("Density", color=THEME["text_muted"], fontsize=8)
        ax2.tick_params(colors=THEME["text_muted"], labelsize=7)
        ax2.legend(facecolor=THEME["card_bg"], edgecolor=THEME["card_border"], labelcolor=THEME["text_muted"], fontsize=6.5, loc="upper right")

    else:
        # Low-rank singular spectrum decay
        ranks = np.arange(1, 65)
        sigmas = np.exp(-0.06 * ranks) + 0.02 * np.random.normal(0, 0.02, size=64)
        sigmas = np.clip(np.sort(sigmas)[::-1], 1e-4, 1.0)
        cum_energy = np.cumsum(sigmas**2) / np.sum(sigmas**2)
        
        ax2.bar(ranks[:16], sigmas[:16], color=acc["primary"], alpha=0.9, label="Retained Rank ($r=16$)")
        ax2.bar(ranks[16:], sigmas[16:], color=THEME["text_dim"], alpha=0.4, label="Truncated Tail")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(ranks, cum_energy * 100, color="#10B981", lw=1.8, label="Cumulative Energy (%)")
        ax2_twin.set_ylabel("Energy Retention (%)", color="#10B981", fontsize=7.5)
        ax2_twin.tick_params(colors="#10B981", labelsize=6.5)
        ax2_twin.set_ylim(50, 105)
        ax2.set_xlabel("Singular Vector Index ($i$)", color=THEME["text_muted"], fontsize=8)
        ax2.set_ylabel("Singular Value $\\sigma_i$", color=THEME["text_muted"], fontsize=8)
        ax2.tick_params(colors=THEME["text_muted"], labelsize=7)
        ax2.legend(facecolor=THEME["card_bg"], edgecolor=THEME["card_border"], labelcolor=THEME["text_muted"], fontsize=6.5, loc="upper center")

    # =========================================================================
    # PANEL C: Theoretical Complexity & Hardware Scaling Profile
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(THEME["card_bg"])
    for spine in ax3.spines.values():
        spine.set_edgecolor(THEME["card_border"])
        spine.set_linewidth(1.2)
    ax3.set_title("(c) Complexity Scaling Profile vs. Context ($L$)", color=THEME["text_main"], fontsize=10, fontweight="bold", pad=8, loc="left")
    ax3.grid(True, color=THEME["grid"], ls="--", lw=0.6, alpha=0.6)

    seq_lens = np.logspace(np.log10(128), np.log10(65536), 50)
    dense_flops = 4 * 768 * (seq_lens**2)
    
    if family in ["ssm", "linear"]:
        module_flops = 8 * 768 * 16 * seq_lens
        mod_label = f"{class_name} $\\mathcal{{O}}(N r d)$"
    elif family == "moe":
        module_flops = 4 * 768 * (768 * 2 / 8) * seq_lens
        mod_label = f"{class_name} $\\mathcal{{O}}(N d \\frac{{k}}{{E}})$"
    elif family == "sparse":
        module_flops = 4 * 768 * seq_lens * np.sqrt(seq_lens)
        mod_label = f"{class_name} $\\mathcal{{O}}(N \\sqrt{{N}} d)$"
    elif family == "kernel":
        module_flops = 6 * 768 * 32 * seq_lens
        mod_label = f"{class_name} $\\mathcal{{O}}(N m d)$"
    else:
        module_flops = 2 * 768 * 768 * seq_lens * 0.25
        mod_label = f"{class_name} (INT4 Quantized)"

    ax3.plot(seq_lens, dense_flops, color="#EF4444", lw=1.8, ls="--", label="Dense Softmax $\\mathcal{O}(N^2 d)$")
    ax3.plot(seq_lens, module_flops, color=acc["primary"], lw=2.2, label=mod_label)
    
    # Fill area between curves to highlight speedup gap
    ax3.fill_between(seq_lens, module_flops, dense_flops, color=acc["primary"], alpha=0.08)

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Sequence Context Length ($L$)", color=THEME["text_muted"], fontsize=8)
    ax3.set_ylabel("Compute Operations (FLOPs)", color=THEME["text_muted"], fontsize=8)
    ax3.tick_params(colors=THEME["text_muted"], labelsize=7)
    ax3.legend(facecolor=THEME["card_bg"], edgecolor=THEME["card_border"], labelcolor=THEME["text_main"], fontsize=6.8, loc="upper left")
    
    # Annotate speedup at 64k
    speedup_64k = dense_flops[-1] / module_flops[-1]
    ax3.annotate(f"{speedup_64k:.0f}× Efficiency Gain\nat 64k Context",
                 xy=(seq_lens[-1], module_flops[-1]),
                 xytext=(seq_lens[-1]*0.12, module_flops[-1]*15),
                 arrowprops=dict(arrowstyle="->", color=acc["primary"], lw=1.2),
                 color=acc["primary"], fontsize=7.2, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=THEME["card_bg"], edgecolor=acc["primary"], lw=0.8))

    plt.savefig(output_pdf, bbox_inches="tight", facecolor=THEME["bg"], dpi=300)
    plt.savefig(output_png, bbox_inches="tight", facecolor=THEME["bg"], dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    test_meta = {
        "module_name": "svd_linear_attention",
        "class_name": "SVDLinearAttention",
        "family": "linear"
    }
    render_ultra_beautified_figure(test_meta, f"{OUTPUT_DIR}/test_beauty.pdf", f"{OUTPUT_DIR}/test_beauty.png")
    print(f"Generated ultra-beautified prototype in {OUTPUT_DIR}/")
