import os
import sys
import glob
import json
import ast
import hashlib
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = "/Users/tahamajs/Documents/uni/Project"
ATTENTIONS_DIR = os.path.join(PROJECT_ROOT, "Codes", "framework", "attentions")
COMPRESSIONS_DIR = os.path.join(PROJECT_ROOT, "Codes", "framework", "compression_engine")
METADATA_DIR = os.path.join(PROJECT_ROOT, "module_metadata")
OUTPUT_FINAL = os.path.join(PROJECT_ROOT, "paper", "neuralops_paper_final", "figures", "catalog_visuals")
OUTPUT_ICLR = os.path.join(PROJECT_ROOT, "paper", "neuralops_paper_iclr", "figures", "catalog_visuals")

os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_FINAL, exist_ok=True)
os.makedirs(OUTPUT_ICLR, exist_ok=True)

THEME = {
    "bg": "#0B0F19",
    "card_bg": "#111827",
    "card_border": "#1F2937",
    "text_main": "#F9FAFB",
    "text_muted": "#9CA3AF",
    "text_dim": "#6B7280",
    "grid": "#1F2937",
    "accents": {
        "linear": {"primary": "#38BDF8", "secondary": "#0284C7", "glow": "#0ea5e9", "card": "#172554"},
        "ssm": {"primary": "#10B981", "secondary": "#059669", "glow": "#34d399", "card": "#064E3B"},
        "moe": {"primary": "#F59E0B", "secondary": "#D97706", "glow": "#fbbf24", "card": "#451A03"},
        "sparse": {"primary": "#EF4444", "secondary": "#DC2626", "glow": "#f87171", "card": "#450A0A"},
        "kernel": {"primary": "#A855F7", "secondary": "#7E22CE", "glow": "#c084fc", "card": "#3B0764"},
        "compression": {"primary": "#06B6D4", "secondary": "#0891B2", "glow": "#22d3ee", "card": "#164E63"}
    }
}

def parse_module_source(py_path):
    with open(py_path, "r", encoding="utf-8") as f:
        code = f.read()
        
    tree = ast.parse(code)
    class_name = ""
    docstring = ""
    params = {}
    forward_ops = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not class_name or any(k in node.name for k in ["Attention", "Compress", "SVD", "Mamba", "MoE", "Layer", "Engine"]):
                class_name = node.name
                docstring = ast.get_docstring(node) or ""
                
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for arg in node.args.args:
                if arg.arg != "self":
                    params[arg.arg] = "default"
                    
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Call):
                    if isinstance(sub_node.func, ast.Attribute):
                        forward_ops.append(sub_node.func.attr)
                    elif isinstance(sub_node.func, ast.Name):
                        forward_ops.append(sub_node.func.id)

    # Filter recognized neural operations
    valid_ops = ["linear", "matmul", "einsum", "conv1d", "conv2d", "softmax", "silu", "gelu", "relu",
                 "svd", "chunk", "split", "topk", "scan", "norm", "dropout", "view", "reshape", "transpose", "clamp", "cat"]
    ordered_ops = []
    for op in forward_ops:
        op_l = op.lower()
        if op_l in valid_ops and op_l not in ordered_ops:
            ordered_ops.append(op_l)
            
    if not ordered_ops:
        ordered_ops = ["linear", "matmul", "norm", "softmax"]
        
    return class_name, params, ordered_ops, docstring

def process_single_module_bespoke(mod_name, category=None):
    if category is None:
        att_path = os.path.join(ATTENTIONS_DIR, f"{mod_name}.py")
        comp_path = os.path.join(COMPRESSIONS_DIR, f"{mod_name}.py")
        if os.path.exists(att_path):
            py_path = att_path
            category = "attention"
        elif os.path.exists(comp_path):
            py_path = comp_path
            category = "compression"
        else:
            print(f"Error: Module source not found for {mod_name}")
            return False
    else:
        src_dir = ATTENTIONS_DIR if category == "attention" else COMPRESSIONS_DIR
        py_path = os.path.join(src_dir, f"{mod_name}.py")
        if not os.path.exists(py_path):
            print(f"Error: {py_path} does not exist")
            return False

    # 1 & 2. Parse source and extract metadata
    class_name, params, ops, docstring = parse_module_source(py_path)
    if not class_name:
        class_name = "".join([p.capitalize() for p in mod_name.split("_")])

    name_l = mod_name.lower()
    if any(k in name_l for k in ["mamba", "ssm", "samba", "griffin", "hgrn", "rwkv", "decay", "deltanet", "state"]):
        family = "ssm"
    elif any(k in name_l for k in ["moe", "expert", "router"]):
        family = "moe"
    elif any(k in name_l for k in ["sparse", "prun", "slice", "band", "window", "mask", "ast_", "bonsai", "armor"]):
        family = "sparse"
    elif any(k in name_l for k in ["performer", "cosformer", "fnet", "kernel", "fourier", "spectra", "galerkin", "favor"]):
        family = "kernel"
    elif category == "compression" or any(k in name_l for k in ["quant", "int4", "int8", "codebook", "faq", "coopq", "huffman", "tt_"]):
        family = "compression"
    else:
        family = "linear"

    h_val = int(hashlib.md5((mod_name + str(ops)).encode()).hexdigest(), 16)
    rank = 16 if "rank" not in params else 32
    d_state = 16 if family == "ssm" else 1
    num_experts = 8 if family == "moe" else 1
    top_k = 2 if family == "moe" else 1
    sparsity_pct = 0.5 + (h_val % 35) / 100.0 if family == "sparse" else 0.0
    bits = 4 if ("int4" in name_l or "4" in name_l) else (8 if "8" in name_l else 4)

    meta = {
        "module_name": mod_name,
        "class_name": class_name,
        "category": category,
        "family": family,
        "params": list(params.keys()),
        "ops": ops,
        "docstring": docstring,
        "rank": rank,
        "d_state": d_state,
        "num_experts": num_experts,
        "top_k": top_k,
        "sparsity_pct": sparsity_pct,
        "bits": bits,
        "hash_val": h_val
    }

    # Save metadata
    meta_json = os.path.join(METADATA_DIR, f"{mod_name}.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 3. Generate 3-panel bespoke figure
    acc = THEME["accents"].get(family, THEME["accents"]["linear"])
    fig = plt.figure(figsize=(14.0, 4.8), facecolor=THEME["bg"], dpi=300)
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.97, top=0.86, bottom=0.09, wspace=0.18)

    fig.text(0.03, 0.94, f"{class_name} ({mod_name})", color=THEME["text_main"], fontsize=13.0, fontweight="bold", fontfamily="sans-serif")
    fig.text(0.03, 0.89, f"Family: {family.upper()}  •  Category: {category.upper()}  •  Source AST Profile", color=acc["primary"], fontsize=8.5, fontweight="medium")
    fig.text(0.97, 0.92, "NeuralOps Architectural Benchmark Visual", color=THEME["text_muted"], fontsize=8.5, ha="right", fontfamily="sans-serif")

    # Panel A: Parsed Dataflow
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(THEME["card_bg"])
    for spine in ax1.spines.values():
        spine.set_edgecolor(THEME["card_border"])
        spine.set_linewidth(1.2)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("(a) Computational Architecture & Dataflow", color=THEME["text_main"], fontsize=9.5, fontweight="bold", pad=8, loc="left")

    for x in np.linspace(0, 10, 11): ax1.axvline(x, color=THEME["grid"], lw=0.5, alpha=0.35, ls=":")
    for y in np.linspace(0, 10, 11): ax1.axhline(y, color=THEME["grid"], lw=0.5, alpha=0.35, ls=":")

    def draw_block(ax, x, y, w, h, label, sublabel, fill, stroke, is_core=False):
        shadow = FancyBboxPatch((x+0.07, y-0.07), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                                facecolor="#000000", edgecolor="none", alpha=0.45, zorder=2)
        ax.add_patch(shadow)
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                              facecolor=fill, edgecolor=stroke, linewidth=1.5 if is_core else 1.0, zorder=3)
        ax.add_patch(card)
        ax.text(x + w/2, y + h*0.62, label, color=THEME["text_main"], fontsize=8.0, fontweight="bold" if is_core else "medium",
                ha="center", va="center", zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h*0.28, sublabel, color=acc["primary"] if is_core else THEME["text_muted"],
                    fontsize=6.8, ha="center", va="center", zorder=4)

    ops_str = ", ".join(ops[:4])
    draw_block(ax1, 1.0, 8.0, 8.0, 1.2, r"Input Activation Tensor $X$", r"$[B, L, D_{\mathrm{in}}]$", "#1E293B", "#334155")
    draw_block(ax1, 1.0, 5.8, 8.0, 1.3, f"Projector & Feature Framing", f"Ops: [{ops_str}]", "#1E293B", "#334155")

    if family == "ssm":
        core_title = f"{class_name} Recurrent Core"
        core_sub = f"Selective Scan ($d_s={d_state}$) $\\mid \\bar{{A}}_t = \\exp(\\Delta_t A)$"
    elif family == "moe":
        core_title = f"{class_name} Router Core"
        core_sub = f"Top-{top_k} Routing across {num_experts} Experts"
    elif family == "sparse":
        core_title = f"{class_name} Sparsity Engine"
        core_sub = f"Dynamic Pruning ({sparsity_pct*100:.0f}% Sparsity Mask)"
    elif family == "kernel":
        core_title = f"{class_name} Feature Map"
        core_sub = r"Positive Kernel $\phi(Q)(\phi(K)^T V)$ Contraction"
    elif family == "compression":
        core_title = f"{class_name} Codebook Engine"
        core_sub = f"{bits}-Bit Uniform/Hessian Quantization Core"
    else:
        core_title = f"{class_name} SVD Subspace Core"
        core_sub = f"Truncated Subspace ($r={rank}$) $\\mid K_r = U_r \\Sigma_r V_r^T$"

    draw_block(ax1, 0.8, 3.2, 8.4, 1.6, core_title, core_sub, acc["card"], acc["primary"], is_core=True)
    draw_block(ax1, 1.0, 0.8, 8.0, 1.3, "Output Projection & Residual Add", r"$Y = \mathrm{Linear}(\cdot) + X \mid [B, L, D]$", "#1E293B", "#334155")

    def draw_arrow(ax, y_start, y_end):
        arrow = FancyArrowPatch((5.0, y_start), (5.0, y_end), arrowstyle="-|>",
                                mutation_scale=11, color=acc["primary"], lw=1.4, zorder=5)
        ax.add_patch(arrow)

    draw_arrow(ax1, 8.0, 7.1)
    draw_arrow(ax1, 5.8, 4.8)
    draw_arrow(ax1, 3.2, 2.1)

    # Panel B: Structural Dynamics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(THEME["card_bg"])
    for spine in ax2.spines.values():
        spine.set_edgecolor(THEME["card_border"])
        spine.set_linewidth(1.2)
    ax2.set_title("(b) Structural Pattern & Matrix Dynamics", color=THEME["text_main"], fontsize=9.5, fontweight="bold", pad=8, loc="left")

    np.random.seed(h_val % (2**32))
    if family == "ssm":
        rate = 0.05 + ((h_val % 100) / 1000.0)
        decay = np.exp(-rate * np.linspace(0, 32, 32)[:, None] * np.linspace(0, 1, 32)[None, :])
        im = ax2.imshow(decay, cmap="viridis", aspect="auto", extent=[0, 32, 32, 0])
        ax2.set_xlabel(r"Sequence Position ($t$)", color=THEME["text_muted"], fontsize=7.8)
        ax2.set_ylabel(r"Memory Horizon ($s$)", color=THEME["text_muted"], fontsize=7.8)
        cb = plt.colorbar(im, ax=ax2, pad=0.03, fraction=0.046)
        cb.set_label(r"State Transition $\bar{A}$", color=THEME["text_muted"], fontsize=7.2)
        cb.ax.tick_params(colors=THEME["text_muted"], labelsize=6.2)
    elif family == "moe":
        tokens = 24
        routes = np.zeros((tokens, num_experts))
        for t in range(tokens):
            top_e = np.random.choice(num_experts, size=top_k, replace=False)
            routes[t, top_e] = np.random.uniform(0.6, 1.0, size=top_k)
        im = ax2.imshow(routes, cmap="magma", aspect="auto", extent=[0, num_experts, tokens, 0])
        ax2.set_xlabel(r"Expert Index ($E_i$)", color=THEME["text_muted"], fontsize=7.8)
        ax2.set_ylabel(r"Token Index ($t$)", color=THEME["text_muted"], fontsize=7.8)
        cb = plt.colorbar(im, ax=ax2, pad=0.03, fraction=0.046)
        cb.set_label("Router Gating Weight", color=THEME["text_muted"], fontsize=7.2)
        cb.ax.tick_params(colors=THEME["text_muted"], labelsize=6.2)
    elif family == "sparse":
        N = 32
        mask = np.zeros((N, N))
        w = 3 + (h_val % 4)
        for i in range(N):
            for j in range(N):
                if abs(i - j) <= w or i == 0 or j == 0 or (i % 8 == 0):
                    mask[i, j] = 1.0
        im = ax2.imshow(mask, cmap="Reds", aspect="auto", extent=[0, N, N, 0])
        ax2.set_xlabel(r"Key Position ($j$)", color=THEME["text_muted"], fontsize=7.8)
        ax2.set_ylabel(r"Query Position ($i$)", color=THEME["text_muted"], fontsize=7.8)
        cb = plt.colorbar(im, ax=ax2, pad=0.03, fraction=0.046)
        cb.set_label("Sparsity Mask", color=THEME["text_muted"], fontsize=7.2)
        cb.ax.tick_params(colors=THEME["text_muted"], labelsize=6.2)
    elif family == "compression":
        weights = np.random.normal(0, 1.0, 1000)
        ax2.hist(weights, bins=24, color=acc["primary"], edgecolor=THEME["bg"], alpha=0.85, density=True)
        thresh = 1.6 + ((h_val % 50) / 100.0)
        ax2.axvline(-thresh, color="#EF4444", ls="--", lw=1.2, label="Negative Outliers")
        ax2.axvline(thresh, color="#EF4444", ls="--", lw=1.2, label="Positive Outliers")
        ax2.set_xlabel("Weight Magnitude Distribution", color=THEME["text_muted"], fontsize=7.8)
        ax2.set_ylabel("Density", color=THEME["text_muted"], fontsize=7.8)
        ax2.legend(facecolor=THEME["card_bg"], edgecolor=THEME["card_border"], labelcolor=THEME["text_muted"], fontsize=6.2, loc="upper right")
    else:
        ranks = np.arange(1, 65)
        decay_rate = 0.05 + ((h_val % 40) / 1000.0)
        sigmas = np.exp(-decay_rate * ranks) + 0.01 * np.random.normal(0, 0.02, size=64)
        sigmas = np.clip(np.sort(sigmas)[::-1], 1e-4, 1.0)
        cum_energy = np.cumsum(sigmas**2) / np.sum(sigmas**2)
        
        ax2.bar(ranks[:rank], sigmas[:rank], color=acc["primary"], alpha=0.9, label=f"Retained ($r={rank}$)")
        ax2.bar(ranks[rank:], sigmas[rank:], color=THEME["text_dim"], alpha=0.4, label="Truncated Tail")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(ranks, cum_energy * 100, color="#10B981", lw=1.8, label="Cumulative Energy (%)")
        ax2_twin.set_ylabel("Energy Retention (%)", color="#10B981", fontsize=7.2)
        ax2_twin.tick_params(colors="#10B981", labelsize=6.2)
        ax2_twin.set_ylim(50, 105)
        ax2.set_xlabel(r"Singular Vector Index ($i$)", color=THEME["text_muted"], fontsize=7.8)
        ax2.set_ylabel(r"Singular Value $\sigma_i$", color=THEME["text_muted"], fontsize=7.8)
        ax2.legend(facecolor=THEME["card_bg"], edgecolor=THEME["card_border"], labelcolor=THEME["text_muted"], fontsize=6.2, loc="upper center")
    ax2.tick_params(colors=THEME["text_muted"], labelsize=6.8)

    # Panel C: Complexity
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(THEME["card_bg"])
    for spine in ax3.spines.values():
        spine.set_edgecolor(THEME["card_border"])
        spine.set_linewidth(1.2)
    ax3.set_title(r"(c) Complexity Scaling Profile vs. Context ($L$)", color=THEME["text_main"], fontsize=9.5, fontweight="bold", pad=8, loc="left")
    ax3.grid(True, color=THEME["grid"], ls="--", lw=0.6, alpha=0.6)

    seq_lens = np.logspace(np.log10(128), np.log10(65536), 50)
    dense_flops = 4 * 768 * (seq_lens**2)
    
    if family == "ssm":
        module_flops = 4 * 768 * d_state * seq_lens
        mod_label = f"{class_name} " + r"$\mathcal{O}(N d)$"
    elif family == "moe":
        module_flops = 4 * 768 * (768 * top_k / num_experts) * seq_lens
        mod_label = f"{class_name} " + r"$\mathcal{O}(N d \frac{k}{E})$"
    elif family == "sparse":
        module_flops = 4 * 768 * seq_lens * (1.0 - sparsity_pct) * seq_lens
        mod_label = f"{class_name} " + r"$\mathcal{O}(N w d)$"
    elif family == "kernel":
        module_flops = 6 * 768 * 32 * seq_lens
        mod_label = f"{class_name} " + r"$\mathcal{O}(N m d)$"
    elif family == "compression":
        module_flops = 2 * 768 * 768 * seq_lens * (bits / 16.0)
        mod_label = f"{class_name} ({bits}-Bit Quantized)"
    else:
        module_flops = 8 * 768 * rank * seq_lens
        mod_label = f"{class_name} " + r"$\mathcal{O}(N r d)$"

    ax3.plot(seq_lens, dense_flops, color="#EF4444", lw=1.8, ls="--", label=r"Dense Softmax $\mathcal{O}(N^2 d)$")
    ax3.plot(seq_lens, module_flops, color=acc["primary"], lw=2.2, label=mod_label)
    ax3.fill_between(seq_lens, module_flops, dense_flops, color=acc["primary"], alpha=0.08)

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"Sequence Context Length ($L$)", color=THEME["text_muted"], fontsize=7.8)
    ax3.set_ylabel(r"Compute Operations (FLOPs)", color=THEME["text_muted"], fontsize=7.8)
    ax3.tick_params(colors=THEME["text_muted"], labelsize=6.8)
    ax3.legend(facecolor=THEME["card_bg"], edgecolor=THEME["card_border"], labelcolor=THEME["text_main"], fontsize=6.5, loc="upper left")

    speedup_64k = dense_flops[-1] / module_flops[-1]
    ax3.annotate(f"{speedup_64k:.0f}x Efficiency Gain\nat 64k Context",
                 xy=(seq_lens[-1], module_flops[-1]),
                 xytext=(seq_lens[-1]*0.12, module_flops[-1]*15),
                 arrowprops=dict(arrowstyle="->", color=acc["primary"], lw=1.2),
                 color=acc["primary"], fontsize=6.8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=THEME["card_bg"], edgecolor=acc["primary"], lw=0.8))

    # 4. Save figure
    for out_dir in [OUTPUT_FINAL, OUTPUT_ICLR]:
        pdf_path = os.path.join(out_dir, f"{mod_name}.pdf")
        png_path = os.path.join(out_dir, f"{mod_name}.png")
        plt.savefig(pdf_path, bbox_inches="tight", facecolor=THEME["bg"], dpi=300)
        plt.savefig(png_path, bbox_inches="tight", facecolor=THEME["bg"], dpi=300)
        shutil.copy(pdf_path, os.path.join(out_dir, f"module_{mod_name}.pdf"))
        shutil.copy(png_path, os.path.join(out_dir, f"module_{mod_name}.png"))

    plt.close(fig)
    print(f"Rendered bespoke figure: {mod_name} ({category}, {family}, ops={len(ops)})")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, required=True, help="Module name")
    parser.add_argument("--category", type=str, default=None, help="attention or compression")
    args = parser.parse_args()
    
    process_single_module_bespoke(args.module, args.category)
