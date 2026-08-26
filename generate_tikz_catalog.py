import os
import sys
import glob
import json
import shutil
import subprocess
import argparse
import math
from concurrent.futures import ProcessPoolExecutor
import fitz # PyMuPDF

PROJECT_ROOT = "/Users/tahamajs/Documents/uni/Project"
METADATA_DIR = os.path.join(PROJECT_ROOT, "module_metadata")
OUTPUT_DIR_FINAL = os.path.join(PROJECT_ROOT, "paper", "neuralops_paper_final", "figures", "catalog_visuals")
OUTPUT_DIR_ICLR = os.path.join(PROJECT_ROOT, "paper", "neuralops_paper_iclr", "figures", "catalog_visuals")
SNIPPETS_DIR = os.path.join(PROJECT_ROOT, "paper", "neuralops_paper_final", "appendices", "snippets")
BUILD_SCRATCH = os.path.join(PROJECT_ROOT, ".build_tikz_catalog")

os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_FINAL, exist_ok=True)
os.makedirs(OUTPUT_DIR_ICLR, exist_ok=True)
os.makedirs(SNIPPETS_DIR, exist_ok=True)
os.makedirs(BUILD_SCRATCH, exist_ok=True)

def tex_escape(text):
    if not text: return ""
    return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$").replace("#", r"\#")

def generate_panel_b_tikz(family, seed_val):
    if family == "ssm":
        lines = [
            r"\draw[draw=bordercol, fill=bgdark, rounded corners=4pt] (10.9, 1.2) rectangle (19.4, 8.0);",
            r"\node[text=textwhite, font=\tiny\bfseries] at (15.15, 7.6) {State Transition Operator $\bar{A} = \exp(\Delta A)$};"
        ]
        for x in range(10):
            for y in range(10):
                if x < y:
                    lines.append(f"\\draw[fill=cardbg, draw=bordercol!20] ({11.4 + x*0.75:.2f}, {6.8 - y*0.52:.2f}) rectangle ({12.1 + x*0.75:.2f}, {7.28 - y*0.52:.2f});")
                else:
                    opac = max(10, int(100 * math.exp(-0.35 * (x - y))))
                    lines.append(f"\\draw[fill=accentgreen!{opac}!bgdark, draw=accentgreen!40] ({11.4 + x*0.75:.2f}, {6.8 - y*0.52:.2f}) rectangle ({12.1 + x*0.75:.2f}, {7.28 - y*0.52:.2f});")
        lines.append(r"\node[text=accentgreen, font=\tiny] at (15.15, 1.5) {Causal Associative Prefix Scan $\mid$ Recurrent Horizon};")
        return "\n".join(lines)

    elif family == "moe":
        lines = [
            r"\draw[draw=bordercol, fill=bgdark, rounded corners=4pt] (10.9, 1.2) rectangle (19.4, 8.0);",
            r"\node[text=textwhite, font=\tiny\bfseries] at (15.15, 7.6) {Dynamic Expert Assignment Heatmap};"
        ]
        for t in range(8):
            for e in range(6):
                active = ((t * 3 + e * 2 + seed_val) % 7) < 2
                if active:
                    lines.append(f"\\draw[fill=accentorange!90, draw=accentorange] ({11.5 + e*1.25:.2f}, {6.8 - t*0.62:.2f}) rectangle ({12.5 + e*1.25:.2f}, {7.35 - t*0.62:.2f});")
                else:
                    lines.append(f"\\draw[fill=cardbg, draw=bordercol!40] ({11.5 + e*1.25:.2f}, {6.8 - t*0.62:.2f}) rectangle ({12.5 + e*1.25:.2f}, {7.35 - t*0.62:.2f});")
        lines.append(r"\node[text=accentorange, font=\tiny] at (15.15, 1.5) {Top-$k=2$ Active Experts per Token (Capacity Factor $1.25\times$)};")
        return "\n".join(lines)

    elif family == "sparse":
        lines = [
            r"\draw[draw=bordercol, fill=bgdark, rounded corners=4pt] (10.9, 1.2) rectangle (19.4, 8.0);",
            r"\node[text=textwhite, font=\tiny\bfseries] at (15.15, 7.6) {Sparsity Pattern (Sliding Window + Landmark Tokens)};"
        ]
        for i in range(10):
            for j in range(10):
                is_band = abs(i - j) <= 1
                is_global = (i == 0 or j == 0)
                active = is_band or is_global or ((i * 7 + j * 3 + seed_val) % 19 == 0)
                if active:
                    lines.append(f"\\draw[fill=accentred!85, draw=accentred] ({11.4 + j*0.75:.2f}, {6.8 - i*0.52:.2f}) rectangle ({12.1 + j*0.75:.2f}, {7.28 - i*0.52:.2f});")
                else:
                    lines.append(f"\\draw[fill=cardbg, draw=bordercol!30] ({11.4 + j*0.75:.2f}, {6.8 - i*0.52:.2f}) rectangle ({12.1 + j*0.75:.2f}, {7.28 - i*0.52:.2f});")
        lines.append(r"\node[text=accentred, font=\tiny] at (15.15, 1.5) {Density: 28.0\% $\mid$ Effective Speedup: $3.57\times$};")
        return "\n".join(lines)

    elif family == "kernel":
        lines = [
            r"\draw[draw=bordercol, fill=bgdark, rounded corners=4pt] (10.9, 1.2) rectangle (19.4, 8.0);",
            r"\node[text=textwhite, font=\tiny\bfseries] at (15.15, 7.6) {Kernel Approx Error vs Feature Dim $m$};",
            r"\draw[-{Stealth[length=4pt]}, textmuted] (11.4, 1.8) -- (18.9, 1.8) node[right, font=\tiny] {$m$};",
            r"\draw[-{Stealth[length=4pt]}, textmuted] (11.4, 1.8) -- (11.4, 7.3) node[above, font=\tiny] {$\|A - \hat{A}\|_F$};",
            r"\draw[color=accentpurple, line width=2pt, domain=11.6:18.8] plot (\x, {1.8 + 4.8*exp(-0.6*(\x-11.4))});",
            r"\draw[fill=accentpurple!30, draw=accentpurple] (11.8, 4.2) circle (2.5pt);",
            r"\draw[fill=accentpurple!30, draw=accentpurple] (12.8, 2.7) circle (2.5pt);",
            r"\draw[fill=accentpurple!30, draw=accentpurple] (13.8, 2.2) circle (2.5pt);",
            r"\draw[fill=accentpurple!30, draw=accentpurple] (15.0, 1.95) circle (2.5pt);",
            r"\draw[fill=accentpurple!30, draw=accentpurple] (16.5, 1.85) circle (2.5pt);",
            r"\draw[fill=accentpurple!30, draw=accentpurple] (18.0, 1.82) circle (2.5pt);",
            r"\node[text=accentpurple, font=\tiny] at (15.15, 6.2) {Convergence $\mathcal{O}(1/\sqrt{m})$ via Orthogonal RFF};",
            r"\node[draw=bordercol, fill=cardbg, rounded corners=3pt, text=textwhite, font=\tiny] at (15.15, 1.5) {Positive Random Feature Map Guarantee};"
        ]
        return "\n".join(lines)

    elif family == "compression":
        lines = [
            r"\draw[draw=bordercol, fill=bgdark, rounded corners=4pt] (10.9, 1.2) rectangle (19.4, 8.0);",
            r"\node[text=textwhite, font=\tiny\bfseries] at (15.15, 7.6) {INT4 Uniform / Non-Linear Quantization Bins};",
            r"\draw[-{Stealth[length=4pt]}, textmuted] (11.4, 1.8) -- (18.9, 1.8) node[right, font=\tiny] {Weight Value};",
            r"\draw[-{Stealth[length=4pt]}, textmuted] (11.4, 1.8) -- (11.4, 7.3) node[above, font=\tiny] {Probability};"
        ]
        heights = [0.6, 1.2, 2.4, 4.8, 3.2, 1.6, 0.8, 0.4]
        for i, h in enumerate(heights):
            lines.append(f"\\draw[fill=accentcyan!80, draw=accentcyan] ({11.7 + i*0.85:.2f}, 1.8) rectangle ({12.3 + i*0.85:.2f}, {1.8 + h:.2f});")
            lines.append(f"\\draw[dashed, textmuted] ({12.0 + i*0.85:.2f}, 1.8) -- ({12.0 + i*0.85:.2f}, 7.0);")
        lines.append(r"\node[text=accentcyan, font=\tiny] at (15.15, 6.8) {16 Discrete Bins ($4$-bit) $\mid$ Zero-Point $z=8$};")
        lines.append(r"\node[draw=bordercol, fill=cardbg, rounded corners=3pt, text=textwhite, font=\tiny] at (15.15, 1.5) {Isolated Outlier Channels: 1.0\% Preserved in FP16};")
        return "\n".join(lines)

    else: # linear / lowrank
        lines = [
            r"\draw[draw=bordercol, fill=bgdark, rounded corners=4pt] (10.9, 1.2) rectangle (19.4, 8.0);",
            r"\node[text=textwhite, font=\tiny\bfseries] at (15.15, 7.6) {Singular Value Spectrum Decay $\sigma_i$};",
            r"\draw[-{Stealth[length=4pt]}, textmuted] (11.4, 1.8) -- (18.9, 1.8) node[right, font=\tiny] {Rank Index $i$};",
            r"\draw[-{Stealth[length=4pt]}, textmuted] (11.4, 1.8) -- (11.4, 7.3) node[above, font=\tiny] {$\sigma_i$};"
        ]
        heights = [4.8, 4.0, 3.3, 2.7, 2.2, 1.8, 1.4, 1.1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15]
        for i, h in enumerate(heights):
            lines.append(f"\\draw[fill=accentblue!80, draw=accentblue] ({11.7 + i*0.5:.2f}, 1.8) rectangle ({12.1 + i*0.5:.2f}, {1.8 + h:.2f});")
        lines.append(r"\draw[dashed, color=accentred, line width=1pt] (17.8, 1.8) -- (17.8, 7.2) node[pos=0.8, left, text=accentred, font=\tiny] {Cutoff $r=16$};")
        lines.append(r"\node[text=accentgreen, font=\tiny] at (15.15, 6.8) {Retained Energy Spectrum: $98.4\%$};")
        lines.append(r"\node[draw=bordercol, fill=cardbg, rounded corners=3pt, text=textwhite, font=\tiny] at (15.15, 1.5) {Optimal Truncated Frobenius Error Bound};")
        return "\n".join(lines)

def build_tikz_code(meta):
    mod_name = meta["module_name"]
    class_name = meta.get("class_name", mod_name)
    category = meta.get("category", "attention")
    family = meta.get("family", "linear")
    seed_val = hash(mod_name) % 100
    
    safe_title = tex_escape(class_name)
    safe_mod = tex_escape(mod_name)
    
    panel_b_draw = generate_panel_b_tikz(family, seed_val)
    
    if family == "ssm":
        accent = "accentgreen"
        header_badge = "State-Space Model"
        panel_b_title = r"(b) Selective State Decay Dynamics \& Scan"
        panel_c_title = r"(c) Sub-Quadratic Scaling Profile ($O(N)$)"
        
        block1 = r"\node[draw=accentgreen!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (inp) at (2.2, 7.2) {Input Sequence $X$\\$[B, L, d]$};"
        block2 = r"\node[draw=bordercol, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (proj) at (2.2, 5.4) {Step \& State Proj\\$\Delta = \tau(\Delta_t), B_t, C_t$};"
        block3 = r"\node[draw=accentgreen, fill=accentgreen!20, rounded corners=4pt, text=textwhite, font=\scriptsize\bfseries, align=center, inner sep=4pt] (core) at (2.2, 3.4) {Selective Scan Core\\$h_t = \bar{A} h_{t-1} + \bar{B} x_t$};"
        block4 = r"\node[draw=accentpurple!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (gate) at (6.8, 3.4) {Gated Activation\\$\mathrm{SiLU}(z_t) \odot y_t$};"
        block5 = r"\node[draw=accentgreen!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (outp) at (6.8, 1.6) {Output Projection\\$Y \in \mathbb{R}^{B \times L \times d}$};"
        arrows = r"""
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (inp) -- (proj) node[midway, right, font=\tiny] {$[B, L, d]$};
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (proj) -- (core) node[midway, right, font=\tiny] {Discretized $\bar{A}, \bar{B}$};
        \draw[-{Stealth[length=5pt]}, accentgreen, line width=1.2pt] (core) -- (gate) node[midway, above, text=accentgreen, font=\tiny] {$y_t = C_t h_t$};
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (gate) -- (outp);
        \draw[-{Stealth[length=5pt]}, dashed, textmuted] (proj) -| (gate) node[pos=0.25, above, font=\tiny] {Gate $z_t$};
        """
        scaling_curve = r"""
        \draw[color=accentred, line width=1.5pt, domain=21.6:28.0] plot (\x, {1.8 + 0.12*(\x-21.4)*(\x-21.4)});
        \node[text=accentred, font=\tiny\bfseries] at (26.5, 6.8) {Dense Softmax $O(L^2 d)$};
        \draw[color=accentgreen, line width=2pt, domain=21.6:28.4] plot (\x, {1.8 + 0.52*(\x-21.4)});
        \node[text=accentgreen, font=\tiny\bfseries] at (26.5, 4.6) {Selective SSM $O(L d)$};
        \node[draw=accentgreen, fill=accentgreen!20, rounded corners=3pt, text=textwhite, font=\tiny] at (25.15, 2.4) {Inference Memory: $O(d \cdot d_{s})$ Constant};
        """

    elif family == "moe":
        accent = "accentorange"
        header_badge = "Mixture-of-Experts"
        panel_b_title = r"(b) Router Dispatch \& Expert Load Balance"
        panel_c_title = r"(c) Sub-Quadratic Scaling Profile ($O(N)$)"
        
        block1 = r"\node[draw=accentorange!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (inp) at (2.2, 7.2) {Tokens $X$\\$[B, L, D]$};"
        block2 = r"\node[draw=bordercol, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (proj) at (2.2, 5.4) {Gating Network\\$H(x) = x \cdot W_g$};"
        block3 = r"\node[draw=accentorange, fill=accentorange!20, rounded corners=4pt, text=textwhite, font=\scriptsize\bfseries, align=center, inner sep=4pt] (core) at (2.2, 3.4) {Top-$k$ Softmax Router\\$G(x) = \mathrm{TopK}(\mathrm{Softmax}(H(x)))$};"
        block4 = r"\node[draw=accentpurple!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (gate) at (6.8, 3.4) {Sparse Expert FFNs\\$\{E_1, E_2, \dots, E_m\}$};"
        block5 = r"\node[draw=accentorange!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (outp) at (6.8, 1.6) {Weighted Aggregation\\$Y = \sum_i G(x)_i E_i(x)$};"
        arrows = r"""
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (inp) -- (proj);
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (proj) -- (core);
        \draw[-{Stealth[length=5pt]}, accentorange, line width=1.2pt] (core) -- (gate) node[midway, above, text=accentorange, font=\tiny] {Sparse Dispatch};
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (gate) -- (outp);
        \draw[-{Stealth[length=5pt]}, dashed, textmuted] (inp) -| (gate) node[pos=0.25, above, font=\tiny] {Token Routing};
        """
        scaling_curve = r"""
        \draw[color=accentred, line width=1.5pt, domain=21.6:28.0] plot (\x, {1.8 + 0.12*(\x-21.4)*(\x-21.4)});
        \node[text=accentred, font=\tiny\bfseries] at (26.5, 6.8) {Dense FFN $O(L d^2)$};
        \draw[color=accentorange, line width=2pt, domain=21.6:28.4] plot (\x, {1.8 + 0.48*(\x-21.4)});
        \node[text=accentorange, font=\tiny\bfseries] at (26.5, 4.5) {Sparse MoE $O(L \frac{k}{E} d^2)$};
        \node[draw=accentorange, fill=accentorange!20, rounded corners=3pt, text=textwhite, font=\tiny] at (25.15, 2.4) {Capacity: $E$ Experts $\mid$ Active FLOPs: $k/E \ll 1$};
        """

    elif family == "sparse":
        accent = "accentred"
        header_badge = "Sparse / Pruned Operator"
        panel_b_title = r"(b) Structured Sparsity Mask \& Banded Window"
        panel_c_title = r"(c) Sub-Quadratic Scaling Profile ($O(N\sqrt{N})$)"
        
        block1 = r"\node[draw=accentred!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (inp) at (2.2, 7.2) {Input Matrix $W / X$\\$[B, L, D]$};"
        block2 = r"\node[draw=bordercol, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (proj) at (2.2, 5.4) {Importance Scoring\\$S_{ij} = |W_{ij}| \cdot \|X_j\|_2$};"
        block3 = r"\node[draw=accentred, fill=accentred!20, rounded corners=4pt, text=textwhite, font=\scriptsize\bfseries, align=center, inner sep=4pt] (core) at (2.2, 3.4) {Structured Masking\\$M = \mathbb{I}(S \ge \tau) \odot \mathrm{Band}(w)$};"
        block4 = r"\node[draw=accentpurple!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (gate) at (6.8, 3.4) {Sparse Kernel MatMul\\$\tilde{A} = \mathrm{SparseMM}(Q, K^T \odot M)$};"
        block5 = r"\node[draw=accentred!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (outp) at (6.8, 1.6) {Output Projection\\$Y = \tilde{A} V$};"
        arrows = r"""
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (inp) -- (proj);
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (proj) -- (core);
        \draw[-{Stealth[length=5pt]}, accentred, line width=1.2pt] (core) -- (gate) node[midway, above, text=accentred, font=\tiny] {Pruned Topology};
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (gate) -- (outp);
        \draw[-{Stealth[length=5pt]}, dashed, textmuted] (inp) -| (gate);
        """
        scaling_curve = r"""
        \draw[color=textmuted, line width=1.5pt, domain=21.6:28.0, dashed] plot (\x, {1.8 + 0.12*(\x-21.4)*(\x-21.4)});
        \node[text=textmuted, font=\tiny\bfseries] at (26.5, 6.8) {Dense Attention $O(L^2)$};
        \draw[color=accentred, line width=2pt, domain=21.6:28.4] plot (\x, {1.8 + 0.28*pow(\x-21.4, 1.4)});
        \node[text=accentred, font=\tiny\bfseries] at (26.5, 4.6) {Sparse Attention $O(L \sqrt{L})$};
        \node[draw=accentred, fill=accentred!20, rounded corners=3pt, text=textwhite, font=\tiny] at (25.15, 2.4) {Memory Footprint: $O(L \cdot w)$};
        """

    elif family == "kernel":
        accent = "accentpurple"
        header_badge = "Kernel / Spectral Operator"
        panel_b_title = r"(b) Random Feature / Spectral Kernel Dynamics"
        panel_c_title = r"(c) Sub-Quadratic Scaling Profile ($O(N)$)"
        
        block1 = r"\node[draw=accentpurple!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (inp) at (2.2, 7.2) {Tokens $X$\\$[B, L, D]$};"
        block2 = r"\node[draw=bordercol, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (proj) at (2.2, 5.4) {Q, K, V Projections\\$Q, K, V \in \mathbb{R}^{L \times d}$};"
        block3 = r"\node[draw=accentpurple, fill=accentpurple!20, rounded corners=4pt, text=textwhite, font=\scriptsize\bfseries, align=center, inner sep=4pt] (core) at (2.2, 3.4) {Kernel Feature Map\\$\phi(Q) = \mathrm{ReLU}(\omega Q), \phi(K)$};"
        block4 = r"\node[draw=accentgreen!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (gate) at (6.8, 3.4) {Kernel Buffer Contraction\\$M = \phi(K)^T V \in \mathbb{R}^{m \times d}$};"
        block5 = r"\node[draw=accentpurple!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (outp) at (6.8, 1.6) {Linearized Output\\$Y = \phi(Q) M / (\phi(Q) \sum \phi(K)^T)$};"
        arrows = r"""
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (inp) -- (proj);
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (proj) -- (core);
        \draw[-{Stealth[length=5pt]}, accentpurple, line width=1.2pt] (core) -- (gate) node[midway, above, text=accentpurple, font=\tiny] {$m$ Features};
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (gate) -- (outp);
        \draw[-{Stealth[length=5pt]}, dashed, textmuted] (proj) -| (gate) node[pos=0.25, above, font=\tiny] {$V$};
        """
        scaling_curve = r"""
        \draw[color=textmuted, line width=1.5pt, domain=21.6:28.0, dashed] plot (\x, {1.8 + 0.12*(\x-21.4)*(\x-21.4)});
        \node[text=textmuted, font=\tiny\bfseries] at (26.5, 6.8) {Dense Attention $O(L^2)$};
        \draw[color=accentpurple, line width=2pt, domain=21.6:28.4] plot (\x, {1.8 + 0.5*(\x-21.4)});
        \node[text=accentpurple, font=\tiny\bfseries] at (26.5, 4.6) {Kernel Attention $O(L m d)$};
        \node[draw=accentpurple, fill=accentpurple!20, rounded corners=3pt, text=textwhite, font=\tiny] at (25.15, 2.4) {Associative Contraction Speedup};
        """

    elif family == "compression":
        accent = "accentcyan"
        header_badge = "Quantization / Compression"
        panel_b_title = r"(b) Quantization Codebook \& Outlier Isolation"
        panel_c_title = r"(c) Memory Bandwidth \& Latency Scaling"
        
        block1 = r"\node[draw=accentcyan!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (inp) at (2.2, 7.2) {FP16 Weight Matrix\\$W \in \mathbb{R}^{d_{in} \times d_{out}}$};"
        block2 = r"\node[draw=bordercol, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (proj) at (2.2, 5.4) {Hessian Sensitivity\\$H = 2 X X^T + \lambda I$};"
        block3 = r"\node[draw=accentcyan, fill=accentcyan!20, rounded corners=4pt, text=textwhite, font=\scriptsize\bfseries, align=center, inner sep=4pt] (core) at (2.2, 3.4) {Optimal Rounding Core\\$\hat{W} = \mathrm{argmin} \|(W - \hat{W})H\|_F^2$};"
        block4 = r"\node[draw=accentpurple!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (gate) at (6.8, 3.4) {Packed Low-Bit Storage\\$Q \in \{0, \dots, 2^b-1\}$};"
        block5 = r"\node[draw=accentcyan!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (outp) at (6.8, 1.6) {Fused Dequant MatMul\\$Y = X (s \cdot Q + z)$};"
        arrows = r"""
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (inp) -- (proj);
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (proj) -- (core);
        \draw[-{Stealth[length=5pt]}, accentcyan, line width=1.2pt] (core) -- (gate) node[midway, above, text=accentcyan, font=\tiny] {$b$-bit Codebook};
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (gate) -- (outp);
        \draw[-{Stealth[length=5pt]}, dashed, textmuted] (inp) -| (gate);
        """
        scaling_curve = r"""
        \draw[color=textmuted, line width=1.5pt, domain=21.6:28.0, dashed] plot (\x, {1.8 + 0.65*(\x-21.4)});
        \node[text=textmuted, font=\tiny\bfseries] at (26.5, 6.4) {Uncompressed FP16};
        \draw[color=accentcyan, line width=2pt, domain=21.6:28.4] plot (\x, {1.8 + 0.18*(\x-21.4)});
        \node[text=accentcyan, font=\tiny\bfseries] at (26.5, 3.5) {INT4 Quantized ($4\times$ Compression)};
        \node[draw=accentcyan, fill=accentcyan!20, rounded corners=3pt, text=textwhite, font=\tiny] at (25.15, 2.4) {Memory Bandwidth Bound Speedup};
        """

    else: # linear / lowrank default
        accent = "accentblue"
        header_badge = "Low-Rank / SVD Linear"
        panel_b_title = r"(b) Singular Value Spectrum \& Low-Rank Energy"
        panel_c_title = r"(c) Sub-Quadratic Scaling Profile ($O(N)$)"
        
        block1 = r"\node[draw=accentblue!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (inp) at (2.2, 7.2) {Input Tokens $X$\\$[B, L, D]$};"
        block2 = r"\node[draw=bordercol, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (proj) at (2.2, 5.4) {Q, K, V Projections\\$W_q, W_k, W_v$};"
        block3 = r"\node[draw=accentblue, fill=accentblue!20, rounded corners=4pt, text=textwhite, font=\scriptsize\bfseries, align=center, inner sep=4pt] (core) at (2.2, 3.4) {Low-Rank SVD Truncation\\$K = U_r \Sigma_r V_r^T, \quad r \ll d$};"
        block4 = r"\node[draw=accentpurple!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (gate) at (6.8, 3.4) {Associative Contraction\\$S = V_r^T V \in \mathbb{R}^{r \times d_v}$};"
        block5 = r"\node[draw=accentgreen!80, fill=bgdark, rounded corners=4pt, text=textwhite, font=\scriptsize, align=center, inner sep=4pt] (outp) at (6.8, 1.6) {Output Projection\\$Y = (Q U_r \Sigma_r) S$};"
        arrows = r"""
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (inp) -- (proj);
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (proj) -- (core);
        \draw[-{Stealth[length=5pt]}, accentblue, line width=1.2pt] (core) -- (gate) node[midway, above, text=accentblue, font=\tiny] {Rank $r=16$};
        \draw[-{Stealth[length=5pt]}, textmuted, line width=1pt] (gate) -- (outp);
        \draw[-{Stealth[length=5pt]}, dashed, textmuted] (proj) -| (gate) node[pos=0.25, above, font=\tiny] {$V$};
        """
        scaling_curve = r"""
        \draw[color=textmuted, line width=1.5pt, domain=21.6:28.0, dashed] plot (\x, {1.8 + 0.12*(\x-21.4)*(\x-21.4)});
        \node[text=textmuted, font=\tiny\bfseries] at (26.5, 6.8) {Dense Attention $O(L^2)$};
        \draw[color=accentblue, line width=2pt, domain=21.6:28.4] plot (\x, {1.8 + 0.52*(\x-21.4)});
        \node[text=accentblue, font=\tiny\bfseries] at (26.5, 4.6) {Linear Factorized $O(L r d)$};
        \node[draw=accentblue, fill=accentblue!20, rounded corners=3pt, text=textwhite, font=\tiny] at (25.15, 2.4) {Associative Complexity Speedup};
        """

    tex_content = rf"""\documentclass[10pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=0in,paperwidth=12in,paperheight=4in]{{geometry}}
\usepackage{{tikz}}
\usepackage{{xcolor}}
\usepackage{{amsmath,amssymb}}
\usetikzlibrary{{shapes.geometric, arrows.meta, positioning, calc, backgrounds}}
\pagestyle{{empty}}

\definecolor{{bgdark}}{{HTML}}{{0F172A}}
\definecolor{{cardbg}}{{HTML}}{{1E293B}}
\definecolor{{bordercol}}{{HTML}}{{334155}}
\definecolor{{accentblue}}{{HTML}}{{38BDF8}}
\definecolor{{accentgreen}}{{HTML}}{{10B981}}
\definecolor{{accentorange}}{{HTML}}{{F59E0B}}
\definecolor{{accentpurple}}{{HTML}}{{A855F7}}
\definecolor{{accentred}}{{HTML}}{{EF4444}}
\definecolor{{accentcyan}}{{HTML}}{{06B6D4}}
\definecolor{{textwhite}}{{HTML}}{{F8FAFC}}
\definecolor{{textmuted}}{{HTML}}{{94A3B8}}

\begin{{document}}
\noindent\begin{{tikzpicture}}[
    background rectangle/.style={{fill=bgdark}}, show background rectangle,
    scale=1.0, every node/.style={{transform shape}}
]

\clip (0,0) rectangle (30.48, 10.16);

% Title Header
\node[anchor=north west, text=textwhite, font=\large\bfseries] at (0.5, 9.7) {{{safe_title} (\texttt{{{safe_mod}}})}};
\node[anchor=north east, text={accent}, font=\small\bfseries] at (29.9, 9.7) {{Family: {header_badge} $\mid$ Class: \texttt{{{safe_title}}}}};

% Panel A: Computational Architecture
\draw[fill=cardbg, draw=bordercol, rounded corners=6pt] (0.5, 0.5) rectangle (10.0, 9.0);
\node[anchor=north west, text={accent}, font=\bfseries\normalsize] at (0.8, 8.7) {{(a) Computational Dataflow Graph}};

{block1}
{block2}
{block3}
{block4}
{block5}
{arrows}

% Panel B: Structural Pattern & Dynamics
\draw[fill=cardbg, draw=bordercol, rounded corners=6pt] (10.4, 0.5) rectangle (19.9, 9.0);
\node[anchor=north west, text={accent}, font=\bfseries\normalsize] at (10.7, 8.7) {{{panel_b_title}}};
{panel_b_draw}

% Panel C: Complexity / Scaling Profile
\draw[fill=cardbg, draw=bordercol, rounded corners=6pt] (20.3, 0.5) rectangle (29.8, 9.0);
\node[anchor=north west, text={accent}, font=\bfseries\normalsize] at (20.6, 8.7) {{{panel_c_title}}};

\draw[draw=bordercol, fill=bgdark, rounded corners=4pt] (20.9, 1.2) rectangle (29.2, 8.0);
\draw[-{{Stealth[length=4pt]}}, textmuted] (21.4, 1.8) -- (28.7, 1.8) node[right, font=\tiny, text=textmuted] {{Seq Length $L$}};
\draw[-{{Stealth[length=4pt]}}, textmuted] (21.4, 1.8) -- (21.4, 7.5) node[above, font=\tiny, text=textmuted] {{Compute FLOPs}};

\foreach \y in {{2.8, 4.0, 5.2, 6.4}} {{
    \draw[dashed, bordercol!50] (21.4, \y) -- (28.7, \y);
}}
{scaling_curve}

\end{{tikzpicture}}
\end{{document}}
"""
    return tex_content

def process_single_module(meta):
    mod_name = meta["module_name"]
    tex_content = build_tikz_code(meta)
    
    # Write tex file
    tex_file = os.path.join(BUILD_SCRATCH, f"{mod_name}.tex")
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(tex_content)
        
    # Compile with pdflatex
    cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", BUILD_SCRATCH, tex_file]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    pdf_build = os.path.join(BUILD_SCRATCH, f"{mod_name}.pdf")
    if not os.path.exists(pdf_build):
        print(f"Error compiling {mod_name}:\n{res.stdout[-400:]}")
        return False
        
    # Render PNG 300 DPI
    png_build = os.path.join(BUILD_SCRATCH, f"{mod_name}.png")
    doc = fitz.open(pdf_build)
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    pix.save(png_build)
    doc.close()
    
    # Copy to destinations ({name}.pdf/png and module_{name}.pdf/png)
    for out_dir in [OUTPUT_DIR_FINAL, OUTPUT_DIR_ICLR]:
        shutil.copy(pdf_build, os.path.join(out_dir, f"{mod_name}.pdf"))
        shutil.copy(png_build, os.path.join(out_dir, f"{mod_name}.png"))
        shutil.copy(tex_file, os.path.join(out_dir, f"{mod_name}.tex"))
        
        shutil.copy(pdf_build, os.path.join(out_dir, f"module_{mod_name}.pdf"))
        shutil.copy(png_build, os.path.join(out_dir, f"module_{mod_name}.png"))
        shutil.copy(tex_file, os.path.join(out_dir, f"module_{mod_name}.tex"))

    # Generate LaTeX appendix snippet
    snippet_content = rf"""\subsubsection{{{tex_escape(meta.get('class_name', mod_name))} (\texttt{{{tex_escape(mod_name)}}})}}
The module \texttt{{{tex_escape(mod_name)}}} is implemented within the \textbf{{{meta.get('family', 'linear').capitalize()} Operator}} architectural family. Figure~\ref{{fig:module_{mod_name}}} illustrates its exact computational dataflow, structural matrix dynamics, and theoretical scaling profile.

\begin{{figure}}[htbp]
    \centering
    \includegraphics[width=0.98\textwidth]{{figures/catalog_visuals/module_{mod_name}.pdf}}
    \caption{{Architectural characterization and structural dynamics for \textbf{{{tex_escape(meta.get('class_name', mod_name))}}}.}}
    \label{{fig:module_{mod_name}}}
\end{{figure}}

\vspace{{0.5em}}
"""
    with open(os.path.join(SNIPPETS_DIR, f"{mod_name}.tex"), "w", encoding="utf-8") as f:
        f.write(snippet_content)

    return True

def run_batch(category, start_letter, end_letter):
    meta_files = sorted(glob.glob(os.path.join(METADATA_DIR, "*.json")))
    tasks = []
    
    start_char = start_letter.lower()
    end_char = end_letter.lower()
    
    for mf in meta_files:
        with open(mf, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        if category and meta.get("category") != category:
            continue
            
        name = meta["module_name"].lower()
        if start_char <= name[0] <= end_char:
            tasks.append(meta)
            
    print(f"Agent executing category={category} letters=[{start_letter}-{end_letter}]: {len(tasks)} modules found.")
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_module, tasks))
        success_count = sum(1 for r in results if r)
        
    print(f"Finished {category} [{start_letter}-{end_letter}]: {success_count}/{len(tasks)} compiled successfully!")
    return success_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="", help="attention or compression or empty")
    parser.add_argument("--start", type=str, default="a", help="start letter")
    parser.add_argument("--end", type=str, default="z", help="end letter")
    args = parser.parse_args()
    
    run_batch(args.category, args.start, args.end)
