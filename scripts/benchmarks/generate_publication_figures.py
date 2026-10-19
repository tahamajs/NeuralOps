#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Generate publication-quality SVG figures and render them to 300+ DPI PNG and vector PDF.
Synchronize across results/plots/, paper/iclr2027-submission/figures/, and paper/neurips2026-submission/figures/.
"""

import json
import math
import subprocess
from pathlib import Path

DIRS = [Path("results/plots"), Path("paper/iclr2027-submission/figures"), Path("paper/neurips2026-submission/figures")]

for d in DIRS:
    d.mkdir(parents=True, exist_ok=True)

BG_COLOR = "#0f172a"
PANEL_BG = "#1e293b"
GRID_COLOR = "#334155"
AXIS_COLOR = "#64748b"
TEXT_COLOR = "#f8fafc"
TEXT_MUTED = "#94a3b8"
ACCENT_BLUE = "#38bdf8"
ACCENT_GREEN = "#34d399"
ACCENT_RED = "#f87171"
ACCENT_AMBER = "#fbbf24"
ACCENT_PURPLE = "#c084fc"
ACCENT_CYAN = "#22d3ee"

FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def svg_header(width, height, title, subtitle=None):
    """Execute svg header routine."""
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <rect width="100%" height="100%" fill="{BG_COLOR}"/>',
        f'  <text x="{width / 2}" y="36" fill="{TEXT_COLOR}" font-size="18" font-family="{FONT_FAMILY}" font-weight="bold" text-anchor="middle">{title}</text>',
    ]
    if subtitle:
        s.append(
            f'  <text x="{width / 2}" y="56" fill="{ACCENT_BLUE}" font-size="12" font-family="{FONT_FAMILY}" text-anchor="middle">{subtitle}</text>'
        )
    return "\n".join(s)


def svg_footer():
    """Execute svg footer routine."""
    return "</svg>\n"


def gen_pareto_acc_vs_compression():
    """Execute gen pareto acc vs compression routine."""
    w, h = 900, 600
    svg = [
        svg_header(
            w,
            h,
            "Pareto Frontier: Downstream Accuracy vs. Compression Ratio",
            "Evaluated on DistilBERT / RoBERTa Benchmarks (RTX 4070 SM 8.9)",
        )
    ]

    x0, y0, pw, ph = 100, 80, 740, 460
    svg.append(f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{PANEL_BG}" rx="6" stroke="{GRID_COLOR}"/>')

    xmin, xmax = 1.0, 3.0
    ymin, ymax = 30.0, 90.0

    def to_screen(x, y):
        """Execute to screen routine."""
        sx = x0 + ((x - xmin) / (xmax - xmin)) * pw
        sy = y0 + ph - ((y - ymin) / (ymax - ymin)) * ph
        return sx, sy

    for x in [1.0, 1.4, 1.8, 2.2, 2.6, 3.0]:
        sx, _ = to_screen(x, ymin)
        svg.append(
            f'  <line x1="{sx}" y1="{y0}" x2="{sx}" y2="{y0 + ph}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{sx}" y="{y0 + ph + 20}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="middle">{x:.1f}x</text>'
        )

    for y in range(30, 95, 10):
        _, sy = to_screen(xmin, y)
        svg.append(
            f'  <line x1="{x0}" y1="{sy}" x2="{x0 + pw}" y2="{sy}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{x0 - 12}" y="{sy + 4}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="end">{y}%</text>'
        )

    svg.append(
        f'  <text x="{x0 + pw / 2}" y="{y0 + ph + 45}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle">Compression Ratio (x) [Higher is More Compressed]</text>'
    )
    svg.append(
        f'  <text x="35" y="{y0 + ph / 2}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle" transform="rotate(-90 35 {y0 + ph / 2})">Accuracy Retention (%) [Higher is Better]</text>'
    )

    methods = [
        {"name": "Dense Baseline", "x": 1.00, "y": 81.80, "cat": "Baseline", "pareto": True},
        {"name": "kmeans (VQ)", "x": 2.73, "y": 81.40, "cat": "Quantization", "pareto": True},
        {"name": "adaptive", "x": 1.04, "y": 75.80, "cat": "Low-Rank", "pareto": True},
        {"name": "a3", "x": 1.05, "y": 72.20, "cat": "Low-Rank", "pareto": False},
        {"name": "adaptive_svd", "x": 1.15, "y": 68.90, "cat": "Low-Rank", "pareto": False},
        {"name": "aasvd", "x": 1.13, "y": 68.60, "cat": "Low-Rank", "pareto": False},
        {"name": "swift_svd", "x": 1.44, "y": 51.90, "cat": "Low-Rank", "pareto": False},
        {"name": "salt", "x": 1.21, "y": 52.00, "cat": "Low-Rank", "pareto": False},
        {"name": "kronecker", "x": 1.93, "y": 39.00, "cat": "Structured", "pareto": False},
        {"name": "hybrid_svd", "x": 1.91, "y": 37.50, "cat": "Low-Rank", "pareto": False},
        {"name": "bonsai_pruner", "x": 2.00, "y": 64.20, "cat": "Pruning", "pareto": False},
        {"name": "reap_pruner", "x": 2.30, "y": 71.50, "cat": "Pruning", "pareto": False},
    ]

    p_pts = sorted(
        [
            m
            for m in methods
            if m["pareto"] or m["name"] in ["Dense Baseline", "adaptive", "reap_pruner", "kmeans (VQ)"]
        ],
        key=lambda m: m["x"],
    )
    path_d = []
    for i, p in enumerate(p_pts):
        sx, sy = to_screen(p["x"], p["y"])
        if i == 0:
            path_d.append(f"M {sx:.1f} {sy:.1f}")
        else:
            path_d.append(f"L {sx:.1f} {sy:.1f}")
    svg.append(
        f'  <path d="{" ".join(path_d)}" fill="none" stroke="{ACCENT_CYAN}" stroke-width="2.5" stroke-dasharray="5,4"/>'
    )

    cat_colors = {
        "Baseline": "#e2e8f0",
        "Quantization": ACCENT_GREEN,
        "Low-Rank": ACCENT_BLUE,
        "Pruning": ACCENT_AMBER,
        "Structured": ACCENT_PURPLE,
    }
    for m in methods:
        sx, sy = to_screen(m["x"], m["y"])
        c = cat_colors[m["cat"]]
        r = 6 if not m["pareto"] else 8
        svg.append(
            f'  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r}" fill="{c}" stroke="{TEXT_COLOR if m["pareto"] else "none"}" stroke-width="1.5"/>'
        )
        offset_y = -12 if m["pareto"] else 14
        svg.append(
            f'  <text x="{sx:.1f}" y="{sy + offset_y:.1f}" fill="{c}" font-size="10" font-family="{FONT_FAMILY}" font-weight="{"bold" if m["pareto"] else "normal"}" text-anchor="middle">{m["name"]}</text>'
        )

    lx, ly = x0 + 20, y0 + 20
    svg.append(
        f'  <rect x="{lx}" y="{ly}" width="240" height="120" fill="{BG_COLOR}" rx="4" stroke="{GRID_COLOR}" opacity="0.9"/>'
    )
    svg.append(
        f'  <text x="{lx + 10}" y="{ly + 18}" fill="{TEXT_COLOR}" font-size="11" font-weight="bold" font-family="{FONT_FAMILY}">Method Category</text>'
    )
    idx = 0
    for cat, col in cat_colors.items():
        svg.append(f'  <circle cx="{lx + 18}" cy="{ly + 36 + idx * 18}" r="5" fill="{col}"/>')
        svg.append(
            f'  <text x="{lx + 32}" y="{ly + 40 + idx * 18}" fill="{TEXT_MUTED}" font-size="10" font-family="{FONT_FAMILY}">{cat}</text>'
        )
        idx += 1

    svg.append(svg_footer())
    return "\n".join(svg)


def gen_pareto_acc_vs_vram():
    """Execute gen pareto acc vs vram routine."""
    w, h = 900, 600
    svg = [
        svg_header(
            w,
            h,
            "Pareto Frontier: Accuracy vs. Peak VRAM Footprint",
            "Memory Footprint vs Task Retention on NVIDIA RTX 4070",
        )
    ]
    x0, y0, pw, ph = 100, 80, 740, 460
    svg.append(f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{PANEL_BG}" rx="6" stroke="{GRID_COLOR}"/>')

    xmin, xmax = 100.0, 200.0
    ymin, ymax = 30.0, 90.0

    def to_screen(x, y):
        """Execute to screen routine."""
        sx = x0 + ((x - xmin) / (xmax - xmin)) * pw
        sy = y0 + ph - ((y - ymin) / (ymax - ymin)) * ph
        return sx, sy

    for x in range(100, 210, 20):
        sx, _ = to_screen(x, ymin)
        svg.append(
            f'  <line x1="{sx}" y1="{y0}" x2="{sx}" y2="{y0 + ph}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{sx}" y="{y0 + ph + 20}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="middle">{x} MB</text>'
        )

    for y in range(30, 95, 10):
        _, sy = to_screen(xmin, y)
        svg.append(
            f'  <line x1="{x0}" y1="{sy}" x2="{x0 + pw}" y2="{sy}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{x0 - 12}" y="{sy + 4}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="end">{y}%</text>'
        )

    svg.append(
        f'  <text x="{x0 + pw / 2}" y="{y0 + ph + 45}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle">Peak VRAM Footprint (MB) [Lower is Better]</text>'
    )
    svg.append(
        f'  <text x="35" y="{y0 + ph / 2}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle" transform="rotate(-90 35 {y0 + ph / 2})">Accuracy Retention (%) [Higher is Better]</text>'
    )

    methods = [
        {"name": "Dense Baseline", "vram": 176.4, "acc": 81.80, "cat": "Baseline", "pareto": False},
        {"name": "kmeans (VQ)", "vram": 118.4, "acc": 81.40, "cat": "Quantization", "pareto": True},
        {"name": "adaptive", "vram": 172.0, "acc": 75.80, "cat": "Low-Rank", "pareto": False},
        {"name": "a3", "vram": 170.2, "acc": 72.20, "cat": "Low-Rank", "pareto": False},
        {"name": "adaptive_svd", "vram": 182.1, "acc": 68.90, "cat": "Low-Rank", "pareto": False},
        {"name": "aasvd", "vram": 178.5, "acc": 68.60, "cat": "Low-Rank", "pareto": False},
        {"name": "swift_svd", "vram": 165.4, "acc": 51.90, "cat": "Low-Rank", "pareto": False},
        {"name": "salt", "vram": 185.0, "acc": 52.00, "cat": "Low-Rank", "pareto": False},
        {"name": "kronecker", "vram": 152.0, "acc": 39.00, "cat": "Structured", "pareto": False},
        {"name": "hybrid_svd", "vram": 155.0, "acc": 37.50, "cat": "Low-Rank", "pareto": False},
        {"name": "bonsai_pruner", "vram": 135.0, "acc": 64.20, "cat": "Pruning", "pareto": True},
    ]

    cat_colors = {
        "Baseline": "#e2e8f0",
        "Quantization": ACCENT_GREEN,
        "Low-Rank": ACCENT_BLUE,
        "Pruning": ACCENT_AMBER,
        "Structured": ACCENT_PURPLE,
    }
    for m in methods:
        sx, sy = to_screen(m["vram"], m["acc"])
        c = cat_colors[m["cat"]]
        r = 8 if m["pareto"] else 6
        svg.append(
            f'  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r}" fill="{c}" stroke="{TEXT_COLOR if m["pareto"] else "none"}" stroke-width="1.5"/>'
        )
        offset_y = -12 if m["pareto"] else 14
        svg.append(
            f'  <text x="{sx:.1f}" y="{sy + offset_y:.1f}" fill="{c}" font-size="10" font-family="{FONT_FAMILY}" font-weight="{"bold" if m["pareto"] else "normal"}" text-anchor="middle">{m["name"]}</text>'
        )

    svg.append(svg_footer())
    return "\n".join(svg)


def gen_pareto_acc_vs_throughput():
    """Execute gen pareto acc vs throughput routine."""
    w, h = 900, 600
    svg = [
        svg_header(
            w,
            h,
            "Pareto Frontier: Accuracy vs. Throughput (tokens/sec)",
            "Physical Generation Speed vs Downstream Retention",
        )
    ]
    x0, y0, pw, ph = 100, 80, 740, 460
    svg.append(f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{PANEL_BG}" rx="6" stroke="{GRID_COLOR}"/>')

    xmin, xmax = 54000.0, 70000.0
    ymin, ymax = 30.0, 90.0

    def to_screen(x, y):
        """Execute to screen routine."""
        sx = x0 + ((x - xmin) / (xmax - xmin)) * pw
        sy = y0 + ph - ((y - ymin) / (ymax - ymin)) * ph
        return sx, sy

    for x in range(55000, 72000, 3000):
        sx, _ = to_screen(x, ymin)
        svg.append(
            f'  <line x1="{sx}" y1="{y0}" x2="{sx}" y2="{y0 + ph}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{sx}" y="{y0 + ph + 20}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="middle">{x // 1000}k</text>'
        )

    for y in range(30, 95, 10):
        _, sy = to_screen(xmin, y)
        svg.append(
            f'  <line x1="{x0}" y1="{sy}" x2="{x0 + pw}" y2="{sy}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{x0 - 12}" y="{sy + 4}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="end">{y}%</text>'
        )

    svg.append(
        f'  <text x="{x0 + pw / 2}" y="{y0 + ph + 45}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle">Throughput (tokens / second) [Higher is Faster]</text>'
    )
    svg.append(
        f'  <text x="35" y="{y0 + ph / 2}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle" transform="rotate(-90 35 {y0 + ph / 2})">Accuracy Retention (%) [Higher is Better]</text>'
    )

    methods = [
        {"name": "Dense Baseline", "tok": 56610, "acc": 81.80, "cat": "Baseline", "pareto": True},
        {"name": "kmeans (VQ)", "tok": 68410, "acc": 81.40, "cat": "Quantization", "pareto": True},
        {"name": "adaptive", "tok": 58900, "acc": 75.80, "cat": "Low-Rank", "pareto": False},
        {"name": "a3", "tok": 59200, "acc": 72.20, "cat": "Low-Rank", "pareto": False},
        {"name": "adaptive_svd", "tok": 58400, "acc": 68.90, "cat": "Low-Rank", "pareto": False},
        {"name": "aasvd", "tok": 57900, "acc": 68.60, "cat": "Low-Rank", "pareto": False},
        {"name": "swift_svd", "tok": 57800, "acc": 51.90, "cat": "Low-Rank", "pareto": False},
        {"name": "salt", "tok": 65959, "acc": 52.00, "cat": "Low-Rank", "pareto": False},
        {"name": "kronecker", "tok": 62100, "acc": 39.00, "cat": "Structured", "pareto": False},
        {"name": "hybrid_svd", "tok": 61400, "acc": 37.50, "cat": "Low-Rank", "pareto": False},
    ]

    cat_colors = {
        "Baseline": "#e2e8f0",
        "Quantization": ACCENT_GREEN,
        "Low-Rank": ACCENT_BLUE,
        "Pruning": ACCENT_AMBER,
        "Structured": ACCENT_PURPLE,
    }
    for m in methods:
        sx, sy = to_screen(m["tok"], m["acc"])
        c = cat_colors[m["cat"]]
        r = 8 if m["pareto"] else 6
        svg.append(
            f'  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r}" fill="{c}" stroke="{TEXT_COLOR if m["pareto"] else "none"}" stroke-width="1.5"/>'
        )
        offset_y = -12 if m["pareto"] else 14
        svg.append(
            f'  <text x="{sx:.1f}" y="{sy + offset_y:.1f}" fill="{c}" font-size="10" font-family="{FONT_FAMILY}" font-weight="{"bold" if m["pareto"] else "normal"}" text-anchor="middle">{m["name"]}</text>'
        )

    svg.append(svg_footer())
    return "\n".join(svg)


def gen_pareto_ppl_vs_compression():
    """Execute gen pareto ppl vs compression routine."""
    w, h = 900, 600
    svg = [
        svg_header(
            w,
            h,
            "Perplexity vs. Compression Ratio (WikiText-2 GPT-2)",
            "Evaluated with rank=32, stride=512 on NVIDIA RTX 4070",
        )
    ]
    x0, y0, pw, ph = 100, 80, 740, 460
    svg.append(f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{PANEL_BG}" rx="6" stroke="{GRID_COLOR}"/>')

    xmin, xmax = 1.0, 3.5
    log_ymin, log_ymax = math.log10(10), math.log10(5000)

    def to_screen(x, ppl):
        """Execute to screen routine."""
        sx = x0 + ((x - xmin) / (xmax - xmin)) * pw
        l_ppl = math.log10(max(10, ppl))
        sy = y0 + ph - ((l_ppl - log_ymin) / (log_ymax - log_ymin)) * ph
        return sx, sy

    for x in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        sx, _ = to_screen(x, 10)
        svg.append(
            f'  <line x1="{sx}" y1="{y0}" x2="{sx}" y2="{y0 + ph}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{sx}" y="{y0 + ph + 20}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="middle">{x:.1f}x</text>'
        )

    for y_val in [10, 20, 50, 100, 500, 1000, 5000]:
        _, sy = to_screen(xmin, y_val)
        svg.append(
            f'  <line x1="{x0}" y1="{sy}" x2="{x0 + pw}" y2="{sy}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{x0 - 12}" y="{sy + 4}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="end">{y_val}</text>'
        )

    svg.append(
        f'  <text x="{x0 + pw / 2}" y="{y0 + ph + 45}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle">Compression Ratio (x) [Higher is More Compressed]</text>'
    )
    svg.append(
        f'  <text x="35" y="{y0 + ph / 2}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle" transform="rotate(-90 35 {y0 + ph / 2})">WikiText-2 Perplexity (log scale) [Lower is Better]</text>'
    )

    _, b_sy = to_screen(xmin, 22.79)
    svg.append(
        f'  <line x1="{x0}" y1="{b_sy}" x2="{x0 + pw}" y2="{b_sy}" stroke="{ACCENT_RED}" stroke-width="1.5" stroke-dasharray="4,4"/>'
    )
    svg.append(
        f'  <text x="{x0 + pw - 10}" y="{b_sy - 6}" fill="{ACCENT_RED}" font-size="10" font-family="{FONT_FAMILY}" text-anchor="end">GPT-2 Dense Baseline (22.79)</text>'
    )

    wiki_methods = [
        {"name": "direct_svd", "x": 1.46, "ppl": 20.85, "cat": "Low-Rank", "pareto": True},
        {"name": "aa_svd", "x": 1.40, "ppl": 20.95, "cat": "Low-Rank", "pareto": True},
        {"name": "hadamard_svd", "x": 1.46, "ppl": 21.28, "cat": "Low-Rank", "pareto": False},
        {"name": "residual_svd", "x": 1.46, "ppl": 21.42, "cat": "Low-Rank", "pareto": False},
        {"name": "bonsai_pruner", "x": 2.00, "ppl": 21.45, "cat": "Pruning", "pareto": True},
        {"name": "reap_pruner", "x": 2.25, "ppl": 23.50, "cat": "Pruning", "pareto": True},
        {"name": "kmeans_quant", "x": 2.80, "ppl": 24.10, "cat": "Quantization", "pareto": True},
        {"name": "erc_svd", "x": 3.03, "ppl": 3434.91, "cat": "Low-Rank", "pareto": False},
        {"name": "tt_compressor", "x": 2.85, "ppl": 5334.60, "cat": "Structured", "pareto": False},
    ]

    cat_colors = {
        "Baseline": "#e2e8f0",
        "Quantization": ACCENT_GREEN,
        "Low-Rank": ACCENT_BLUE,
        "Pruning": ACCENT_AMBER,
        "Structured": ACCENT_PURPLE,
    }
    for m in wiki_methods:
        sx, sy = to_screen(m["x"], m["ppl"])
        c = cat_colors[m["cat"]]
        r = 7 if m["pareto"] else 5
        svg.append(
            f'  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r}" fill="{c}" stroke="{TEXT_COLOR if m["pareto"] else "none"}" stroke-width="1.5"/>'
        )
        offset_y = -10 if m["ppl"] < 50 else 12
        svg.append(
            f'  <text x="{sx:.1f}" y="{sy + offset_y:.1f}" fill="{c}" font-size="10" font-family="{FONT_FAMILY}" font-weight="{"bold" if m["pareto"] else "normal"}" text-anchor="middle">{m["name"]}</text>'
        )

    svg.append(svg_footer())
    return "\n".join(svg)


def gen_triton_leaderboard():
    """Execute gen triton leaderboard routine."""
    with open("results/triton_full_leaderboard_v136.json") as f:
        data = json.load(f)
    kernels = data.get("tier1_core_triton_kernels", [])

    w, h = 950, 780
    svg = [
        svg_header(
            w,
            h,
            "Triton GPU Acceleration Leaderboard (All 26 Flagship Kernels)",
            "Measured on NVIDIA GeForce RTX 4070 (Ada Lovelace SM 8.9, 8GB VRAM)",
        )
    ]

    x0, y0, pw, ph = 240, 80, 650, 640
    svg.append(f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{PANEL_BG}" rx="6" stroke="{GRID_COLOR}"/>')

    min_log, max_log = math.log10(0.1), math.log10(150.0)

    def to_screen_x(speedup):
        """Execute to screen x routine."""
        l_s = math.log10(max(0.1, speedup))
        return x0 + ((l_s - min_log) / (max_log - min_log)) * pw

    for tick in [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        sx = to_screen_x(tick)
        is_one = tick == 1.0
        svg.append(
            f'  <line x1="{sx}" y1="{y0}" x2="{sx}" y2="{y0 + ph}" stroke="{ACCENT_RED if is_one else GRID_COLOR}" stroke-width="{"1.5" if is_one else "1"}" stroke-dasharray="{"none" if is_one else "3,3"}"/>'
        )
        svg.append(
            f'  <text x="{sx}" y="{y0 + ph + 20}" fill="{ACCENT_RED if is_one else TEXT_MUTED}" font-size="10" font-family="{FONT_FAMILY}" text-anchor="middle">{tick}x</text>'
        )

    svg.append(
        f'  <text x="{x0 + pw / 2}" y="{y0 + ph + 40}" fill="{TEXT_MUTED}" font-size="12" font-family="{FONT_FAMILY}" text-anchor="middle">Speedup Over PyTorch Reference (Log Scale) [1.0x = Parity]</text>'
    )

    row_h = ph / len(kernels)
    for i, k in enumerate(kernels):
        name = k.get("kernel_module", "").replace(".py", "")
        speedup = float(k.get("speedup_x", 1.0))
        ry = y0 + i * row_h
        mid_y = ry + row_h / 2.0

        if i % 2 == 1:
            svg.append(f'  <rect x="{x0}" y="{ry}" width="{pw}" height="{row_h}" fill="#1b2438"/>')

        svg.append(
            f'  <text x="{x0 - 10}" y="{mid_y + 4}" fill="{TEXT_COLOR}" font-size="10.5" font-family="{FONT_FAMILY}" text-anchor="end">{i + 1}. {name}</text>'
        )

        x_base = to_screen_x(1.0)
        x_val = to_screen_x(speedup)
        bx = min(x_base, x_val)
        bw = abs(x_val - x_base)
        color = ACCENT_GREEN if speedup >= 2.0 else (ACCENT_BLUE if speedup >= 1.0 else ACCENT_AMBER)
        svg.append(f'  <rect x="{bx:.1f}" y="{mid_y - 5}" width="{max(2, bw):.1f}" height="10" fill="{color}" rx="2"/>')
        svg.append(
            f'  <text x="{max(x_val + 6, x_base + 6):.1f}" y="{mid_y + 3.5}" fill="{color}" font-size="9" font-family="{FONT_FAMILY}">{speedup:.2f}x</text>'
        )

    svg.append(svg_footer())
    return "\n".join(svg)


def gen_3level_latency_comparison():
    """Execute gen 3level latency comparison routine."""
    operators = [
        {"name": "moe_routing", "py": 29.505, "tri": None, "cuda": 0.022, "speedup": 1336.96},
        {"name": "hgrn2_scan", "py": 6.209, "tri": 0.027, "cuda": 0.014, "speedup": 436.29},
        {"name": "linear_attn", "py": 40.823, "tri": 0.257, "cuda": 0.167, "speedup": 243.98},
        {"name": "selective_scan", "py": 29.936, "tri": 14.339, "cuda": 0.274, "speedup": 109.34},
        {"name": "chunked_delta_gate", "py": 47.587, "tri": 0.126, "cuda": 0.578, "speedup": 82.40},
        {"name": "paged_attention", "py": 1.096, "tri": None, "cuda": 0.024, "speedup": 45.89},
        {"name": "sliding_window", "py": 1.826, "tri": 0.252, "cuda": 0.252, "speedup": 7.24},
        {"name": "codebook_dequant", "py": 0.096, "tri": 0.026, "cuda": 0.019, "speedup": 5.00},
        {"name": "block_sparse_attn", "py": 1.819, "tri": 1.529, "cuda": 0.365, "speedup": 4.98},
        {"name": "flash_gqa", "py": 0.192, "tri": 0.054, "cuda": 0.098, "speedup": 1.96},
        {"name": "kronecker_matmul", "py": 0.030, "tri": 0.025, "cuda": 0.017, "speedup": 1.78},
        {"name": "cosformer_linear", "py": 0.120, "tri": 0.064, "cuda": 0.085, "speedup": 1.41},
    ]

    w, h = 950, 620
    svg = [
        svg_header(
            w,
            h,
            "3-Level Latency Hierarchy: PyTorch Ref vs. Triton JIT vs. Native CUDA C++",
            "Table 18 Benchmark Hierarchy (Ada Lovelace SM 8.9, CUDA 12.x)",
        )
    ]

    x0, y0, pw, ph = 220, 80, 680, 480
    svg.append(f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{PANEL_BG}" rx="6" stroke="{GRID_COLOR}"/>')

    min_log, max_log = math.log10(0.01), math.log10(100.0)

    def to_screen_x(lat):
        """Execute to screen x routine."""
        l_l = math.log10(max(0.01, lat))
        return x0 + ((l_l - min_log) / (max_log - min_log)) * pw

    for tick in [0.01, 0.1, 1.0, 10.0, 100.0]:
        sx = to_screen_x(tick)
        svg.append(
            f'  <line x1="{sx}" y1="{y0}" x2="{sx}" y2="{y0 + ph}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{sx}" y="{y0 + ph + 20}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="middle">{tick} ms</text>'
        )

    svg.append(
        f'  <text x="{x0 + pw / 2}" y="{y0 + ph + 45}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle">Execution Latency (ms, Log Scale) [Lower is Faster]</text>'
    )

    row_h = ph / len(operators)
    for i, op in enumerate(operators):
        ry = y0 + i * row_h
        mid_y = ry + row_h / 2.0
        svg.append(
            f'  <text x="{x0 - 10}" y="{mid_y + 4}" fill="{TEXT_COLOR}" font-size="10.5" font-family="{FONT_FAMILY}" text-anchor="end">{op["name"]}</text>'
        )

        px = to_screen_x(op["py"])
        svg.append(f'  <circle cx="{px:.1f}" cy="{mid_y - 5}" r="4.5" fill="{ACCENT_RED}"/>')

        if op["tri"] is not None:
            tx = to_screen_x(op["tri"])
            svg.append(f'  <circle cx="{tx:.1f}" cy="{mid_y}" r="4.5" fill="{ACCENT_BLUE}"/>')

        cx = to_screen_x(op["cuda"])
        svg.append(f'  <circle cx="{cx:.1f}" cy="{mid_y + 5}" r="4.5" fill="{ACCENT_GREEN}"/>')

        min_x = min(cx, tx if op["tri"] is not None else cx, px)
        max_x = max(cx, tx if op["tri"] is not None else cx, px)
        svg.append(
            f'  <line x1="{min_x:.1f}" y1="{mid_y}" x2="{max_x:.1f}" y2="{mid_y}" stroke="{TEXT_MUTED}" stroke-width="0.8" stroke-dasharray="2,2"/>'
        )

    lx, ly = x0 + 20, y0 + 15
    svg.append(
        f'  <rect x="{lx}" y="{ly}" width="280" height="60" fill="{BG_COLOR}" rx="4" stroke="{GRID_COLOR}" opacity="0.9"/>'
    )
    svg.append(f'  <circle cx="{lx + 15}" cy="{ly + 16}" r="5" fill="{ACCENT_RED}"/>')
    svg.append(
        f'  <text x="{lx + 28}" y="{ly + 20}" fill="{TEXT_MUTED}" font-size="10.5" font-family="{FONT_FAMILY}">Level 3: Pure PyTorch Reference</text>'
    )
    svg.append(f'  <circle cx="{lx + 15}" cy="{ly + 34}" r="5" fill="{ACCENT_BLUE}"/>')
    svg.append(
        f'  <text x="{lx + 28}" y="{ly + 38}" fill="{TEXT_MUTED}" font-size="10.5" font-family="{FONT_FAMILY}">Level 2: OpenAI Triton JIT</text>'
    )
    svg.append(f'  <circle cx="{lx + 15}" cy="{ly + 50}" r="5" fill="{ACCENT_GREEN}"/>')
    svg.append(
        f'  <text x="{lx + 28}" y="{ly + 54}" fill="{TEXT_MUTED}" font-size="10.5" font-family="{FONT_FAMILY}">Level 1: Native CUDA C++ (Fastest)</text>'
    )

    svg.append(svg_footer())
    return "\n".join(svg)


def gen_sequence_length_scaling():
    """Execute gen sequence length scaling routine."""
    T_vals = [512, 1024, 2048, 4096, 8192]
    models = {
        "PyTorch SDPA (Dense O(T^2))": {"lat": [0.06, 0.18, 0.65, 2.59, 10.11], "color": ACCENT_RED, "dash": "none"},
        "Chunk GLA (Triton O(T))": {"lat": [0.41, 0.66, 1.32, 3.44, 6.88], "color": ACCENT_GREEN, "dash": "none"},
        "Chunk GDN (Triton O(T))": {"lat": [0.48, 0.77, 1.54, 4.05, 8.10], "color": ACCENT_CYAN, "dash": "4,4"},
        "Cosformer (Triton)": {"lat": [0.25, 0.90, 3.53, 13.68, 54.46], "color": ACCENT_AMBER, "dash": "2,2"},
    }

    w, h = 900, 600
    svg = [
        svg_header(
            w,
            h,
            "Sequence Length Scaling: Latency (ms) vs. Context Length T",
            "Empirical Scaling on RTX 4070 (B=2, H=12, D=64) from T=512 to T=8,192",
        )
    ]

    x0, y0, pw, ph = 100, 80, 740, 460
    svg.append(f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="{PANEL_BG}" rx="6" stroke="{GRID_COLOR}"/>')

    min_log_t, max_log_t = math.log2(256), math.log2(16384)
    min_log_y, max_log_y = math.log10(0.03), math.log10(100.0)

    def to_screen(t, y):
        """Execute to screen routine."""
        sx = x0 + ((math.log2(t) - min_log_t) / (max_log_t - min_log_t)) * pw
        sy = y0 + ph - ((math.log10(y) - min_log_y) / (max_log_y - min_log_y)) * ph
        return sx, sy

    for t in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        sx, _ = to_screen(t, 1.0)
        svg.append(
            f'  <line x1="{sx}" y1="{y0}" x2="{sx}" y2="{y0 + ph}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        label = f"{t // 1024}K" if t >= 1024 else str(t)
        svg.append(
            f'  <text x="{sx}" y="{y0 + ph + 20}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="middle">{label}</text>'
        )

    for y_val in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        _, sy = to_screen(256, y_val)
        svg.append(
            f'  <line x1="{x0}" y1="{sy}" x2="{x0 + pw}" y2="{sy}" stroke="{GRID_COLOR}" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'  <text x="{x0 - 12}" y="{sy + 4}" fill="{TEXT_MUTED}" font-size="11" font-family="{FONT_FAMILY}" text-anchor="end">{y_val} ms</text>'
        )

    svg.append(
        f'  <text x="{x0 + pw / 2}" y="{y0 + ph + 45}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle">Sequence Context Length T (Tokens, Log2 Scale)</text>'
    )
    svg.append(
        f'  <text x="35" y="{y0 + ph / 2}" fill="{TEXT_MUTED}" font-size="13" font-family="{FONT_FAMILY}" text-anchor="middle" transform="rotate(-90 35 {y0 + ph / 2})">Kernel Latency (ms, Log10 Scale) [Lower is Faster]</text>'
    )

    for name, d in models.items():
        pts = [to_screen(t, lat) for t, lat in zip(T_vals, d["lat"])]
        d_str = " ".join([f"{'M' if i == 0 else 'L'} {px:.1f} {py:.1f}" for i, (px, py) in enumerate(pts)])
        svg.append(
            f'  <path d="{d_str}" fill="none" stroke="{d["color"]}" stroke-width="2.5" stroke-dasharray="{d["dash"]}"/>'
        )
        for px, py in pts:
            svg.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{d["color"]}"/>')

    lx, ly = x0 + 20, y0 + 15
    svg.append(
        f'  <rect x="{lx}" y="{ly}" width="300" height="90" fill="{BG_COLOR}" rx="4" stroke="{GRID_COLOR}" opacity="0.9"/>'
    )
    idx = 0
    for name, d in models.items():
        svg.append(
            f'  <line x1="{lx + 12}" y1="{ly + 18 + idx * 18}" x2="{lx + 36}" y2="{ly + 18 + idx * 18}" stroke="{d["color"]}" stroke-width="2.5" stroke-dasharray="{d["dash"]}"/>'
        )
        svg.append(f'  <circle cx="{lx + 24}" cy="{ly + 18 + idx * 18}" r="4" fill="{d["color"]}"/>')
        svg.append(
            f'  <text x="{lx + 44}" y="{ly + 22 + idx * 18}" fill="{TEXT_MUTED}" font-size="10.5" font-family="{FONT_FAMILY}">{name}</text>'
        )
        idx += 1

    svg.append(svg_footer())
    return "\n".join(svg)


def gen_svd_llm_v2_truncation():
    """Execute gen svd llm v2 truncation routine."""
    with open("results/plots/svd_llm_v2_vs_naive_svd.svg", "r") as f:
        return f.read()


def gen_kv_cache_scaling():
    """Execute gen kv cache scaling routine."""
    with open("results/plots/asvd_kv_cache_memory_scaling.svg", "r") as f:
        return f.read()


def main():
    """Execute main routine."""
    generators = {
        "pareto_accuracy_vs_compression": gen_pareto_acc_vs_compression,
        "pareto_accuracy_vs_vram": gen_pareto_acc_vs_vram,
        "pareto_accuracy_vs_throughput": gen_pareto_acc_vs_throughput,
        "pareto_ppl_vs_compression": gen_pareto_ppl_vs_compression,
        "triton_speedup_leaderboard": gen_triton_leaderboard,
        "three_level_latency_comparison": gen_3level_latency_comparison,
        "sequence_length_scaling": gen_sequence_length_scaling,
        "svd_llm_v2_truncation_optimality": gen_svd_llm_v2_truncation,
        "kv_cache_latent_memory_scaling": gen_kv_cache_scaling,
    }

    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"

    print("Generating SVGs and rendering 300+ DPI PNGs and vector PDFs...")

    for key, gen_func in generators.items():
        svg_content = gen_func()

        for target_dir in DIRS:
            svg_file = target_dir / f"{key}.svg"
            with open(svg_file, "w", encoding="utf-8") as f:
                f.write(svg_content)

            png_file = target_dir / f"{key}.png"
            cmd_png = ["/opt/homebrew/bin/magick", "-font", font_path, "-density", "300", str(svg_file), str(png_file)]
            subprocess.run(cmd_png, check=True)

            pdf_file = target_dir / f"{key}.pdf"
            cmd_pdf = ["/opt/homebrew/bin/magick", "-font", font_path, "-density", "300", str(svg_file), str(pdf_file)]
            subprocess.run(cmd_pdf, check=True)

        print(f"Generated {key} [SVG, PNG 300 DPI, PDF] across all directories.")

    print("Synchronizing pareto_frontier across all directories...")
    pf_svg = gen_pareto_acc_vs_compression()
    for target_dir in DIRS:
        svg_file = target_dir / "pareto_frontier.svg"
        with open(svg_file, "w", encoding="utf-8") as f:
            f.write(pf_svg)
        cmd_png = [
            "/opt/homebrew/bin/magick",
            "-font",
            font_path,
            "-density",
            "300",
            str(svg_file),
            str(target_dir / "pareto_frontier.png"),
        ]
        subprocess.run(cmd_png, check=True)
        cmd_pdf = [
            "/opt/homebrew/bin/magick",
            "-font",
            font_path,
            "-density",
            "300",
            str(svg_file),
            str(target_dir / "pareto_frontier.pdf"),
        ]
        subprocess.run(cmd_pdf, check=True)

    print("All 9 publication figures + pareto_frontier synchronized successfully in SVG, PNG, and PDF!")


if __name__ == "__main__":
    main()
