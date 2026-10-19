#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Academic LaTeX and Pareto Exporter

Automated LaTeX table generation and Pareto frontier export tools for research papers.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Union


def compute_pareto_frontier(
    records: Sequence[Dict[str, Any]],
    minimize_keys: Sequence[str] = (
        "latency_ms",
        "latency_triton_ms",
        "peak_vram_mb",
        "ppl",
        "mse",
        "hbm_bytes_fused",
        "error",
    ),
    maximize_keys: Sequence[str] = (
        "speedup_x",
        "speedup",
        "throughput_tokens_per_sec",
        "cosine_sim",
        "achieved_tflops",
        "sram_bytes_saved_mb",
        "accuracy",
    ),
) -> List[bool]:
    """Identify multi-objective Pareto-optimal records.

    A record i is Pareto-optimal (non-dominated) if no other record j is at least
    as good as i on all active objectives and strictly better on at least one objective.

    Returns:
        List[bool] of length len(records) where True indicates Pareto optimality.
    """
    n = len(records)
    if n == 0:
        return []
    if n == 1:
        return [True]

    active_min = [k for k in minimize_keys if any(k in r for r in records)]
    active_max = [k for k in maximize_keys if any(k in r for r in records)]

    if not active_min and not active_max:
        return [True] * n

    is_pareto = [True] * n

    for i in range(n):
        r_i = records[i]
        for j in range(n):
            if i == j:
                continue
            r_j = records[j]

            j_as_good = True
            j_strictly_better = False

            for k in active_min:
                val_i = float(r_i.get(k, float("inf")))
                val_j = float(r_j.get(k, float("inf")))
                if val_j > val_i:
                    j_as_good = False
                    break
                if val_j < val_i:
                    j_strictly_better = True

            if not j_as_good:
                continue

            for k in active_max:
                val_i = float(r_i.get(k, float("-inf")))
                val_j = float(r_j.get(k, float("-inf")))
                if val_j < val_i:
                    j_as_good = False
                    break
                if val_j > val_i:
                    j_strictly_better = True

            if j_as_good and j_strictly_better:
                is_pareto[i] = False
                break

    return is_pareto


def _is_lower_better(col_name: str) -> bool:
    """Determine optimization direction from column name."""
    c = col_name.lower()
    return any(
        k in c
        for k in (
            "latency",
            "ms",
            "sec",
            "vram",
            "mem",
            "byte",
            "ppl",
            "loss",
            "err",
            "mse",
            "param",
        )
    ) and not any(k in c for k in ("speedup", "saved", "tflops", "throughput", "tok/s", "acc", "cos"))


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters."""
    special = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    regex = re.compile("|".join(re.escape(k) for k in special.keys()))
    return regex.sub(lambda m: special[m.group(0)], str(text))


def export_benchmark_table(
    records: Sequence[Dict[str, Any]],
    columns: Optional[Sequence[str]] = None,
    format: str = "latex",
    caption: str = "Empirical Hardware Benchmark Results on NeuralOps.",
    label: str = "tab:neuralops_benchmark",
    highlight_best: bool = True,
    mark_pareto: bool = True,
) -> str:
    """Export benchmark records into LaTeX booktabs, Markdown, or CSV.

    Features:
      - Automatic Pareto-optimal detection ($^\\star$ / ★)
      - Column direction detection with arrows ($\\downarrow$ / $\\uparrow$)
      - Bolded best and underlined second-best formatting in LaTeX & Markdown
    """
    if not records:
        return ""

    if columns is None:
        preferred = [
            "module",
            "module_name",
            "name",
            "latency_pytorch_ms",
            "latency_triton_ms",
            "speedup_x",
            "achieved_tflops",
            "sram_bytes_saved_mb",
            "peak_vram_mb",
            "bottleneck_regime",
        ]
        columns = [c for c in preferred if any(c in r for r in records)]
        if not columns:
            columns = list(records[0].keys())

    pareto_mask = compute_pareto_frontier(records) if mark_pareto else [False] * len(records)

    best_vals: Dict[str, float] = {}
    second_vals: Dict[str, float] = {}

    for col in columns:
        numeric_vals = []
        for r in records:
            val = r.get(col)
            if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
                numeric_vals.append(float(val))

        if numeric_vals:
            unique_vals = sorted(list(set(numeric_vals)))
            lower_better = _is_lower_better(col)
            if lower_better:
                best_vals[col] = unique_vals[0]
                second_vals[col] = unique_vals[1] if len(unique_vals) > 1 else unique_vals[0]
            else:
                best_vals[col] = unique_vals[-1]
                second_vals[col] = unique_vals[-2] if len(unique_vals) > 1 else unique_vals[-1]

    fmt = format.lower()
    if fmt == "csv":
        import io
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(list(columns) + (["pareto_optimal"] if mark_pareto else []))
        for i, r in enumerate(records):
            row = [r.get(c, "") for c in columns]
            if mark_pareto:
                row.append(pareto_mask[i])
            writer.writerow(row)
        return output.getvalue()

    elif fmt == "markdown":
        lines = []
        header_row = []
        for c in columns:
            if any(k in c.lower() for k in ("name", "module", "regime")):
                header_row.append(c.replace('_', ' ').title())
            else:
                arrow = " ↓" if _is_lower_better(c) else " ↑"
                header_row.append(f"{c.replace('_', ' ').title()}{arrow}")
        lines.append("| " + " | ".join(header_row) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

        for i, r in enumerate(records):
            row_items = []
            for j, c in enumerate(columns):
                val = r.get(c, "")
                if isinstance(val, (int, float)):
                    f_val = float(val)
                    val_str = f"{f_val:.2f}" if isinstance(val, float) else str(val)
                    if highlight_best and c in best_vals and abs(f_val - best_vals[c]) < 1e-6:
                        val_str = f"**{val_str}**"
                else:
                    val_str = str(val)

                if j == 0 and mark_pareto and pareto_mask[i]:
                    val_str = f"{val_str} ★"
                row_items.append(val_str)
            lines.append("| " + " | ".join(row_items) + " |")

        return "\n".join(lines)

    else:
        col_align = []
        for c in columns:
            col_align.append("l" if any(k in c.lower() for k in ("name", "module", "regime")) else "r")
        align_str = "".join(col_align)

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            rf"\begin{{tabular}}{{{align_str}}}",
            r"\toprule",
        ]

        headers = []
        for c in columns:
            esc = _escape_latex(c.replace("_", " ").title())
            if any(k in c.lower() for k in ("name", "module", "regime")):
                headers.append(esc)
            else:
                arrow = r"$\downarrow$" if _is_lower_better(c) else r"$\uparrow$"
                headers.append(f"{esc} ({arrow})")

        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\midrule")

        for i, r in enumerate(records):
            row_items = []
            for j, c in enumerate(columns):
                val = r.get(c, "")
                if isinstance(val, (int, float)):
                    f_val = float(val)
                    val_str = f"{f_val:.2f}" if isinstance(val, float) else str(val)
                    if highlight_best and c in best_vals and abs(f_val - best_vals[c]) < 1e-6:
                        val_str = rf"\textbf{{{val_str}}}"
                    elif highlight_best and c in second_vals and abs(f_val - second_vals[c]) < 1e-6 and abs(best_vals[c] - second_vals[c]) > 1e-6:
                        val_str = rf"\underline{{{val_str}}}"
                else:
                    val_str = _escape_latex(str(val))

                if j == 0 and mark_pareto and pareto_mask[i]:
                    val_str = f"{val_str}$^\\star$"

                row_items.append(val_str)
            lines.append(" & ".join(row_items) + r" \\")

        lines.extend([
            r"\bottomrule",
            rf"\end{{tabular}}",
            rf"\caption{{{caption} $^\star$ marks Pareto-optimal operating points. Bold indicates best, underline second best.}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ])
        return "\n".join(lines)


__all__ = [
    "compute_pareto_frontier",
    "export_benchmark_table",
]
