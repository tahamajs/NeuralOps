"""Report generators for module analysis outputs (CSV and Markdown)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from neuralops.analysis.module_analysis.constants import CSV_OUT, MD_OUT


ANALYSIS_FIELDNAMES = [
    "module_name",
    "file_path",
    "module_type",
    "source_dir",
    "category",
    "sub_category",
    "paper_reference",
    "paper_arxiv",
    "paper_year",
    "paper_venue",
    "architectures",
    "has_registry_entry",
    "registry_names",
    "dataset_sst2_acc",
    "dataset_mnli_acc",
    "dataset_agnews_acc",
    "dataset_gpt2_ppl",
    "compression_ratio",
    "params_before",
    "params_after",
    "paper_07_results_status",
    "has_fabricated_tables",
    "csv_source",
    "csv_timestamp",
    "fix_priority",
]


def write_csv_report(analysis_rows: List[Dict[str, Any]], output_path: Path = CSV_OUT) -> None:
    """Write analysis rows to CSV file.

    Args:
        analysis_rows: List of module analysis record dicts.
        output_path: Target CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_FIELDNAMES)
        writer.writeheader()
        for r in analysis_rows:
            writer.writerow(r)


def write_markdown_report(analysis_rows: List[Dict[str, Any]], output_path: Path = MD_OUT) -> None:
    """Generate and write comprehensive Markdown summary report.

    Args:
        analysis_rows: List of module analysis record dicts.
        output_path: Target Markdown path.
    """
    cat_counts: Dict[str, int] = defaultdict(int)
    prio_counts: Dict[str, int] = defaultdict(int)
    with_paper = sum(1 for r in analysis_rows if r["paper_reference"] != "None")
    with_arxiv = sum(1 for r in analysis_rows if r["paper_arxiv"] != "None")
    with_sst2 = sum(1 for r in analysis_rows if r["dataset_sst2_acc"] != "N/A")
    with_mnli = sum(1 for r in analysis_rows if r["dataset_mnli_acc"] != "N/A")
    with_agnews = sum(1 for r in analysis_rows if r["dataset_agnews_acc"] != "N/A")
    with_gpt2 = sum(1 for r in analysis_rows if r["dataset_gpt2_ppl"] != "N/A")
    with_fab = sum(1 for r in analysis_rows if r["has_fabricated_tables"] == "YES")

    for r in analysis_rows:
        cat_counts[r["category"]] += 1
        prio_counts[r["fix_priority"]] += 1

    md_lines = [
        "# Comprehensive 217-Module Analysis Report\n",
        "This document provides the exhaustive audit and empirical validation across all 217 transformer compression ",
        "and linear attention modules in the research framework. All metrics strictly trace to empirical benchmark CSV records.\n\n",
        "## 1. Executive Summary & Category Breakdown\n\n",
        "| Category | Count | Proportion | Meaning |\n",
        "|:---|:---:|:---:|:---|\n",
    ]
    for cat in ["VALID", "FALLBACK", "BROKEN", "QUESTIONABLE", "NO_DATA"]:
        n = cat_counts[cat]
        pct = (100.0 * n / len(analysis_rows)) if analysis_rows else 0.0
        md_lines.append(f"| **{cat}** | {n} | {pct:.1f}% | {cat} empirical evaluation status |\n")

    md_lines.extend(
        [
            "\n## 2. Statistical Highlights\n\n",
            f"- **Total Evaluated Modules**: {len(analysis_rows)}\n",
            f"- **Modules with Academic Literature Reference**: {with_paper} ({100.0 * with_paper / max(1, len(analysis_rows)):.1f}%)\n",
            f"- **Modules with arXiv ID**: {with_arxiv} ({100.0 * with_arxiv / max(1, len(analysis_rows)):.1f}%)\n",
            f"- **Modules with Empirical SST-2 Benchmark**: {with_sst2} ({100.0 * with_sst2 / max(1, len(analysis_rows)):.1f}%)\n",
            f"- **Modules with Empirical MNLI Benchmark**: {with_mnli} ({100.0 * with_mnli / max(1, len(analysis_rows)):.1f}%)\n",
            f"- **Modules with Empirical AG News Benchmark**: {with_agnews} ({100.0 * with_agnews / max(1, len(analysis_rows)):.1f}%)\n",
            f"- **Modules with GPT-2 WikiText-2 PPL**: {with_gpt2} ({100.0 * with_gpt2 / max(1, len(analysis_rows)):.1f}%)\n",
            f"- **Modules with Quarantined Fabricated Tables**: {with_fab} ({100.0 * with_fab / max(1, len(analysis_rows)):.1f}%)\n\n",
            "## 3. Actionable Priority Fix List\n\n",
            f"- **P0 (Broken - Immediate submission blockers)**: {prio_counts['P0']} modules\n",
            f"- **P1 (Fallback Compressors - Fixable via Cholesky whitening / LoRA unfreezing)**: {prio_counts['P1']} modules\n",
            f"- **P2 (Fallback Attention - Inherent linear attention replacements, move to Related Work)**: {prio_counts['P2']} modules\n",
            f"- **P3 (No Data - Needs evaluation run)**: {prio_counts['P3']} modules\n",
            f"- **NONE (Valid - Production-hardened with genuine compression)**: {prio_counts['NONE']} modules\n\n",
            "### P0 Action Items (Immediate Fixes)\n\n",
        ]
    )

    for r in analysis_rows:
        if r["fix_priority"] == "P0":
            md_lines.append(
                f"- `{r['module_name']}`: {r['file_path']} (Category: {r['category']}, Sub: {r['sub_category']})\n"
            )

    md_lines.extend(
        [
            "\n## 4. Complete 217-Module Master Table\n\n",
            "| Module Name | Type | Category | Sub-Category | SST-2 Acc | MNLI Acc | Comp Ratio | Priority | Source CSV |\n",
            "|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|\n",
        ]
    )

    for r in analysis_rows:
        src = r["csv_source"]
        if len(src) > 35:
            src = "..." + src[-32:]
        md_lines.append(
            f"| `{r['module_name']}` | {r['module_type']} | **{r['category']}** | {r['sub_category']} | "
            f"{r['dataset_sst2_acc']} | {r['dataset_mnli_acc']} | {r['compression_ratio']} | **{r['fix_priority']}** | `{src}` |\n"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)


__all__ = ["ANALYSIS_FIELDNAMES", "write_csv_report", "write_markdown_report"]
