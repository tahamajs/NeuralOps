#!/usr/bin/env python3
"""
Automated Leaderboard Aggregator & Report Generator.
Merges multi-worker results from summary_worker_*.csv into a unified,
deduplicated, publication-ready leaderboard with tiers, speedup, and metrics.
"""

import os
import sys
import glob
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

SWEEP_DIR = Path("results/05_perplexity_seqlen/gpt2_wikitext2_sweep")

def aggregate():
    csv_files = glob.glob(str(SWEEP_DIR / "summary_worker_*.csv"))
    if not csv_files:
        csv_files = glob.glob(str(SWEEP_DIR / "summary*.csv"))
    
    modules: Dict[str, Dict[str, Any]] = {}

    for f in csv_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for r in reader:
                    name = r.get("module_name", "").strip()
                    if not name or name == "full":
                        continue
                    
                    try:
                        ppl = float(r.get("perplexity", -1.0))
                    except (ValueError, TypeError):
                        ppl = -1.0
                    
                    # Store or update if more recent / better
                    if name not in modules:
                        modules[name] = r
                    else:
                        existing_ppl = float(modules[name].get("perplexity", -1.0))
                        # Prefer valid positive PPL over negative, and lower PPL if both positive
                        if existing_ppl <= 0 < ppl:
                            modules[name] = r
                        elif ppl > 0 and existing_ppl > 0 and ppl < existing_ppl:
                            modules[name] = r
                        elif r.get("timestamp", "") > modules[name].get("timestamp", "") and existing_ppl <= 0:
                            modules[name] = r
        except Exception as e:
            print(f"Warning reading {f}: {e}")

    rows = list(modules.values())
    if not rows:
        print("No module results found.")
        return

    # Sort valid modules by PPL ascending
    valid_rows = [r for r in rows if float(r.get("perplexity", -1.0)) > 0]
    invalid_rows = [r for r in rows if float(r.get("perplexity", -1.0)) <= 0]
    valid_rows.sort(key=lambda r: float(r["perplexity"]))

    baseline_ppl = float(valid_rows[0].get("baseline_ppl", 21.71)) if valid_rows else 21.71

    # Tiers
    tier1 = [r for r in valid_rows if float(r["perplexity"]) < 50.0]
    tier2 = [r for r in valid_rows if 50.0 <= float(r["perplexity"]) < 500.0]
    tier3 = [r for r in valid_rows if 500.0 <= float(r["perplexity"]) < 2000.0]
    tier4 = [r for r in valid_rows if float(r["perplexity"]) >= 2000.0]

    report = []
    report.append("# GPT-2 WikiText-2 Attention & Compression Leaderboard\n")
    report.append(f"- **Baseline GPT-2 PPL**: `{baseline_ppl:.2f}`")
    report.append(f"- **Total Unique Modules Evaluated**: `{len(rows)}`")
    report.append(f"- **Valid PPL Results**: `{len(valid_rows)}`")
    report.append(f"- **Near-Lossless (PPL < 50)**: `{len(tier1)}` modules\n")

    report.append("## Tier 1: Near-Lossless Representations (PPL < 50)\n")
    report.append("| Rank | Module | PPL | PPL Ratio | Comp Ratio | Layers | Throughput (tok/s) |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, r in enumerate(tier1, 1):
        ppl = float(r["perplexity"])
        ratio = ppl / baseline_ppl
        comp = float(r.get("compression_rate", 1.0))
        inj = r.get("num_injected_attention_layers", "12")
        thr = float(r.get("throughput_tok_per_sec", 0.0))
        report.append(f"| {i} | `{r['module_name']}` | **{ppl:.2f}** | **{ratio:.2f}x** | {comp:.3f}x | {inj}/12 | {thr:.1f} |")

    report.append("\n## Tier 2: Strong Compression / Hybrid (PPL 50 - 500)\n")
    report.append("| Rank | Module | PPL | PPL Ratio | Comp Ratio | Layers | Throughput (tok/s) |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, r in enumerate(tier2, len(tier1) + 1):
        ppl = float(r["perplexity"])
        ratio = ppl / baseline_ppl
        comp = float(r.get("compression_rate", 1.0))
        inj = r.get("num_injected_attention_layers", "12")
        thr = float(r.get("throughput_tok_per_sec", 0.0))
        report.append(f"| {i} | `{r['module_name']}` | {ppl:.2f} | {ratio:.2f}x | {comp:.3f}x | {inj}/12 | {thr:.1f} |")

    report.append("\n## Tier 3: Moderate Representation Shift (PPL 500 - 2,000)\n")
    report.append("| Rank | Module | PPL | PPL Ratio | Comp Ratio | Layers |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, r in enumerate(tier3[:15], len(tier1) + len(tier2) + 1):
        ppl = float(r["perplexity"])
        ratio = ppl / baseline_ppl
        comp = float(r.get("compression_rate", 1.0))
        inj = r.get("num_injected_attention_layers", "12")
        report.append(f"| {i} | `{r['module_name']}` | {ppl:.2f} | {ratio:.2f}x | {comp:.3f}x | {inj}/12 |")

    out_md = SWEEP_DIR / "LEADERBOARD.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    print(f"Generated leaderboard with {len(rows)} modules at {out_md}")
    print("\n" + "\n".join(report[:40]))

if __name__ == "__main__":
    aggregate()
