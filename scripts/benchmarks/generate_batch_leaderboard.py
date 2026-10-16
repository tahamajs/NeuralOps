#!/usr/bin/env python3
"""
Combine batch results from all models into a cross-method leaderboard.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

RESULTS_DIR = Path("results/batch_validation")
MODELS = ["synth_block", "roberta", "gpt2"]
RANKS = [8, 16, 32, 64]

def main():
    # Load all aggregates
    all_data: Dict[str, Dict[str, Any]] = {}
    for model in MODELS:
        path = RESULTS_DIR / model / f"{model}_batch_results.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                all_data[model] = loaded.get("aggregate", {})

    print("\n" + "=" * 140)
    print("CROSS-METHOD LEADERBOARD (mean rel_err ± std over 10 seeds)")
    print("=" * 140)

    methods: Set[str] = set()
    for model in all_data:
        for key in all_data[model]:
            if "method" in all_data[model][key]:
                methods.add(all_data[model][key]["method"])

    sorted_methods = sorted(list(methods))

    for rank in RANKS:
        print(f"\n--- Rank = {rank} ---")
        header = f"{'Method':<26s}"
        for model in MODELS:
            header += f"  {model[:12]:>22s}"
        print(header)
        print("-" * len(header))

        for method in sorted_methods:
            row = f"{method:<26s}"
            for model in MODELS:
                key = f"{method}_rank{rank}"
                if model in all_data and key in all_data[model]:
                    a = all_data[model][key]
                    if a.get("status") == "SUCCESS" and "rel_err_mean" in a:
                        row += f"  {a['rel_err_mean']:.4f} ± {a['rel_err_std']:.4f}"
                    else:
                        row += f"  {'FAILED':>22s}"
                else:
                    row += f"  {'—':>22s}"
            print(row)

    # Save as Markdown
    md_path = RESULTS_DIR / "LEADERBOARD.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🏆 Cross-Method Batch Validation Leaderboard\n\n")
        f.write("All results are reported as **mean ± std** across 10 random seeds on physical NVIDIA RTX 4070 (SM 8.9).\n\n")
        for rank in RANKS:
            f.write(f"## Rank = {rank}\n\n")
            f.write("| Method | " + " | ".join(MODELS) + " |\n")
            f.write("|:---|" + "|".join([":---:"] * len(MODELS)) + "|\n")
            for method in sorted_methods:
                row = f"| **{method}** |"
                for model in MODELS:
                    key = f"{method}_rank{rank}"
                    if model in all_data and key in all_data[model]:
                        a = all_data[model][key]
                        if a.get("status") == "SUCCESS" and "rel_err_mean" in a:
                            row += f" {a['rel_err_mean']:.4f} ± {a['rel_err_std']:.4f} |"
                        else:
                            row += " FAILED |"
                    else:
                        row += " — |"
                f.write(row + "\n")
            f.write("\n")

    print(f"\nLeaderboard written to {md_path}")

if __name__ == "__main__":
    main()
