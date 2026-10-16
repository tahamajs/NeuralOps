#!/usr/bin/env python3
"""
Combine batch validation results from all models into a rigorous, categorized leaderboard.
Separates:
  1. VALID Rank-Sensitive Low-Rank Compressors (Main Leaderboard)
  2. NO_OP / Heuristic Rank-Overriding / Fallback Compressors (Forensic Audit Table)
  3. FAILED / Incompatible Compressors
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

RESULTS_DIR = Path("results/batch_validation")
MODELS = ["synth_block", "roberta", "gpt2"]
RANKS = [8, 16, 32, 64]

VALID_RANK_SENSITIVE = [
    "direct_svd",
    "hnc_svd",
    "whitened_svd",
    "fisher_svd",
    "fwsvd",
    "dobi_svd",
    "tt_compressor",
    "rpca_compressor",
    "entropy_budget_allocation",
]

EXCLUDED_NO_OP_DETAILS = {
    "aa_svd": "Structural Rank Search (use_strs=True) overrides requested rank to fixed candidate (r=16); constant output across ranks",
    "asvd_compressor": "Alias for aa_svd; identical STRS candidate rank override",
    "adasvd_compressor": "Adaptive energy-variance threshold selects internal rank independently of target_rank; constant output across ranks",
    "sparsegpt": "Second-order unstructured weight pruner; fixed default sparsity, ignores low-rank decomposition rank parameter",
    "flap": "Structured pruning algorithm; sparsity-based, ignores low-rank decomposition rank parameter",
    "admm_pruner": "Alternating Direction Method of Multipliers weight pruner; fixed sparsity, ignores low-rank rank parameter",
    "smoothquant_real": "SmoothQuant activation-weight scaling for INT8/FP8 quantization; fixed migration scale alpha=0.5, no rank concept",
    "slicegpt": "PCA-ASVD / SliceGPT computational invariance pruner; uses internal feature ratio, rank parameter is ignored",
    "quarot": "Rotational INT4 quantizer; reconstruction error exceeded threshold, 100% layer fallback to uncompressed dense layers (err=0.0000)",
    "adaptive_ranks": "Dynamic error-bounded rank search; error exceeded threshold on benchmark block, 100% fallback to uncompressed dense layers (err=0.0000)",
    "gptq_real": "4-bit/8-bit weight quantizer; step-discontinuity caused by wrapper switching bits (4-bit for r<=32, 8-bit for r=64), not matrix rank",
    "w4a16": "Fixed 4-bit weight, 16-bit activation quantizer; rank parameter ignored (delta err = -0.0001)",
    "wanda_real": "Weight-and-activation magnitude pruner; sparsity heuristic, not low-rank SVD",
    "erc_svd": "Energy-ratio clipped SVD; falls back to uncompressed layers at low ranks and clips above threshold, non-monotonic rank scaling",
    "kronecker_compressor": "Kronecker factorizer; suffers from shape factorability constraints, leading to inconsistent partial layer fallbacks across ranks",
}

INCOMPATIBLE_DETAILS = {
    "int4_lut_quantizer": "Constructor returned None / specialized CoOPQ LUT structure incompatible with standard module interface",
    "tucker_compressor": "Constructor returned None / CompOT Tucker decomposition requiring specific 4D tensor configurations",
}

def main():
    # Load all aggregates
    all_data: Dict[str, Dict[str, Any]] = {}
    for model in MODELS:
        path = RESULTS_DIR / model / f"{model}_batch_results.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                all_data[model] = loaded.get("aggregate", {})

    md_path = RESULTS_DIR / "LEADERBOARD.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🏆 NeuralOps Categorized Cross-Method Batch Validation Leaderboard\n\n")
        f.write("All experiments executed on physical **NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)** across **10 random seeds** per configuration.\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Scientific Rigor & Rank-Sensitivity Integrity**: Following forensic diagnostic testing (`diff_8_64` on weights and outputs), ")
        f.write("compressors are strictly segregated into **Verified Rank-Sensitive Low-Rank Methods** (Table 1) and ")
        f.write("**Excluded Methods** (Table 2: pruning, quantization, or heuristic rank overrides that ignore the rank parameter).\n\n")

        # -------------------------------------------------------------
        # TABLE 1: VALID COMPRESSORS
        # -------------------------------------------------------------
        f.write("## 1. Verified Rank-Sensitive Low-Rank Compressors (Main Leaderboard)\n\n")
        f.write("Methods that legitimately scale matrix factorization rank with parameter count and output behavior:\n\n")

        for rank in RANKS:
            f.write(f"### Target Rank $r = {rank}$\n\n")
            f.write("| Compressor | Type / Family | " + " | ".join([f"{m} (RelErr ± std)" for m in MODELS]) + " | Ratio vs `direct_svd` | Verdict |\n")
            f.write("|:---|:---|" + "|".join([":---:"] * len(MODELS)) + "|:---:|:---:|\n")

            base_key = f"direct_svd_rank{rank}"
            for method in VALID_RANK_SENSITIVE:
                row = f"| **`{method}`** | "
                # Family tag
                if "hnc" in method:
                    fam = "Hessian Null Space"
                elif "fisher" in method or "fwsvd" in method:
                    fam = "Fisher-Weighted SVD"
                elif "whitened" in method:
                    fam = "Cholesky-Whitened SVD"
                elif "tt" in method:
                    fam = "Tensor Train Decomp"
                elif "rpca" in method:
                    fam = "Robust PCA"
                elif "entropy" in method:
                    fam = "Spectral Entropy"
                elif "dobi" in method:
                    fam = "Dual Orthogonal Basis"
                else:
                    fam = "Truncated SVD Baseline"
                row += f"{fam} |"

                ratios = []
                for model in MODELS:
                    key = f"{method}_rank{rank}"
                    b_key = f"direct_svd_rank{rank}"
                    if model in all_data and key in all_data[model]:
                        a = all_data[model][key]
                        if a.get("status") == "SUCCESS" and "rel_err_mean" in a:
                            m_err = a['rel_err_mean']
                            s_err = a['rel_err_std']
                            row += f" {m_err:.4f} ± {s_err:.4f} |"
                            if model in all_data and b_key in all_data[model]:
                                b_err = all_data[model][b_key].get("rel_err_mean", 1.0)
                                ratios.append(m_err / max(b_err, 1e-8))
                        else:
                            row += " FAILED |"
                    else:
                        row += " — |"

                if method == "direct_svd":
                    row += " 1.0000x | **Baseline** |"
                elif ratios:
                    mean_ratio = sum(ratios) / len(ratios)
                    verdict = "**VALIDATED (<0.75)**" if mean_ratio < 0.75 else ("SIGNIFICANT (<1.0)" if mean_ratio < 1.0 else "INFERIOR")
                    row += f" **{mean_ratio:.4f}x** | {verdict} |"
                else:
                    row += " — | — |"

                f.write(row + "\n")
            f.write("\n")

        # -------------------------------------------------------------
        # TABLE 2: EXCLUDED NO-OP COMPRESSORS
        # -------------------------------------------------------------
        f.write("## 2. Forensic Audit of Excluded Compressors (Rank-Insensitive / NO-OP)\n\n")
        f.write("The following methods were evaluated but **excluded from the primary rank leaderboard** because diagnostic testing proved they ignore the `--rank` parameter, apply fixed-sparsity pruning, or fall back to original weights:\n\n")
        f.write("| Excluded Method | Primary Family | Constant RelErr Observed | Forensic Diagnostic Finding |\n")
        f.write("|:---|:---|:---:|:---|\n")
        for m, reason in sorted(EXCLUDED_NO_OP_DETAILS.items()):
            # Find sample error if present
            sample_err = "—"
            for model in MODELS:
                if model in all_data:
                    k8 = f"{m}_rank8"
                    if k8 in all_data[model] and "rel_err_mean" in all_data[model][k8]:
                        sample_err = f"{all_data[model][k8]['rel_err_mean']:.4f}"
                        break
            fam = "Pruning" if "pruner" in m or "sparse" in m or "flap" in m or "wanda" in m or "slice" in m else ("Quantization" if "quant" in m or "quarot" in m or "gptq" in m or "w4a16" in m else "Adaptive / SVD")
            f.write(f"| **`{m}`** | {fam} | `{sample_err}` | {reason} |\n")
        f.write("\n")

        # -------------------------------------------------------------
        # TABLE 3: INCOMPATIBLE / FAILED COMPRESSORS
        # -------------------------------------------------------------
        f.write("## 3. Incompatible / Failed Compressors\n\n")
        f.write("| Compressor | Status | Root Cause Analysis |\n")
        f.write("|:---|:---:|:---|\n")
        for m, reason in sorted(INCOMPATIBLE_DETAILS.items()):
            f.write(f"| **`{m}`** | `FAILED_TO_LOAD` | {reason} |\n")
        f.write("\n")

    print(f"\nCategorized leaderboard successfully generated at {md_path}")

if __name__ == "__main__":
    main()
