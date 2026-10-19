#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Automated Consistency & Invariant Auditor for README.md.

Validates all 12 architectural, registry, benchmark, and arithmetic invariants
to prevent regressions across releases.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys


def audit_readme_consistency() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    readme_path = os.path.join(repo_root, "README.md")
    errors: list[str] = []

    print("=" * 70)
    print("NeuralOps README Consistency & Invariants Audit")
    print("=" * 70)

    if not os.path.exists(readme_path):
        print("FAIL: README.md not found!")
        return 1

    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()

    # -------------------------------------------------------------
    # 1. Registry Counts Verification (131 Attentions, 72 Compressors)
    # -------------------------------------------------------------
    print("[1/12] Verifying Attention and Compressor Registry Invariants...")
    attn_bench_path = os.path.join(repo_root, "results", "triton_all_131_attentions_benchmark.json")
    comp_bench_path = os.path.join(repo_root, "results", "triton_all_72_compressors_benchmark.json")
    
    if os.path.exists(attn_bench_path):
        with open(attn_bench_path, encoding="utf-8") as f:
            attn_data = json.load(f)
        if len(attn_data) != 131:
            errors.append(f"Expected 131 items in attention benchmark JSON, found {len(attn_data)}")
    else:
        errors.append(f"Attention benchmark JSON missing: {attn_bench_path}")

    if os.path.exists(comp_bench_path):
        with open(comp_bench_path, encoding="utf-8") as f:
            comp_data = json.load(f)
        if len(comp_data) != 72:
            errors.append(f"Expected 72 items in compressor benchmark JSON, found {len(comp_data)}")
    else:
        errors.append(f"Compressor benchmark JSON missing: {comp_bench_path}")

    if "131 canonical attention mechanisms" not in readme:
        errors.append("README tagline missing '131 canonical attention mechanisms'")
    if "72 compression engines" not in readme:
        errors.append("README tagline missing '72 compression engines'")
    if "203 out of 203 modules passed" not in readme:
        errors.append("README missing '203 out of 203 modules passed'")

    # -------------------------------------------------------------
    # 2. Section 5 Footnote Clarification
    # -------------------------------------------------------------
    print("[2/12] Verifying Section 5 Footnote Scope...")
    if "Registry Module Scope Footnote" not in readme:
        errors.append("Section 5 missing 'Registry Module Scope Footnote' clarifying 203 modules = 131 + 72")
    elif "203 registered modules comprise exactly 131 attention mechanisms" not in readme:
        errors.append("Section 5 footnote does not explicitly state 131 attention + 72 compression breakdown")

    # -------------------------------------------------------------
    # 3. Triton Kernels Disambiguation (26 Modules vs 62 Operators)
    # -------------------------------------------------------------
    print("[3/12] Verifying Triton Modules & Unified Operators Disambiguation...")
    if "26 categorized Triton kernel modules" not in readme and "26 Categorized Triton Kernel Modules" not in readme:
        errors.append("README does not clarify 26 categorized Triton kernel modules")
    if "62 unified" not in readme.lower() and "62 Unified" not in readme:
        errors.append("README does not mention 62 unified Triton operators / JIT kernels")

    # Verify 4-tier Triton architecture numbers via AST
    triton_init_path = os.path.join(repo_root, "neuralops", "core", "triton_kernels", "__init__.py")
    if os.path.exists(triton_init_path):
        with open(triton_init_path, encoding="utf-8") as f:
            triton_tree = ast.parse(f.read())
        tier_counts: dict[str, int] = {}
        for node in triton_tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in ("TRITON_STANDALONE_KERNELS", "TRITON_CORE_MODULES", "TRITON_ROOT_MODULES", "TRITON_EXTENDED_KERNELS"):
                    if isinstance(node.value, ast.List):
                        tier_counts[node.target.id] = len(node.value.elts)

        standalone_c = tier_counts.get("TRITON_STANDALONE_KERNELS", 0)
        core_c = tier_counts.get("TRITON_CORE_MODULES", 0)
        root_c = tier_counts.get("TRITON_ROOT_MODULES", 0)
        ext_c = tier_counts.get("TRITON_EXTENDED_KERNELS", 0)
        tier_sum = standalone_c + core_c + root_c + ext_c

        if standalone_c != 13:
            errors.append(f"Expected 13 standalone kernels in TRITON_STANDALONE_KERNELS, found {standalone_c}")
        if core_c != 27:
            errors.append(f"Expected 27 core modules in TRITON_CORE_MODULES, found {core_c}")
        if root_c != 2:
            errors.append(f"Expected 2 root modules in TRITON_ROOT_MODULES, found {root_c}")
        if ext_c != 20:
            errors.append(f"Expected 20 extended kernels in TRITON_EXTENDED_KERNELS, found {ext_c}")
        if tier_sum != 62:
            errors.append(f"Expected 62 unified Triton kernels across tiers, got {tier_sum}")

    # -------------------------------------------------------------
    # 4. Native CUDA Kernels Count (114) & 110 Packages
    # -------------------------------------------------------------
    print("[4/12] Verifying Native CUDA Kernels & Package Counts...")
    if "114 native CUDA C++ kernels" not in readme and "114 Native CUDA C++ Kernels" not in readme:
        errors.append("README does not state 114 native CUDA C++ kernels")
    if "110" not in readme:
        errors.append("README missing mention of 110 modular CUDA packages")

    # -------------------------------------------------------------
    # 5. Section 2.3 Production-Certified Native CUDA C++ Kernels
    # -------------------------------------------------------------
    print("[5/12] Verifying Section 2.3 Production-Certified CUDA Kernels...")
    if "The 16 Production-Certified Native CUDA C++ Kernels" not in readme:
        errors.append("Section 2.3 missing 'The 16 Production-Certified Native CUDA C++ Kernels' table heading")
    flagship_rows = re.findall(
        r"\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*\*\*([^*]+)\*\*\s*\|\s*([0-9.]+\s*ms)\s*\|\s*\*\*([0-9.]+x)\*\*",
        readme,
    )
    if len(flagship_rows) != 16:
        errors.append(f"Expected 16 production-certified CUDA kernel rows in Section 2.3, found {len(flagship_rows)}")

    # -------------------------------------------------------------
    # 6. Master Jupyter Notebooks Count (27)
    # -------------------------------------------------------------
    print("[6/12] Verifying Master Jupyter Notebooks Invariant...")
    nb_files = glob.glob(os.path.join(repo_root, "notebooks", "*.ipynb"))
    if len(nb_files) != 27:
        errors.append(f"Expected 27 master Jupyter notebooks in notebooks/, found {len(nb_files)}")
    if "27 master Jupyter notebooks" not in readme and "27 Master Jupyter Notebooks" not in readme:
        errors.append("README missing mention of 27 master Jupyter notebooks")

    # -------------------------------------------------------------
    # 7. Benchmark Speedup Parity with JSON Artifact
    # -------------------------------------------------------------
    print("[7/12] Verifying Speedup Claims against Benchmark Artifact...")
    benchmark_json_path = os.path.join(repo_root, "results", "triton_all_26_kernels_benchmark.json")
    if not os.path.exists(benchmark_json_path):
        errors.append(f"Benchmark artifact not found: {benchmark_json_path}")
    else:
        with open(benchmark_json_path, encoding="utf-8") as f:
            bench_data = json.load(f)
        if len(bench_data) != 26:
            errors.append(f"Expected 26 benchmarked kernels in JSON, found {len(bench_data)}")

        bench_map = {item["kernel"]: item["speedup"] for item in bench_data}
        hgrn2_speedup = bench_map.get("hgrn2_scan")
        gla_speedup = bench_map.get("gla_scan")
        gdn_speedup = bench_map.get("gdn_scan")

        if hgrn2_speedup != 157.9:
            errors.append(f"Benchmark JSON hgrn2_scan speedup changed: {hgrn2_speedup} (expected 157.9)")
        if gla_speedup != 142.82:
            errors.append(f"Benchmark JSON gla_scan speedup changed: {gla_speedup} (expected 142.82)")
        if gdn_speedup != 137.69:
            errors.append(f"Benchmark JSON gdn_scan speedup changed: {gdn_speedup} (expected 137.69)")

        # Verify README references 157.9x as peak speedup
        if "157.9x" not in readme:
            errors.append("README does not reference authoritative peak speedup of 157.9x (hgrn2_scan)")

        # Verify outdated 179.1x or 161.2x claims are eliminated
        if "179.1x" in readme:
            errors.append("README still contains outdated/unaligned 179.1x speedup claim")
        if "161.2x" in readme:
            errors.append("README still contains outdated 161.2x speedup claim in Section 3/16")

    # -------------------------------------------------------------
    # 8. Section 16 Spectrum Breakdown Arithmetic
    # -------------------------------------------------------------
    print("[8/12] Verifying Section 16 Spectrum Breakdown Arithmetic...")
    spectrum_match = re.search(
        r"High-Speedup Kernels \(> 2\.0x\)\s*:\s*(\d+)\s*modules.*?"
        r"Moderate-Speedup Kernels \(1\.0x-2\.0x\):\s*(\d+)\s*modules.*?"
        r"Near-Parity / Slower \(< 1\.0x\)\s*:\s*(\d+)\s*module",
        readme,
        re.DOTALL,
    )
    if not spectrum_match:
        errors.append("Could not parse Section 16 Spectrum Breakdown counts")
    else:
        s_counts = [int(g) for g in spectrum_match.groups()]
        s_total = sum(s_counts)
        if s_total != 26:
            errors.append(f"Section 16 spectrum modules sum to {s_total} ({s_counts}), expected exactly 26")
        if s_counts != [13, 12, 1]:
            errors.append(f"Section 16 spectrum distribution is {s_counts}, expected [13, 12, 1]")

    # -------------------------------------------------------------
    # 9. Remote Worktree Path Consistency (/mnt/d/SVD_Project)
    # -------------------------------------------------------------
    print("[9/12] Verifying Remote Worktree Path Consistency...")
    if "/mnt/d/NeuralOps" in readme:
        errors.append("Found outdated remote worktree path '/mnt/d/NeuralOps', expected '/mnt/d/SVD_Project'")
    if "/mnt/d/SVD_Project" not in readme:
        errors.append("Expected remote worktree path '/mnt/d/SVD_Project' not found in README")

    # -------------------------------------------------------------
    # 10. Release Report References & File Existence
    # -------------------------------------------------------------
    print("[10/12] Verifying Release Report Reference & Target File...")
    if "V1_4_0_RELEASE_REPORT.md" in readme:
        errors.append("Found legacy 'V1_4_0_RELEASE_REPORT.md' reference in README, expected 'V1_4_3_RELEASE_REPORT.md'")
    if "V1_4_3_RELEASE_REPORT.md" not in readme and "V1_4_2_RELEASE_REPORT.md" not in readme:
        errors.append("Expected 'V1_4_3_RELEASE_REPORT.md' reference not found in Section 22")
    rel_report_path = os.path.join(repo_root, "docs", "reports", "V1_4_3_RELEASE_REPORT.md")
    if not os.path.exists(rel_report_path):
        rel_report_path_prev = os.path.join(repo_root, "docs", "reports", "V1_4_2_RELEASE_REPORT.md")
        if not os.path.exists(rel_report_path_prev):
            errors.append(f"Target release report file missing: {rel_report_path}")

    # -------------------------------------------------------------
    # 11. Local Markdown Link Validation
    # -------------------------------------------------------------
    print("[11/12] Verifying All Local Markdown Links in README.md...")
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", readme)
    broken_links: list[str] = []
    for text, link in links:
        if link.startswith(("http://", "https://", "#", "mailto:")):
            continue
        clean_path = link.split("#")[0].split("?")[0]
        if not clean_path:
            continue
        target = os.path.join(repo_root, clean_path)
        if not os.path.exists(target):
            broken_links.append(f"'{text}' -> {clean_path}")

    if broken_links:
        errors.append(f"Found {len(broken_links)} broken local links: {', '.join(broken_links[:5])}")

    # -------------------------------------------------------------
    # 12. Hugging Face Links & Model Zoo Verification
    # -------------------------------------------------------------
    print("[12/12] Verifying Hugging Face Model Zoo references...")
    if "https://huggingface.co/tahamajs/distilbert-svd-attention" not in readme:
        errors.append("Hugging Face model URL 'https://huggingface.co/tahamajs/distilbert-svd-attention' not found")
    model_cards = [
        "docs/model_cards/distilbert-base-fwsvd-rank64.md",
        "docs/model_cards/gpt2-medium-fwsvd-rank64.md",
        "docs/model_cards/roberta-base-fwsvd-rank32.md",
    ]
    for mc in model_cards:
        if not os.path.exists(os.path.join(repo_root, mc)):
            errors.append(f"Missing model card file: {mc}")

    # -------------------------------------------------------------
    # Final Report
    # -------------------------------------------------------------
    print("=" * 70)
    if errors:
        print(f"FAILED: Found {len(errors)} consistency error(s):")
        for idx, err in enumerate(errors, 1):
            print(f"  {idx}. {err}")
        print("=" * 70)
        return 1

    print("SUCCESS: 0 errors found! All 12 README consistency invariants verified.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(audit_readme_consistency())
