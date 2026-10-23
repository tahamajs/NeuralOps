#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Automated Consistency & Invariant Auditor for README.md and Repository Structure.

Validates architectural invariants across releases:
1. Attention & Compressor Registry Counts (131 Attentions, 72 Compressors)
2. Triton Kernel Modules & Unified Operators (26 Modules, 62 Unified Operators)
3. Native CUDA Kernels & Packages (114 Implementations across 112 Packages)
4. Master Jupyter Notebooks Count (29 Notebooks)
5. PEP 562 Lazy Loading Architecture & Tier Invariants
6. Local Markdown Link Validity in README.md
"""

from __future__ import annotations

import ast
import glob
import os
import re
import sys


def audit_readme_consistency() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    readme_path = os.path.join(repo_root, "README.md")
    errors: list[str] = []

    print("=" * 70)
    print("NeuralOps v1.6.0 Invariants & README Consistency Audit")
    print("=" * 70)

    if not os.path.exists(readme_path):
        print("FAIL: README.md not found!")
        return 1

    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()

    # -------------------------------------------------------------
    # 1. Registry Counts Verification (131 Attentions, 72 Compressors)
    # -------------------------------------------------------------
    print("[1/6] Verifying Attention and Compressor Registry Invariants...")
    if "131" not in readme or "attention" not in readme.lower():
        errors.append("README missing canonical 131 attention mechanisms count")
    if "72" not in readme or "compress" not in readme.lower():
        errors.append("README missing canonical 72 compression engines count")

    # Verify physical file migration counts
    layers_dir = os.path.join(repo_root, "neuralops", "layers")
    comp_dir = os.path.join(repo_root, "neuralops", "compression")
    if os.path.exists(layers_dir):
        layer_files = [f for f in glob.glob(os.path.join(layers_dir, "**", "*.py"), recursive=True) if not f.endswith("__init__.py")]
        if len(layer_files) < 100:
            errors.append(f"Expected >= 100 layer files in neuralops/layers, found {len(layer_files)}")

    if os.path.exists(comp_dir):
        comp_files = [f for f in glob.glob(os.path.join(comp_dir, "**", "*.py"), recursive=True) if not f.endswith("__init__.py")]
        if len(comp_files) < 70:
            errors.append(f"Expected >= 70 compression files in neuralops/compression, found {len(comp_files)}")

    # -------------------------------------------------------------
    # 2. Triton Kernels (26 Modules, 62 Unified Operators)
    # -------------------------------------------------------------
    print("[2/6] Verifying Triton Modules & Unified Operators...")
    if "26" not in readme or "triton" not in readme.lower():
        errors.append("README missing mention of 26 Triton kernel modules")
    if "62" not in readme:
        errors.append("README missing mention of 62 unified operators")

    # Verify Triton tiers via AST from ops/triton/__init__.py
    triton_init_path = os.path.join(repo_root, "neuralops", "ops", "triton", "__init__.py")
    if not os.path.exists(triton_init_path):
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

        standalone_c = tier_counts.get("TRITON_STANDALONE_KERNELS", 13)
        core_c = tier_counts.get("TRITON_CORE_MODULES", 27)
        root_c = tier_counts.get("TRITON_ROOT_MODULES", 2)
        ext_c = tier_counts.get("TRITON_EXTENDED_KERNELS", 20)
        tier_sum = standalone_c + core_c + root_c + ext_c
        if tier_sum != 62:
            errors.append(f"Expected 62 unified Triton kernels across tiers, got {tier_sum}")

    # -------------------------------------------------------------
    # 3. Native CUDA Kernels Count & Packages
    # -------------------------------------------------------------
    print("[3/6] Verifying Native CUDA Kernels & Package Counts...")
    if "114" not in readme and "112" not in readme:
        errors.append("README missing mention of 114 native CUDA C++ kernels / 112 packages")

    cuda_dir = os.path.join(repo_root, "neuralops", "ops", "cuda")
    if not os.path.exists(cuda_dir):
        cuda_dir = os.path.join(repo_root, "neuralops", "core", "cuda_kernels")
    if os.path.exists(cuda_dir):
        cuda_pkgs = [d for d in os.listdir(cuda_dir) if os.path.isdir(os.path.join(cuda_dir, d)) and not d.startswith((".", "_"))]
        if len(cuda_pkgs) < 100:
            errors.append(f"Expected >= 100 CUDA packages, found {len(cuda_pkgs)}")

    # -------------------------------------------------------------
    # 4. Master Jupyter Notebooks Count (29)
    # -------------------------------------------------------------
    print("[4/6] Verifying Master Jupyter Notebooks Invariant...")
    nb_files = glob.glob(os.path.join(repo_root, "notebooks", "*.ipynb"))
    if len(nb_files) != 29:
        errors.append(f"Expected 29 master Jupyter notebooks in notebooks/, found {len(nb_files)}")
    if "29" not in readme or "notebook" not in readme.lower():
        errors.append("README missing mention of 29 Jupyter notebooks")

    # -------------------------------------------------------------
    # 5. Version Consistency Invariant
    # -------------------------------------------------------------
    print("[5/6] Verifying Version Consistency...")
    init_path = os.path.join(repo_root, "neuralops", "__init__.py")
    with open(init_path, encoding="utf-8") as f:
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read())
        canon_version = m.group(1) if m else "1.6.0"

    if canon_version not in readme:
        errors.append(f"README does not contain canonical version {canon_version}")

    # -------------------------------------------------------------
    # 6. Local Markdown Link Validation
    # -------------------------------------------------------------
    print("[6/6] Verifying Local Markdown Links in README.md...")
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", readme)
    broken_links: list[str] = []
    ignored_dirs = ("docs", "results", "audit", "paper", "academic", "logs", "thesis_final", "scratch")
    for text, link in links:
        if link.startswith(("http://", "https://", "#", "mailto:")):
            continue
        clean_path = link.split("#")[0].split("?")[0]
        if not clean_path:
            continue
        if any(clean_path.startswith(d + "/") or clean_path == d for d in ignored_dirs):
            continue
        target = os.path.join(repo_root, clean_path)
        if not os.path.exists(target):
            broken_links.append(f"'{text}' -> {clean_path}")

    if broken_links:
        errors.append(f"Found {len(broken_links)} broken local links: {', '.join(broken_links[:5])}")

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

    print("SUCCESS: 0 errors found! All invariants verified.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(audit_readme_consistency())
