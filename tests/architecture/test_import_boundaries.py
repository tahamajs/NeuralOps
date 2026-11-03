#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Automated Static AST Architectural Boundary and Import Policy Tests.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Forbidden dependency directional mapping
FORBIDDEN_DEPENDENCIES = {
    "neuralops/layers": [
        "neuralops.runners",
        "neuralops.reporting",
        "neuralops.results_saver",
        "neuralops.cli",
    ],
    "neuralops/ops": [
        "neuralops.runners",
        "neuralops.reporting",
        "neuralops.results_saver",
        "neuralops.cli",
    ],
    "neuralops/operations": [
        "neuralops.runners",
        "neuralops.cli",
    ],
    "neuralops/recipe": [
        "neuralops.runners",
        "neuralops.inference",
        "neuralops.pipeline_transform",
    ],
}


def test_ast_import_boundaries():
    """Statically verifies that low-level subsystems never import higher-level workflow components."""
    violations = []

    for rel_dir, banned_imports in FORBIDDEN_DEPENDENCIES.items():
        abs_dir = REPO_ROOT / rel_dir
        if not abs_dir.exists():
            continue

        for root, _, files in os.walk(abs_dir):
            if "__pycache__" in root:
                continue
            for file_name in files:
                if not file_name.endswith(".py"):
                    continue

                file_path = Path(root) / file_name
                try:
                    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
                except Exception as exc:
                    violations.append(f"AST parse error in {file_path}: {exc}")
                    continue

                for node in ast.walk(tree):
                    imported_module = ""
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_module = alias.name
                            for banned in banned_imports:
                                if imported_module.startswith(banned):
                                    rel_path = file_path.relative_to(REPO_ROOT)
                                    violations.append(
                                        f"{rel_path}:{node.lineno} imports forbidden '{imported_module}' (rule: {rel_dir} -> {banned})"
                                    )
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported_module = node.module
                        for banned in banned_imports:
                            if imported_module.startswith(banned):
                                rel_path = file_path.relative_to(REPO_ROOT)
                                violations.append(
                                    f"{rel_path}:{node.lineno} imports forbidden '{imported_module}' (rule: {rel_dir} -> {banned})"
                                )

    assert not violations, "Architectural boundary violations found:\n" + "\n".join(violations)


def test_canonical_catalog_counts():
    """Verifies that the static catalog retains the exact canonical invariant counts."""
    catalog_path = REPO_ROOT / "neuralops" / "benchmarks" / "static_catalog.json"
    assert catalog_path.is_file(), f"Catalog file not found at {catalog_path}"

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    attentions = catalog.get("attentions", [])
    compressors = catalog.get("compressors", [])

    assert len(attentions) == 131, (
        f"Expected exactly 131 canonical attentions, found {len(attentions)}"
    )
    assert len(compressors) == 72, (
        f"Expected exactly 72 canonical compressors, found {len(compressors)}"
    )


def test_triton_kernel_modules_count():
    """Verifies that neuralops/core/triton_kernels contains exactly 26 canonical modules."""
    triton_dir = REPO_ROOT / "neuralops" / "core" / "triton_kernels"
    assert triton_dir.is_dir(), f"Triton kernels dir not found at {triton_dir}"

    py_files = [
        f.stem
        for f in triton_dir.glob("*.py")
        if f.is_file() and f.stem != "__init__" and not f.stem.startswith(".")
    ]
    assert len(py_files) == 26, (
        f"Expected exactly 26 canonical Triton modules in neuralops/core/triton_kernels, found {len(py_files)}"
    )
