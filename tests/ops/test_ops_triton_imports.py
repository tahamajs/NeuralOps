#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying that Triton imports from CUDA autograd interfaces resolve cleanly.
"""

from __future__ import annotations

import ast
from pathlib import Path
import pytest


def test_rmsnorm_triton_import_resolves():
    """Verify rmsnorm_triton resolves to a real export in elementwise."""
    from neuralops.ops.triton.elementwise import rmsnorm_triton, fused_rmsnorm

    assert rmsnorm_triton is not None
    assert rmsnorm_triton is fused_rmsnorm


def test_all_autograd_triton_imports_resolve():
    """Verify every from neuralops.ops.triton.* import in *_autograd.py files resolves."""
    root_dir = Path(__file__).resolve().parent.parent
    cuda_dir = root_dir / "neuralops" / "ops" / "cuda"

    def get_module_symbols(mod_name: str) -> tuple[set[str] | None, str | None]:
        parts = mod_name.split(".")
        rel_path = Path(*parts[1:])  # drop "neuralops"
        file_path = root_dir / "neuralops" / rel_path.with_suffix(".py")
        if not file_path.exists():
            dir_init = root_dir / "neuralops" / rel_path / "__init__.py"
            if dir_init.exists():
                file_path = dir_init
            else:
                return None, f"File {file_path} does not exist"

        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        symbols = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.add(target.id)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    asname = alias.asname or alias.name
                    symbols.add(asname)
        return symbols, None

    autograd_files = sorted(cuda_dir.glob("**/*autograd*.py"))
    assert len(autograd_files) > 0, "No autograd files found"

    unresolved = []
    checked_count = 0

    for af in autograd_files:
        tree = ast.parse(af.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and ("ops.triton" in node.module or "core.triton_kernels" in node.module)
            ):
                symbols, err = get_module_symbols(node.module)
                for alias in node.names:
                    checked_count += 1
                    if err:
                        unresolved.append((str(af), node.module, alias.name, err))
                    elif symbols is None or alias.name not in symbols:
                        unresolved.append(
                            (str(af), node.module, alias.name, f"Symbol {alias.name} not in {node.module}")
                        )

    assert checked_count > 0, "No Triton imports found to check"
    assert not unresolved, f"Unresolved Triton imports in autograd files: {unresolved}"
