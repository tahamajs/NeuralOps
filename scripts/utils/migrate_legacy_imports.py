#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
AST-based Codemod Transformer for NeuralOps Legacy Imports.

Safely rewrites deprecated import paths across Python source files using
the ast.NodeTransformer API. Only modifies Python files (*.py) in target
directories (e.g., tests/ and scripts/). Comments, docstrings, and markdown
files are strictly untouched.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Optional


MODULE_MAPPINGS = [
    ("neuralops.attentions", "neuralops.layers"),
    ("neuralops.compression_engine", "neuralops.compression"),
    ("neuralops.training", "neuralops.trainer"),
    ("neuralops.serving", "neuralops.export.serving"),
]


class ImportMigrator(ast.NodeTransformer):
    """AST Transformer that rewrites legacy NeuralOps import modules."""

    def __init__(self) -> None:
        super().__init__()
        self.modified = False

    def _remap_module(self, mod_name: Optional[str]) -> Optional[str]:
        if not mod_name:
            return mod_name
        for old_prefix, new_prefix in MODULE_MAPPINGS:
            if mod_name == old_prefix:
                self.modified = True
                return new_prefix
            if mod_name.startswith(old_prefix + "."):
                self.modified = True
                return new_prefix + mod_name[len(old_prefix):]
        return mod_name

    def visit_Import(self, node: ast.Import) -> ast.Import:
        new_names = []
        for alias in node.names:
            remapped = self._remap_module(alias.name)
            if remapped != alias.name:
                new_names.append(ast.alias(name=remapped, asname=alias.asname))
            else:
                new_names.append(alias)
        node.names = new_names
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.module:
            remapped = self._remap_module(node.module)
            if remapped != node.module:
                node.module = remapped
        return node


def process_file(file_path: Path) -> bool:
    """Parses, transforms, and writes back Python file if imports changed."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except Exception as e:
        print(f"Skipping {file_path}: parse error {e}", file=sys.stderr)
        return False

    transformer = ImportMigrator()
    new_tree = transformer.visit(tree)

    if transformer.modified:
        ast.fix_missing_locations(new_tree)
        # unparse is standard in Python 3.9+
        new_source = ast.unparse(new_tree) + "\n"
        file_path.write_text(new_source, encoding="utf-8")
        return True
    return False


def run_migration(target_dirs: list[str]) -> int:
    """Migrates all python files in the given target directories."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    total_modified = 0

    for target in target_dirs:
        target_path = repo_root / target
        if not target_path.exists():
            continue
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.endswith(".py"):
                    full_p = Path(root) / file
                    if process_file(full_p):
                        total_modified += 1
                        print(f"Migrated AST imports: {full_p.relative_to(repo_root)}")

    return total_modified


if __name__ == "__main__":
    targets = ["tests", "scripts"] if len(sys.argv) < 2 else sys.argv[1:]
    count = run_migration(targets)
    print(f"\nAST Migration Complete. Modified {count} files.")
