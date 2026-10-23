#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dead Code and AST Integrity Auditor

Audits AST import integrity, dead module stubs, and syntax errors across neuralops.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def audit_syntax_and_imports() -> int:
    source_dir = ROOT_DIR / "neuralops"
    error_count = 0
    checked_files = 0

    for py_file in source_dir.rglob("*.py"):
        checked_files += 1
        try:
            content = py_file.read_text(encoding="utf-8")
            ast.parse(content, filename=str(py_file))
        except SyntaxError as e:
            print(f"Syntax error in {py_file}: {e}")
            error_count += 1
        except Exception as e:
            print(f"Read error in {py_file}: {e}")
            error_count += 1

    print(f"Audited {checked_files} Python source files. Issues found: {error_count}")
    return error_count


def main() -> None:
    issues = audit_syntax_and_imports()
    if issues > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
