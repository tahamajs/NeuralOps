#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Static audit script to verify no silent error swallowing in data-producing paths.

Checks for 'except Exception: pass' and bare 'except:' patterns in fixed
files and functions (scoped to P0 data-producing paths fixed in AGT-06:
neuralops/api.py, neuralops/main.py, and neuralops/export/exporter.py).
The explicit ALLOW_LIST starts empty; any future exception requires a written justification.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Explicit allow-list for known acceptable swallow sites (starts empty)
ALLOW_LIST: list[str] = []

# Scoped target files from Task 2
TARGET_FILES = [
    ROOT / "neuralops" / "api.py",
    ROOT / "neuralops" / "main.py",
    ROOT / "neuralops" / "export" / "exporter.py",
]

# Fixed function scopes within the target files
FIXED_FUNCTIONS = {
    "api.py": {"replace_attention"},
    "main.py": {"run_eval", "run_compress"},
    "exporter.py": {"load_exported_model"},
}


def check_file(file_path: Path) -> list[str]:
    violations: list[str] = []
    if not file_path.exists():
        return [f"File not found: {file_path}"]

    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # 1. Regex check for literal single-line bare except: or except Exception: pass
    bare_except_re = re.compile(r"^\s*except\s*:\s*(#.*)?$")
    except_pass_re = re.compile(r"^\s*except\s+Exception\s*:\s*pass\s*(#.*)?$")

    for idx, line in enumerate(lines, start=1):
        rel_pos = f"{file_path.relative_to(ROOT)}:{idx}"
        if bare_except_re.match(line):
            if rel_pos not in ALLOW_LIST:
                violations.append(f"{rel_pos}: Bare 'except:' found")
        if except_pass_re.match(line):
            if rel_pos not in ALLOW_LIST:
                violations.append(f"{rel_pos}: 'except Exception: pass' found")

    # 2. AST check within the fixed function scopes for multi-line except ...: pass
    fname = file_path.name
    target_funcs = FIXED_FUNCTIONS.get(fname, set())
    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        return [f"SyntaxError parsing {file_path}: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_funcs:
            for child in ast.walk(node):
                if isinstance(child, ast.ExceptHandler):
                    rel_pos = f"{file_path.relative_to(ROOT)}:{child.lineno}"
                    # Check for bare except
                    if child.type is None:
                        if rel_pos not in ALLOW_LIST:
                            violations.append(f"{rel_pos}: Bare 'except:' in {node.name}")
                    # Check for pass body
                    if len(child.body) == 1 and isinstance(child.body[0], ast.Pass):
                        if rel_pos not in ALLOW_LIST:
                            violations.append(f"{rel_pos}: Silent pass handler in {node.name}")

    return violations


def main() -> int:
    all_violations: list[str] = []
    for f in TARGET_FILES:
        all_violations.extend(check_file(f))

    if all_violations:
        print("FAIL: Silent swallowing violations detected:")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: No silent swallowing detected in fixed data-producing paths.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
