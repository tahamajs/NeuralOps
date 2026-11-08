#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Static audit script to verify no silent error swallowing in data-producing paths.

Checks for 'except Exception: pass', bare 'except:', and silent pass exception
handlers across all critical and deferred paths:
  - neuralops/api.py
  - neuralops/main.py
  - neuralops/export/exporter.py
  - neuralops/__init__.py
  - neuralops/_optional.py
  - neuralops/compression/__init__.py
  - neuralops/cli/main.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Explicit allow-list for known acceptable swallow sites (each requires justification)
ALLOW_LIST: list[str] = [
    # neuralops/main.py:32 - Bootstrap stderr UTF-8 encoding reconfigure polyfill
    "neuralops/main.py:32",
]

# Scoped target files covered by AGT-06 and AGT-55 (F-84)
TARGET_FILES = [
    ROOT / "neuralops" / "api.py",
    ROOT / "neuralops" / "main.py",
    ROOT / "neuralops" / "export" / "exporter.py",
    ROOT / "neuralops" / "__init__.py",
    ROOT / "neuralops" / "_optional.py",
    ROOT / "neuralops" / "compression" / "__init__.py",
    ROOT / "neuralops" / "cli" / "main.py",
]

# Critical function scopes within target files requiring zero-silent-swallow enforcement
FIXED_FUNCTIONS = {
    "api.py": {"replace_attention", "_decode_tokens", "list_attentions", "heal_distributed"},
    "main.py": {"run_eval", "run_compress", "run_train"},
    "exporter.py": {"load_exported_model"},
    "__init__.py": {"_patch_torch_compat", "_patch_pascal_svd_if_needed", "find_spec"},
    "_optional.py": {"_require_torch"},
    "compression/__init__.py": {"__dir__"},
}


def check_file(file_path: Path) -> list[str]:
    violations: list[str] = []
    if not file_path.exists():
        return [f"File not found: {file_path}"]

    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # 1. Regex check for literal single-line bare except: or except Exception: pass
    bare_except_re = re.compile(r"^\s*except\s*:\s*(#.*)?$")
    except_pass_re = re.compile(r"^\s*except(\s+Exception)?\s*:\s*pass\s*(#.*)?$")

    for idx, line in enumerate(lines, start=1):
        rel_pos = f"{file_path.relative_to(ROOT)}:{idx}"
        if bare_except_re.match(line):
            if rel_pos not in ALLOW_LIST:
                violations.append(f"{rel_pos}: Bare 'except:' found")
        if except_pass_re.match(line):
            if rel_pos not in ALLOW_LIST:
                violations.append(f"{rel_pos}: 'except Exception: pass' found")

    # 2. AST check across the entire file for any bare except or except Exception -> pass
    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        return [f"SyntaxError parsing {file_path}: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            rel_pos = f"{file_path.relative_to(ROOT)}:{node.lineno}"
            # Check for bare except
            if node.type is None:
                if rel_pos not in ALLOW_LIST:
                    violations.append(f"{rel_pos}: Bare 'except:' handler found")

            # Check for except Exception: pass (or BaseException: pass)
            is_generic_exc = False
            if node.type is not None:
                if isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                    is_generic_exc = True

            if is_generic_exc and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                if rel_pos not in ALLOW_LIST:
                    violations.append(f"{rel_pos}: Generic 'except Exception: pass' handler found")

    # 3. AST check within fixed function scopes for ANY silent pass handler
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fname = file_path.name
            target_funcs = FIXED_FUNCTIONS.get(fname, set())
            if node.name in target_funcs:
                for child in ast.walk(node):
                    if isinstance(child, ast.ExceptHandler):
                        rel_pos = f"{file_path.relative_to(ROOT)}:{child.lineno}"
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
