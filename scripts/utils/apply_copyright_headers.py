#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Utility to apply standardized Copyright & License header to all source files based on file type and repo name (NeuralOps).
"""

from __future__ import annotations

from pathlib import Path

PY_HEADER = """# Copyright (c) 2024-2026, Taha Majlesi
"""

CPP_HEADER = """// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""

EXCLUDE_PARTS = {".git", "__pycache__", "build", "dist", ".venv", "venv", "archive", "_archive", ".ipynb_checkpoints"}


def apply_header_to_py(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return False

    if "Copyright (c) 2024-2026, Taha Majlesi" in content:
        return False

    lines = content.splitlines(keepends=True)
    shebang_idx = 0

    while shebang_idx < len(lines) and (
        lines[shebang_idx].startswith("#!") or "coding:" in lines[shebang_idx]
    ):
        shebang_idx += 1

    prefix = "".join(lines[:shebang_idx])
    suffix = "".join(lines[shebang_idx:])

    new_content = prefix + (PY_HEADER if not prefix.endswith("\n\n") else PY_HEADER.lstrip()) + suffix.lstrip("\n")
    path.write_text(new_content, encoding="utf-8")
    return True


def apply_header_to_cpp(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return False

    if "Copyright (c) 2024-2026, Taha Majlesi" in content:
        return False

    new_content = CPP_HEADER + content.lstrip("\n")
    path.write_text(new_content, encoding="utf-8")
    return True


def main() -> None:
    root = Path(".")
    target_dirs = ["neuralops", "scripts", "benchmarks", "evaluation", "tests"]
    py_count = 0
    cpp_count = 0

    for d in target_dirs:
        dir_path = root / d
        if not dir_path.is_dir():
            continue

        for p in sorted(dir_path.rglob("*")):
            if any(part in p.parts for part in EXCLUDE_PARTS) or not p.is_file():
                continue

            if p.suffix == ".py":
                if apply_header_to_py(p):
                    py_count += 1
            elif p.suffix in {".cu", ".cuh", ".cpp", ".h"}:
                if apply_header_to_cpp(p):
                    cpp_count += 1

    print(f"Applied NeuralOps copyright headers to {py_count} Python files and {cpp_count} C++/CUDA files.")


if __name__ == "__main__":
    main()
