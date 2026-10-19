# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Utility to remove inline and mid-code comments across the repository, keeping standard top-level license headers, docstrings, and static analysis directives.

Author: Taha Majlesi
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

EXCLUDE_PARTS = {".git", "__pycache__", "build", "dist", ".venv", "venv", "archive", "_archive", ".ipynb_checkpoints"}
PRESERVED_DIRECTIVES = ("type: ignore", "noqa", "pragma:", "fmt:", "pylint:", "mypy:")


def strip_py_comments(source: str) -> str:
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline))
    except Exception:
        return source

    lines = source.splitlines(keepends=True)

    copyright_lines = set()
    in_header = False
    for idx, l in enumerate(lines[:15]):
        if idx == 0 and l.startswith("#!"):
            copyright_lines.add(1)
            continue
        if "Copyright" in l:
            in_header = True
        if in_header:
            if l.startswith("#") or not l.strip():
                copyright_lines.add(idx + 1)
            else:
                break

    new_lines = []
    for line_idx, line in enumerate(lines, start=1):
        if line_idx in copyright_lines:
            new_lines.append(line)
            continue

        stripped = line.strip()
        if stripped.startswith("#"):
            if any(k in stripped for k in PRESERVED_DIRECTIVES):
                new_lines.append(line)
            continue

        line_comments = [t for t in tokens if t.type == tokenize.COMMENT and t.start[0] == line_idx]
        if line_comments:
            c_tok = line_comments[0]
            if any(k in c_tok.string for k in PRESERVED_DIRECTIVES):
                new_lines.append(line)
            else:
                col = c_tok.start[1]
                code_part = line[:col].rstrip() + "\n"
                new_lines.append(code_part)
        else:
            new_lines.append(line)

    return "".join(new_lines)


def strip_cpp_comments(source: str) -> str:
    lines = source.splitlines(keepends=True)
    copyright_lines = set()
    in_header = False
    for idx, l in enumerate(lines[:15]):
        if "Copyright" in l:
            in_header = True
        if in_header:
            if l.startswith("//") or not l.strip():
                copyright_lines.add(idx + 1)
            else:
                break

    new_lines = []
    for idx, line in enumerate(lines, start=1):
        if idx in copyright_lines:
            new_lines.append(line)
            continue

        s = line.strip()
        if s.startswith("//"):
            continue

        if "//" in line:
            parts = line.split("//")
            q_count = parts[0].count('"')
            if q_count % 2 == 0:
                new_lines.append(parts[0].rstrip() + "\n")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "".join(new_lines)


def main() -> None:
    root = Path(".")
    target_dirs = ["neuralops", "scripts", "benchmarks", "evaluation", "tests"]
    py_modified = 0
    cpp_modified = 0

    for d in target_dirs:
        dir_path = root / d
        if not dir_path.is_dir():
            continue

        for p in sorted(dir_path.rglob("*")):
            if any(part in p.parts for part in EXCLUDE_PARTS) or not p.is_file():
                continue

            try:
                src = p.read_text(encoding="utf-8")
            except Exception:
                continue

            if p.suffix == ".py":
                cleaned = strip_py_comments(src)
                if cleaned != src:
                    import ast

                    try:
                        ast.parse(cleaned, filename=str(p))
                        p.write_text(cleaned, encoding="utf-8")
                        py_modified += 1
                    except Exception as exc:
                        print(f"Skipping {p} due to parse error after stripping: {exc}")
            elif p.suffix in {".cu", ".cuh", ".cpp", ".h"}:
                cleaned = strip_cpp_comments(src)
                if cleaned != src:
                    p.write_text(cleaned, encoding="utf-8")
                    cpp_modified += 1

    print(f"Successfully stripped mid-code comments from {py_modified} Python files and {cpp_modified} C++/CUDA files.")


if __name__ == "__main__":
    main()
