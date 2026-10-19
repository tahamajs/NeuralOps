"""Utility to remove redundant 'Author: Taha Majlesi' lines across the codebase.

The repository now uniformly uses the standard top-level MIT license header:
# Copyright (c) 2024-2026, Taha Majlesi
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EXCLUDE_PARTS = {".git", "__pycache__", "build", "dist", ".venv", "venv", "archive", "_archive", ".ipynb_checkpoints"}


def clean_content(text: str) -> str:
    text = re.sub(r"(?m)^[ \t]*(?://|#|<!--)\s*Author:\s*Taha Majlesi\s*(?:-->)?\r?\n?", "", text)
    text = re.sub(r"(?m)^[ \t]*\*\s*Author:\s*Taha Majlesi\r?\n?", "", text)
    text = re.sub(r"(?m)^[ \t]*Author:\s*Taha Majlesi\r?\n?", "", text)
    text = re.sub(r"\n\s*\n(\s*\"\"\"|\s*\'\'\')", r"\n\1", text)
    return text


def main() -> None:
    root = Path(".")
    target_dirs = ["neuralops", "scripts", "benchmarks", "evaluation", "tests", "docs", "paper", "thesis_final", "academic"]
    modified = 0

    root_files = [p for p in root.glob("*") if p.is_file() and p.suffix in {".py", ".md", ".sh", ".toml", ".yml", ".yaml"}]

    all_files = list(root_files)
    for d in target_dirs:
        dir_path = root / d
        if not dir_path.is_dir():
            continue
        for p in dir_path.rglob("*"):
            if any(part in p.parts for part in EXCLUDE_PARTS) or not p.is_file():
                continue
            if p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".md", ".sh"}:
                all_files.append(p)

    for p in sorted(set(all_files)):
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue

        new_content = clean_content(content)
        if new_content != content:
            p.write_text(new_content, encoding="utf-8")
            modified += 1

    nb_cleaned = 0
    for p in root.glob("notebooks/*.ipynb"):
        text = p.read_text(encoding="utf-8")
        if "Author: Taha Majlesi" in text:
            data = json.loads(text)
            for cell in data.get("cells", []):
                if cell.get("cell_type") in ("markdown", "code"):
                    src = cell.get("source", [])
                    if isinstance(src, list):
                        new_src = [line for line in src if "Author: Taha Majlesi" not in line]
                        cell["source"] = new_src
                    elif isinstance(src, str):
                        cell["source"] = re.sub(r"Author:\s*Taha Majlesi\n?", "", src)
            p.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            nb_cleaned += 1

    print(f"Cleaned redundant Author lines from {modified} source files and {nb_cleaned} notebooks.")


if __name__ == "__main__":
    main()
