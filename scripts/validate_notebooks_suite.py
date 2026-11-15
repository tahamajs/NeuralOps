#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# AST and JSON integrity scanner for all 94 Jupyter notebooks in notebooks/.

import ast
import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
OUTPUT_REPORT = Path("audits/notebooks/ast_validation_results.json")
OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

notebook_files = sorted(list(NOTEBOOKS_DIR.glob("*.ipynb")))
print(f"Validating {len(notebook_files)} notebooks...")

results = {}
total_cells = 0
total_syntax_errors = 0

for nb_path in notebook_files:
    nb_name = nb_path.name
    try:
        data = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        results[nb_name] = {"valid_json": False, "error": f"JSONDecodeError: {exc}"}
        continue

    cells = data.get("cells", [])
    nb_cell_count = len(cells)
    total_cells += nb_cell_count
    cell_errors = []

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # Strip IPython magics (% and !) before AST validation
            clean_lines = []
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("%") or stripped.startswith("!"):
                    clean_lines.append(f"# {line}")
                else:
                    clean_lines.append(line)
            clean_source = "\n".join(clean_lines)

            try:
                ast.parse(clean_source)
            except SyntaxError as exc:
                cell_errors.append({
                    "cell_index": idx,
                    "line": exc.lineno,
                    "error": str(exc.msg)
                })

    if cell_errors:
        total_syntax_errors += len(cell_errors)
        results[nb_name] = {
            "valid_json": True,
            "cell_count": nb_cell_count,
            "syntax_errors": cell_errors
        }
    else:
        results[nb_name] = {
            "valid_json": True,
            "cell_count": nb_cell_count,
            "syntax_errors": []
        }

OUTPUT_REPORT.write_text(json.dumps(results, indent=2))
print(f"Validated {len(notebook_files)} notebooks ({total_cells} cells).")
print(f"Total syntax errors detected: {total_syntax_errors}")
print(f"Report written to {OUTPUT_REPORT}")
