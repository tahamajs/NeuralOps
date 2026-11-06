#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Query operations for Model Zoo and Notebook demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuralops.ui.queries.ledger import _get_connection


def list_notebooks() -> list[dict[str, Any]]:
    """Scan and list all 94 canonical demonstration notebooks in notebooks/."""
    repo_root = Path(__file__).resolve().parents[3]
    notebooks_dir = repo_root / "notebooks"
    if not notebooks_dir.is_dir():
        return []

    notebook_paths = sorted([p for p in notebooks_dir.glob("*.ipynb")])
    notebooks: list[dict[str, Any]] = []

    for p in notebook_paths:
        name = p.name
        size_kb = round(p.stat().st_size / 1024, 1)

        # Extract title and description from first markdown cell
        title = p.stem.replace("_", " ").title()
        description = ""
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                cells = data.get("cells", [])
                for cell in cells:
                    if cell.get("cell_type") == "markdown":
                        src_lines = cell.get("source", [])
                        text = "".join(src_lines).strip()
                        # Extract first meaningful heading/line
                        for line in text.split("\n"):
                            clean_line = line.strip().lstrip("#").strip()
                            if clean_line:
                                if not description:
                                    description = clean_line
                                elif len(description) < 40:
                                    description += " - " + clean_line
                                break
                        if description:
                            break
        except Exception:
            description = "NeuralOps Demonstration Tutorial"

        if not description:
            description = "NeuralOps Tutorial Notebook"

        notebooks.append(
            {
                "filename": name,
                "title": title,
                "description": description,
                "size_kb": size_kb,
                "path": str(p.relative_to(repo_root)),
            }
        )

    return notebooks


def list_model_zoo() -> list[dict[str, Any]]:
    """Retrieve distinct model architectures, backbones, and calibration summaries from master ledger."""
    query = """
        SELECT backbone, dataset,
               COUNT(*) as evaluation_count,
               AVG(accuracy) as mean_accuracy,
               MIN(latency_triton_ms) as min_latency_ms,
               MAX(speedup_x) as max_speedup,
               AVG(peak_vram_mb) as mean_vram_mb
        FROM results
        WHERE backbone IS NOT NULL AND backbone != ''
        GROUP BY backbone, dataset
        ORDER BY evaluation_count DESC
    """
    models: list[dict[str, Any]] = []
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        for r in rows:
            backbone = r["backbone"]
            # Clean display name
            display_name = backbone.split("\\")[-1].split("/")[-1]
            dataset = r["dataset"] or "general"
            eval_count = r["evaluation_count"]
            mean_acc = round(float(r["mean_accuracy"]), 2) if r["mean_accuracy"] is not None else None
            min_lat = round(float(r["min_latency_ms"]), 3) if r["min_latency_ms"] is not None else None
            max_spd = round(float(r["max_speedup"]), 2) if r["max_speedup"] is not None else None
            vram = round(float(r["mean_vram_mb"]), 1) if r["mean_vram_mb"] is not None else None

            models.append(
                {
                    "backbone": backbone,
                    "name": display_name,
                    "dataset": dataset,
                    "evaluation_count": eval_count,
                    "mean_accuracy": mean_acc,
                    "min_latency_ms": min_lat,
                    "max_speedup": max_spd,
                    "mean_vram_mb": vram,
                }
            )
    finally:
        conn.close()

    return models
