#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation Report Generator

Generates structured markdown and JSON summary reports from validation suites.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates Markdown and JSON reports from validation suite executions."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.entries: List[Dict[str, Any]] = []

    def add(
        self,
        category: str,
        name: str,
        status: str,
        value: Optional[Any] = None,
        notes: str = "",
    ) -> None:
        """Add a validation outcome record to the report."""
        self.entries.append(
            {
                "category": category,
                "name": name,
                "status": status,
                "value": value,
                "notes": notes,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def save(self, filename: str) -> str:
        """Save report entries to both Markdown and JSON files."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Validation Report\n\nGenerated: {datetime.now()}\n\n")
            f.write("| Category | Name | Status | Value | Notes |\n")
            f.write("|----------|------|--------|-------|-------|\n")
            for e in self.entries:
                val_str = str(e["value"]) if e["value"] is not None else ""
                f.write(f"| {e['category']} | {e['name']} | {e['status']} | {val_str} | {e['notes']} |\n")

        json_path = f"{os.path.splitext(path)[0]}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)

        logger.info("Report saved to %s and %s", path, json_path)
        return path


__all__ = ["ReportGenerator"]
