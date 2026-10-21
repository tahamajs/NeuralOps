#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Experiment Metrics Tracker

Tracks metrics in memory and serializes run histories to JSON format.
"""

import json
from pathlib import Path
from typing import Any


class MetricsTracker:
    """Collects metrics during training and saves them as JSON."""

    def __init__(self, save_path: str | None = None) -> None:
        self.save_path = save_path
        self.records: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Append one metric record to the in-memory run history."""
        entry = dict(metrics)
        if step is not None:
            entry["step"] = step
        self.records.append(entry)

    def to_dataframe(self) -> Any:
        """Return the collected records as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.records)

    def save(self, path: str | None = None) -> None:
        """Write records as JSON Lines for ``.jsonl`` paths, otherwise JSON."""
        p = path or self.save_path
        if p:
            output_path = Path(p)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as output_file:
                if output_path.suffix.lower() == ".jsonl":
                    for record in self.records:
                        output_file.write(json.dumps(record, default=str) + "\n")
                else:
                    json.dump(self.records, output_file, indent=2, ensure_ascii=False, default=str)

    def flush(self) -> None:
        """Persist the in-memory history when a save path was configured."""
        if self.save_path:
            self.save(self.save_path)
