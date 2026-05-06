"""
MetricsTracker: collect, save, and load training metrics (loss, accuracy, etc.).
Supports JSON and CSV formats.
"""

import json
import csv
import os
from typing import Any, Dict, List, Optional
import logging

class MetricsTracker:
    """
    Simple step‑wise metrics recorder.

    Usage:
        tracker = MetricsTracker("metrics.json")
        tracker.log({"loss": 2.3, "lr": 1e-4}, step=100)
        tracker.log({"loss": 2.1}, step=200)
        tracker.save()        # flushes to disk
        tracker.to_csv("metrics.csv")
    """
    def __init__(self, filepath: str, logger: Optional[logging.Logger] = None):
        self.filepath = filepath
        self.records: List[Dict[str, Any]] = []
        self.logger = logger or logging.getLogger(__name__)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, flush: bool = False):
        """Record a metrics dict, optionally associated with a step."""
        entry = {"step": step} if step is not None else {}
        entry.update(metrics)
        entry["timestamp"] = _now_iso()
        self.records.append(entry)
        if flush:
            self._flush()

    def _flush(self):
        """Append to JSON file."""
        if self.filepath:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "a", encoding="utf-8") as f:
                for rec in self.records:
                    f.write(json.dumps(rec) + "\n")
            self.records.clear()

    def save(self, filepath: Optional[str] = None):
        """Save all accumulated records to a file (default: self.filepath)."""
        target = filepath or self.filepath
        self._flush()
        if self.logger:
            self.logger.info(f"Metrics saved to {target}")

    def to_csv(self, csv_path: str):
        """Export all records (from file or memory) to a CSV file."""
        all_records = self._load_all()
        if not all_records:
            return
        # Collect all possible column names
        columns = set()
        for rec in all_records:
            columns.update(rec.keys())
        columns = sorted(columns)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_records)
        if self.logger:
            self.logger.info(f"Metrics CSV exported to {csv_path}")

    def _load_all(self) -> List[Dict[str, Any]]:
        """Load all records from the JSON file (if any)."""
        records = []
        if self.filepath and os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        # Also include any unsaved in‑memory records
        records.extend(self.records)
        return records

    def get_best(self, metric: str, mode: str = "min") -> Dict[str, Any]:
        """Return the best record according to a metric ('min' or 'max')."""
        all_records = self._load_all()
        if not all_records:
            return {}
        best = all_records[0]
        for rec in all_records[1:]:
            if mode == "min" and rec.get(metric, float("inf")) < best.get(metric, float("inf")):
                best = rec
            elif mode == "max" and rec.get(metric, float("-inf")) > best.get(metric, float("-inf")):
                best = rec
        return best

    def clear(self):
        """Clear in‑memory records (does not delete file)."""
        self.records.clear()


def _now_iso() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()