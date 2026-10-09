"""Intelligent run deduplication and O(1) index management."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from neuralops.benchmarks.results.schema import BenchmarkResult

LOG = logging.getLogger(__name__)


class Deduplicator:
    """Manages O(1) run indexing and deduplication checks."""

    def __init__(self, output_dir: Path | str = "results", rerun_after_days: int = 30) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.output_dir / "run_index.json"
        self.rerun_after_days = rerun_after_days
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception as e:
                LOG.warning("Could not load run_index.json, starting fresh: %s", e)
        return {}

    def save_index(self) -> None:
        """Persist index to disk."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as fp:
                json.dump(self._index, fp, indent=2)
        except Exception as e:
            LOG.warning("Could not persist run_index.json: %s", e)

    def should_skip(self, run_key: str, force_rerun: bool = False) -> tuple[bool, Optional[str]]:
        """
        Determine if run should be skipped based on duplication status.
        Returns: (should_skip, reason)
        """
        if force_rerun:
            return False, "force_rerun=True overrides deduplication"

        if run_key not in self._index:
            return False, "no existing run with matching key"

        entry = self._index[run_key]
        status = entry.get("status")

        if status != "success":
            return False, f"previous run failed ({status}), allowing retry"

        # Check timestamp age
        ts_str = entry.get("timestamp")
        if ts_str and self.rerun_after_days > 0:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
                if age_days > self.rerun_after_days:
                    return False, f"existing run is {age_days:.1f} days old (> {self.rerun_after_days}d), allowing rerun"
            except Exception:
                pass

        return True, f"Identical successful run already exists ({entry.get('run_id')})"

    def record_run(self, result: BenchmarkResult, result_path: Path | str) -> None:
        """Register or update completed benchmark run in O(1) deduplication index."""
        run_key = result.compute_run_key()
        self._index[run_key] = {
            "run_id": result.run_id,
            "timestamp": result.timestamp_end or result.timestamp_start,
            "status": result.status,
            "module": result.module.get("name", "") if isinstance(result.module, dict) else str(result.module),
            "model": result.model.get("name", "") if isinstance(result.model, dict) else str(result.model),
            "dataset": result.dataset.get("name", "") if isinstance(result.dataset, dict) else str(result.dataset),
            "rank": result.config.get("rank", 0) if isinstance(result.config, dict) else 0,
            "accuracy": (result.metrics or {}).get("accuracy"),
            "result_path": str(result_path),
        }
        self.save_index()


__all__ = ["Deduplicator"]
