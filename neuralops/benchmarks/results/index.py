"""High-level query API for querying, filtering, and comparing benchmark results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from neuralops.benchmarks.results.schema import BenchmarkResult

LOG = logging.getLogger(__name__)


class ResultsIndex:
    """Query engine for exploring and analyzing benchmark runs."""

    def __init__(self, output_dir: Path | str = "results") -> None:
        self.output_dir = Path(output_dir)
        self.jsonl_file = self.output_dir / "results.jsonl"
        self._cached_results: Optional[List[BenchmarkResult]] = None

    def _load_results(self, force_reload: bool = False) -> List[BenchmarkResult]:
        """Lazy loader for all individual JSON benchmark results."""
        if self._cached_results is not None and not force_reload:
            return self._cached_results

        results: List[BenchmarkResult] = []
        for path in self.output_dir.rglob("*.json"):
            if path.name in ("run_index.json", "run_state.json", "package.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if "module" in data and ("status" in data or "run_metadata" in data):
                    results.append(BenchmarkResult.from_dict(data))
            except Exception:
                continue

        self._cached_results = results
        return self._cached_results

    def find(
        self,
        module: Optional[str] = None,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        rank: Optional[int] = None,
        status: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """Query benchmark results with multi-criteria filtering."""
        records = self._load_results()
        filtered = []
        for r in records:
            if module and (r.module.get("name") if isinstance(r.module, dict) else r.module) != module:
                continue
            if model and (r.model.get("name") if isinstance(r.model, dict) else r.model) != model:
                continue
            if dataset and (r.dataset.get("name") if isinstance(r.dataset, dict) else r.dataset) != dataset:
                continue
            if rank is not None and r.config.get("rank") != rank:
                continue
            if status and r.status != status:
                continue
            filtered.append(r)
        return filtered

    def best(self, model: str, dataset: str, metric: str = "accuracy") -> Optional[BenchmarkResult]:
        """Find the top-performing benchmark result for a model/dataset pair."""
        candidates = self.find(model=model, dataset=dataset, status="success")
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda r: float((r.metrics or {}).get(metric, 0.0) or 0.0),
        )

    def compare(
        self,
        modules: List[str],
        model: str,
        dataset: str,
    ) -> List[BenchmarkResult]:
        """Compare specific modules on identical model and dataset configuration."""
        matched = []
        for mod in modules:
            runs = self.find(module=mod, model=model, dataset=dataset)
            if runs:
                matched.append(runs[-1])
        return matched

    def failures(self) -> List[BenchmarkResult]:
        """Return all failed runs."""
        return [r for r in self._load_results() if r.status != "success"]

    def to_dataframe(self) -> pd.DataFrame:
        """Export indexed benchmark runs to Pandas DataFrame."""
        rows = []
        for r in self._load_results():
            row: Dict[str, Any] = {
                "run_id": r.run_id,
                "module": r.module.get("name") if isinstance(r.module, dict) else r.module,
                "module_type": r.module.get("type") if isinstance(r.module, dict) else "",
                "model": r.model.get("name") if isinstance(r.model, dict) else r.model,
                "dataset": r.dataset.get("name") if isinstance(r.dataset, dict) else r.dataset,
                "rank": r.config.get("rank"),
                "status": r.status,
                "duration_s": (r.resources or {}).get("duration_s"),
                "peak_vram_mb": (r.resources or {}).get("peak_vram_mb"),
            }
            if r.metrics:
                for k, v in r.metrics.items():
                    row[f"metric_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)


__all__ = ["ResultsIndex"]
