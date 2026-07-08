"""Local JSON persistence for drift reports."""

from pathlib import Path
from typing import Optional

from ..report import DriftReport


class LocalDriftBackend:
    def __init__(self, base_dir: str = "./drift_reports"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, report: DriftReport) -> str:
        path = self.base_dir / f"{run_id}_drift.json"
        report.save(str(path))
        return str(path)

    def load(self, run_id: str) -> Optional[DriftReport]:
        path = self.base_dir / f"{run_id}_drift.json"
        if path.exists():
            return DriftReport.load(str(path))
        return None
