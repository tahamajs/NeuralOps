"""Simple in‑memory metrics tracker with JSON serialisation."""

import json
import os
from typing import Any


class MetricsTracker:
    """Collects metrics during training and saves them as JSON."""

    def __init__(self, save_path: str | None = None) -> None:
        self.save_path = save_path
        self.records: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int = None) -> None:
        """Log."""
        entry = dict(metrics)
        if step is not None:
            entry["step"] = step
        self.records.append(entry)

    def to_dataframe(self) -> Any:
        """To dataframe."""
        import pandas as pd

        return pd.DataFrame(self.records)

    def save(self, path: str | None = None) -> None:
        """Save."""
        p = path or self.save_path
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(self.records, f, indent=2)

    def flush(self) -> None:
        """Flush."""
        if self.save_path:
            self.save(self.save_path)
