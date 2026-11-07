#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Phase 1: Structured Telemetry and Benchmark Logging Engine.

Writes high-precision, timestamped JSONL audit trails with execution phase,
hardware context, and status codes for production regression tracking.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BenchmarkLogger:
    """Thread-safe structured JSONL benchmark telemetry logger."""

    def __init__(self, run_id: str, log_dir: str | Path = "results/logs") -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"benchmark_{self.run_id}.jsonl"
        self._lock = threading.Lock()

    def log(
        self,
        phase: str,
        module: str,
        status: str,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Emit a structured log entry."""
        entry: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "module": module,
            "status": status,
        }
        if metrics:
            entry["metrics"] = metrics
        if error:
            entry["error"] = error

        line = json.dumps(entry)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return entry

    def info(self, phase: str, message: str) -> None:
        """Log human-readable informational message to stderr."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        sys.stderr.write(f"[{ts}] [{phase}] {message}\n")
        sys.stderr.flush()
