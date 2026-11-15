"""
Canonical record definition and BenchmarkStatus enumeration for Results v2.
Follows AGENTS.md Section 4 rules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from neuralops.core.exceptions import NeuralOpsError


class LedgerError(NeuralOpsError):
    """Raised when an operation on the benchmark ledger fails."""


class ChecksumMismatchError(LedgerError, ValueError):
    """Raised when record checksum verification fails."""


class BenchmarkStatus(str, Enum):
    """Canonical benchmark execution status values."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    HARDWARE_ERROR = "HARDWARE_ERROR"
    TIMEOUT = "TIMEOUT"
    OOM = "OOM"


@dataclass(frozen=True)
class CanonicalRecord:
    """Canonical benchmark result record for all NeuralOps benchmarks."""

    run_id: str
    timestamp: str  # ISO 8601 UTC
    git_commit: str
    git_dirty: bool
    suite: str
    module: str
    model: str
    dataset: str
    device: str
    status: str  # from BenchmarkStatus enum only
    metrics: dict[str, float]
    config: dict[str, Any]
    telemetry: dict[str, Any]
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanonicalRecord:
        """Construct record from dictionary, parsing embedded JSON fields if necessary."""
        clean_data = dict(data)
        if isinstance(clean_data.get("git_dirty"), int):
            clean_data["git_dirty"] = bool(clean_data["git_dirty"])
        if isinstance(clean_data.get("metrics"), str):
            clean_data["metrics"] = json.loads(clean_data["metrics"])
        if isinstance(clean_data.get("config"), str):
            clean_data["config"] = json.loads(clean_data["config"])
        if isinstance(clean_data.get("telemetry"), str):
            clean_data["telemetry"] = json.loads(clean_data["telemetry"])
        return cls(**clean_data)


__all__ = [
    "BenchmarkStatus",
    "CanonicalRecord",
    "ChecksumMismatchError",
    "LedgerError",
]
