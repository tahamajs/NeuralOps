"""BenchmarkResult and metadata dataclasses matching canonical schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import subprocess
from typing import Any
import uuid


def _get_git_commit() -> str:
    """Return the current HEAD git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _is_git_dirty() -> bool:
    """Check if git working tree has uncommitted changes."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


@dataclass
class ModuleMetadata:
    """Metadata describing candidate attention or compression module."""

    name: str
    type: str  # "attention" | "compressor"
    family: str = "unknown"
    supports_causal: bool = False
    supports_gqa: bool = False
    supports_rope: bool = False
    requires_cuda: bool = False
    timeout_s: int = 1800


@dataclass
class ModelMetadata:
    """Metadata describing evaluated target model architecture."""

    name: str
    family: str = "unknown"
    num_params: int = 0
    num_params_after: int | None = None
    hidden_size: int = 0
    num_heads: int = 0
    num_layers: int = 0


@dataclass
class DatasetMetadata:
    """Metadata describing evaluation dataset task."""

    name: str
    task: str = "classification"
    num_labels: int = 2
    num_train: int = 0
    num_eval: int = 0


@dataclass
class BenchmarkResult:
    """Canonical schema for every single benchmark evaluation run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_name: str = "benchmark_run"
    timestamp_start: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    timestamp_end: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: str = field(default_factory=_get_git_commit)
    git_dirty: bool = field(default_factory=_is_git_dirty)

    module: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    metrics: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None

    status: str = "success"  # "success" | "failed" | "skipped"
    error: str | None = None
    traceback: str | None = None
    warnings: list[str] = field(default_factory=list)

    timeout_s: int = 1800
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert result dataclass to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Construct BenchmarkResult from dictionary."""
        return cls(**data)


__all__ = [
    "ModuleMetadata",
    "ModelMetadata",
    "DatasetMetadata",
    "BenchmarkResult",
]
