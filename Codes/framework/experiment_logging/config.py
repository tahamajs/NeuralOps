"""Logging configuration dataclass."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoggingConfig:
    """Complete logging & experiment tracking configuration."""

    log_level: str = "INFO"
    log_format: str = "structured"  # "structured" (JSON lines) or "human"
    log_to_console: bool = True
    log_to_file: bool = True
    log_dir: str = "./logs"
    experiment_name: str = "experiment"
    run_id: str | None = None
    use_timestamp: bool = True
    # Rank-specific and thread-safe logging
    rank: int | None = None  # FIX: Local rank for DDP isolated file logging
    rank_specific_logs: bool = True  # FIX: Enforce isolated rank_{i}.log files per process
    use_queue_handler: bool = True  # FIX: Enforce thread-safe asynchronous logging via QueueHandler
    # Metrics tracking
    metrics_enabled: bool = True
    metrics_flush_every: int = 10  # flush metrics file every N logs
    # Extra metadata added to every event
    extra_metadata: dict[str, Any] = field(default_factory=dict)

