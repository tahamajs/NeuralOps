"""Logging configuration dataclass."""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LoggingConfig:
    # Directory that will contain all run subdirectories
    log_base_dir: str = "logs"
    # Experiment name (used as subfolder prefix)
    experiment_name: str = "experiment"
    # If True, append current timestamp to experiment_name for a unique run dir
    use_timestamp: bool = True
    # Optional explicit run id – if provided, used instead of timestamp
    run_id: Optional[str] = None

    # Console output
    console_level: str = "INFO"
    colored_console: bool = True

    # Plain text file (inside the run directory)
    file_level: str = "DEBUG"
    log_filename: str = "run.log"

    # Structured JSON log file (one JSON object per line)
    json_log: bool = True
    json_filename: str = "run.jsonl"

    # Metrics tracking
    track_metrics: bool = True
    metrics_filename: str = "metrics.json"   # or "metrics.csv" if CSV output

    # Rotating logs? Not per‑run; we use timestamped directories, so no need.
    # Max size for text log before rotation (0 = no rotation)
    max_log_size_mb: int = 0
    backup_count: int = 5