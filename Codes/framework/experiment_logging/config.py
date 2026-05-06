from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class LoggingConfig:
    """Complete logging & experiment tracking configuration."""
    log_level: str = "INFO"
    log_format: str = "structured"       # "structured" (JSON lines), "human"
    log_to_console: bool = True
    log_to_file: bool = True
    log_dir: str = "./logs"
    experiment_name: str = "experiment"
    run_id: Optional[str] = None
    use_timestamp: bool = True
    # Metrics tracking
    metrics_enabled: bool = True
    metrics_flush_every: int = 10         # flush metrics file every N logs
    # Extra metadata added to every event
    extra_metadata: Dict[str, Any] = field(default_factory=dict)