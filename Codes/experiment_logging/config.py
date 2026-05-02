"""Logging configuration dataclass."""
from dataclasses import dataclass

@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    experiment_name: str = "experiment"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    json_log: bool = True
    colored_console: bool = True