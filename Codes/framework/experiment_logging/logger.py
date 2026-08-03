"""Structured logger setup with JSON and human‑readable formats."""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from .config import LoggingConfig


class StructuredFormatter(logging.Formatter):
    """JSON‑line formatter for structured logging."""

    def format(self, record) -> Any:
        """Format."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            entry.update(record.extra_data)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(config: LoggingConfig = None) -> logging.Logger:
    """Create a logger configured according to the given config."""
    cfg = config or LoggingConfig()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    os.makedirs(cfg.log_dir, exist_ok=True)

    if cfg.log_format == "structured":
        fmt = StructuredFormatter()
    else:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if cfg.log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if cfg.log_to_file:
        fname = f"{cfg.experiment_name}_{cfg.run_id or ''}.log".rstrip("_").lstrip("_")
        fh = RotatingFileHandler(
            os.path.join(cfg.log_dir, fname), maxBytes=5 * 1024 * 1024, backupCount=3
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    logger = logging.getLogger("framework")
    logger.info(
        "Logging initialised. Experiment: %s, Run: %s", cfg.experiment_name, cfg.run_id
    )
    return logger


def get_logger(name: str = "framework") -> logging.Logger:
    """Return a logger for the given name."""
    return logging.getLogger(name)
