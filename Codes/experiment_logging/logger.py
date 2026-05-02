"""
Professional logging setup: colored console, plain text file, JSON structured logs.
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Optional

# Try importing colorlog for colored console output
try:
    import colorlog
    _has_colorlog = True
except ImportError:
    _has_colorlog = False


class ColoredFormatter(logging.Formatter):
    """Formatter with colored log levels."""
    def __init__(self, use_colors=True):
        super().__init__()
        self.use_colors = use_colors and _has_colorlog
        if self.use_colors:
            self.color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                }
            )
        else:
            self.fallback_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S"
            )

    def format(self, record):
        if self.use_colors:
            return self.color_formatter.format(record)
        return self.fallback_formatter.format(record)


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for file output."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        # Include extra fields if present
        if hasattr(record, "metrics"):
            log_entry["metrics"] = record.metrics
        if hasattr(record, "experiment_id"):
            log_entry["experiment_id"] = record.experiment_id
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    log_dir: str = "logs",
    experiment_name: str = "experiment",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    json_log: bool = True,
    colored_console: bool = True,
) -> logging.Logger:
    """
    Set up professional logging with console, text file, and JSON outputs.

    Returns:
        logging.Logger instance (you can use as `logger.info(...)`)
    """
    os.makedirs(log_dir, exist_ok=True)

    # Root logger for the experiment
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # remove any previous handlers

    # ---- Console Handler (colored if possible) ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(ColoredFormatter(use_colors=colored_console))
    logger.addHandler(console_handler)

    # ---- Plain Text File Handler ----
    txt_log_path = os.path.join(log_dir, f"{experiment_name}.log")
    file_handler = logging.FileHandler(txt_log_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    # ---- JSON Structured Log Handler ----
    if json_log:
        json_log_path = os.path.join(log_dir, f"{experiment_name}.jsonl")
        json_handler = logging.FileHandler(json_log_path, encoding="utf-8")
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

    logger.info(f"🚀 Logging initialized – logs: {log_dir}")
    return logger