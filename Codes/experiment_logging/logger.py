"""
Professional logging setup: colored console, plain text file, JSON structured logs.
Automatically creates a unique run directory based on experiment name and timestamp.
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

from .config import LoggingConfig

# Try importing colorlog for colored console output
try:
    import colorlog
    _has_colorlog = True
except ImportError:
    _has_colorlog = False


class ColoredFormatter(logging.Formatter):
    """Formatter with colored log levels (if colorlog is available)."""
    def __init__(self, use_colors: bool = True):
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
        # Include extra fields if present (e.g., from adapter)
        for attr in ["metrics", "experiment_id", "step"]:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        return json.dumps(log_entry, ensure_ascii=False)


def _build_run_dir(cfg: LoggingConfig) -> str:
    """Create and return the unique run directory path."""
    os.makedirs(cfg.log_base_dir, exist_ok=True)

    # Build unique run name
    if cfg.run_id:
        run_name = cfg.run_id
    elif cfg.use_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.experiment_name}_{timestamp}"
    else:
        run_name = cfg.experiment_name

    run_dir = os.path.join(cfg.log_base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logging(cfg: LoggingConfig) -> logging.Logger:
    """
    Set up professional logging with console, text file, and JSON outputs.
    Returns a logger for the root of the experiment (cfg.experiment_name).

    Example:
        cfg = LoggingConfig(experiment_name="mrpc_aurora")
        logger = setup_logging(cfg)
        logger.info("Training started")
    """
    run_dir = _build_run_dir(cfg)
    logger = logging.getLogger(cfg.experiment_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # remove any previous handlers

    # ---- Console Handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, cfg.console_level.upper(), logging.INFO))
    console_handler.setFormatter(ColoredFormatter(use_colors=cfg.colored_console))
    logger.addHandler(console_handler)

    # ---- Plain Text File Handler ----
    txt_log_path = os.path.join(run_dir, cfg.log_filename)
    if cfg.max_log_size_mb > 0:
        file_handler = logging.handlers.RotatingFileHandler(
            txt_log_path,
            maxBytes=cfg.max_log_size_mb * 1024 * 1024,
            backupCount=cfg.backup_count,
            encoding="utf-8"
        )
    else:
        file_handler = logging.FileHandler(txt_log_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, cfg.file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    # ---- JSON Structured Log Handler ----
    if cfg.json_log:
        json_log_path = os.path.join(run_dir, cfg.json_filename)
        json_handler = logging.FileHandler(json_log_path, encoding="utf-8")
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

    logger.info(f"🚀 Logging initialized. Run directory: {run_dir}")
    # Store run_dir in logger for later access (via logging.getLogger(...).run_dir)
    logger.run_dir = run_dir
    return logger


def get_logger(name: str, cfg: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Return a child logger. If cfg is provided and the root logger hasn't been set up,
    it will set up logging first. Usually you call setup_logging once at the start,
    then use get_logger for submodules.
    """
    logger = logging.getLogger(name)
    if cfg is not None and not logger.hasHandlers():
        setup_logging(cfg)
    return logger