"""Core logging utilities: Colored console, file, JSON structured logs."""
import logging
import logging.handlers
import json
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

# Try importing colorlog for colored console output
try:
    import colorlog
    _has_colorlog = True
except ImportError:
    _has_colorlog = False

class ColoredFormatter(logging.Formatter):
    """Formatter with colored log levels."""
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and _has_colorlog
        if self.use_colors:
            self.formatter = colorlog.ColoredFormatter(
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
            self.formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S"
            )

    def format(self, record):
        if self.use_colors:
            return self.formatter.format(record)
        return super().format(record)

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
        if hasattr(record, "metrics"):
            log_entry["metrics"] = record.metrics
        if hasattr(record, "experiment_id"):
            log_entry["experiment_id"] = record.experiment_id
        return json.dumps(log_entry, ensure_ascii=False)

class ExperimentLogger(logging.Logger):
    """Logger that can attach metrics and experiment context."""
    def __init__(self, name: str = "experiment", level=logging.INFO):
        super().__init__(name, level)
        self.metrics = {}
        self.experiment_id = None

    def log_metric(self, key: str, value: float, step: int = None):
        """Log a scalar metric."""
        self.metrics[key] = value
        extra = {"metrics": {key: value}}
        if step is not None:
            extra["step"] = step
        self.info(f"📊 {key}: {value:.4f}", extra=extra)

    def set_experiment_id(self, exp_id: str):
        self.experiment_id = exp_id

    def info(self, msg, extra=None, **kwargs):
        if extra:
            record = logging.LogRecord(
                self.name, logging.INFO, None, None, msg, (), None
            )
            record.created = time.time()
            record.msecs = 0
            if "metrics" in extra:
                record.metrics = extra["metrics"]
            if "step" in extra:
                record.step = extra["step"]
            self.handle(record)
        else:
            super().info(msg, **kwargs)

def setup_logging(
    log_dir: str = "logs",
    experiment_name: str = "experiment",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    json_log: bool = True,
    colored_console: bool = True,
) -> ExperimentLogger:
    """Set up logging with console and optional JSON file output."""
    os.makedirs(log_dir, exist_ok=True)
    logger = ExperimentLogger(experiment_name, level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)  # Allow all levels, handlers filter

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_formatter = ColoredFormatter(use_colors=colored_console)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (plain text)
    txt_log = os.path.join(log_dir, f"{experiment_name}.log")
    file_handler = logging.FileHandler(txt_log, encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    # JSON structured log
    if json_log:
        json_log_path = os.path.join(log_dir, f"{experiment_name}.jsonl")
        json_handler = logging.FileHandler(json_log_path, encoding="utf-8")
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

    logger.info(f"🚀 Logging initialized – logs: {log_dir}")
    return logger
