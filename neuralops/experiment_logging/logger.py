"""
Structured and thread-safe logger setup with JSON and human-readable formats.
Implements QueueHandler / QueueListener for non-blocking logging across threads and
ensures each DDP worker rank writes to an isolated log file (rank_{LOCAL_RANK}.log).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import sys
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any

from .config import LoggingConfig

# Active listeners registry for clean shutdown
_ACTIVE_LISTENERS: list[QueueListener] = []


def get_rank(configured_rank: int | None = None) -> int:
    """
    Determine the current process rank in single-node or DDP multi-GPU environments.

    Args:
        configured_rank: Explicit rank if specified in config.

    Returns:
        int: Process local/global rank (0 for single-node / master).
    """
    if configured_rank is not None:
        return int(configured_rank)

    for env_key in (
        "LOCAL_RANK",
        "RANK",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        val = os.environ.get(env_key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except (
        AttributeError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        ImportError,
    ):
        pass

    return 0


class StructuredFormatter(logging.Formatter):
    """JSON-line formatter for structured logging with rank metadata."""

    def __init__(self, rank: int = 0) -> None:
        super().__init__()
        self.rank = rank

    def format(self, record: logging.LogRecord) -> str:
        """Format log record into JSON string."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "rank": getattr(record, "rank", self.rank),
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            entry.update(record.extra_data)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def shutdown_logging() -> None:
    """
    Stop all background QueueListener workers and flush pending log events.
    """
    global _ACTIVE_LISTENERS
    for listener in _ACTIVE_LISTENERS:
        try:
            listener.stop()
        except (
            AttributeError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            ImportError,
        ):
            pass
    _ACTIVE_LISTENERS.clear()


atexit.register(shutdown_logging)


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """
    Create and configure a thread-safe logger with isolated rank logging.

    Args:
        config: LoggingConfig dataclass specifying log parameters.

    Returns:
        logging.Logger: Configured logger instance.
    """
    cfg = config or LoggingConfig()
    rank = get_rank(cfg.rank)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    os.makedirs(cfg.log_dir, exist_ok=True)

    if cfg.log_format == "structured":
        fmt = StructuredFormatter(rank=rank)
    else:
        fmt = logging.Formatter(
            f"%(asctime)s [Rank {rank}] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    destination_handlers: list[logging.Handler] = []

    if cfg.log_to_console:
        # Stream logs to stdout (typically rank 0 only or all ranks if formatted)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        destination_handlers.append(ch)

    if cfg.log_to_file:
        # FIX: Ensure each DDP rank writes to an isolated log file (rank_{LOCAL_RANK}.log)
        if cfg.rank_specific_logs:
            fname = f"rank_{rank}.log"
        else:
            fname = f"{cfg.experiment_name}_{cfg.run_id or ''}_rank_{rank}.log".strip("_")

        log_file_path = os.path.join(cfg.log_dir, fname)
        fh = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        destination_handlers.append(fh)

    if cfg.use_queue_handler:
        # FIX: Implement Python logging with QueueHandler and QueueListener for thread safety and non-blocking I/O
        log_queue: queue.Queue = queue.Queue(-1)
        queue_handler = QueueHandler(log_queue)
        root.addHandler(queue_handler)

        listener = QueueListener(log_queue, *destination_handlers, respect_handler_level=True)
        listener.start()
        _ACTIVE_LISTENERS.append(listener)
    else:
        for handler in destination_handlers:
            root.addHandler(handler)

    logger = logging.getLogger("neuralops")
    logger.info(
        "Logging initialised. Experiment: %s, Run: %s, Rank: %d, ThreadSafeQueue: %s",
        cfg.experiment_name,
        cfg.run_id,
        rank,
        cfg.use_queue_handler,
        extra={"rank": rank},
    )
    return logger


def get_logger(name: str = "neuralops") -> logging.Logger:
    """Return a logger for the given name."""
    return logging.getLogger(name)
