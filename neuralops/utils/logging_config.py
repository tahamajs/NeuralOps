"""Central logging configuration and warning suppression utilities for neuralops.

Ensures that repeated warnings, deprecated notices, or recurrent exceptions
from hot forward() loops do not flood standard output or log streams.

Examples
--------
>>> from neuralops.utils.logging_config import configure_logging, warn_once
>>> configure_logging(level="INFO", dedup=True)
>>> warn_once("module_fallback", "Using fallback attention implementation")
"""

from __future__ import annotations

import logging
from typing import Dict, Set, Tuple

_SEEN_WARNING_KEYS: Set[str] = set()
_CONFIGURED: bool = False


class DedupFilter(logging.Filter):
    """Filter that suppresses duplicate log records based on call site."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: Set[Tuple[str, int, int]] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        """Allow each unique log message through only once per call site.

        Parameters
        ----------
        record : logging.LogRecord
            Active logging record.

        Returns
        -------
        bool
            True if record is new, False if already observed.
        """
        key = (record.module, record.lineno, record.levelno)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def warn_once(
    key: str,
    msg: str,
    *args,
    logger: logging.Logger | None = None,
    level: int = logging.WARNING,
) -> None:
    """Log a warning or error message exactly once per process.

    Parameters
    ----------
    key : str
        Unique dedup identifier for the message site.
    msg : str
        Message template or string to log.
    *args : Any
        Formatting arguments for the log record.
    logger : Optional[logging.Logger], default=None
        Target logger instance. If None, root logger is used.
    level : int, default=logging.WARNING
        Logging level.
    """
    global _SEEN_WARNING_KEYS
    if key in _SEEN_WARNING_KEYS:
        return
    _SEEN_WARNING_KEYS.add(key)

    target_logger = logger if logger is not None else logging.getLogger()
    target_logger.log(level, msg, *args)


def configure_logging(level: str = "INFO", dedup: bool = True) -> None:
    """Configure root logger with optional duplicate suppression filter.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
    dedup : bool, default=True
        Whether to attach DedupFilter suppressing repeated records.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        if dedup:
            handler.addFilter(DedupFilter())
        root.addHandler(handler)
    else:
        if dedup:
            for handler in root.handlers:
                handler.addFilter(DedupFilter())

    _CONFIGURED = True


__all__ = [
    "DedupFilter",
    "warn_once",
    "configure_logging",
]
