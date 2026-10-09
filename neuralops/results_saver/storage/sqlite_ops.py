"""Sanitization, formatting, and file locking utilities for result storage."""

from __future__ import annotations

import contextlib
import logging
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)


def sanitize_json_value(value: Any) -> Any:
    """Convert runtime values to strict JSON-safe values.

    JSON permits neither NaN nor Infinity. Non-finite numeric values are represented
    as null (None) to prevent corrupting downstream dashboards and aggregation.

    Args:
        value: Any Python object or primitive.

    Returns:
        JSON-compliant primitive, list, or dictionary.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json_value(v) for v in value]
    return value


@contextlib.contextmanager
def sqlite_file_lock(lock_file_path: str | Path, timeout: float = 30.0) -> Any:
    """Acquire an exclusive cross-process / multi-GPU file lock to serialize SQLite writes.

    Args:
        lock_file_path: Path to the lock file.
        timeout: Maximum seconds to wait before falling back or raising.

    Yields:
        None while holding the lock.
    """
    lock_path = Path(lock_file_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd: int | None = None
    start_time = time.time()
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        has_fcntl = True
        try:
            import fcntl
        except ImportError:
            has_fcntl = False

        if has_fcntl:
            acquired = False
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    time.sleep(0.05)

            if not acquired:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except (
                    AttributeError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    OSError,
                ):
                    pass

        yield
    finally:
        if lock_fd is not None:
            try:
                import fcntl

                fcntl.flock(lock_fd, fcntl.LOCK_UN)
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
            try:
                os.close(lock_fd)
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


def configure_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    """Configure high-performance, concurrency-safe SQLite pragmas for multi-GPU training.

    Enables Write-Ahead Logging (WAL) and NORMAL synchronous mode.

    Args:
        conn: Open SQLite database connection.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA busy_timeout=30000;")
    cursor.close()
