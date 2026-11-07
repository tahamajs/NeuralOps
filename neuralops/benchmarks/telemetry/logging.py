#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Console Stream & Log Redirection Manager

Mirrors stdout and stderr to stdout_stderr.log inside the isolated run directory
in real time while preserving live terminal streaming.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path
from typing import TextIO


class _TeeStream(io.TextIOBase):
    """Multiplexes writes to both terminal stream and a file stream."""

    def __init__(self, original_stream: TextIO, file_stream: TextIO, lock: threading.Lock) -> None:
        self.original_stream = original_stream
        self.file_stream = file_stream
        self.lock = lock

    def write(self, s: str) -> int:
        with self.lock:
            self.original_stream.write(s)
            self.file_stream.write(s)
            self.file_stream.flush()
        return len(s)

    def flush(self) -> None:
        with self.lock:
            self.original_stream.flush()
            self.file_stream.flush()

    def fileno(self) -> int:
        return self.original_stream.fileno()

    def isatty(self) -> bool:
        return self.original_stream.isatty()


class ConsoleLogCapture:
    """
    Context manager that mirrors stdout and stderr to stdout_stderr.log.

    Usage:
        with ConsoleLogCapture(run_dir):
            print("This appears in terminal and in stdout_stderr.log")
    """

    def __init__(self, log_dir: str | Path, filename: str = "stdout_stderr.log") -> None:
        self.log_dir = Path(log_dir)
        self.log_file_path = self.log_dir / filename
        self._lock = threading.Lock()
        self._file_handle: TextIO | None = None
        self._orig_stdout: TextIO | None = None
        self._orig_stderr: TextIO | None = None

    def __enter__(self) -> "ConsoleLogCapture":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(self.log_file_path, "a", encoding="utf-8")

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        sys.stdout = _TeeStream(self._orig_stdout, self._file_handle, self._lock)
        sys.stderr = _TeeStream(self._orig_stderr, self._file_handle, self._lock)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr

        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.flush()
            self._file_handle.close()
