#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Run directory management and file-locking primitives for CLI runs."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from neuralops.cli.core.errors import ExitCode, RunDirectoryLockedError


def create_run_directory(
    output_dir: Path | str,
    run_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create and validate a run output directory.

    If output_dir exists and is non-empty, requires overwrite=True.
    """
    out_path = Path(output_dir).resolve()
    if out_path.exists() and any(out_path.iterdir()):
        if not overwrite:
            raise RunDirectoryLockedError(
                f"Output directory '{out_path}' exists and is not empty. "
                "Specify --overwrite to reuse or provide a different --output-dir.",
                exit_code=ExitCode.FILE_LOCK_FAILED,
            )
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


@contextlib.contextmanager
def run_directory_lock(output_dir: Path | str) -> Iterator[Path]:
    """Acquire an exclusive advisory file lock on a run directory via .neuralops.lock.

    Raises RunDirectoryLockedError if another process holds the lock.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    lock_file = out_path / ".neuralops.lock"
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as err:
            raise RunDirectoryLockedError(
                f"Another process is currently executing in directory: {out_path}",
                exit_code=ExitCode.FILE_LOCK_FAILED,
            ) from err

        lock_payload = {
            "pid": os.getpid(),
            "timestamp": time.time(),
        }
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(lock_payload).encode("utf-8"))
        os.fsync(fd)

        yield out_path
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def atomic_write_json(file_path: Path | str, data: Any, indent: int = 2) -> None:
    """Safely write JSON to a file by writing to a temporary file first and atomically replacing."""
    target = Path(file_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=str(target.parent), delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, indent=indent, default=str)
        tf.write("\n")
        tf.flush()
        os.fsync(tf.fileno())
        temp_name = tf.name

    shutil.move(temp_name, str(target))
