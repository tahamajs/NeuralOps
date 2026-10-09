"""Structured dual-format logging for neuralops benchmarks."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any


class JSONLHandler(logging.Handler):
    """Logging handler that outputs structured JSON Lines."""

    def __init__(self, filepath: Path | str) -> None:
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": getattr(record, "benchmark_module", record.module),
                "line": record.lineno,
            }
            if record.exc_info:
                log_entry["traceback"] = self.format(record)
            self._file.write(json.dumps(log_entry) + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            super().close()


def setup_run_logger(
    output_dir: Path | str,
    run_id: str,
    module_name: str = "",
) -> tuple[logging.Logger, Path, Path]:
    """
    Setup dual-format logging (human-readable .log and structured .jsonl).
    Returns (logger, log_path, jsonl_path).
    """
    out_dir = Path(output_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"{run_id}.log"
    jsonl_path = logs_dir / f"{run_id}.jsonl"

    logger = logging.getLogger(f"neuralops.benchmarks.run.{run_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 1. Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 2. Text File Handler (.log)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 3. JSONL Handler (.jsonl)
    jsonl_handler = JSONLHandler(jsonl_path)
    jsonl_handler.setLevel(logging.DEBUG)
    logger.addHandler(jsonl_handler)

    logger.propagate = False
    return logger, log_path, jsonl_path
