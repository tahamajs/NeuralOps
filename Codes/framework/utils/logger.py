# framework/utils/logger.py
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_log_level(logger_name: str, level: int):
    logging.getLogger(logger_name).setLevel(level)
