import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    name: str = "neuralops",
) -> logging.Logger:
    """
    Configure and return a structured logger for NeuralOps agent components.
    Ensures idempotency by replacing existing handlers cleanly.
    """
    if isinstance(level, str):
        level_int = getattr(logging, level.upper(), logging.INFO)
    else:
        level_int = int(level)

    logger = logging.getLogger(name)
    logger.setLevel(level_int)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_int)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler if specified
    if log_file:
        try:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(file_path), encoding="utf-8")
            fh.setLevel(level_int)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            logger.warning("Failed to initialize file logger at %s: %s", log_file, e)

    # Prevent propagation to avoid duplicate logs in root handlers if configured
    logger.propagate = False

    return logger
