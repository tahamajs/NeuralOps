"""
Automatic integration between logging, ResultsSaver, and the experiment runner.
Call `auto_setup(run_config)` once at the start of training.
"""

from pathlib import Path

from .config import LoggingConfig
from .logger import setup_logging
from .tracker import MetricsTracker


def auto_setup(
    run_config: dict,
    run_id: str,
    base_dir: Path,
) -> dict:
    """
    Create logger and tracker from run config, returning them as a dict.
    Also writes config snapshot to the run directory.

    Args:
        run_config: dictionary containing keys such as `log_level`, `log_format`,
                    `save_logs`, `exp_name`, etc.
        run_id: unique identifier for this run.
        base_dir: root directory where run artifacts are stored.

    Returns:
        A dictionary with keys `logger`, `tracker`, and `config`.
    """
    log_cfg = LoggingConfig(
        log_level=run_config.get("log_level", "INFO"),
        log_format=run_config.get("log_format", "structured"),
        log_to_console=True,
        log_to_file=run_config.get("save_logs", True),
        log_dir=str(base_dir / run_id / "logs"),
        experiment_name=run_config.get("exp_name", "training"),
        run_id=run_id,
        use_timestamp=True,
        metrics_enabled=True,
    )
    logger = setup_logging(log_cfg)

    metrics_path = base_dir / run_id / "logs" / "metrics.jsonl"
    tracker = MetricsTracker(str(metrics_path))

    # Save config snapshot for reproducibility
    import json

    config_dir = base_dir / run_id / "artifacts"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    logger.info("Auto‑setup complete. Run ID: %s", run_id)
    return {"logger": logger, "tracker": tracker, "config": log_cfg}
