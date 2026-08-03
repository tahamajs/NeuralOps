import logging

logger = logging.getLogger(__name__)


def push(run_id, run_dir, cfg) -> None:
    """Push."""
    logger.info(
        "Docker registry push not implemented – configure docker_options in save_config"
    )
