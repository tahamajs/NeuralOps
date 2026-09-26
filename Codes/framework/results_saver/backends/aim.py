import logging

logger = logging.getLogger(__name__)


def track(run_id, run_dir, metrics, params, cfg) -> None:
    """Track."""
    try:
        from aim import Run
    except ImportError:
        logger.warning("aim not installed – skipping AimStack tracking")
        return
    run = Run(repo=str(run_dir / "aim_repo"))
    run["hparams"] = params
    for k, v in metrics.items():
        run.track(v, name=k)
    logger.info("Logged to AimStack – run %s", run_id)
