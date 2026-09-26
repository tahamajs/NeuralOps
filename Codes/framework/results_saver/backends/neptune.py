import logging

logger = logging.getLogger(__name__)


def track(run_id, run_dir, metrics, params, cfg) -> None:
    """Track."""
    try:
        import neptune
    except ImportError:
        logger.warning("neptune not installed – skipping Neptune tracking")
        return
    project = cfg.cloud_options.get("neptune_project", "")
    if not project:
        logger.warning("Neptune project not configured")
        return
    run = neptune.init_run(project=project, name=run_id)
    run["parameters"] = params
    run["metrics"] = metrics
    artifacts = run_dir / "artifacts" / "model"
    if artifacts.exists():
        run["artifacts"].upload(str(artifacts))
    run.stop()
    logger.info("Logged to Neptune – run %s", run_id)
