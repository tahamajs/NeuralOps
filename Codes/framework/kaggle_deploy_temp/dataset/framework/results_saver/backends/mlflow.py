import logging

logger = logging.getLogger(__name__)


def track(run_id, run_dir, metrics, params, cfg) -> None:
    """Track."""
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed – skipping MLflow tracking")
        return
    mlflow.set_tracking_uri(cfg.cloud_options.get("mlflow_uri", ""))
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        artifacts = run_dir / "artifacts"
        if artifacts.exists():
            mlflow.log_artifacts(str(artifacts))
    logger.info("Logged to MLflow – run %s", run_id)
