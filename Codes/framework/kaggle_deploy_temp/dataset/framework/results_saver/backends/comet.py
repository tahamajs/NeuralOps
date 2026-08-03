import logging

logger = logging.getLogger(__name__)


def track(run_id, run_dir, metrics, params, cfg) -> None:
    """Track."""
    try:
        from comet_ml import Experiment
    except ImportError:
        logger.warning("comet_ml not installed – skipping Comet tracking")
        return
    project = cfg.cloud_options.get("comet_project", "")
    exp = Experiment(project_name=project)
    exp.log_parameters(params)
    exp.log_metrics(metrics)
    model_file = run_dir / "artifacts" / "model" / "pytorch_model.bin"
    if model_file.exists():
        exp.log_model("model", str(model_file))
    exp.end()
    logger.info("Logged to Comet ML – run %s", run_id)
