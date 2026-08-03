import logging

logger = logging.getLogger(__name__)


def track(run_id, run_dir, metrics, params, cfg) -> None:
    """Log results as a ZenML step artifact."""
    try:
        from zenml.artifacts.artifact_config import ArtifactConfig
        from zenml.client import Client
    except ImportError:
        logger.warning("zenml not installed – skipping ZenML tracking")
        return
    client = Client()
    # Save metrics as a JSON artifact
    import json

    artifact_path = run_dir / "metrics.json"
    with open(artifact_path, "w") as f:
        json.dump(metrics, f)
    client.create_run_step_artifact(
        name=f"run_{run_id}_metrics",
        artifact=artifact_path,
        step_name=f"training_run_{run_id}",
    )
    logger.info("Logged to ZenML – run %s", run_id)
