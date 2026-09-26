import logging

logger = logging.getLogger(__name__)


def track(run_id, run_dir, metrics, params, cfg) -> None:
    """Track."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed – skipping W&B tracking")
        return
    wandb.init(project="transformer-toolkit", name=run_id, config=params, reinit=True)
    wandb.log(metrics)
    artifacts = run_dir / "artifacts" / "model"
    if artifacts.exists():
        art = wandb.Artifact(f"model-{run_id}", type="model")
        art.add_dir(str(artifacts))
        wandb.log_artifact(art)
    wandb.finish()
    logger.info("Logged to W&B – run %s", run_id)
