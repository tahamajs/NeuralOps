import logging

logger = logging.getLogger(__name__)


def push(run_id, run_dir, cfg) -> None:
    """Push model as OCI artifact using ORAS CLI."""
    try:
        import subprocess

        subprocess.run(["oras", "version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ORAS CLI not installed – skipping OCI push")
        return
    registry = cfg.docker_options.get("registry", "ghcr.io/myorg")
    tag = cfg.docker_options.get("tag", "latest")
    artifact = f"{registry}/{run_id}:{tag}"
    model_dir = run_dir / "artifacts" / "model"
    if not model_dir.exists():
        return
    subprocess.run(
        [
            "oras",
            "push",
            artifact,
            str(model_dir / "pytorch_model.bin"),
            "--artifact-type",
            "application/vnd.oci.image.config.v1+json",
        ],
        check=True,
    )
    logger.info("Pushed OCI artifact %s", artifact)
