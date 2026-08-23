import os
import logging

logger = logging.getLogger(__name__)


def push(run_dir, cfg) -> None:
    """Commit results to a lakeFS branch."""
    try:
        import lakefs
    except ImportError:
        logger.warning("lakefs not installed – skipping lakeFS push")
        return
    client = lakefs.Client(
        host=cfg.cloud_options.get("lakefs_host", "http://localhost:8000"),
        username=cfg.cloud_options.get("lakefs_access_key", ""),
        password=cfg.cloud_options.get("lakefs_secret_key", ""),
    )
    repo = client.get_repository(cfg.cloud_options.get("lakefs_repo", "my-repo"))
    branch = repo.branch(cfg.cloud_options.get("lakefs_branch", "main"))
    for root, _, files in os.walk(run_dir):
        for f in files:
            local = os.path.join(root, f)
            rel = os.path.relpath(local, os.path.dirname(str(run_dir)))
            with open(local, "rb") as data:
                branch.object(rel).upload(data.read())
    branch.commit(f"Results for run {run_dir.name}")
    logger.info("Committed results to lakeFS branch %s", branch.id)
