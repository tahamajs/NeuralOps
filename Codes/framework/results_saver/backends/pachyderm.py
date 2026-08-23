import os
import logging

logger = logging.getLogger(__name__)


def push(run_dir, cfg) -> None:
    """Commit results to Pachyderm data repository."""
    try:
        import python_pachyderm
    except ImportError:
        logger.warning("pachyderm not installed – skipping Pachyderm push")
        return
    client = python_pachyderm.Client(
        host=cfg.cloud_options.get("pachyderm_host", "localhost"),
        port=cfg.cloud_options.get("pachyderm_port", 30650),
    )
    repo = cfg.cloud_options.get("pachyderm_repo", "experiments")
    with client.commit(repo, "master") as commit:
        for root, _, files in os.walk(run_dir):
            for f in files:
                local = os.path.join(root, f)
                rel = os.path.relpath(local, os.path.dirname(str(run_dir)))
                client.put_file_bytes(commit, rel, open(local, "rb").read())
    client.finish_commit(commit)
    logger.info("Pushed results to Pachyderm repo %s", repo)
