import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def dvc_track(run_dir, cfg) -> None:
    """Dvc track."""
    try:
        subprocess.run(["dvc", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("DVC not installed or not on PATH – skipping DVC tracking")
        return
    relative = os.path.relpath(run_dir, os.getcwd())
    subprocess.run(["dvc", "add", relative], check=True)
    subprocess.run(["git", "add", f"{relative}.dvc"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"DVC track run {os.path.basename(run_dir)}"],
        check=True,
    )
    if cfg.dvc_options.get("auto_push"):
        subprocess.run(["dvc", "push"], check=True)
    logger.info("DVC tracking completed for run %s", os.path.basename(run_dir))
