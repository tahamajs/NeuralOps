import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def git_commit_and_push(run_dir, cfg) -> None:
    """Git commit and push."""
    try:
        import git
    except ImportError:
        logger.warning("gitpython not installed – skipping Git push")
        return
    repo = git.Repo(search_parent_directories=True)
    # Add run_dir to repo index
    relative = os.path.relpath(run_dir, repo.working_dir)
    if cfg.git_options.get("use_git_lfs"):
        threshold = cfg.git_options.get("git_lfs_threshold_mb", 10) * 1024 * 1024
        for f in run_dir.rglob("*"):
            if f.is_file() and f.stat().st_size > threshold:
                subprocess.run(
                    ["git", "lfs", "track", str(f.relative_to(repo.working_dir))]
                )
    repo.index.add([relative])
    tag = cfg.git_options.get("tag_pattern", "v{run_id}").format(
        run_id=os.path.basename(run_dir)
    )
    repo.index.commit(f"Results for run {os.path.basename(run_dir)}")
    repo.create_tag(tag)
    if cfg.git_options.get("push_to_remote"):
        remote = repo.remote()
        remote.push()
        remote.push(tags=True)
    logger.info("Committed and pushed results to Git with tag %s", tag)
