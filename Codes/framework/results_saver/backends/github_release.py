from typing import Any
import logging
import os

logger = logging.getLogger(__name__)


def create_release(run_id, run_dir, metrics, cfg) -> Any:
    """Build and return release."""
    try:
        from github import Github
    except ImportError:
        logger.warning("PyGithub not installed – skipping GitHub release")
        return None
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("GITHUB_REPO")
    if not token or not repo_name:
        logger.warning("GITHUB_TOKEN or GITHUB_REPO not set – skipping release")
        return None
    g = Github(token)
    repo = g.get_repo(repo_name)
    tag = cfg.release_options.get("tag_pattern", "v{run_id}").format(
        run_id=run_id, **metrics
    )
    notes = cfg.release_options.get(
        "release_notes_template", "Results for {run_id}"
    ).format(run_id=run_id, **metrics)
    release = repo.create_git_release(tag, tag, notes)
    for art in cfg.release_options.get("attach_artifacts", []):
        fp = run_dir / "artifacts" / art
        if fp.exists():
            release.upload_asset(str(fp))
    logger.info("Created GitHub release %s", tag)
    return release.html_url
