import datetime
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict


def get_git_info() -> dict:
    """Return the git info."""
    try:
        from git import Repo

        repo = Repo(".", search_parent_directories=True)
        return {
            "commit": repo.head.commit.hexsha,
            "branch": repo.active_branch.name,
            "dirty": repo.is_dirty(untracked_files=True),
            "remote": repo.remotes.origin.url if repo.remotes else None,
        }
    except Exception:
        return {}


def get_dvc_commit() -> str:
    """Return the dvc commit."""
    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "*.dvc"],
            capture_output=True,
            text=True,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except:
        return ""


def get_pip_freeze() -> str:
    """Return the pip freeze."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
        )
        return r.stdout
    except:
        return ""


def get_conda_env() -> str:
    """Return the conda env."""
    try:
        r = subprocess.run(["conda", "env", "export"], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else ""
    except:
        return ""


def capture_environment(output_dir: str) -> Dict[str, Any]:
    """Capture environment."""
    env = {
        "timestamp": datetime.datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "pip_freeze": get_pip_freeze(),
        "git": get_git_info(),
        "dvc_commit": get_dvc_commit(),
    }
    conda = get_conda_env()
    if conda:
        env["conda_env"] = conda
    with open(os.path.join(output_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    return env
