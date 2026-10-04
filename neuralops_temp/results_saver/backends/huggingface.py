"""
neuralops.results_saver.backends.huggingface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optional Hugging Face Hub upload backend for ResultsSaver.

Requires: huggingface_hub  (pip install huggingface_hub)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def push(
    run_id: str,
    run_dir: Path,
    config: dict[str, Any],
) -> str | None:
    """
    Upload all artifacts in run_dir to Hugging Face Hub.

    Parameters
    ----------
    run_id  : Unique experiment identifier (used as repo path suffix).
    run_dir : Local directory containing checkpoints, metrics, etc.
    config  : SaveConfig dict; must contain ``hf_options.repo_id`` to upload.

    Returns
    -------
    str | None
        Hugging Face repo URL on success, None if upload is skipped or fails.
    """
    hf_opts: dict[str, Any] = config.get("hf_options", {}) if isinstance(config, dict) else {}
    repo_id: str | None = hf_opts.get("repo_id")

    if not repo_id:
        logger.debug("HuggingFace upload skipped — hf_options.repo_id not set.")
        return None

    try:
        from huggingface_hub import (
            HfApi,  # type: ignore[import]
            upload_folder,
        )
    except ImportError:
        logger.warning(
            "huggingface_hub not installed — skipping HF upload for run '%s'. "
            "Install with: pip install huggingface_hub",
            run_id,
        )
        return None

    run_dir = Path(run_dir)
    if not run_dir.exists():
        logger.warning("run_dir does not exist: %s — skipping HF upload.", run_dir)
        return None

    private: bool = hf_opts.get("private", True)
    token: str | None = hf_opts.get("token") or None  # falls back to HF_TOKEN env var

    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        upload_folder(
            folder_path=str(run_dir),
            repo_id=repo_id,
            token=token,
            commit_message=f"Upload run artifacts: {run_id}",
        )
        url = f"https://huggingface.co/{repo_id}"
        logger.info("Uploaded run '%s' to %s", run_id, url)
        return url
    except Exception as exc:
        logger.error("HuggingFace upload failed for run '%s': %s", run_id, exc)
        return None
