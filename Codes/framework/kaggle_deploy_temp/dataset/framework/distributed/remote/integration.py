"""
Integration with MegaOrchestrator – automatically decides whether to
train locally or dispatch to remote workers.
"""

from typing import Any, Dict, Optional

from .coordinator import RemoteCoordinator


def launch_remote_training(
    config: Dict[str, Any],
    run_id: str,
) -> Optional[str]:
    """
    Called by MegaOrchestrator when distributed.remote.enabled = True.

    If enough remote workers are available, dispatches the job and returns
    the job_id.  Otherwise, falls back to local training.

    Returns:
        job_id if remote launch succeeded, None if fallback needed.
    """
    remote_cfg = config.get("distributed", {}).get("remote", {})
    if not remote_cfg.get("enabled"):
        return None

    num_gpus = remote_cfg.get("num_gpus_needed", 1)
    coord = RemoteCoordinator()
    if coord.total_available_gpus < num_gpus:
        return None  # caller should fall back to single‑node

    job_id = coord.launch_job(config, num_gpus_needed=num_gpus)
    return job_id
