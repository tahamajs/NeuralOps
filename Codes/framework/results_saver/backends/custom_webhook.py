import json
import logging

import requests

logger = logging.getLogger(__name__)


def push(run_dir, cfg) -> None:
    """POST results to a user-defined webhook URL."""
    webhook_url = cfg.cloud_options.get("custom_webhook_url")
    if not webhook_url:
        logger.warning("No custom webhook URL configured – skipping")
        return
    # Send a summary JSON
    summary = {
        "run_id": run_dir.name,
        "metrics": json.load(open(run_dir / "artifacts" / "final_metrics.json")),
        "artifact_count": sum(1 for _ in run_dir.rglob("*")),
    }
    try:
        requests.post(webhook_url, json=summary, timeout=10)
        logger.info("Sent results to webhook %s", webhook_url)
    except Exception:
        logger.exception("Custom webhook push failed")
