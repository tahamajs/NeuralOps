import logging

import requests

logger = logging.getLogger(__name__)


def track(run_id, run_dir, metrics, params, cfg) -> None:
    """Send metadata and lineage to OpenMetadata via REST API."""
    om_cfg = cfg.metadata_store
    if om_cfg != "openmetadata":
        return
    url = cfg.cloud_options.get("openmetadata_url", "http://localhost:8585/api/v1")
    try:
        # Create a simple dataset/ML model entry
        payload = {
            "name": f"model_{run_id}",
            "displayName": f"Model {run_id}",
            "description": f"Training run {run_id} with accuracy {metrics.get('eval_accuracy')}",
            "serviceType": "MLModel",
            "owner": {"id": cfg.cloud_options.get("om_owner_id", "admin")},
            "tags": [{"name": "transformer-toolkit"}],
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(f"{url}/mlmodels", json=payload, headers=headers)
        resp.raise_for_status()
        logger.info("Pushed metadata to OpenMetadata – run %s", run_id)
    except Exception:
        logger.exception("OpenMetadata push failed")
