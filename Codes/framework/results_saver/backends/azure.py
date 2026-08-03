import logging
import os

logger = logging.getLogger(__name__)


def push(run_dir, cfg) -> None:
    """Push."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.warning("azure-storage-blob not installed – skipping Azure push")
        return
    conn_str = cfg.cloud_options.get("azure_conn_str")
    container = cfg.cloud_options.get("azure_container")
    if not conn_str or not container:
        logger.warning("Azure connection string or container not configured")
        return
    service = BlobServiceClient.from_connection_string(conn_str)
    cc = service.get_container_client(container)
    for root, _, files in os.walk(run_dir):
        for f in files:
            local = os.path.join(root, f)
            with open(local, "rb") as data:
                cc.upload_blob(
                    name=os.path.relpath(local, os.path.dirname(str(run_dir))),
                    data=data,
                )
    logger.info("Uploaded results to Azure container %s", container)
