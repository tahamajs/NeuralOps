import logging
import os

logger = logging.getLogger(__name__)


def push(run_dir, cfg) -> None:
    """Push."""
    try:
        from google.cloud import storage
    except ImportError:
        logger.warning("google-cloud-storage not installed – skipping GCP push")
        return
    bucket_name = cfg.cloud_options.get("gcp_bucket")
    if not bucket_name:
        logger.warning("No GCP bucket configured")
        return
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(run_dir):
        for f in files:
            local = os.path.join(root, f)
            blob = bucket.blob(os.path.relpath(local, os.path.dirname(str(run_dir))))
            blob.upload_from_filename(local)
    logger.info("Uploaded results to GCP bucket %s", bucket_name)
