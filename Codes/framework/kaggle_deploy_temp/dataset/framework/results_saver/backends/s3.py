import logging
import os

logger = logging.getLogger(__name__)


def push(run_dir, cfg) -> None:
    """Push."""
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed – skipping S3 push")
        return
    bucket = cfg.cloud_options.get("aws_s3_bucket")
    if not bucket:
        logger.warning("No S3 bucket configured")
        return
    s3 = boto3.client("s3")
    for root, _, files in os.walk(run_dir):
        for f in files:
            local = os.path.join(root, f)
            key = os.path.relpath(local, os.path.dirname(str(run_dir)))
            s3.upload_file(local, bucket, key)
    logger.info("Uploaded results to S3 bucket %s", bucket)
