import logging
import os

logger = logging.getLogger(__name__)


def upload_to_s3(local_path, bucket, region=None) -> None:
    """Upload to s3."""
    try:
        import boto3

        s3 = boto3.client("s3", region_name=region)
        key = os.path.basename(local_path)
        s3.upload_file(local_path, bucket, key)
        logger.info("Uploaded %s to s3://%s/%s", local_path, bucket, key)
    except ImportError:
        logger.warning("boto3 not installed – S3 upload skipped")
    except Exception as e:
        logger.exception("S3 upload failed: %s", e)


def download_from_s3(local_path, bucket, region=None) -> None:
    """Download from s3."""
    try:
        import boto3

        s3 = boto3.client("s3", region_name=region)
        key = os.path.basename(local_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)
    except Exception as e:
        logger.exception("S3 download failed: %s", e)


def delete_from_s3(local_path, bucket, region=None) -> None:
    """Delete from s3."""
    try:
        import boto3

        s3 = boto3.client("s3", region_name=region)
        key = os.path.basename(local_path)
        s3.delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted s3://%s/%s", bucket, key)
    except Exception as e:
        logger.exception("S3 delete failed: %s", e)
