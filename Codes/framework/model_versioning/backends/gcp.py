import logging
import os

logger = logging.getLogger(__name__)


def upload_to_gcp(local_path, bucket) -> None:
    """Upload to gcp."""
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(os.path.basename(local_path))
        blob.upload_from_filename(local_path)
        logger.info("Uploaded %s to gs://%s/%s", local_path, bucket, blob.name)
    except ImportError:
        logger.warning("google-cloud-storage not installed – GCP upload skipped")
    except Exception as e:
        logger.exception("GCP upload failed: %s", e)


def download_from_gcp(local_path, bucket) -> None:
    """Download from gcp."""
    pass  # implement with client.download_blob_to_file
