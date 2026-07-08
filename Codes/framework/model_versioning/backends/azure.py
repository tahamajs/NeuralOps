import logging
import os

logger = logging.getLogger(__name__)


def upload_to_azure(local_path, conn_str, container):
    try:
        from azure.storage.blob import BlobServiceClient

        service = BlobServiceClient.from_connection_string(conn_str)
        cc = service.get_container_client(container)
        with open(local_path, "rb") as data:
            cc.upload_blob(name=os.path.basename(local_path), data=data)
        logger.info("Uploaded to Azure container %s", container)
    except ImportError:
        logger.warning("azure-storage-blob not installed – Azure upload skipped")
    except Exception as e:
        logger.exception("Azure upload failed: %s", e)
