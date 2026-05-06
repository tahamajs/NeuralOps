import os, logging
logger = logging.getLogger(__name__)

def push_to_hub(local_dir, repo_name, token=None):
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_name, exist_ok=True)
        api.upload_folder(folder_path=local_dir, repo_id=repo_name, token=token)
        logger.info("Pushed to HuggingFace Hub: %s", repo_name)
    except ImportError:
        logger.warning("huggingface_hub not installed – HF Hub push skipped")
    except Exception as e:
        logger.exception("HF Hub push failed: %s", e)