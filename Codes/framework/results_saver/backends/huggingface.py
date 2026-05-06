import logging, os
logger = logging.getLogger(__name__)

def push(run_id, run_dir, cfg):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed – skipping HF Hub push")
        return
    username = os.environ.get("HUGGINGFACE_USERNAME", "user")
    repo = f"{username}/model-{run_id}"
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    model_dir = run_dir / "artifacts" / "model"
    if model_dir.exists():
        api.upload_folder(folder_path=str(model_dir), repo_id=repo)
    logger.info("Pushed model to HuggingFace Hub: %s", repo)