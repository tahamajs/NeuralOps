import logging
import os

logger = logging.getLogger(__name__)


def push(run_id, run_dir, cfg=None) -> None:
    """
    Push run directory results (models, plots, metrics, reports) to HuggingFace Hub.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed – skipping HF Hub push")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    username = os.environ.get("HUGGINGFACE_USERNAME") or os.environ.get("USER", "user")
    
    hf_opts = getattr(cfg, "hf_options", {}) if cfg else {}
    if isinstance(cfg, dict):
        hf_opts = cfg.get("hf_options", {})
        
    repo_id = hf_opts.get("repo_id") or f"{username}/model-{run_id}"
    private = hf_opts.get("private", False)
    
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
        
        # Upload model directory or full run directory if exists
        model_dir = run_dir / "artifacts" / "model"
        upload_path = run_dir if run_dir.exists() else model_dir
        
        api.upload_folder(
            folder_path=str(upload_path),
            repo_id=repo_id,
            commit_message=f"Upload results for run {run_id}",
        )
        logger.info("Successfully pushed complete run results to HuggingFace Hub: https://huggingface.co/%s", repo_id)
    except Exception as e:
        logger.error("Failed to push results to HuggingFace Hub (%s): %s", repo_id, e)

