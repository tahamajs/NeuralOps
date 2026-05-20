# framework/checkpoint_hf.py
import os, json, logging
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

logger = logging.getLogger(__name__)


class HuggingFaceCheckpointUploader:
    """
    Automatically uploads training checkpoints and final results to
    a dedicated HuggingFace repository.

    Repository: {username}/neuralops-run-{run_id}
    """
    def __init__(
        self,
        run_id: str,
        token: Optional[str] = None,
        username: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        upload_every_n_steps: int = 500,
        max_checkpoints_to_keep: int = 5,
    ):
        self.run_id = run_id
        self.token = token or os.environ.get("HUGGINGFACE_TOKEN")
        self.username = username or os.environ.get("HUGGINGFACE_USERNAME", "neuralops")
        self.checkpoint_dir = checkpoint_dir or f"./checkpoints/{run_id}"
        self.upload_every_n_steps = upload_every_n_steps
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.repo_id = f"{self.username}/neuralops-run-{run_id}"
        self._api: Optional[HfApi] = None
        self._uploaded_checkpoints = []

    @property
    def api(self) -> HfApi:
        if self._api is None:
            if not self.token:
                raise ValueError("HuggingFace token not set. Set HUGGINGFACE_TOKEN or log in via HF.")
            self._api = HfApi(token=self.token)
        return self._api

    def init_repo(self):
        """Create the HF repository if it doesn't exist (private by default)."""
        create_repo(self.repo_id, token=self.token, repo_type="model", exist_ok=True, private=True)
        logger.info(f"Initialized HF repo: {self.repo_id}")

    def upload_checkpoint(self, step: int, checkpoint_path: Optional[str] = None):
        """
        Upload a single checkpoint (the whole checkpoint directory) to HF.
        Stores it under `checkpoints/step_{step}/`.
        """
        self.init_repo()
        local_dir = checkpoint_path or self.checkpoint_dir
        if not os.path.isdir(local_dir):
            logger.warning(f"Checkpoint directory not found: {local_dir}")
            return

        remote_folder = f"checkpoints/step_{step}"
        logger.info(f"Uploading checkpoint at step {step} to {self.repo_id}/{remote_folder}")
        upload_folder(
            folder_path=local_dir,
            path_in_repo=remote_folder,
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token,
        )

        self._uploaded_checkpoints.append(step)

        # Clean up old checkpoints beyond the limit
        if len(self._uploaded_checkpoints) > self.max_checkpoints_to_keep:
            oldest_step = self._uploaded_checkpoints.pop(0)
            # HF doesn't provide a simple delete folder API, we can try to delete
            # by listing files and removing them. For simplicity, we just log.
            logger.info(f"Old checkpoint at step {oldest_step} can be pruned (manual step).")

    def upload_final_model(self, model_dir: str, metrics: dict, final: bool = True):
        """
        Upload the final model and a metrics/README to the root of the repo.
        """
        self.init_repo()
        # Upload model files
        upload_folder(
            folder_path=model_dir,
            path_in_repo="model",
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token,
        )

        # Create and upload README.md with metrics
        readme_content = self._generate_readme(metrics)
        readme_path = os.path.join(model_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token,
        )

        # Optionally make the repo public now that training is complete
        if final:
            self.api.update_repo_visibility(
                repo_id=self.repo_id, repo_type="model", private=False, token=self.token
            )

        logger.info(f"Final model and metrics uploaded to {self.repo_id}")

    def _generate_readme(self, metrics: dict) -> str:
        run_id = self.run_id
        content = f"""---
tags:
- neuralops
- training-run
metrics:
"""
        for key, value in metrics.items():
            content += f"- {key}: {value}\n"
        content += f"""---

# Training Run: {run_id}

## Metrics
| Metric | Value |
|--------|-------|
"""
        for key, value in metrics.items():
            content += f"| {key} | {value} |\n"
        content += "\nThis model was automatically uploaded by NeuralOps.\n"
        return content

    def make_public(self):
        """Make the HF repo public (e.g., after training completes)."""
        self.api.update_repo_visibility(
            repo_id=self.repo_id, repo_type="model", private=False, token=self.token
        )
        logger.info(f"Made repo {self.repo_id} public")