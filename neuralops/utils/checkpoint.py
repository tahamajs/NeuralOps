# framework/utils/checkpoint.py
import json
import logging
import time
from pathlib import Path


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        experiment_id: int,
        results: list[dict],
        checkpoint_name: str = "checkpoint.json",
        metadata: dict | None = None,
    ) -> Path:
        """Persist checkpoint to disk."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        data = {
            "experiment_id": experiment_id,
            "results": results,
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if metadata:
            data["metadata"] = metadata
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self, checkpoint_name: str = "checkpoint.json"
    ) -> tuple[int, list[dict], dict | None]:
        """Load checkpoint from disk or config."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            logging.info(f"No checkpoint found at {checkpoint_path}")
            return 0, [], None
        with open(checkpoint_path) as f:
            data = json.load(f)
        logging.info(f"Checkpoint loaded: {checkpoint_path}")
        return (
            data.get("experiment_id", 0),
            data.get("results", []),
            data.get("metadata"),
        )

    def delete_checkpoint(self, checkpoint_name: str = "checkpoint.json") -> None:
        """Delete checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logging.info(f"Deleted checkpoint: {checkpoint_path}")

    def list_checkpoints(self) -> list[Path]:
        """List checkpoints."""
        return list(self.checkpoint_dir.glob("*.json"))

    def get_latest_checkpoint(self) -> Path | None:
        """Return the latest checkpoint."""
        checkpoints = self.list_checkpoints()
        return (
            max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else None
        )
