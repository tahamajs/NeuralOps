# framework/utils/checkpoint.py
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        experiment_id: int,
        results: List[Dict],
        checkpoint_name: str = "checkpoint.json",
        metadata: Optional[Dict] = None
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        data = {
            'experiment_id': experiment_id,
            'results': results,
            'timestamp': time.time(),
            'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        if metadata:
            data['metadata'] = metadata
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_name: str = "checkpoint.json") -> Tuple[int, List[Dict], Optional[Dict]]:
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            logging.info(f"No checkpoint found at {checkpoint_path}")
            return 0, [], None
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Checkpoint loaded: {checkpoint_path}")
        return data.get('experiment_id', 0), data.get('results', []), data.get('metadata')

    def delete_checkpoint(self, checkpoint_name: str = "checkpoint.json"):
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logging.info(f"Deleted checkpoint: {checkpoint_path}")

    def list_checkpoints(self) -> List[Path]:
        return list(self.checkpoint_dir.glob("*.json"))

    def get_latest_checkpoint(self) -> Optional[Path]:
        checkpoints = self.list_checkpoints()
        return max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else None