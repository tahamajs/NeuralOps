import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import logging

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, experiment_id: int, results: List[Dict], checkpoint_name: str = "checkpoint.json"):
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_data = {'experiment_id': experiment_id, 'results': results, 'timestamp': time.time()}
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_name: str = "checkpoint.json") -> Tuple[int, List[Dict]]:
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            return 0, []
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            logging.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data['experiment_id'], checkpoint_data['results']
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return 0, []