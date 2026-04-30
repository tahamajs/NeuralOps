# framework/utils/checkpoint.py
"""
Checkpoint management for saving and loading experiment state.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging


class CheckpointManager:
    """
    Manage experiment checkpoints for resumable experiments.
    
    Example:
        >>> manager = CheckpointManager("./checkpoints")
        >>> manager.save_checkpoint(experiment_id=1, results=results)
        >>> experiment_id, results = manager.load_checkpoint()
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        experiment_id: int,
        results: List[Dict],
        checkpoint_name: str = "checkpoint.json",
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save experiment checkpoint.
        
        Args:
            experiment_id: Current experiment ID
            results: List of completed results
            checkpoint_name: Name of checkpoint file
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            'experiment_id': experiment_id,
            'results': results,
            'timestamp': time.time(),
            'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        if metadata:
            checkpoint_data['metadata'] = metadata
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logging.info(f"✅ Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_name: str = "checkpoint.json") -> Tuple[int, List[Dict], Optional[Dict]]:
        """
        Load experiment checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint file
            
        Returns:
            Tuple of (experiment_id, results, metadata)
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            logging.info(f"No checkpoint found at {checkpoint_path}")
            return 0, [], None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            logging.info(f"✅ Checkpoint loaded: {checkpoint_path}")
            logging.info(f"   Experiment ID: {checkpoint_data['experiment_id']}")
            logging.info(f"   Completed results: {len(checkpoint_data['results'])}")
            
            return (
                checkpoint_data.get('experiment_id', 0),
                checkpoint_data.get('results', []),
                checkpoint_data.get('metadata')
            )
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return 0, [], None
    
    def delete_checkpoint(self, checkpoint_name: str = "checkpoint.json"):
        """Delete checkpoint file."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logging.info(f"Deleted checkpoint: {checkpoint_path}")
    
    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint files."""
        return list(self.checkpoint_dir.glob("*.json"))
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint file."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)