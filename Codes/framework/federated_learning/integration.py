"""Integration with Django runner and ResultsSaver."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from .config import FLConfig
from .server import FLServer
from .client import FLClientTrainer
from .data_partitioning import DataPartitioner

def run_federated_experiment(
    config_dict: Dict[str, Any],
    model_builder: Callable[[], torch.nn.Module],
    dataset: Any,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run a federated learning experiment from configuration.

    Args:
        config_dict: FL configuration dictionary
        model_builder: function that returns a fresh model instance
        dataset: full training dataset (will be partitioned)
        output_dir: directory to save results

    Returns:
        dict with final metrics and round history
    """
    cfg = FLConfig(**{k: v for k, v in config_dict.items() if k in FLConfig.__dataclass_fields__})

    # Partition data across clients
    partitioner = DataPartitioner(
        num_clients=cfg.num_clients,
        partition_type=cfg.partition_type,
        alpha=cfg.non_iid_alpha,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
    )
    client_data = partitioner.partition(dataset)

    # Build global model
    global_model = model_builder()

    # Create server
    server = FLServer(cfg, global_model)

    # Client trainer
    trainer = FLClientTrainer(cfg)

    # Define client training function
    def train_fn(model, data):
        return trainer.train_from_data(model, data, epochs=cfg.local_epochs,
                                       batch_size=cfg.local_batch_size)

    # Optional evaluation function (must be provided by user)
    eval_fn = None  # user can pass their own

    # Run FL
    result = server.run(client_data, train_fn, eval_fn)

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(output_dir / "fl_results.json", 'w') as f:
            # Convert round history to JSON‑serialisable
            serialisable = {
                "final_accuracy": result.get("accuracy", 0),
                "round_history": [{k: float(v) if isinstance(v, (np.floating, float)) else v
                                   for k, v in r.items()}
                                  for r in result["round_history"]],
            }
            json.dump(serialisable, f, indent=2)

    return result