"""Integration with Django runner – called by MegaOrchestrator."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from .client import FLClientTrainer
from .config import FLConfig
from .data_partitioning import DataPartitioner
from .server import FLServer


def run_federated_experiment(
    config_dict: Dict[str, Any],
    model_builder: Callable[[], torch.nn.Module],
    dataset: Any,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run federated experiment."""
    cfg = FLConfig(
        **{k: v for k, v in config_dict.items() if k in FLConfig.__dataclass_fields__}
    )
    partitioner = DataPartitioner(
        num_clients=cfg.num_clients,
        partition_type=cfg.partition_type,
        alpha=cfg.non_iid_alpha,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
    )
    client_data = partitioner.partition(dataset)
    global_model = model_builder()
    server = FLServer(cfg, global_model)
    trainer = FLClientTrainer(cfg)

    def train_fn(model, data) -> Any:
        """Train fn."""
        return trainer.train_from_data(
            model, data, epochs=cfg.local_epochs, batch_size=cfg.local_batch_size
        )

    result = server.run(client_data, train_fn)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        import json

        # Convert any numpy types to plain floats
        serialisable = {
            "final_accuracy": result.get("accuracy", 0),
            "round_history": [
                {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in r.items()
                }
                for r in result.get("round_history", [])
            ],
        }
        with open(output_dir / "fl_results.json", "w") as f:
            json.dump(serialisable, f, indent=2)
    return result
