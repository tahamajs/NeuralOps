"""Elastic training — dynamic scaling with checkpoint support."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ElasticConfig:
    min_nodes: int = 1
    max_nodes: int = 8
    nproc_per_node: int = 1
    rdzv_backend: str = "c10d"
    rdzv_endpoint: str = "localhost:29500"
    run_id: str = "elastic-run"
    checkpoint_dir: str = "./elastic_checkpoints"
    checkpoint_every_steps: int = 500
    max_restarts: int = 3


@dataclass
class ElasticState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "best_metric": self.best_metric,
                },
                f,
            )

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            d = json.load(f)
        self.epoch = d["epoch"]
        self.global_step = d["global_step"]
        self.best_metric = d["best_metric"]


class ElasticTrainer:
    """Wraps a training loop with elastic scaling and checkpointing."""

    def __init__(self, config: ElasticConfig = None):
        self.cfg = config or ElasticConfig()
        self.state = ElasticState()
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer) -> str:
        path = os.path.join(
            self.cfg.checkpoint_dir, f"ckpt_step_{self.state.global_step}.pt"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "state": self.state,
            },
            path,
        )
        logger.info("Elastic checkpoint saved to %s", path)
        return path

    def load_checkpoint(self, model, optimizer) -> Optional[dict]:
        if not os.path.isdir(self.cfg.checkpoint_dir):
            return None
        files = sorted(
            [f for f in os.listdir(self.cfg.checkpoint_dir) if f.endswith(".pt")],
            reverse=True,
        )
        if not files:
            return None
        ckpt = torch.load(
            os.path.join(self.cfg.checkpoint_dir, files[0]), map_location="cpu"
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        self.state = ckpt["state"]
        logger.info("Loaded elastic checkpoint from %s", files[0])
        return ckpt
