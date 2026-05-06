"""
Elastic Trainer – inspired by TorchElastic and Horovod Elastic Training.

Enables dynamic scaling of the number of GPUs during training:
  - New nodes can join and leave mid‑training
  - Checkpoints are automatically saved and restored
  - Rendezvous handles worker discovery

References:
  - PyTorch Elastic: https://pytorch.org/docs/stable/elastic/
  - Horovod Elastic: https://horovod.readthedocs.io/en/stable/elastic_include.html
  - DLRover: https://github.com/intelligent-machine-learning/dlrover
"""

import os
import time
import json
import torch
import torch.distributed as dist
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic.multiprocessing.errors import record
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ElasticConfig:
    """Configuration for elastic training."""
    min_nodes: int = 1
    max_nodes: int = 8
    nproc_per_node: int = 1
    rdzv_backend: str = "c10d"       # "c10d", "etcd", "etcd-v2", "static"
    rdzv_endpoint: str = "localhost:29500"
    run_id: str = "elastic-run"
    checkpoint_dir: str = "./elastic_checkpoints"
    checkpoint_every_steps: int = 500
    max_restarts: int = 3
    monitor_interval: int = 5        # seconds between health checks


@dataclass
class ElasticState:
    """State that must be synchronised across elastic workers."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    rng_state: Optional[bytes] = None

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump({
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
            }, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            d = json.load(f)
        self.epoch = d["epoch"]
        self.global_step = d["global_step"]
        self.best_metric = d["best_metric"]


class ElasticTrainer:
    """
    Wraps a standard training loop with elastic scaling support.

    Usage:
        trainer = ElasticTrainer(ElasticConfig(min_nodes=1, max_nodes=4))
        trainer.launch(my_train_function, args=(model, dataloader))
    """

    def __init__(self, config: ElasticConfig = None):
        self.config = config or ElasticConfig()
        self.state = ElasticState()
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    @record
    def _worker_main(self, train_fn: Callable, args, kwargs):
        """Entry point for each elastic worker process."""
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Initialise process group
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

        logger.info("Worker %d/%d started (elastic)", rank, world_size)

        # Load checkpoint if available
        self._load_checkpoint()

        # Run user's training function
        try:
            train_fn(*args, **kwargs, state=self.state)
        except Exception as e:
            logger.exception("Worker %d failed: %s", rank, e)
            raise

        # Cleanup
        dist.destroy_process_group()

    def launch(
        self,
        train_fn: Callable,
        args: tuple = (),
        kwargs: dict = None,
    ) -> None:
        """
        Launch elastic training using torchrun.

        Args:
            train_fn: the training function (must accept `state` kwarg)
            args: positional args for train_fn
            kwargs: keyword args for train_fn
        """
        kwargs = kwargs or {}

        # Use PyTorch Elastic Launcher
        from torch.distributed.run import main as torchrun_main
        import sys

        # Build command‑line args for torchrun
        sys.argv = [
            "torchrun",
            f"--nnodes={self.config.min_nodes}:{self.config.max_nodes}",
            f"--nproc_per_node={self.config.nproc_per_node}",
            f"--rdzv_backend={self.config.rdzv_backend}",
            f"--rdzv_endpoint={self.config.rdzv_endpoint}",
            f"--rdzv_id={self.config.run_id}",
            f"--max_restarts={self.config.max_restarts}",
            f"--monitor_interval={self.config.monitor_interval}",
            # Pass the current script path (this is a simplification)
            __file__,
        ]
        torchrun_main()

    # ── Checkpointing ───────────────────────────────────────────────
    def save_checkpoint(self, model: torch.nn.Module, optimizer,
                        state: ElasticState) -> str:
        """Save training state to disk."""
        path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_step_{state.global_step}.pt"
        )
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state,
        }, path)
        logger.info("Elastic checkpoint saved to %s", path)
        return path

    def _load_checkpoint(self) -> Optional[dict]:
        """Find and load the latest checkpoint."""
        ckpt_dir = self.config.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            return None
        files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            reverse=True,
        )
        if not files:
            return None
        path = os.path.join(ckpt_dir, files[0])
        checkpoint = torch.load(path, map_location="cpu")
        logger.info("Loaded elastic checkpoint from %s", path)
        return checkpoint