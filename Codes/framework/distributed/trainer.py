"""DistributedTrainer — the single entry point that wraps any training function."""

import logging
import os

import torch

from .allocator import AllocationStrategy, GPUAllocator
from .config import DistributedConfig
from .strategies import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Wraps a training function with automatic distributed setup.

    Usage:
        def train(model, dataloader, device): ...

        cfg = DistributedConfig(strategy="ddp", num_gpus=2)
        trainer = DistributedTrainer(cfg)
        trainer.run(train, model, dataloader)
    """

    def __init__(self, config: DistributedConfig):
        self.cfg = config
        self._strategy = None
        self._gpu_ids = None

    def run(self, train_fn, *args, **kwargs):
        """Run training with full distributed setup."""
        # 1. Allocate GPUs if not already set
        if self.cfg.auto_discover and not self.cfg.gpu_ids:
            alloc = GPUAllocator()
            strategy = AllocationStrategy(self.cfg.allocation_strategy)
            gpus = alloc.allocate(
                num=self.cfg.num_gpus or alloc.total_gpus,
                strategy=strategy,
                job_id=kwargs.get("job_id", "local"),
                min_mem=self.cfg.min_free_memory_mb,
                prefer_same_node=self.cfg.prefer_same_node,
            )
            self._gpu_ids = [g.index for g in gpus]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self._gpu_ids))
        elif self.cfg.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.cfg.gpu_ids))

        # 2. Setup strategy
        strategy_cls = STRATEGY_REGISTRY.resolve(self.cfg.strategy)
        self._strategy = strategy_cls(self.cfg)
        self._strategy.setup()

        # 3. Wrap model if provided
        model = kwargs.get("model")
        if model is not None:
            kwargs["model"] = self._strategy.wrap_model(model)

        # 4. Set device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        kwargs["device"] = device

        # 5. Run training
        try:
            result = train_fn(*args, **kwargs)
        finally:
            self._strategy.cleanup()
            if self._gpu_ids:
                GPUAllocator().release(kwargs.get("job_id", "local"))

        return result
