"""FSDP – Fully Sharded Data Parallel (PyTorch 1.12+)."""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from .base import BaseStrategy, STRATEGY_REGISTRY

@STRATEGY_REGISTRY.register("fsdp")
class FSDPStrategy(BaseStrategy):
    def setup(self):
        dist.init_process_group(backend="nccl")

    def wrap_model(self, model, **kwargs):
        return FSDP(model)

    def cleanup(self):
        dist.destroy_process_group()