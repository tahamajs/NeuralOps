"""FSDP – Fully Sharded Data Parallel (PyTorch 1.12+)."""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .base import STRATEGY_REGISTRY, BaseStrategy


@STRATEGY_REGISTRY.register("fsdp")
class FSDPStrategy(BaseStrategy):
    def setup(self) -> None:
        """Initialise resources and configuration."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def wrap_model(self, model, **kwargs):
        # Automatically wrap transformer layers
        """Wrap model."""
        from transformers.models.bert.modeling_bert import BertLayer  # example

        auto_wrap_policy = transformer_auto_wrap_policy(BertLayer)
        return FSDP(model, auto_wrap_policy=auto_wrap_policy)

    def cleanup(self) -> None:
        """Cleanup."""
        if dist.is_initialized():
            dist.destroy_process_group()


import os  # needed in FSDP strategy
