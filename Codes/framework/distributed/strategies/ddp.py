"""PyTorch DistributedDataParallel – single machine, multi‑GPU."""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .base import BaseStrategy, STRATEGY_REGISTRY

@STRATEGY_REGISTRY.register("ddp")
class DDPStrategy(BaseStrategy):
    def setup(self):
        """Called once per process to initialise the process group."""
        os.environ.setdefault("MASTER_ADDR", self.config.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.config.master_port))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def wrap_model(self, model, **kwargs):
        device_id = int(os.environ.get("LOCAL_RANK", 0))
        model = model.to(device_id)
        return DDP(model, device_ids=[device_id], output_device=device_id)

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()