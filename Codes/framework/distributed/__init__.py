# framework/distributed/__init__.py
from .strategies import (
    DDPStrategy, DeepSpeedStrategy, FSDPStrategy, DistributedConfig
)

# framework/distributed/strategies.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
import torch.distributed.fsdp as fsdp

class DistributedConfig:
    def __init__(self, strategy="ddp", num_gpus=1, **kwargs):
        self.strategy = strategy
        self.num_gpus = num_gpus
        self.extra = kwargs

def wrap_model_with_distributed(model, config: DistributedConfig):
    if config.strategy == "ddp":
        model = DDP(model)
    elif config.strategy == "deepspeed":
        engine, *_ = deepspeed.initialize(model=model, config_params=config.extra)
        model = engine
    elif config.strategy == "fsdp":
        model = fsdp.FullyShardedDataParallel(model)
    return model