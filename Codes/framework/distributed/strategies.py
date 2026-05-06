import torch
import torch.distributed as dist

class DistributedConfig:
    def __init__(self, strategy="ddp", num_gpus=1, **kwargs):
        self.strategy = strategy
        self.num_gpus = num_gpus
        self.extra = kwargs

def wrap_model_with_distributed(model, config: DistributedConfig):
    if not torch.cuda.is_available() or config.strategy == "none":
        return model
    if config.strategy == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.strategy == "fsdp":
        model = torch.distributed.fsdp.FullyShardedDataParallel(model)
    return model

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()