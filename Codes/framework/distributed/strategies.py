"""Pluggable distributed strategies — DDP, FSDP, DeepSpeed."""

import logging
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from framework.core.registry import Registry

STRATEGY_REGISTRY = Registry("distributed_strategies")
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def wrap_model(self, model, **kw): ...
    def setup(self):
        pass

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()


@STRATEGY_REGISTRY.register("none")
class NoStrategy(BaseStrategy):
    def wrap_model(self, model, **kw):
        return model


@STRATEGY_REGISTRY.register("ddp")
class DDPStrategy(BaseStrategy):
    def setup(self):
        os.environ.setdefault("MASTER_ADDR", self.config.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.config.master_port))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def wrap_model(self, model, **kw):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = model.to(local_rank)
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


@STRATEGY_REGISTRY.register("fsdp")
class FSDPStrategy(BaseStrategy):
    def setup(self):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def wrap_model(self, model, **kw):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.bert.modeling_bert import BertLayer

        return FSDP(model, auto_wrap_policy=transformer_auto_wrap_policy(BertLayer))


@STRATEGY_REGISTRY.register("deepspeed")
class DeepSpeedStrategy(BaseStrategy):
    def wrap_model(self, model, **kw):
        try:
            import deepspeed
        except ImportError:
            raise ImportError("Install deepspeed: pip install deepspeed")
        ds_cfg = self.config.deepspeed_config or {}
        ds_cfg.setdefault("train_batch_size", "auto")
        ds_cfg.setdefault("train_micro_batch_size_per_gpu", "auto")
        ds_cfg.setdefault("optimizer", {"type": "AdamW", "params": {"lr": "auto"}})
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config_params=ds_cfg
        )
        return engine
