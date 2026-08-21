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
    def wrap_model(self, model, **kw):
        """Wrap the model with the distributed strategy and return it."""
        ...
    def setup(self):
        """Initialise resources and configuration."""
        pass

    def cleanup(self):
        """Cleanup."""
        if dist.is_initialized():
            dist.destroy_process_group()


@STRATEGY_REGISTRY.register("none")
class NoStrategy(BaseStrategy):
    def wrap_model(self, model, **kw):
        """Wrap model."""
        return model


@STRATEGY_REGISTRY.register("ddp")
class DDPStrategy(BaseStrategy):
    def setup(self):
        """Initialise resources and configuration."""
        os.environ.setdefault("MASTER_ADDR", self.config.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.config.master_port))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def wrap_model(self, model, **kw):
        """Wrap model."""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = model.to(local_rank)
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


@STRATEGY_REGISTRY.register("fsdp")
class FSDPStrategy(BaseStrategy):
    def setup(self):
        """Initialise resources and configuration."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def wrap_model(self, model, **kw):
        """Wrap model."""
        import functools
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.bert.modeling_bert import BertLayer

        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={BertLayer}
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return FSDP(model, auto_wrap_policy=my_auto_wrap_policy, device_id=local_rank)


@STRATEGY_REGISTRY.register("deepspeed")
class DeepSpeedStrategy(BaseStrategy):
    def wrap_model(self, model, **kw):
        """Wrap model."""
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
