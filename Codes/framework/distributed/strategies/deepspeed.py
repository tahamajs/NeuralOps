"""DeepSpeed ZeRO strategy – single / multi‑node."""

import os
import torch
from .base import BaseStrategy, STRATEGY_REGISTRY

try:
    import deepspeed
except ImportError:
    deepspeed = None

@STRATEGY_REGISTRY.register("deepspeed")
class DeepSpeedStrategy(BaseStrategy):
    def wrap_model(self, model, **kwargs):
        if deepspeed is None:
            raise ImportError("DeepSpeedStrategy requires the optional 'deepspeed' package.")
        ds_config = self.config.deepspeed_config or {}
        ds_config.setdefault("train_batch_size", "auto")
        ds_config.setdefault("train_micro_batch_size_per_gpu", "auto")
        ds_config.setdefault(
            "optimizer",
            {"type": "AdamW", "params": {"lr": "auto"}},
        )

        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params=ds_config,
        )
        return engine

    def setup(self):
        pass   # DeepSpeed does its own init

    def cleanup(self):
        pass
