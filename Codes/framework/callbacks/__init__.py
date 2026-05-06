# framework/callbacks/__init__.py (append at end)

import torch
from framework.core.registry import Registry

CALLBACK_REGISTRY = Registry("callback")

# We'll create factory functions for each callback that accept config kwargs.

def _make_callback(name, cls):
    CALLBACK_REGISTRY.register(name)(lambda **kw: cls(**kw))

from .resources import MemoryCallback, TimeCallback, GPUStatsCallback
from .training import EarlyStoppingCallback, CheckpointCallback, LRSchedulerCallback
from .monitoring import WandbCallback, TensorBoardCallback, TerminalCallback, WebSocketCallback
from .analysis import SVDAttentionCallback, AttentionHeatmapCallback, GradientNormCallback, LayerwiseAnalysisCallback, ActivationStatsCallback
from .custom import EMACallback, SWACallback, GradientClippingCallback, FreezeLayersCallback, UnfreezeScheduleCallback, NanCheckerCallback, ActivationCheckpointingCallback

_make_callback("memory", MemoryCallback)
_make_callback("time", TimeCallback)
_make_callback("gpu_stats", GPUStatsCallback)
_make_callback("early_stopping", EarlyStoppingCallback)
_make_callback("checkpoint", CheckpointCallback)
_make_callback("lr_scheduler", LRSchedulerCallback)
_make_callback("wandb", WandbCallback)
_make_callback("tensorboard", TensorBoardCallback)
_make_callback("terminal", TerminalCallback)
_make_callback("websocket", WebSocketCallback)
_make_callback("svd_attention", SVDAttentionCallback)
_make_callback("attention_heatmap", AttentionHeatmapCallback)
_make_callback("gradient_norm", GradientNormCallback)
_make_callback("layerwise_analysis", LayerwiseAnalysisCallback)
_make_callback("activation_stats", ActivationStatsCallback)
_make_callback("ema", EMACallback)
_make_callback("swa", SWACallback)
_make_callback("gradient_clipping", GradientClippingCallback)
_make_callback("freeze_layers", FreezeLayersCallback)
_make_callback("unfreeze_schedule", UnfreezeScheduleCallback)
_make_callback("nan_checker", NanCheckerCallback)
_make_callback("activation_checkpointing", ActivationCheckpointingCallback)