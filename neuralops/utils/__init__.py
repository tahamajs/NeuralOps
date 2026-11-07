#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Utils Subpackage.
"""

from . import (
    architecture,
    checkpoint,
    config,
    device,
    helpers,
    layer_norm_fix,
    logger,
    logging_config,
    metrics,
    paths,
    profiler,
    replace_attention,
    results_saver,
    seed,
    varlen_padding,
)
from .logging_config import configure_logging, warn_once
from .results_saver import ResultsSaver
from .varlen_padding import CuTeTiledVarlenPadding, construct_cu_seqlens_from_lengths, pad_input, unpad_input

__all__ = [
    "architecture",
    "checkpoint",
    "config",
    "device",
    "helpers",
    "layer_norm_fix",
    "logger",
    "metrics",
    "paths",
    "profiler",
    "replace_attention",
    "results_saver",
    "ResultsSaver",
    "seed",
    "varlen_padding",
    "pad_input",
    "unpad_input",
    "construct_cu_seqlens_from_lengths",
    "CuTeTiledVarlenPadding",
]

from .architecture import get_compression_targets, get_hidden_size, get_layer_count, get_model_info  # noqa: E402
from .compat import init_output_scale  # noqa: E402
from .config import merge_configs  # noqa: E402
from .helpers import count_parameters, format_time, load_json, save_json, set_seed, unflatten_dict  # noqa: E402
from .logger import get_logger, set_log_level  # noqa: E402
from .metrics import compute_metrics_for_task  # noqa: E402
from .paths import resolve_path  # noqa: E402
from .profiler import measure_inference_latency, measure_memory_usage, profile_context  # noqa: E402
from .qwen_compat import (  # noqa: E402
    is_architecture_error,
    is_qwen_attention,
    qwen_config_proxy,
    qwen_module_policy,
    qwen_projection,
)

__all__ += [
    "format_time",
    "save_json",
    "load_json",
    "unflatten_dict",
    "resolve_path",
    "merge_configs",
    "set_log_level",
    "get_logger",
    "count_parameters",
    "set_seed",
    "measure_inference_latency",
    "measure_memory_usage",
    "profile_context",
    "get_model_info",
    "get_layer_count",
    "get_hidden_size",
    "get_compression_targets",
    "init_output_scale",
    "is_qwen_attention",
    "qwen_config_proxy",
    "qwen_projection",
    "qwen_module_policy",
    "is_architecture_error",
    "compute_metrics_for_task",
    "CheckpointManager",
]

from neuralops.utils import (
    architecture,
    attention_profiler,
    checkpoint,
    compat,
    config,
    device,
    helpers,
    hf_git_exporter,
    layer_norm_fix,
    logger,
    lora,
    metrics,
    model_family,
    paths,
    profiler,
    qwen_compat,
    replace_attention,
    results_saver,
    seed,
)

from typing import Any

from .checkpoint import CheckpointManager  # noqa: E402


def __getattr__(name: str) -> Any:
    if name == "replace_attention":
        from neuralops.api import replace_attention

        return replace_attention
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
