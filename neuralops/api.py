# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Public Clean API

Unified public interface enabling drop-in attention replacement, model compression, benchmarking, and export.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    if exc.name == "torch" or (exc.name and exc.name.startswith("torch.")):
        from neuralops.core.exceptions import torch_install_error

        raise torch_install_error("neuralops.api", exc) from exc
    raise

# Dedicated named logger for API operations (Rule & policy requirement)
logger = logging.getLogger("NeuralOps.API")

# Core modular functions re-exported from neuralops.api_modules
from neuralops.api_modules.attention_surgery import (
    compose_attention,
    replace_attention,
    transplant,
    transplant_attention,
)
from neuralops.api_modules.benchmarking import (
    available_backends,
    benchmark,
    benchmark_all,
    certify,
    compare_attentions_ablation,
    get_version,
    has_triton,
    info,
    quick_benchmark,
    validate,
)
from neuralops.api_modules.compression import (
    compress,
    compress_model,
)
from neuralops.api_modules.export import (
    compile_to_so,
    export,
    export_model,
    load_model,
    push_to_hub,
)
from neuralops.api_modules.inference import (
    TextGenerationPipeline,
    pipeline,
    serve,
)
from neuralops.api_modules.registry_ops import (
    get_attention,
    get_attention_class,
    get_attention_module,
    get_compressor_class,
    list_attentions,
    list_compressors,
    list_families,
    probe_optimal_batch_size,
    register_custom_attention,
    register_custom_compressor,
    scaffold_module,
    unregister_custom_module,
    validate_module_contract,
)
from neuralops.api_modules.shadow_twin import (
    collect_surgery_distillation_loss,
    finalize_surgery,
    set_surgery_alpha,
    set_surgery_stage,
)
from neuralops.api_modules.training import (
    distill,
    evaluate,
    heal_distributed,
    load_and_prepare_dataset,
    surgical_transplant_and_heal_pipeline,
    train,
    train_grpo,
)

# Canonical universal factories from core.registry
from neuralops.core.registry import create_attention, create_compressor
from neuralops.export.exporter import load_exported_model

# Downstream submodule imports for canonical re-exports
from neuralops.analysis.inspector import compare, summary
from neuralops.core.state_cache import NeuralOpsStateCache, benchmark_autoregressive_decoding
from neuralops.core.triton_kernels.autotuner import select_optimal_triton_config
from neuralops.export.surgery_manifest import merge_for_inference
from neuralops.recipes import oneshot, recommend
from neuralops.trainer import NeuralOpsTrainer
from neuralops.transformers import (
    AutoNeuralOpsForCausalLM,
    AutoNeuralOpsForSequenceClassification,
    AutoNeuralOpsModel,
    patch_transformers,
)
from neuralops.utils.hub_integration import load_attention_from_hub, save_attention_only

__all__ = [
    "load_model",
    "replace_attention",
    "transplant_attention",
    "transplant",
    "compose_attention",
    "probe_optimal_batch_size",
    "serve",
    "compress",
    "get_attention_module",
    "quick_benchmark",
    "export_model",
    "export",
    "push_to_hub",
    "load_exported_model",
    "collect_surgery_distillation_loss",
    "set_surgery_alpha",
    "set_surgery_stage",
    "finalize_surgery",
    "list_attentions",
    "list_compressors",
    "get_attention_class",
    "get_compressor_class",
    "create_attention",
    "create_compressor",
    "list_families",
    "register_custom_attention",
    "register_custom_compressor",
    "unregister_custom_module",
    "validate_module_contract",
    "scaffold_module",
    "compare_attentions_ablation",
    "surgical_transplant_and_heal_pipeline",
    "heal_distributed",
    "train",
    "distill",
    "evaluate",
    "load_and_prepare_dataset",
    "train_grpo",
    "benchmark",
    "benchmark_all",
    "info",
    "available_backends",
    "has_triton",
    "validate",
    "certify",
    "get_version",
    "NeuralOpsStateCache",
    "benchmark_autoregressive_decoding",
    "select_optimal_triton_config",
    "oneshot",
    "recommend",
    "summary",
    "compare",
    "patch_transformers",
    "AutoNeuralOpsModel",
    "AutoNeuralOpsForCausalLM",
    "AutoNeuralOpsForSequenceClassification",
    "merge_for_inference",
    "NeuralOpsTrainer",
    "save_attention_only",
    "load_attention_from_hub",
    "pipeline",
    "TextGenerationPipeline",
]

# Canonical Public API symbol mapping (F-79 single source of truth)
_PUBLIC_API: Dict[str, str] = {name: "neuralops.api" for name in __all__}
