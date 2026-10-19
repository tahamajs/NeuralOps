#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Models Core Modules.
"""

from __future__ import annotations

import logging
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from neuralops.models.configuration import NeuralOpsConfig
from neuralops.models.modeling import (
    NeuralOpsPreTrainedModel,
    NeuralOpsModel,
    NeuralOpsForCausalLM,
)
from neuralops.models.modeling_block import NeuralOpsBlock
from neuralops.models.modeling_rope import NeuralOpsRotaryEmbedding
from neuralops.models.modeling_norm import NeuralOpsRMSNorm, build_norm_layer
from neuralops.models.modeling_mlp import NeuralOpsSwiGLU, build_mlp
from neuralops.models.hybrid_spec import normalize_hybrid_spec, get_layer_spec
from neuralops.models.hub import (
    scan_for_secrets,
    generate_model_card,
    push_model_to_hub,
    load_model_from_hub,
    list_org_models,
)

from neuralops.models.cache import UnifiedHybridCache
from neuralops.models.cuda_graph import CUDAGraphDecodeRunner
from neuralops.compression_engine.svd.direct_svd import merge_for_inference
from neuralops.models.paged_state import PagedStateTable, RadixPrefixStateCache

logger = logging.getLogger(__name__)

__all__ = [
    "PagedStateTable",
    "RadixPrefixStateCache",
    "UnifiedHybridCache",
    "CUDAGraphDecodeRunner",
    "merge_for_inference",

    "NeuralOpsConfig",
    "NeuralOpsPreTrainedModel",
    "NeuralOpsModel",
    "NeuralOpsForCausalLM",
    "NeuralOpsBlock",
    "NeuralOpsRotaryEmbedding",
    "NeuralOpsRMSNorm",
    "build_norm_layer",
    "NeuralOpsSwiGLU",
    "build_mlp",
    "normalize_hybrid_spec",
    "get_layer_spec",
    "scan_for_secrets",
    "generate_model_card",
    "push_model_to_hub",
    "load_model_from_hub",
    "list_org_models",
]


def _register_with_transformers() -> None:
    """Idempotently register NeuralOps architectures with Transformers Auto classes."""
    try:
        AutoConfig.register("neuralops", NeuralOpsConfig, exist_ok=True)
    except Exception as exc:
        logger.debug("AutoConfig registration note: %s", exc)

    try:
        AutoModel.register(NeuralOpsConfig, NeuralOpsModel, exist_ok=True)
    except Exception as exc:
        logger.debug("AutoModel registration note: %s", exc)

    try:
        AutoModelForCausalLM.register(NeuralOpsConfig, NeuralOpsForCausalLM, exist_ok=True)
    except Exception as exc:
        logger.debug("AutoModelForCausalLM registration note: %s", exc)


_register_with_transformers()

from neuralops.models.gla_hf_model import GLAConfig, GLAModel, GLABlock

__all__ += ["GLAConfig", "GLAModel", "GLABlock"]

try:
    AutoConfig.register("gla", GLAConfig, exist_ok=True)
    AutoModel.register(GLAConfig, GLAModel, exist_ok=True)
    AutoModelForCausalLM.register(GLAConfig, GLAModel, exist_ok=True)
except Exception as exc:
    logger.debug("GLA Auto registration note: %s", exc)

from neuralops.export.exporter import load_model
__all__ += ["load_model"]
