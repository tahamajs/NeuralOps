# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Integrations Subpackage.
"""

from __future__ import annotations

from neuralops.integrations.hf_automodel import (
    register_hf_automodel,
    AutoNeuralOpsModel,
    AutoNeuralOpsForCausalLM,
    AutoNeuralOpsForSequenceClassification,
    patch_model_attention,
    patch_transformers,
)
from neuralops.integrations.distributed_compile import (
    compile_attention_module,
    get_deepspeed_config,
    apply_fsdp_wrapping,
)
from neuralops.integrations.vllm import (
    NeuralOpsvLLMMetadata,
    NeuralOpsvLLMImpl,
    NeuralOpsvLLMAttentionBackend,
    NeuralOpsForCausalLMvLLM,
    register_neuralops_vllm_plugin,
)
from neuralops.integrations.serving import (
    VLLMAttentionAdapter,
    TGIAttentionAdapter,
    SGLangRadixLinearCacheAdapter,
)
from neuralops.integrations.tgi import (
    TGIServingAdapter,
    TGIRequest,
    TGIRequestBatch,
    TGIRequestSlot,
    TGIStreamingIterator,
)
from neuralops.integrations.sglang import (
    SGLangRadixAdapter,
    RadixPrefixNode,
)
from neuralops.integrations.tensorrt_llm import (
    TensorRTLLMPluginBuilder,
)

__all__ = [
    "register_hf_automodel",
    "AutoNeuralOpsModel",
    "AutoNeuralOpsForCausalLM",
    "AutoNeuralOpsForSequenceClassification",
    "patch_model_attention",
    "patch_transformers",
    "compile_attention_module",
    "get_deepspeed_config",
    "apply_fsdp_wrapping",
    "NeuralOpsvLLMMetadata",
    "NeuralOpsvLLMImpl",
    "NeuralOpsvLLMAttentionBackend",
    "NeuralOpsForCausalLMvLLM",
    "register_neuralops_vllm_plugin",
    "VLLMAttentionAdapter",
    "TGIAttentionAdapter",
    "SGLangRadixLinearCacheAdapter",
    "TGIServingAdapter",
    "TGIRequest",
    "TGIRequestBatch",
    "TGIRequestSlot",
    "TGIStreamingIterator",
    "SGLangRadixAdapter",
    "RadixPrefixNode",
    "TensorRTLLMPluginBuilder",
    "CompressedTensorsExporter",
    "NeuralOpsLinearMethod",
    "register_neuralops_linear_method",
]

from neuralops.integrations.compressed_tensors import CompressedTensorsExporter
from neuralops.integrations.vllm_plugin import NeuralOpsLinearMethod, register_neuralops_linear_method

from neuralops.integrations import compressed_tensors
from neuralops.integrations import distributed_compile
from neuralops.integrations import hf_automodel
from neuralops.integrations import serving
from neuralops.integrations import vllm
from neuralops.integrations import vllm_plugin
from neuralops.integrations import tgi
from neuralops.integrations import sglang
from neuralops.integrations import tensorrt_llm
