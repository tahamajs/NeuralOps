#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Lazy public exports for optional framework integrations."""

from importlib import import_module

_EXPORTS = {
    "register_hf_automodel": ("hf_automodel", "register_hf_automodel"),
    "AutoNeuralOpsModel": ("hf_automodel", "AutoNeuralOpsModel"),
    "AutoNeuralOpsForCausalLM": ("hf_automodel", "AutoNeuralOpsForCausalLM"),
    "AutoNeuralOpsForSequenceClassification": ("hf_automodel", "AutoNeuralOpsForSequenceClassification"),
    "patch_model_attention": ("hf_automodel", "patch_model_attention"),
    "patch_transformers": ("hf_automodel", "patch_transformers"),
    "compile_attention_module": ("distributed_compile", "compile_attention_module"),
    "get_deepspeed_config": ("distributed_compile", "get_deepspeed_config"),
    "apply_fsdp_wrapping": ("distributed_compile", "apply_fsdp_wrapping"),
    "NeuralOpsvLLMMetadata": ("vllm", "NeuralOpsvLLMMetadata"),
    "NeuralOpsvLLMImpl": ("vllm", "NeuralOpsvLLMImpl"),
    "NeuralOpsvLLMAttentionBackend": ("vllm", "NeuralOpsvLLMAttentionBackend"),
    "NeuralOpsForCausalLMvLLM": ("vllm", "NeuralOpsForCausalLMvLLM"),
    "register_neuralops_vllm_plugin": ("vllm", "register_neuralops_vllm_plugin"),
    "VLLMAttentionAdapter": ("serving", "VLLMAttentionAdapter"),
    "TGIAttentionAdapter": ("serving", "TGIAttentionAdapter"),
    "SGLangRadixLinearCacheAdapter": ("serving", "SGLangRadixLinearCacheAdapter"),
    "TGIServingAdapter": ("tgi", "TGIServingAdapter"),
    "TGIRequest": ("tgi", "TGIRequest"),
    "TGIRequestBatch": ("tgi", "TGIRequestBatch"),
    "TGIRequestSlot": ("tgi", "TGIRequestSlot"),
    "TGIStreamingIterator": ("tgi", "TGIStreamingIterator"),
    "SGLangRadixAdapter": ("sglang", "SGLangRadixAdapter"),
    "RadixPrefixNode": ("sglang", "RadixPrefixNode"),
    "TensorRTLLMPluginBuilder": ("tensorrt_llm", "TensorRTLLMPluginBuilder"),
    "CompressedTensorsExporter": ("compressed_tensors", "CompressedTensorsExporter"),
    "NeuralOpsLinearMethod": ("vllm_plugin", "NeuralOpsLinearMethod"),
    "register_neuralops_linear_method": ("vllm_plugin", "register_neuralops_linear_method"),
}
_MODULES = {
    "compressed_tensors", "distributed_compile", "hf_automodel", "serving",
    "sglang", "tensorrt_llm", "tgi", "vllm", "vllm_plugin",
}

__all__ = [*_EXPORTS, *_MODULES]


def __getattr__(name: str):
    """Import framework-specific dependencies only when an integration is used."""
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    elif name in _MODULES:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
