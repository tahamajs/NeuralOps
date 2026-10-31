#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Surgery Subpackage.
"""

from neuralops.surgery.scanner import (
    AttentionTargetSpec,
    Conv1DLinearAdapter,
    UniversalArchitectureScanner,
)
from neuralops.surgery.endoskeletal_shell import (
    SidecarGateBranch,
    UniversalAttentionBridge,
    register_neuralops_with_huggingface,
    transfer_pretrained_qkv_weights,
)

from neuralops.surgery.topology_router import (
    AsymmetricCrossAttentionRouter,
    BiDirectionalScanWrapper,
    extract_binary_token_mask,
    sanitize_kv_for_padding_mask,
)

from neuralops.surgery.shadow_twin import (
    check_surgery_divergence,
    collect_surgery_distillation_loss,
    compute_linear_cka_loss,
    compute_operator_cloning_loss,
    enable_shadow_twin,
    finalize_surgery,
    patch_bridge_for_shadow_twin,
    set_surgery_alpha,
    set_surgery_stage,
)

from neuralops.surgery.hf_cache_bridge import (
    NeuralOpsHybridHFCache,
    wrap_model_for_generation,
)

from neuralops.surgery.universal_calibrator import (
    UniversalActivationCalibrator,
    compress_model_universal,
    transplant_and_compress,
)
from neuralops.surgery.param_recovery import get_recovery_params
from neuralops.surgery.surgery_manifest import (
    AttentionSurgeryEntry,
    CompressionSurgeryEntry,
    SurgeryManifest,
    build_manifest_from_model,
    merge_for_inference,
    record_attention_surgery,
    record_compression_surgery,
    replay_surgery_manifest,
)
from neuralops.surgery.surgical_io import (
    extract_surgical_manifest,
    load_surgical_checkpoint,
    save_surgical_checkpoint,
)

from neuralops.api import (
    replace_attention,
    transplant_attention,
)

__all__ = [
    "AttentionTargetSpec",
    "Conv1DLinearAdapter",
    "UniversalArchitectureScanner",
    "SidecarGateBranch",
    "UniversalAttentionBridge",
    "register_neuralops_with_huggingface",
    "transfer_pretrained_qkv_weights",
    "extract_binary_token_mask",
    "sanitize_kv_for_padding_mask",
    "BiDirectionalScanWrapper",
    "AsymmetricCrossAttentionRouter",
    "compute_operator_cloning_loss",
    "compute_linear_cka_loss",
    "enable_shadow_twin",
    "set_surgery_alpha",
    "set_surgery_stage",
    "collect_surgery_distillation_loss",
    "finalize_surgery",
    "patch_bridge_for_shadow_twin",
    "check_surgery_divergence",
    "NeuralOpsHybridHFCache",
    "wrap_model_for_generation",
    "UniversalActivationCalibrator",
    "compress_model_universal",
    "transplant_and_compress",
    "get_recovery_params",
    "SurgeryManifest",
    "AttentionSurgeryEntry",
    "CompressionSurgeryEntry",
    "record_attention_surgery",
    "record_compression_surgery",
    "build_manifest_from_model",
    "replay_surgery_manifest",
    "merge_for_inference",
    "extract_surgical_manifest",
    "save_surgical_checkpoint",
    "load_surgical_checkpoint",
    "replace_attention",
    "transplant_attention",
]
