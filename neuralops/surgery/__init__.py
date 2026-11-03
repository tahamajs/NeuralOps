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

from typing import Any

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
    HFCacheBridge,
    NeuralOpsHybridHFCache,
    PolymorphicGenerationCacheBridge,
    wrap_model_for_generation,
)

from neuralops.surgery.director import (
    ArchitectureProfile,
    CrossAttentionLinearBridge,
    GatheredParametersContext,
    InvariantPreservingLinearBridge,
    UniversalSurgeryDirector,
    apply_rope_pre_feature_map,
    apply_rotary_pos_emb,
    compute_decayed_state_bound,
    profile_model_topology,
    re_tie_embeddings,
    verify_tied_embeddings,
)
from neuralops.surgery.fx_surgery import (
    FXAttentionGraphRewriter,
    NeuralOpsAttentionTracer,
    compile_linear_surgery,
    rewriter_linear_attention_fx,
)
from neuralops.surgery.streaming_distill import (
    CheckpointedTeacherBridge,
    MicroChunkedShadowTwinDistiller,
    StreamingCKALoss,
    get_vram_utilization,
)
from neuralops.surgery.mla_surgery import (
    DeepSeekMLASurgeryBridge,
    DualStreamLatentState,
)
from neuralops.surgery.moa_router import (
    AttentionEntropyProfile,
    MoARouter,
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
from neuralops.surgery.healing import (
    calibrate_attention_subspace,
    heal_replaced_model,
)

from neuralops.surgery.model_surgery import ModelSurgery
from neuralops.surgery.registry import (
    SURGERY_REGISTRY,
    SurgeryMetadata,
    get_surgery,
    get_surgeries_for_model,
    list_surgeries,
    register_surgery,
)
from neuralops.surgery.pipeline import (
    ApplyReport,
    SurgeryOp,
    SurgeryRecipe,
)
from neuralops.surgery.validation import (
    CheckResult,
    ValidationReport,
    validate_before_surgery,
)
from neuralops.surgery.recipe import (
    load_recipe,
    save_recipe,
    save_recipe_to_hub,
    validate_recipe_schema,
)
from neuralops.surgery.attention.replace import replace_attention
from neuralops.surgery.mlp.shrink import shrink_mlp
from neuralops.surgery.layers.prune import prune_layers
from neuralops.surgery.weights.quantize import quantize_int4
from neuralops.surgery.recovery.distill import distill_from_teacher

def apply_recipe(model: Any, recipe: SurgeryRecipe, **kwargs: Any) -> tuple[Any, ApplyReport]:
    """Execute a validated surgery recipe onto target model."""
    return recipe.apply(model, **kwargs)


__all__ = [
    "apply_recipe",
    "SURGERY_REGISTRY",
    "SurgeryMetadata",
    "register_surgery",
    "list_surgeries",
    "get_surgery",
    "get_surgeries_for_model",
    "SurgeryOp",
    "SurgeryRecipe",
    "ApplyReport",
    "CheckResult",
    "ValidationReport",
    "validate_before_surgery",
    "load_recipe",
    "save_recipe",
    "save_recipe_to_hub",
    "validate_recipe_schema",
    "replace_attention",
    "shrink_mlp",
    "prune_layers",
    "quantize_int4",
    "distill_from_teacher",
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
    "HFCacheBridge",
    "NeuralOpsHybridHFCache",
    "PolymorphicGenerationCacheBridge",
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
    "heal_replaced_model",
    "calibrate_attention_subspace",
    "ModelSurgery",
    "ArchitectureProfile",
    "CrossAttentionLinearBridge",
    "GatheredParametersContext",
    "InvariantPreservingLinearBridge",
    "UniversalSurgeryDirector",
    "apply_rope_pre_feature_map",
    "apply_rotary_pos_emb",
    "compute_decayed_state_bound",
    "profile_model_topology",
    "re_tie_embeddings",
    "verify_tied_embeddings",
    "FXAttentionGraphRewriter",
    "NeuralOpsAttentionTracer",
    "compile_linear_surgery",
    "rewriter_linear_attention_fx",
    "CheckpointedTeacherBridge",
    "MicroChunkedShadowTwinDistiller",
    "StreamingCKALoss",
    "get_vram_utilization",
    "DeepSeekMLASurgeryBridge",
    "DualStreamLatentState",
    "AttentionEntropyProfile",
    "MoARouter",
]
