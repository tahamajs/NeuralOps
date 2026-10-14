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
    "NeuralOpsHybridHFCache",
    "wrap_model_for_generation",
    "UniversalActivationCalibrator",
    "compress_model_universal",
    "transplant_and_compress",
    "replace_attention",
    "transplant_attention",
]
