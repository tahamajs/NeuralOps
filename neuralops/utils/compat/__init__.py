"""Family-aware, opt-in compatibility helpers for attention modules."""

from __future__ import annotations

from .aliases import install_projection_aliases
from .architectures import (
    ALLOWED_FFN_PATTERNS,
    SENSITIVE_PATTERNS,
    FamilyModuleWrapper,
    causal_mask_from_attention_mask,
    convert_mask,
    fp32_linalg,
    get_dynamic_compression_rank,
    init_output_scale,
    is_decoder_family,
    is_decoder_projection_name,
    maybe_apply_rotary,
    module_family_support,
    repeat_kv,
    should_compress_linear,
    should_compress_qwen3_5,
    wrap_for_family,
)
from .param_recovery import (
    PROTECTED_KEYWORDS,
    SVD_FACTOR_KEYWORDS,
    _is_protected_param,
    get_recovery_params,
)

__all__ = [
    "ALLOWED_FFN_PATTERNS",
    "FamilyModuleWrapper",
    "PROTECTED_KEYWORDS",
    "SENSITIVE_PATTERNS",
    "SVD_FACTOR_KEYWORDS",
    "_is_protected_param",
    "causal_mask_from_attention_mask",
    "convert_mask",
    "fp32_linalg",
    "get_dynamic_compression_rank",
    "get_recovery_params",
    "init_output_scale",
    "install_projection_aliases",
    "is_decoder_family",
    "is_decoder_projection_name",
    "maybe_apply_rotary",
    "module_family_support",
    "repeat_kv",
    "should_compress_linear",
    "should_compress_qwen3_5",
    "wrap_for_family",
]

# Auto-wired package members
from neuralops.utils.compat import aliases
from neuralops.utils.compat import architectures
from neuralops.utils.compat import param_recovery

from neuralops.utils.compat import aliases
from neuralops.utils.compat import architectures
from neuralops.utils.compat import param_recovery
