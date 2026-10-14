"""
neuralops/core/legacy_aliases.py

Deprecated attention mechanism and compressor aliases for backward compatibility.
All canonical attention/compressor keys are in registry.py.

DO NOT add new aliases here. This module will be removed in v2.0.
"""
import warnings

# Legacy attention key aliases (deprecated string -> canonical string)
LEGACY_ATTENTION_ALIASES: dict[str, str] = {
    # spectron variants
    "aegis_spectron_fp8": "spectron_fp8",
    "aegis_spectron": "spectron",
    # svd variants
    "svd_linear_attn": "svd_linear_attention",
    "triton_svd": "triton_svd_linear_attention",
    # gated variants
    "decay_la": "decay_linear_attention",
    "gated_la": "gated_linear_attention",
    # ssm variants
    "ssm_mamba": "mamba_block",
    "selective_state": "selective_ssm",
}

# Legacy compressor key aliases
LEGACY_COMPRESSOR_ALIASES: dict[str, str] = {}


def get_canonical_attention_key(key: str) -> str | None:
    """Returns canonical key if the given key is a deprecated alias, else None."""
    canonical = LEGACY_ATTENTION_ALIASES.get(key)
    if canonical is not None:
        warnings.warn(
            f"'{key}' is deprecated. Use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return canonical


def get_canonical_compressor_key(key: str) -> str | None:
    """Returns canonical key if the given key is a deprecated alias, else None."""
    canonical = LEGACY_COMPRESSOR_ALIASES.get(key)
    if canonical is not None:
        warnings.warn(
            f"'{key}' is deprecated. Use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return canonical
