"""
Auto-import all attention modules and export their registered classes.
"""

import logging
import os
import sys
import importlib
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["ATTENTION_REGISTRY"]

_curr_dir = os.path.dirname(__file__)
for file in os.listdir(_curr_dir):
    if file.endswith(".py") and not file.startswith(".") and not file.startswith("_"):
        module_name = file[:-3]
        try:
            mod = __import__(f"framework.attentions.{module_name}", fromlist=[module_name])
            __all__.append(module_name)
            for attr in dir(mod):
                if not attr.startswith("_"):
                    val = getattr(mod, attr)
                    if isinstance(val, type):
                        globals()[attr] = val
                        __all__.append(attr)
        except Exception as e:
            logger.warning(f"Skipping attention module '{module_name}' due to: {e}")

# Compatibility for legacy callers that used a capitalized module path on
# case-sensitive systems.  The implementation lives in ``mamba.py``.
try:
    _mamba_mod = importlib.import_module(".mamba", __name__)
    sys.modules.setdefault(f"{__name__}.Mamba", _mamba_mod)
except Exception as e:
    logger.warning("Could not install Mamba compatibility alias: %s", e)

_LEGACY_MAP = {
    "gqa": "gqa_attention",
    "galerkin": "galerkin_attention",
    "galerkin_inject": "galerkin_attention",
    "adaptive_span_inject": "adaptive_span_attention",
    "adaptive_sparse_inject": "adaptive_sparse_attention",
    "adasplash_inject": "adasplash_attention",
    "aegis_hybrid_attention": "aegis_hybrid",
    "cluster_inject": "cluster_attention",
    "compressed_mha": "compressed_mha_attention",
    "cosformer_inject": "cosformer",
    "hla_inject": "hla_attention",
    "kda_inject": "kda_attention",
    "leka_v1_inject": "leka_v1",
    "leka_v2_inject": "leka_v2",
    "leka_v3_inject": "leka_v3",
    "mamba_attention": "mamba_block",
    "moba": "moba_attention",
    "nha_compressor": "nha_attention",
    "rpc": "rpc_attention",
    "salt": "salt_attention",
    "soa_fused_module": "SOAFusedAttention",
    "soa_module": "soa_attn",
    "spectra_v2_attention": "spectra_v2",
    "svd_compressed_attention": "svd_compressed",
    "svd_llm_v2": "svd_llm_v2_attention",
    "svd_proj_attn": "svd_projection_attention",
    "token_compressed": "token_compressed_attention",
    "tt": "token_compressed_attention",
    "upi_mamba": "upi_stuffed_mamba",
}


def __getattr__(name: str):
    if name in _LEGACY_MAP:
        target = _LEGACY_MAP[name]
        mod = importlib.import_module(f".{target}", __name__)
        sys.modules[f"{__name__}.{name}"] = mod
        return mod
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

