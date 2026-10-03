"""
Auto-import all compression_engine modules and export their registered classes.
"""

import logging
import os

from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["COMPRESSOR_REGISTRY"]

_curr_dir = os.path.dirname(__file__)
for file in os.listdir(_curr_dir):
    if file.endswith(".py") and not file.startswith(".") and not file.startswith("_"):
        module_name = file[:-3]
        try:
            mod = __import__(
                f"neuralops.compression_engine.{module_name}", fromlist=[module_name]
            )
            __all__.append(module_name)
            for attr in dir(mod):
                if not attr.startswith("_"):
                    val = getattr(mod, attr)
                    if isinstance(val, type):
                        globals()[attr] = val
                        __all__.append(attr)
        except Exception as e:
            logger.warning(
                f"Skipping compression_engine module '{module_name}' due to: {e}"
            )

import importlib
import sys

_LEGACY_MAP = {
    "aasvd": "aa_svd_compressor",
    "adasvd": "adasvd_compressor",
    "cur_moe": "cur_moe_compressor",
    "frandi": "frandi_compressor",
    "grecodec": "grecodec_compressor",
    "hybridkv": "hybridkv_compressor",
    "kvtc": "kvtc_compressor",
    "modegpt": "modegpt_compressor",
    "oats": "oats_compressor",
    "proxsparse_pruner": "proxsparse",
    "rpca": "rpca_compressor",
    "saes_svd": "saes_svd_v2",
    "saes_svd_compressor": "saes_svd_v2",
    "sdrm": "sdrm_compressor",
    "slicegpt_compressor": "slicegpt",
    "soka": "soka_compressor",
    "swift_svd_compressor": "swift_svd_act",
    "zerokv_neo": "zerokv_neo_compressor",
    "cur_linear": "cur_compressor",
    "kronecker_linear": "kronecker_compressor",
    "tt_linear": "tt_compressor",
    "hashed_linear": "hashed_net_compressor",
    "come": "come_compressor",
    "cure": "cur_moe_compressor",
    "dobi": "dobi_svd",
    "dynamic_entropy": "dynamic_rank",
    "tomoe": "tomoe_compressor",
    "adaptive_compressor": "adaptive_ranks",
    "pruning_utils": "neuralops.core.pruning_utils",
}


def __getattr__(name: str):
    if name in _LEGACY_MAP:
        target = _LEGACY_MAP[name]
        if target.startswith("neuralops."):
            mod = importlib.import_module(target)
        else:
            mod = importlib.import_module(f".{target}", __name__)
        sys.modules[f"{__name__}.{name}"] = mod
        return mod
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
