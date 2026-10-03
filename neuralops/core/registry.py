"""
Central registry system for the Efficient Transformer Toolkit.
All framework components (attentions, compressors, trainers, visualizers, etc.)
are stored in registries, enabling dynamic discovery and plug‑and‑play integration.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_ATTENTION_FILE_MAP: dict[str, str] = {
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

_COMPRESSOR_FILE_MAP: dict[str, str] = {
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
}


class Registry:
    """A named dictionary that maps string keys to classes or factory functions with capability metadata."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._dict: dict[str, Callable] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._loaded: bool = False

    def register(
        self,
        key: str | None = None,
        supports_causal: bool | None = None,
        supports_bidirectional: bool | None = None,
        supports_cross_attention: bool | None = None,
        supports_gqa: bool | None = None,
        **extra_metadata: Any,
    ) -> Callable:
        """
        Decorator to register a class/factory under the given key with capability flags.

        Usage:
            @ATTENTION_REGISTRY.register("griffin", supports_causal=True, supports_bidirectional=False)
            class GriffinAttention(BaseAttention): ...
        """

        def decorator(obj: Callable) -> Callable:
            final_key = key if key is not None else obj.__name__
            k = final_key.lower()
            # If key already exists and maps to a class, do not let a plain function overwrite it
            if (
                k in self._dict
                and isinstance(self._dict[k], type)
                and not isinstance(obj, type)
            ):
                logger.debug(
                    "Registry '%s': Skipping overwrite of class %s by function %s for key '%s'",
                    self.name,
                    self._dict[k].__name__,
                    obj.__name__,
                    k,
                )
                return obj

            self._dict[k] = obj

            # Derive capabilities from explicit kwargs or attributes on the registered object
            caps = {
                "supports_causal": getattr(
                    obj,
                    "supports_causal",
                    True if supports_causal is None else supports_causal,
                ),
                "supports_bidirectional": getattr(
                    obj,
                    "supports_bidirectional",
                    True if supports_bidirectional is None else supports_bidirectional,
                ),
                "supports_cross_attention": getattr(
                    obj,
                    "supports_cross_attention",
                    (
                        True
                        if supports_cross_attention is None
                        else supports_cross_attention
                    ),
                ),
                "supports_gqa": getattr(
                    obj, "supports_gqa", False if supports_gqa is None else supports_gqa
                ),
            }
            caps.update(extra_metadata)
            self._metadata[k] = caps
            return obj

        return decorator

    def get_metadata(self, key: str) -> dict[str, Any]:
        """Retrieve capability metadata for a registered component."""
        self._try_lazy_load(key)
        k = key.lower()
        if k in self._metadata:
            return dict(self._metadata[k])
        obj = self.get(key)
        if obj is not None:
            return {
                "supports_causal": getattr(obj, "supports_causal", True),
                "supports_bidirectional": getattr(obj, "supports_bidirectional", True),
                "supports_cross_attention": getattr(
                    obj, "supports_cross_attention", True
                ),
                "supports_gqa": getattr(obj, "supports_gqa", False),
            }
        return {}

    def _try_lazy_load(self, key: str) -> None:
        """Attempt to lazily load package modules when a key is queried."""
        k = key.lower()
        if k in self._dict:
            return
        if self.name == "compressors":
            canonical = _COMPRESSOR_FILE_MAP.get(k, k)
            try:
                import neuralops.compression_engine as _ce

                _ = getattr(_ce, key, None)
                if k not in self._dict:
                    import importlib

                    importlib.import_module(f"neuralops.compression_engine.{canonical}")
            except ImportError:
                pass
            except Exception as e:
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise
                logger.warning(
                    "Error during lazy loading of compressor '%s': %s", key, e
                )
        elif self.name == "attentions":
            canonical = _ATTENTION_FILE_MAP.get(k, k)
            try:
                import neuralops.attentions as _att

                _ = getattr(_att, key, None)
                if k not in self._dict:
                    import importlib

                    importlib.import_module(f"neuralops.attentions.{canonical}")
            except ImportError:
                pass
            except Exception as e:
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise
                logger.warning(
                    "Error during lazy loading of attention '%s': %s", key, e
                )

    def get(self, key: str) -> Callable | None:
        """Safely retrieve a registered component, or None if not found."""
        self._try_lazy_load(key)
        return self._dict.get(key.lower())

    def resolve(self, key: str) -> Callable:
        """Retrieve a registered component and raise a descriptive error if missing."""
        self._try_lazy_load(key)
        k = key.lower()
        if k not in self._dict:
            raise KeyError(
                f"Unknown {self.name.rstrip('s')} '{key}'. "
                f"Available: {list(self._dict.keys())}"
            )
        return self._dict[k]

    def _load_all(self) -> None:
        """Attempt to load all modules in the associated package."""
        if self._loaded:
            return
        if self.name == "compressors":
            try:
                pass
            except Exception as e:
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise
                logger.warning("Error loading compression_engine: %s", e)
        elif self.name == "attentions":
            try:
                pass
            except Exception as e:
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise
                logger.warning("Error loading attentions: %s", e)

    def keys(self) -> list[str]:
        """Return a list of all registered keys (in original casing)."""
        self._load_all()
        return list(self._dict.keys())

    def list_available(self) -> list[str]:
        """Return a list of all registered keys."""
        return self.keys()

    def list(self) -> list[str]:
        """Return a list of all registered keys."""
        return self.keys()

    def items(self):
        """Return registered key-value pairs."""
        return self._dict.items()

    def values(self):
        """Return registered values."""
        return self._dict.values()

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, key: str) -> Callable:
        return self.resolve(key)

    def __contains__(self, key: str) -> bool:
        self._try_lazy_load(key)
        return key.lower() in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, {len(self)} items)"


# ----------------------------------------------------------------------
# Global registries – populated by their respective packages
# ----------------------------------------------------------------------
ATTENTION_REGISTRY = Registry("attentions")  # attention modules (Nexus, LEKA, ...)
COMPRESSOR_REGISTRY = Registry("compressors")  # compression strategies (SVD, TT, ...)
METHOD_REGISTRY = Registry("methods")  # training methods (LoRA, R-Drop, ...)
AUX_REGISTRY = Registry(
    "auxiliary"
)  # auxiliary modules (RoPE, SSM, token pruning, ...)

# Optional registries (if needed)
MODEL_REGISTRY = Registry("models")  # different base models
DATASET_REGISTRY = Registry("datasets")  # GLUE, SuperGLUE, etc.
LOSS_REGISTRY = Registry("losses")  # custom loss functions
TRAINER_REGISTRY = Registry("trainers")  # trainer classes
VISUALIZER_REGISTRY = Registry("visualizers")  # visualization utilities


HELPER_SUFFIXES = (
    "_inject",
    "_injection",
    "_fn",
    "_inject_fn",
    "_attn_inject",
    "_module_inject",
    "_compressor_fn",
    "_factory",
    "_quantize",
    "_compressor",
)


def is_helper_module(name: str) -> bool:
    """Returns True if the given module name corresponds to an injection helper or factory alias."""
    if not name:
        return False
    n = name.lower()
    if n.startswith("inject_"):
        return True
    return any(n.endswith(suf) for suf in HELPER_SUFFIXES)


def list_all_components() -> dict:
    """Convenience function that returns all global registry keys."""
    return {
        "attentions": ATTENTION_REGISTRY.keys(),
        "compressors": COMPRESSOR_REGISTRY.keys(),
        "methods": METHOD_REGISTRY.keys(),
        "auxiliary": AUX_REGISTRY.keys(),
        "models": MODEL_REGISTRY.keys(),
        "datasets": DATASET_REGISTRY.keys(),
        "losses": LOSS_REGISTRY.keys(),
        "trainers": TRAINER_REGISTRY.keys(),
        "visualizers": VISUALIZER_REGISTRY.keys(),
    }


def get_evaluatable_modules(include_compressors: bool = True) -> list[str]:
    """Returns all genuine registered attention and compressor architectures, excluding helper aliases."""
    attns = [k for k in ATTENTION_REGISTRY.keys() if not is_helper_module(k)]
    comps = (
        [k for k in COMPRESSOR_REGISTRY.keys() if not is_helper_module(k)]
        if include_compressors
        else []
    )
    return sorted(list(set(attns + comps)))


def get_module_capabilities(name: str) -> dict[str, Any]:
    """Return capability flags for either attention or compressor module."""
    k = name.lower()
    if k in ATTENTION_REGISTRY:
        return ATTENTION_REGISTRY.get_metadata(k)
    if k in COMPRESSOR_REGISTRY:
        return COMPRESSOR_REGISTRY.get_metadata(k)
    return {}


def is_module_declared_compatible(name: str, family: str) -> tuple[bool, str]:
    """Check whether a module's declared capabilities satisfy a model family's architectural requirements."""
    if name == "full":
        return True, ""
    family = family.lower()
    caps = get_module_capabilities(name)
    if not caps:
        return True, ""
    is_encoder = family in {"bert", "roberta", "distilbert"}
    is_decoder = family in {
        "gpt",
        "gpt2",
        "qwen",
        "qwen3_5",
        "llama",
        "mistral",
        "falcon",
    }
    is_enc_dec = family in {"t5"}
    if is_encoder and not caps.get("supports_bidirectional", True):
        return False, "declared_causal_only_incompatible_with_encoder"
    if is_decoder and not caps.get("supports_causal", True):
        return False, "declared_bidirectional_only_incompatible_with_decoder"
    if is_enc_dec and not caps.get("supports_cross_attention", True):
        return False, "declared_lacks_cross_attention_for_encoder_decoder"
    return True, ""
