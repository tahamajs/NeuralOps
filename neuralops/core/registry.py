"""
Central Component Registry

Decorators and registry catalogs managing discovery and instantiation of attention and compression modules.

Author: Taha Majlesi
"""

from __future__ import annotations

import builtins
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

DEPRECATED_ALIASES: dict[str, str] = {
    "chela_slay_attention": "chela_slay",
    "cosformer_attention": "cosformer",
    "decay_linear_attn": "decay_linear_attention",
    "eat_attention": "eat",
    "gated_linear_attn": "gated_linear_attention",
    "hgrn2_outer": "hgrn2_gated_outer",
}

from neuralops.core.legacy_aliases import (
    DEPRECATED_ALIASES as _DEPRECATED_ALIASES,
    handle_deprecated_alias as _handle_deprecated,
    ATTENTION_FILE_MAP as _ATTENTION_FILE_MAP,
    COMPRESSOR_FILE_MAP as _COMPRESSOR_FILE_MAP,
)


_UNIVERSAL_WRAPPER_CACHE: dict[type, type] = {}


def _make_universal_attention_class(target_cls: type) -> type:
    """Wraps an attention class so direct instantiation cls(dim=256, num_heads=4) works seamlessly."""
    if not isinstance(target_cls, type):
        return target_cls
    if getattr(target_cls, "_is_universal_attention_wrapped", False):
        return target_cls
    if target_cls in _UNIVERSAL_WRAPPER_CACHE:
        return _UNIVERSAL_WRAPPER_CACHE[target_cls]

    class UniversalAttentionWrapper(target_cls):
        _is_universal_attention_wrapped = True

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            args_list = list(args)
            resolved_dim = kwargs.get("dim") or kwargs.get("hidden_size") or kwargs.get("d_model") or kwargs.get("embed_dim")
            resolved_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads") or kwargs.get("n_head")

            if len(args_list) > 0 and isinstance(args_list[0], int):
                resolved_dim = args_list.pop(0)
                if len(args_list) > 0 and isinstance(args_list[0], int):
                    resolved_heads = args_list.pop(0)

            config = kwargs.get("config")
            if len(args_list) > 0 and not isinstance(args_list[0], int):
                config = args_list[0]

            if resolved_dim is not None:
                if config is None:
                    from transformers import BertConfig
                    h = resolved_heads or max(1, resolved_dim // 64)
                    config = BertConfig(hidden_size=resolved_dim, num_attention_heads=h)
                else:
                    try:
                        config.hidden_size = resolved_dim
                        config.n_embd = resolved_dim
                        config.d_model = resolved_dim
                        if resolved_heads:
                            config.num_attention_heads = resolved_heads
                            config.num_heads = resolved_heads
                    except Exception:
                        pass
                kwargs["config"] = config
                kwargs["hidden_size"] = resolved_dim
                kwargs["dim"] = resolved_dim
                if resolved_heads:
                    kwargs["num_heads"] = resolved_heads
                    kwargs["num_attention_heads"] = resolved_heads

            try:
                return super().__new__(cls)
            except Exception:
                return target_cls.__new__(target_cls)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            args_list = list(args)
            resolved_dim = kwargs.get("dim") or kwargs.get("hidden_size") or kwargs.get("d_model") or kwargs.get("embed_dim")
            resolved_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads") or kwargs.get("n_head")

            if len(args_list) > 0 and isinstance(args_list[0], int):
                resolved_dim = args_list.pop(0)
                if len(args_list) > 0 and isinstance(args_list[0], int):
                    resolved_heads = args_list.pop(0)

            config = kwargs.get("config")
            if len(args_list) > 0 and not isinstance(args_list[0], int):
                config = args_list[0]

            if resolved_dim is not None:
                if config is None:
                    from transformers import BertConfig
                    h = resolved_heads or max(1, resolved_dim // 64)
                    config = BertConfig(hidden_size=resolved_dim, num_attention_heads=h)
                else:
                    try:
                        config.hidden_size = resolved_dim
                        config.n_embd = resolved_dim
                        config.d_model = resolved_dim
                        if resolved_heads:
                            config.num_attention_heads = resolved_heads
                            config.num_heads = resolved_heads
                    except Exception:
                        pass
                kwargs["config"] = config
                kwargs["hidden_size"] = resolved_dim
                kwargs["dim"] = resolved_dim
                if resolved_heads:
                    kwargs["num_heads"] = resolved_heads
                    kwargs["num_attention_heads"] = resolved_heads

            init_attempts = [
                lambda: super(UniversalAttentionWrapper, self).__init__(*args_list, **kwargs),
                lambda: super(UniversalAttentionWrapper, self).__init__(config=config, **{k: v for k, v in kwargs.items() if k != "config"}),
                lambda: super(UniversalAttentionWrapper, self).__init__(config, **{k: v for k, v in kwargs.items() if k != "config"}),
                lambda: super(UniversalAttentionWrapper, self).__init__(resolved_dim, resolved_heads, **kwargs),
                lambda: super(UniversalAttentionWrapper, self).__init__(resolved_dim, **kwargs),
                lambda: super(UniversalAttentionWrapper, self).__init__(**kwargs),
            ]
            last_err = None
            for attempt in init_attempts:
                try:
                    attempt()
                    return
                except TypeError as e:
                    last_err = e
                    continue
            if last_err:
                raise last_err

    UniversalAttentionWrapper.__name__ = target_cls.__name__
    UniversalAttentionWrapper.__qualname__ = target_cls.__qualname__
    UniversalAttentionWrapper.__module__ = target_cls.__module__
    UniversalAttentionWrapper.__doc__ = target_cls.__doc__
    _UNIVERSAL_WRAPPER_CACHE[target_cls] = UniversalAttentionWrapper
    return UniversalAttentionWrapper



class Registry:
    """A named dictionary that maps string keys to classes or factory functions with capability metadata."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._dict: dict[str, Callable] = {}
        self._aliases: dict[str, Callable] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._loaded: bool = False

    def register(
        self,
        key: str | None | Callable = None,
        supports_causal: bool | None = None,
        supports_bidirectional: bool | None = None,
        supports_cross_attention: bool | None = None,
        supports_gqa: bool | None = None,
        is_alias: bool = False,
        **extra_metadata: Any,
    ) -> Callable:
        """
        Decorator to register a class/factory under the given key with capability flags.

        Usage:
            @ATTENTION_REGISTRY.register("griffin", supports_causal=True, supports_bidirectional=False)
            class GriffinAttention(BaseAttention): ...

            @ATTENTION_REGISTRY.register
            class GriffinAttention(BaseAttention): ...
        """
        if callable(key) and not isinstance(key, str):
            obj = key
            final_key = getattr(obj, "__name__", str(obj))
            k = final_key.lower()
            if is_alias:
                self._aliases[k] = obj
            else:
                self._dict[k] = obj
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
                    (True if supports_cross_attention is None else supports_cross_attention),
                ),
                "supports_gqa": getattr(
                    obj, "supports_gqa", False if supports_gqa is None else supports_gqa
                ),
            }
            if self.name == "compressors":
                caps["hardware_accelerated"] = getattr(
                    obj,
                    "hardware_accelerated",
                    extra_metadata.get("hardware_accelerated", True),
                )
            caps.update(extra_metadata)
            self._metadata[k] = caps
            return obj

        def decorator(obj: Callable) -> Callable:
            """Inner registration decorator wrapping target callable.

            Args:
                obj: The class or callable to register.

            Returns:
                Callable: The registered object unmodified.
            """
            final_key = key if key is not None else obj.__name__
            k = final_key.lower()
            if is_alias:
                self._aliases[k] = obj
            else:
                # If key already exists and maps to a class, do not let a plain function overwrite it
                if k in self._dict and isinstance(self._dict[k], type) and not isinstance(obj, type):
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
                    (True if supports_cross_attention is None else supports_cross_attention),
                ),
                "supports_gqa": getattr(
                    obj, "supports_gqa", False if supports_gqa is None else supports_gqa
                ),
            }
            if self.name == "compressors":
                caps["hardware_accelerated"] = getattr(
                    obj,
                    "hardware_accelerated",
                    extra_metadata.get("hardware_accelerated", True),
                )
            caps.update(extra_metadata)
            self._metadata[k] = caps
            return obj

        return decorator

    def get_metadata(self, key: str) -> dict[str, Any]:
        """Retrieve capability metadata for a registered component."""
        if key in DEPRECATED_ALIASES:
            key = DEPRECATED_ALIASES[key]
        try:
            k = _handle_deprecated(key).lower()
        except ValueError:
            k = key.lower()
        if self.name == "attentions" and k in _ATTENTION_FILE_MAP:
            mapped = _ATTENTION_FILE_MAP[k].lower()
            self._try_lazy_load(mapped)
            if k in self._metadata:
                return dict(self._metadata[k])
            if mapped in self._metadata:
                return dict(self._metadata[mapped])
            bare = mapped.split(".")[-1]
            if bare in self._metadata:
                return dict(self._metadata[bare])
        elif self.name == "compressors" and k in _COMPRESSOR_FILE_MAP:
            mapped = _COMPRESSOR_FILE_MAP[k].lower()
            self._try_lazy_load(mapped)
            if k in self._metadata:
                return dict(self._metadata[k])
            if mapped in self._metadata:
                return dict(self._metadata[mapped])
            bare = mapped.split(".")[-1]
            if bare in self._metadata:
                return dict(self._metadata[bare])
        self._try_lazy_load(k)
        if k in self._metadata:
            return dict(self._metadata[k])
        obj = self.get(key)
        if obj is not None:
            res = {
                "supports_causal": getattr(obj, "supports_causal", True),
                "supports_bidirectional": getattr(obj, "supports_bidirectional", True),
                "supports_cross_attention": getattr(obj, "supports_cross_attention", True),
                "supports_gqa": getattr(obj, "supports_gqa", False),
            }
            if self.name == "compressors":
                res["hardware_accelerated"] = getattr(obj, "hardware_accelerated", True)
            return res
        return {}

    def _try_lazy_load(self, key: str) -> None:
        """Attempt to lazily load package modules when a key is queried."""
        k = key.lower()
        if k in self._dict or k in self._aliases:
            return
        if self.name == "compressors":
            canonical = _COMPRESSOR_FILE_MAP.get(k, k)
            try:
                import neuralops.compression_engine as _ce

                _ = getattr(_ce, key, None)
                if k not in self._dict and k not in self._aliases:
                    import importlib

                    try:
                        importlib.import_module(f"neuralops.compression_engine.{canonical}")
                    except (ImportError, ModuleNotFoundError):
                        bare = canonical.split(".")[-1]
                        for subpkg in ('svd', 'quantization', 'pruning', 'low_rank', 'tensor_decomp', 'kv_cache', 'experimental'):
                            try:
                                importlib.import_module(f"neuralops.compression_engine.{subpkg}.{bare}")
                                break
                            except (ImportError, ModuleNotFoundError):
                                continue
            except ImportError:
                pass
            except Exception as e:
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise
                logger.warning("Error during lazy loading of compressor '%s': %s", key, e)
        elif self.name == "attentions":
            if k == "ttt_linear_attention":
                try:
                    import neuralops.trainer.ttt.test_time_trainer  # noqa: F401
                except Exception:
                    pass
                return
            canonical = _ATTENTION_FILE_MAP.get(k, k)
            try:
                import neuralops.attentions as _att

                _ = getattr(_att, key, None)
                if k not in self._dict and k not in self._aliases:
                    import importlib

                    try:
                        importlib.import_module(f"neuralops.attentions.{canonical}")
                    except (ImportError, ModuleNotFoundError):
                        bare = canonical.split(".")[-1]
                        for subpkg in ('modern', 'svd', 'adaptive', 'gated', 'linear', 'sparse', 'ssm', 'hybrid', 'dense', 'fast_kernels', 'experimental'):
                            try:
                                importlib.import_module(f"neuralops.attentions.{subpkg}.{bare}")
                                break
                            except (ImportError, ModuleNotFoundError):
                                continue
            except ImportError:
                pass
            except Exception as e:
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise
                logger.warning("Error during lazy loading of attention '%s': %s", key, e)

    def get(self, key: str, unwrap: bool = False) -> Callable | None:
        """Safely retrieve a registered component, or None if not found."""
        if key in DEPRECATED_ALIASES:
            canonical = DEPRECATED_ALIASES[key]
            import warnings
            warnings.warn(
                f"'{key}' is deprecated. Use '{canonical}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            key = canonical
        try:
            k = _handle_deprecated(key)
        except ValueError:
            return None
        self._try_lazy_load(k)
        k_lower = k.lower()
        obj = self._dict.get(k_lower)
        if obj is None:
            obj = self._aliases.get(k_lower)
        if obj is None and self.name == "attentions" and k_lower in _ATTENTION_FILE_MAP:
            can = _ATTENTION_FILE_MAP[k_lower]
            self._try_lazy_load(can)
            bare = can.split(".")[-1].lower()
            obj = self._dict.get(can.lower()) or self._aliases.get(can.lower()) or self._dict.get(bare) or self._aliases.get(bare)
        if obj is None and self.name == "compressors" and k_lower in _COMPRESSOR_FILE_MAP:
            can = _COMPRESSOR_FILE_MAP[k_lower]
            self._try_lazy_load(can)
            bare = can.split(".")[-1].lower()
            obj = self._dict.get(can.lower()) or self._aliases.get(can.lower()) or self._dict.get(bare) or self._aliases.get(bare)
        if unwrap:
            return obj
        if obj is not None and self.name == "attentions" and isinstance(obj, type):
            return _make_universal_attention_class(obj)
        return obj

    def resolve(self, key: str) -> Callable:
        """Retrieve a registered component and raise a descriptive error if missing."""
        if key in DEPRECATED_ALIASES:
            canonical = DEPRECATED_ALIASES[key]
            import warnings
            warnings.warn(
                f"'{key}' is deprecated. Use '{canonical}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            key = canonical
        try:
            k = _handle_deprecated(key)
        except ValueError as e:
            raise KeyError(str(e)) from e
        self._try_lazy_load(k)
        k_lower = k.lower()
        obj = self._dict.get(k_lower)
        if obj is None:
            obj = self._aliases.get(k_lower)
        if obj is None and self.name == "attentions" and k_lower in _ATTENTION_FILE_MAP:
            can = _ATTENTION_FILE_MAP[k_lower]
            self._try_lazy_load(can)
            bare = can.split(".")[-1].lower()
            obj = self._dict.get(can.lower()) or self._aliases.get(can.lower()) or self._dict.get(bare) or self._aliases.get(bare)
        if obj is None and self.name == "compressors" and k_lower in _COMPRESSOR_FILE_MAP:
            can = _COMPRESSOR_FILE_MAP[k_lower]
            self._try_lazy_load(can)
            bare = can.split(".")[-1].lower()
            obj = self._dict.get(can.lower()) or self._aliases.get(can.lower()) or self._dict.get(bare) or self._aliases.get(bare)
        if obj is None:
            raise KeyError(
                f"Unknown {self.name.rstrip('s')} '{key}'. Available: {list(self._dict.keys())}"
            )
        if self.name == "attentions" and isinstance(obj, type):
            return _make_universal_attention_class(obj)
        return obj

    def _load_all(self) -> None:
        """Attempt to load all modules in the associated package."""
        if self._loaded:
            return
        if self.name == "compressors":
            try:
                import neuralops.compression_engine  # noqa: F401
                self._loaded = True
            except Exception as e:
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise
                logger.warning("Error loading compression_engine: %s", e)
        elif self.name == "attentions":
            try:
                import neuralops.attentions  # noqa: F401
                try:
                    import neuralops.trainer.ttt.test_time_trainer  # noqa: F401
                except Exception:
                    pass
                self._loaded = True
            except Exception as e:
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise
                logger.warning("Error loading attentions: %s", e)

    def keys(self) -> builtins.list[str]:
        """Return a list of all registered keys (canonical only)."""
        self._load_all()
        return [k for k in self._dict.keys() if k not in DEPRECATED_ALIASES]

    def list_available(self) -> builtins.list[str]:
        """Return a list of all registered keys."""
        return self.keys()

    def list(self) -> builtins.list[str]:
        """Return a list of all registered keys."""
        return self.keys()

    def items(self) -> Any:
        """Return registered key-value pairs."""
        return self._dict.items()

    def values(self) -> Any:
        """Return registered values."""
        return self._dict.values()

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, key: str) -> Callable:
        return self.resolve(key)

    def __contains__(self, key: str) -> bool:
        try:
            k = _handle_deprecated(key)
        except ValueError:
            return False
        self._try_lazy_load(k)
        k_lower = k.lower()
        if (k_lower in self._dict) or (k_lower in self._aliases):
            return True
        if self.name == "attentions" and k_lower in _ATTENTION_FILE_MAP:
            can = _ATTENTION_FILE_MAP[k_lower].lower()
            self._try_lazy_load(can)
            bare = can.split(".")[-1]
            return (can in self._dict) or (can in self._aliases) or (bare in self._dict) or (bare in self._aliases)
        if self.name == "compressors" and k_lower in _COMPRESSOR_FILE_MAP:
            can = _COMPRESSOR_FILE_MAP[k_lower].lower()
            self._try_lazy_load(can)
            bare = can.split(".")[-1]
            return (can in self._dict) or (can in self._aliases) or (bare in self._dict) or (bare in self._aliases)
        return False

    def __len__(self) -> int:
        self._load_all()
        return len(self._dict)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, {len(self)} items)"


# ----------------------------------------------------------------------
# Global registries – populated by their respective packages
# ----------------------------------------------------------------------
ATTENTION_REGISTRY = Registry("attentions")  # attention modules (Nexus, LEKA, ...)
COMPRESSOR_REGISTRY = Registry("compressors")  # compression strategies (SVD, TT, ...)
METHOD_REGISTRY = Registry("methods")  # training methods (LoRA, R-Drop, ...)
AUX_REGISTRY = Registry("auxiliary")  # auxiliary modules (RoPE, SSM, token pruning, ...)


def register_attention(
    key: str | None | Callable = None,
    supports_causal: bool | None = None,
    supports_bidirectional: bool | None = None,
    supports_cross_attention: bool | None = None,
    supports_gqa: bool | None = None,
    is_alias: bool = False,
    **extra_metadata: Any,
) -> Callable:
    """Decorator or helper to register attention mechanism with capability metadata."""
    if callable(key) and not isinstance(key, str):
        target = key
        return ATTENTION_REGISTRY.register(
            supports_causal=supports_causal,
            supports_bidirectional=supports_bidirectional,
            supports_cross_attention=supports_cross_attention,
            supports_gqa=supports_gqa,
            is_alias=is_alias,
            **extra_metadata,
        )(target)
    return ATTENTION_REGISTRY.register(
        key=key,
        supports_causal=supports_causal,
        supports_bidirectional=supports_bidirectional,
        supports_cross_attention=supports_cross_attention,
        supports_gqa=supports_gqa,
        is_alias=is_alias,
        **extra_metadata,
    )


def register_compressor(
    key: str | None | Callable = None,
    is_alias: bool = False,
    hardware_accelerated: bool = True,
    **extra_metadata: Any,
) -> Callable:
    """Decorator or helper to register compression strategy with capability metadata."""
    if callable(key) and not isinstance(key, str):
        target = key
        return COMPRESSOR_REGISTRY.register(
            is_alias=is_alias,
            hardware_accelerated=hardware_accelerated,
            **extra_metadata,
        )(target)
    return COMPRESSOR_REGISTRY.register(
        key=key,
        is_alias=is_alias,
        hardware_accelerated=hardware_accelerated,
        **extra_metadata,
    )


def list_attention_mechanisms() -> list[str]:
    """Returns canonical list of all registered attention mechanisms."""
    return ATTENTION_REGISTRY.keys()


def list_attentions() -> list[str]:
    """Returns canonical list of all registered attention mechanisms."""
    return ATTENTION_REGISTRY.keys()


def list_compressors() -> list[str]:
    """Returns canonical list of all registered compressor names."""
    return COMPRESSOR_REGISTRY.keys()


# Dynamically attach list_attention_mechanisms to neuralops top-level package if loaded
import sys as _sys
if "neuralops" in _sys.modules:
    setattr(_sys.modules["neuralops"], "list_attention_mechanisms", list_attention_mechanisms)

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


def create_attention(
    name: str,
    config: Any = None,
    dim: int | None = None,
    num_heads: int | None = None,
    embed_dim: int | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> Any:
    """Universal attention factory instantiating any registered attention mechanism or ComposableAttention."""
    if name == "composable" or name.startswith("compose(") or "proj=" in name:
        from neuralops.core.composable import ComposableAttention
        return ComposableAttention(embed_dim=embed_dim or dim or 768, num_heads=num_heads or 12, **kwargs)

    cls = ATTENTION_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown attention type '{name}'. Registered: {list(ATTENTION_REGISTRY.keys())}")

    resolved_dim = (
        embed_dim
        or dim
        or kwargs.pop("d_model", None)
        or kwargs.pop("hidden_size", None)
        or getattr(config, "hidden_size", None)
        or getattr(config, "n_embd", None)
        or getattr(config, "d_model", None)
    )
    resolved_heads = (
        num_heads
        or kwargs.pop("n_heads", None)
        or kwargs.pop("heads", None)
        or kwargs.pop("num_attention_heads", None)
        or getattr(config, "num_attention_heads", None)
        or getattr(config, "num_heads", None)
        or getattr(config, "n_head", None)
    )
    resolved_rank = rank or kwargs.pop("r", None)

    # Adjust heads if dim is not divisible
    if resolved_dim is not None and resolved_heads is not None and resolved_heads > 0:
        if resolved_dim % resolved_heads != 0:
            for cand in [16, 12, 8, 4, 2, 1]:
                if resolved_dim % cand == 0:
                    resolved_heads = cand
                    break

    call_kwargs = dict(kwargs)
    if resolved_dim is not None:
        call_kwargs["dim"] = resolved_dim
        call_kwargs["embed_dim"] = resolved_dim
        call_kwargs["hidden_size"] = resolved_dim
        call_kwargs["d_model"] = resolved_dim
    if resolved_heads is not None:
        call_kwargs["num_heads"] = resolved_heads
        call_kwargs["n_heads"] = resolved_heads
        call_kwargs["num_attention_heads"] = resolved_heads
    if resolved_rank is not None:
        call_kwargs["rank"] = resolved_rank
        call_kwargs["r"] = resolved_rank

    if config is None and resolved_dim is not None:
        from transformers import BertConfig
        h = resolved_heads or max(1, resolved_dim // 64)
        config = BertConfig(hidden_size=resolved_dim, num_attention_heads=h)
        call_kwargs["config"] = config

    # Dynamic multi-attempt constructor resolution
    init_attempts = []
    if config is not None:
        init_attempts.append(lambda: cls(config=config, **{k: v for k, v in call_kwargs.items() if k != "config"}))
        init_attempts.append(lambda: cls(config, **{k: v for k, v in call_kwargs.items() if k != "config"}))
    if resolved_dim is not None and resolved_heads is not None:
        init_attempts.append(lambda: cls(embed_dim=resolved_dim, num_heads=resolved_heads, **{k: v for k, v in call_kwargs.items() if k not in ("embed_dim", "num_heads")}))
        init_attempts.append(lambda: cls(dim=resolved_dim, num_heads=resolved_heads, **{k: v for k, v in call_kwargs.items() if k not in ("dim", "num_heads")}))
        init_attempts.append(lambda: cls(resolved_dim, num_heads=resolved_heads, **{k: v for k, v in call_kwargs.items() if k not in ("embed_dim", "dim", "num_heads")}))
    init_attempts.append(lambda: cls(**call_kwargs))
    init_attempts.append(lambda: cls())

    last_err = None
    for attempt in init_attempts:
        try:
            return attempt()
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return cls(**call_kwargs)


def create_compressor(name: str, model: Any = None, rank: int | None = None, **kwargs: Any) -> Any:
    """Universal compressor factory instantiating any registered compression engine."""
    cls = COMPRESSOR_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Compressor '{name}' not found in COMPRESSOR_REGISTRY.")

    call_kwargs = dict(kwargs)
    resolved_rank = rank or call_kwargs.pop("r", None)
    if resolved_rank is not None:
        call_kwargs["rank"] = resolved_rank

    if model is not None:
        call_kwargs["model"] = model

    attempts = [
        lambda: cls(**call_kwargs),
    ]
    if model is not None:
        attempts.append(lambda: cls(model, **{k: v for k, v in call_kwargs.items() if k != "model"}))
    else:
        import torch.nn as nn
        dummy_host = nn.Sequential(nn.Linear(64, 64))
        attempts.append(lambda: cls(model=dummy_host, **call_kwargs))
        attempts.append(lambda: cls(dummy_host, **call_kwargs))

    last_err = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return cls(**call_kwargs)


# Register HNC-SVD Compressor aliases without altering canonical 72 counts
try:
    from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor
    COMPRESSOR_REGISTRY.register("svd", is_alias=True)(DirectSVDCompressor)
    COMPRESSOR_REGISTRY.register("hnc_svd", is_alias=True)(DirectSVDCompressor)
    COMPRESSOR_REGISTRY.register("hnc_svd_compressor", is_alias=True)(DirectSVDCompressor)
except Exception as _e_hnc:
    logger.debug("Failed registering hnc_svd aliases: %s", _e_hnc)

try:
    from neuralops.compression_engine.low_rank.svd_llm_compressor import SVDLLMCompressor
    COMPRESSOR_REGISTRY.register("svd_llm", is_alias=True)(SVDLLMCompressor)
except Exception as _e_svd_llm:
    logger.debug("Failed registering svd_llm alias: %s", _e_svd_llm)



