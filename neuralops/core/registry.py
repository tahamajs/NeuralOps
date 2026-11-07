#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Central Component Registry & Parameter Synchronization Engine

Architectural Role (NeuralOps):
This module implements the unified registry and dynamic discovery system underpinning
the attention and compression families in NeuralOps. Components define their
taxonomy via their `FAMILY` attribute (defaulting to their subpackage name).

1. Dynamic Component Registries:
   - ATTENTION_REGISTRY: Registered attention operators grouped by `FAMILY`.
   - COMPRESSOR_REGISTRY: Registered model compression engines grouped by `FAMILY`.

2. PEP 562 Lazy Loading Architecture:
   Eager loading of operator classes at import time would inflate startup latency.
   The registry decouples registration metadata from concrete class imports:
   operator modules are loaded on demand via PEP 562 and meta-path hooks.

3. Universal Parameter Synchronization Factory:
   create_attention() and create_compressor() provide unified instantiation contracts,
   automatically reconciling dimension keys (d_model, hidden_size, embed_dim), head
   counts (n_heads, num_attention_heads), and causal masking flags.
"""

from __future__ import annotations

import builtins
import importlib
import logging
from collections.abc import Callable
from typing import Any, cast

from neuralops.core.exceptions import RegistryError

logger = logging.getLogger(__name__)

LOAD_ERRORS: list[tuple[str, str]] = []


def load_errors() -> list[tuple[str, str]]:
    """Return a copy of all errors encountered during registry loading."""
    return list(LOAD_ERRORS)

DEPRECATED_ALIASES: dict[str, str] = {
    "aegis_v6": "gated_hybrid_kernel_attention",
    "chela_slay_attention": "chela_slay",
    "cosformer_attention": "cosformer",
    "decay_linear_attn": "decay_linear_attention",
    "eat_attention": "eat",
    "gated_linear_attention": "decay_linear_attention",
    "gated_linear_attn": "decay_linear_attention",
    "hgrn2_outer": "hgrn2_gated_outer",
    "native_hybrid_attention": "nha_attention",
    "svd_seq": "svd_attention",
}

from neuralops.core.legacy_aliases import ATTENTION_FILE_MAP as _ATTENTION_FILE_MAP
from neuralops.core.legacy_aliases import COMPRESSOR_FILE_MAP as _COMPRESSOR_FILE_MAP
from neuralops.core.legacy_aliases import DEPRECATED_ALIASES as _DEPRECATED_ALIASES
from neuralops.core.legacy_aliases import handle_deprecated_alias as _handle_deprecated

DEPRECATED_ALIASES.update(_DEPRECATED_ALIASES)


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
            resolved_dim = (
                kwargs.get("dim") or kwargs.get("hidden_size") or kwargs.get("d_model") or kwargs.get("embed_dim")
            )
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
                    except Exception as e:
                        logger.warning("Could not set dimension attributes on config: %s", e)
                kwargs["config"] = config
                kwargs["hidden_size"] = resolved_dim
                kwargs["dim"] = resolved_dim
                if resolved_heads:
                    kwargs["num_heads"] = resolved_heads
                    kwargs["num_attention_heads"] = resolved_heads

            try:
                return super().__new__(cls)
            except Exception as e:
                logger.warning(
                    "UniversalAttentionWrapper.__new__ super failed (%s); allocating target_cls directly",
                    e,
                )
                return object.__new__(target_cls)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            args_list = list(args)
            resolved_dim = (
                kwargs.get("dim") or kwargs.get("hidden_size") or kwargs.get("d_model") or kwargs.get("embed_dim")
            )
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
                    except Exception as e:
                        logger.warning("Could not set dimension attributes on config: %s", e)
                kwargs["config"] = config
                kwargs["hidden_size"] = resolved_dim
                kwargs["dim"] = resolved_dim
                if resolved_heads:
                    kwargs["num_heads"] = resolved_heads
                    kwargs["num_attention_heads"] = resolved_heads

            init_attempts = [
                lambda: super(UniversalAttentionWrapper, self).__init__(*args_list, **kwargs),
                lambda: super(UniversalAttentionWrapper, self).__init__(
                    config=config, **{k: v for k, v in kwargs.items() if k != "config"}
                ),
                lambda: super(UniversalAttentionWrapper, self).__init__(
                    config, **{k: v for k, v in kwargs.items() if k != "config"}
                ),
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


class _LazyRegistryDict(dict):
    """Dictionary proxy that lazily triggers _load_all() upon iteration or bulk inspection."""

    def __init__(self, registry: Registry, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._registry = registry

    def values(self) -> Any:
        self._registry._load_all()
        return super().values()

    def items(self) -> Any:
        self._registry._load_all()
        return super().items()

    def keys(self) -> Any:
        self._registry._load_all()
        return super().keys()

    def __iter__(self) -> Any:
        self._registry._load_all()
        return super().__iter__()

    def __len__(self) -> int:
        self._registry._load_all()
        return super().__len__()


class Registry:
    """A named dictionary that maps string keys to classes or factory functions with capability metadata."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._dict: dict[str, Callable] = _LazyRegistryDict(self)
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
                "supports_gqa": getattr(obj, "supports_gqa", False if supports_gqa is None else supports_gqa),
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
                if k in self._dict:
                    raise RegistryError(f"Alias '{k}' collides with canonical key in registry '{self.name}'.")
                self._aliases[k] = obj
            else:
                if k in self._dict and self._dict[k] is not obj:
                    raise RegistryError(
                        f"Canonical key '{k}' already registered to {self._dict[k]} in registry '{self.name}'."
                    )
                if k in self._aliases:
                    raise RegistryError(
                        f"Canonical key '{k}' collides with an existing alias in registry '{self.name}'."
                    )
                self._dict[k] = obj

            # Ensure FAMILY attribute is set on registered component
            if getattr(obj, "FAMILY", None) in (None, "uncategorized"):
                mod = getattr(obj, "__module__", "")
                parts = mod.split(".")
                if len(parts) >= 3 and parts[0] == "neuralops" and parts[1] in ("layers", "compression"):
                    try:
                        obj.FAMILY = parts[2]  # type: ignore[attr-defined]
                    except (AttributeError, TypeError):
                        pass
                elif "ttt" in mod:
                    try:
                        obj.FAMILY = "linear"  # type: ignore[attr-defined]
                    except (AttributeError, TypeError):
                        pass

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
                "supports_gqa": getattr(obj, "supports_gqa", False if supports_gqa is None else supports_gqa),
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
                import neuralops.compression as _ce

                _ = getattr(_ce, key, None)
                if k not in self._dict and k not in self._aliases:
                    import importlib

                    try:
                        importlib.import_module(f"neuralops.compression.{canonical}")
                    except (ImportError, ModuleNotFoundError):
                        bare = canonical.split(".")[-1]
                        for subpkg in (
                            "svd",
                            "quantization",
                            "pruning",
                            "low_rank",
                            "tensor_decomp",
                            "kv_cache",
                            "experimental",
                        ):
                            try:
                                importlib.import_module(f"neuralops.compression.{subpkg}.{bare}")
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
                except Exception as e:
                    logger.warning("Failed to import ttt_linear_attention trainer: %s", e)
                return
            canonical = _ATTENTION_FILE_MAP.get(k, k)
            try:
                import neuralops.layers as _att

                _ = getattr(_att, key, None)
                if k not in self._dict and k not in self._aliases:
                    import importlib

                    try:
                        importlib.import_module(f"neuralops.layers.{canonical}")
                    except (ImportError, ModuleNotFoundError):
                        bare = canonical.split(".")[-1]
                        for subpkg in (
                            "modern",
                            "svd",
                            "adaptive",
                            "gated",
                            "linear",
                            "sparse",
                            "ssm",
                            "hybrid",
                            "dense",
                            "fast_kernels",
                            "experimental",
                        ):
                            try:
                                importlib.import_module(f"neuralops.layers.{subpkg}.{bare}")
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
            obj = (
                self._dict.get(can.lower())
                or self._aliases.get(can.lower())
                or self._dict.get(bare)
                or self._aliases.get(bare)
            )
        if obj is None and self.name == "compressors" and k_lower in _COMPRESSOR_FILE_MAP:
            can = _COMPRESSOR_FILE_MAP[k_lower]
            self._try_lazy_load(can)
            bare = can.split(".")[-1].lower()
            obj = (
                self._dict.get(can.lower())
                or self._aliases.get(can.lower())
                or self._dict.get(bare)
                or self._aliases.get(bare)
            )
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
            obj = (
                self._dict.get(can.lower())
                or self._aliases.get(can.lower())
                or self._dict.get(bare)
                or self._aliases.get(bare)
            )
        if obj is None and self.name == "compressors" and k_lower in _COMPRESSOR_FILE_MAP:
            can = _COMPRESSOR_FILE_MAP[k_lower]
            self._try_lazy_load(can)
            bare = can.split(".")[-1].lower()
            obj = (
                self._dict.get(can.lower())
                or self._aliases.get(can.lower())
                or self._dict.get(bare)
                or self._aliases.get(bare)
            )
        if obj is None:
            raise KeyError(f"Unknown {self.name.rstrip('s')} '{key}'. Available: {list(self._dict.keys())}")
        if self.name == "attentions" and isinstance(obj, type):
            return _make_universal_attention_class(obj)
        return obj

    def load_errors(self) -> builtins.list[tuple[str, str]]:
        """Return a copy of all errors encountered during registry loading."""
        return list(LOAD_ERRORS)

    def _load_all(self) -> None:
        """Attempt to load all modules in the associated package."""
        if self._loaded:
            return
        if self.name == "compressors":
            try:
                package = importlib.import_module("neuralops.compression")
                for subpackage in package._SUBPACKAGES:
                    sub_mod = importlib.import_module(f"neuralops.compression.{subpackage}")
                    for mod_path in getattr(sub_mod, "_FAMILY_MODULES", {}).values():
                        try:
                            importlib.import_module(mod_path)
                        except Exception as e:
                            logger.error("Failed to eagerly load compressor module %s: %s", mod_path, e)
                            LOAD_ERRORS.append((mod_path, str(e)))
                self._loaded = True
            except Exception as e:
                logger.error("Error loading compression: %s", e)
                LOAD_ERRORS.append(("neuralops.compression", str(e)))
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise
        elif self.name == "attentions":
            try:
                package = importlib.import_module("neuralops.layers")
                for subpackage in package._SUBPACKAGES:
                    sub_mod = importlib.import_module(f"neuralops.layers.{subpackage}")
                    for mod_path in getattr(sub_mod, "_FAMILY_MODULES", {}).values():
                        try:
                            importlib.import_module(mod_path)
                        except Exception as e:
                            logger.error("Failed to eagerly load attention module %s: %s", mod_path, e)
                            LOAD_ERRORS.append((mod_path, str(e)))
                try:
                    import neuralops.trainer.ttt.test_time_trainer  # noqa: F401
                except Exception as e:
                    logger.error("Failed to eagerly import ttt_linear_attention: %s", e)
                    LOAD_ERRORS.append(("neuralops.trainer.ttt.test_time_trainer", str(e)))
                self._loaded = True
            except Exception as e:
                logger.error("Error loading layers: %s", e)
                LOAD_ERRORS.append(("neuralops.layers", str(e)))
                import os

                if os.environ.get("SVD_DEBUG") == "1":
                    raise

    def canonical_keys(self) -> builtins.list[str]:
        """Return a list of all registered keys (canonical only)."""
        self._load_all()
        return [k for k in self._dict.keys() if k not in DEPRECATED_ALIASES]

    def distinct_classes(self) -> builtins.list[type]:
        """Return a list of unique classes registered under canonical keys."""
        self._load_all()
        classes = []
        seen = set()
        for k in self.canonical_keys():
            v = self._dict[k]
            if isinstance(v, type) and v not in seen:
                seen.add(v)
                classes.append(v)
        return classes

    def keys(self) -> builtins.list[str]:
        """Return a list of all registered keys (canonical only)."""
        return self.canonical_keys()

    def list_available(self) -> builtins.list[str]:
        """Return a list of all registered keys."""
        return self.keys()

    def list(self) -> builtins.list[str]:
        """Return a list of all registered keys."""
        return self.keys()

    def items(self) -> Any:
        """Return registered key-value pairs."""
        self._load_all()
        return self._dict.items()

    def values(self) -> Any:
        """Return registered values."""
        self._load_all()
        return self._dict.values()

    def __iter__(self):
        self._load_all()
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


ATTENTION_REGISTRY = Registry("attentions")
COMPRESSOR_REGISTRY = Registry("compressors")
METHOD_REGISTRY = Registry("methods")
AUX_REGISTRY = Registry("auxiliary")


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
    return list_attentions()


import sys as _sys

if "neuralops" in _sys.modules:
    _sys.modules["neuralops"].list_attention_mechanisms = list_attention_mechanisms  # type: ignore[attr-defined]

MODEL_REGISTRY = Registry("models")
DATASET_REGISTRY = Registry("datasets")
LOSS_REGISTRY = Registry("losses")
TRAINER_REGISTRY = Registry("trainers")
VISUALIZER_REGISTRY = Registry("visualizers")


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
    comps = [k for k in COMPRESSOR_REGISTRY.keys() if not is_helper_module(k)] if include_compressors else []
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
    """Universal attention factory instantiating any registered attention mechanism or ComposableAttention.

    Normalizes architectural hyperparameter aliases (embed_dim, dim, d_model, hidden_size,
    num_heads, n_heads, heads, rank, r). If no config is provided and dim is specified,
    constructs a synthetic BertConfig with config._is_synthetic = True. Validates that
    num_heads divides dim evenly, raising ValueError if it does not.
    """
    if name == "composable" or name.startswith("compose(") or "proj=" in name:
        from neuralops.core.composable import ComposableAttention

        return ComposableAttention(embed_dim=embed_dim or dim or 768, num_heads=num_heads or 12, **kwargs)

    cls = ATTENTION_REGISTRY.get(name)
    if cls is None:
        from neuralops.core.exceptions import AttentionNotRegisteredError

        raise AttentionNotRegisteredError(
            f"Unknown attention type '{name}'. Registered: {list(ATTENTION_REGISTRY.keys())}"
        )

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

    if resolved_dim is not None and resolved_heads is not None and resolved_heads > 0:
        if resolved_dim % resolved_heads != 0:
            raise ValueError(f"num_heads ({resolved_heads}) must divide dim ({resolved_dim}) evenly.")

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
        is_dec = kwargs.get("is_decoder", kwargs.get("is_causal", True))
        config = BertConfig(hidden_size=resolved_dim, num_attention_heads=h, is_decoder=is_dec)
        config.is_causal = is_dec
        config._is_synthetic = True
        call_kwargs["config"] = config

    init_attempts = []
    if config is not None:
        init_attempts.append(lambda: cls(config=config, **{k: v for k, v in call_kwargs.items() if k != "config"}))
        init_attempts.append(lambda: cls(config, **{k: v for k, v in call_kwargs.items() if k != "config"}))
    if resolved_dim is not None and resolved_heads is not None:
        init_attempts.append(
            lambda: cls(
                embed_dim=resolved_dim,
                num_heads=resolved_heads,
                **{k: v for k, v in call_kwargs.items() if k not in ("embed_dim", "num_heads")},
            )
        )
        init_attempts.append(
            lambda: cls(
                dim=resolved_dim,
                num_heads=resolved_heads,
                **{k: v for k, v in call_kwargs.items() if k not in ("dim", "num_heads")},
            )
        )
        init_attempts.append(
            lambda: cls(
                resolved_dim,
                num_heads=resolved_heads,
                **{k: v for k, v in call_kwargs.items() if k not in ("embed_dim", "dim", "num_heads")},
            )
        )
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
        from neuralops.core.exceptions import CompressorNotRegisteredError

        raise CompressorNotRegisteredError(f"Compressor '{name}' not found in COMPRESSOR_REGISTRY.")

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


def get_attention(name: str) -> type:
    """Retrieve an attention class by registered name."""
    cls = ATTENTION_REGISTRY.get(name)
    if cls is None:
        from neuralops.core.exceptions import AttentionNotRegisteredError

        raise AttentionNotRegisteredError(f"Attention '{name}' not found in ATTENTION_REGISTRY.")
    return cast(type, cls)


def get_compressor(name: str, **kwargs: Any) -> Any:
    """Retrieve a compressor class by registered name, or instantiate if kwargs provided."""
    cls = COMPRESSOR_REGISTRY.get(name)
    if cls is None:
        from neuralops.core.exceptions import CompressorNotRegisteredError

        raise CompressorNotRegisteredError(f"Compressor '{name}' not found in COMPRESSOR_REGISTRY.")
    if kwargs:
        return cls(**kwargs)
    return cls


def list_attentions(include_extensions: bool = False) -> list[str]:
    """List all registered attention names."""
    names = sorted(list(ATTENTION_REGISTRY.keys()))
    if not include_extensions:
        names = [n for n in names if not n.startswith("fla:")]
    return names


def list_compressors() -> list[str]:
    """List all registered compressor names."""
    return sorted(list(COMPRESSOR_REGISTRY.keys()))
