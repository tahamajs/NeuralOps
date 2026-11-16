# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Registry query, module instantiation, extensibility, and contract scaffolding:
list_attentions, list_compressors, get_attention_class, get_compressor_class,
get_attention, get_attention_module, list_families, register_custom_attention,
register_custom_compressor, unregister_custom_module, validate_module_contract,
scaffold_module, probe_optimal_batch_size.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

logger = logging.getLogger("NeuralOps.API")


def _ensure_registries_loaded() -> None:
    """Ensure all subpackages registering attentions and compressors are imported."""
    try:
        import neuralops.layers  # noqa: F401
    except (ImportError, AttributeError) as exc:
        logger.warning("Error loading neuralops.layers: %s", exc)
    try:
        import neuralops.trainer.ttt.test_time_trainer  # noqa: F401
    except (ImportError, AttributeError) as exc:
        logger.warning("Error loading neuralops.trainer.ttt: %s", exc)
    try:
        import neuralops.compression  # noqa: F401
    except (ImportError, AttributeError) as exc:
        logger.warning("Error loading neuralops.compression: %s", exc)


def list_attentions(include_extensions: bool = False) -> List[str]:
    """Return a sorted list of all available canonical attention module names."""
    from neuralops.core.registry import list_attentions as _registry_list_attentions

    _ensure_registries_loaded()
    canonical = list(_registry_list_attentions())
    if include_extensions:
        try:
            from neuralops.integrations.fla import list_fla_attentions

            canonical.extend(list_fla_attentions())
        except ImportError as e:
            logger.warning("Optional FLA integration unavailable when listing attentions: %s", e)
        except Exception as e:
            logger.error("Failed to query FLA extension attentions: %s", e)
            raise
    return sorted(list(set(canonical)))


def list_compressors() -> List[str]:
    """Return a sorted list of all available canonical compressor names."""
    _ensure_registries_loaded()
    return sorted(list(COMPRESSOR_REGISTRY.keys()))


def probe_optimal_batch_size(*args: Any, **kwargs: Any) -> Any:
    """Lazily invoke the optional evaluation batch-size probe.

    Keeping the import inside the call makes the public NeuralOps API importable
    when evaluation extras are not installed, while source checkouts and full
    installations still use the canonical evaluation implementation.
    """
    from neuralops.evaluation.eval_pipeline import probe_optimal_batch_size as _probe

    return _probe(*args, **kwargs)


def get_attention_class(name: str) -> Type[Any]:
    """Retrieve the class or constructor for a registered attention module.

    Args:
        name: Attention module identifier from `list_attentions()`.

    Returns:
        Registered attention class.
    """
    if name.startswith("fla:"):
        from neuralops.core.exceptions import IntegrationNotImplementedError
        from neuralops.integrations.fla import EXTENSION_ATTENTION_REGISTRY, SUPPORTED_FLA_ATTENTIONS, fla_is_available

        if not fla_is_available():
            raise IntegrationNotImplementedError(
                f"Cannot load FLA attention '{name}': the 'fla' package is not installed. "
                "Install it via: pip install 'neuralops[fla]' (requires Linux + CUDA)."
            )
        if name not in EXTENSION_ATTENTION_REGISTRY:
            raise IntegrationNotImplementedError(
                f"Unknown FLA attention '{name}'. Supported FLA attentions: {SUPPORTED_FLA_ATTENTIONS}"
            )
        return EXTENSION_ATTENTION_REGISTRY[name]

    if name not in ATTENTION_REGISTRY:
        from neuralops.core.exceptions import AttentionNotRegisteredError

        raise AttentionNotRegisteredError(f"Unknown attention '{name}'. Available: {list_attentions()}")
    return ATTENTION_REGISTRY[name]


def get_compressor_class(name: str) -> Type[Any]:
    """Retrieve the class or constructor for a registered compressor engine.

    Args:
        name: Compressor engine identifier from `list_compressors()`.

    Returns:
        Registered compressor class.
    """
    if name not in COMPRESSOR_REGISTRY:
        from neuralops.core.exceptions import CompressorNotRegisteredError

        raise CompressorNotRegisteredError(f"Unknown compressor '{name}'. Available: {list_compressors()}")
    return COMPRESSOR_REGISTRY[name]


def list_families() -> Dict[str, List[str]]:
    """Return grouped categorization of all registered attention modules and compressors.

    Returns:
        Dictionary mapping family names (e.g., 'attention_linear', 'compression_svd') to module lists.
    """
    families: Dict[str, List[str]] = {}

    for name, cls in ATTENTION_REGISTRY.items():
        mod = getattr(cls, "__module__", "")
        subfam = mod.split(".")[2] if "neuralops.attentions." in mod else "other"
        group_key = f"attention_{subfam}"
        families.setdefault(group_key, []).append(name)

    for name, cls in COMPRESSOR_REGISTRY.items():
        mod = getattr(cls, "__module__", "")
        subfam = mod.split(".")[2] if "neuralops.compression_engine." in mod else "other"
        group_key = f"compression_{subfam}"
        families.setdefault(group_key, []).append(name)

    for k in families:
        families[k] = sorted(families[k])

    return families


# Canonical attention lookup triple: get_attention is canonical; get_attention_class and get_attention_module are aliases
get_attention = get_attention_class
get_attention_module = get_attention_class


def register_custom_attention(*args: Any, **kwargs: Any) -> Any:
    """Register a custom attention mechanism with full framework compatibility."""
    from neuralops.core.extensibility import register_custom_attention as _reg

    return _reg(*args, **kwargs)


def register_custom_compressor(*args: Any, **kwargs: Any) -> Any:
    """Register a custom compression engine with full framework compatibility."""
    from neuralops.core.extensibility import register_custom_compressor as _reg

    return _reg(*args, **kwargs)


def unregister_custom_module(name: str, category: Optional[str] = None) -> bool:
    """Cleanly unregister a custom module to restore the canonical registry counts."""
    from neuralops.core.extensibility import unregister_custom_module as _unreg

    return _unreg(name, category)


def validate_module_contract(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Automated 6-point contract diagnostic check for any module."""
    from neuralops.core.extensibility import validate_module_contract as _val

    return _val(*args, **kwargs)


def scaffold_module(
    name: str, category: str = "attention", family: str = "linear-scan", output_file: Optional[str] = None
) -> str:
    """Generate boilerplate code for a new custom module."""
    from neuralops.core.extensibility import scaffold_module as _scaff

    return _scaff(name, category=category, family=family, output_file=output_file)
