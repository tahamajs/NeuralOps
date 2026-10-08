"""Auto-discovery and capability detection for all 232 modules."""
from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

from neuralops.benchmarks.result import ModuleMetadata
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

LOG = logging.getLogger(__name__)

OVERRIDES_PATH = Path(__file__).parent / "discovery_overrides.json"


def _load_overrides() -> dict[str, dict[str, Any]]:
    """Load manual capability overrides if file exists."""
    if OVERRIDES_PATH.exists():
        try:
            with open(OVERRIDES_PATH, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception as e:
            LOG.warning("Failed to load discovery_overrides.json: %s", e)
    return {}


def inspect_module_capabilities(
    name: str,
    cls: type,
    module_type: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> ModuleMetadata:
    """
    Inspect capability flags for a module class using:
    1. discovery_overrides.json
    2. Explicit class attributes
    3. Signature reflection heuristics
    """
    if overrides is None:
        overrides = _load_overrides()

    mod_overrides = overrides.get(name, {})

    sig = str(inspect.signature(cls.__init__)) if hasattr(cls, "__init__") else ""

    # Causal
    if "supports_causal" in mod_overrides:
        causal = mod_overrides["supports_causal"]
    elif hasattr(cls, "supports_causal"):
        causal = bool(getattr(cls, "supports_causal"))
    elif hasattr(cls, "is_causal"):
        causal = bool(getattr(cls, "is_causal"))
    elif "causal" in sig or "is_causal" in sig:
        causal = True
    else:
        causal = (module_type == "compressor")

    # GQA
    if "supports_gqa" in mod_overrides:
        gqa = mod_overrides["supports_gqa"]
    elif hasattr(cls, "supports_gqa"):
        gqa = bool(getattr(cls, "supports_gqa"))
    elif "num_kv_heads" in sig or "gqa" in sig:
        gqa = True
    else:
        gqa = False

    # RoPE
    if "supports_rope" in mod_overrides:
        rope = mod_overrides["supports_rope"]
    elif hasattr(cls, "supports_rope"):
        rope = bool(getattr(cls, "supports_rope"))
    elif "rope" in sig or "rotary" in sig or "position_embeddings" in sig:
        rope = True
    else:
        rope = False

    # CUDA
    if "requires_cuda" in mod_overrides:
        cuda = mod_overrides["requires_cuda"]
    elif hasattr(cls, "requires_cuda"):
        cuda = bool(getattr(cls, "requires_cuda"))
    else:
        cuda = False

    # Timeout
    timeout_s = mod_overrides.get("timeout_s", 1800)

    # Family deduction from module name
    parts = name.split("_")
    family = parts[0] if parts else "unknown"

    return ModuleMetadata(
        name=name,
        type=module_type,
        family=family,
        supports_causal=causal,
        supports_gqa=gqa,
        supports_rope=rope,
        requires_cuda=cuda,
        timeout_s=timeout_s,
    )


def discover_all_modules() -> list[ModuleMetadata]:
    """Discover and return capabilities for all 232 registered modules."""

    overrides = _load_overrides()
    discovered: list[ModuleMetadata] = []

    for k in sorted(ATTENTION_REGISTRY.keys()):
        meta = inspect_module_capabilities(k, ATTENTION_REGISTRY[k], "attention", overrides)
        discovered.append(meta)

    for k in sorted(COMPRESSOR_REGISTRY.keys()):
        meta = inspect_module_capabilities(k, COMPRESSOR_REGISTRY[k], "compressor", overrides)
        discovered.append(meta)

    return discovered


def discover_modules() -> dict[str, ModuleMetadata]:
    """Discover all registered modules and return a mapping of name to ModuleMetadata."""
    modules = discover_all_modules()
    return {m.name: m for m in modules}


__all__ = [
    "ModuleMetadata",
    "inspect_module_capabilities",
    "discover_all_modules",
    "discover_modules",
]

