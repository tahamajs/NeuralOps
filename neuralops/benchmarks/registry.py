#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Module, Model, Dataset, and Benchmark registry indexing for NeuralOps.

Implements registry accessors, benchmark decorator registration, and canonical
mappings for the 12 operational benchmark modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Canonical specification and capability metadata for a benchmark target.

    Adheres strictly to the Unified Benchmark CLI & Integrity Specification.
    """
    canonical_name: str
    category: str
    target_name: str
    requires_cuda: bool = False
    requires_triton: bool = False
    supports_causal: bool = True
    supports_backward: bool = True
    supports_gqa: bool = False
    supports_rope: bool = False
    default_timeout_s: int = 1800
    reference_backend: str = "pytorch"
    is_deprecated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


CANONICAL_BENCHMARK_MODELS: list[str] = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "gpt2",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

CANONICAL_BENCHMARK_DATASETS: list[str] = [
    "sst2",
    "mrpc",
    "qnli",
    "qqp",
    "mnli",
    "wikitext2",
]

# The 12 canonical operational modes formalized in UNIFIED_BENCHMARK_SPECIFICATION
OPERATIONAL_BENCHMARK_MODES: list[str] = [
    "checkpoint",              # Mode 1: Checkpoint evaluation
    "sweep-attentions",        # Mode 2: Sequential attention sweep (131 keys)
    "sweep-compressors",       # Mode 3: Sequential compression sweep (72 keys)
    "flash-compute",           # Mode 4: Flash compute saturation (% peak TFLOPS)
    "serving-poisson",         # Mode 5: Poisson cloud serving simulation
    "ruler-niah",              # Mode 6: Extended context needle-in-a-haystack
    "perplexity",              # Mode 7: Autoregressive perplexity quality
    "lm-eval",                 # Mode 8: Multi-choice reasoning & accuracy
    "ridge-healing",           # Mode 9: Closed-form calibration healing
    "distributed-mfu",         # Mode 10: Multi-rank distributed MFU
    "multi-seed",              # Mode 11: Statistical multi-seed invariance
    "chaos-stress",            # Mode 12: Hardware dynamic range & chaos stress
]


class BaseBenchmark(ABC):
    """Abstract base class for all registered benchmarks in NeuralOps."""

    name: str = "base_benchmark"
    mode: str = "any"
    hardware: str = "any"

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    def run(self, model_adapter: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute benchmark against model adapter and return metrics dictionary."""
        raise NotImplementedError


# Canonical benchmark registry mapping benchmark name to benchmark entry (class, callable, or metadata dict)
BENCHMARK_REGISTRY: Dict[str, Union[Type[BaseBenchmark], Callable[..., Any], Dict[str, Any]]] = {}

# Detailed metadata index for registered benchmarks
_BENCHMARK_METADATA: Dict[str, Dict[str, Any]] = {}


def register_benchmark(
    name: str,
    mode: str = "any",
    hardware: str = "any",
    description: str = "",
) -> Callable[[Any], Any]:
    """Decorator to register a benchmark function or BaseBenchmark subclass in BENCHMARK_REGISTRY."""
    if mode not in OPERATIONAL_BENCHMARK_MODES and mode not in ("any", "all"):
        raise ValueError(
            f"Invalid benchmark mode '{mode}'. Must be one of {OPERATIONAL_BENCHMARK_MODES} or 'any'/'all'."
        )

    def decorator(target: Any) -> Any:
        if name in BENCHMARK_REGISTRY and BENCHMARK_REGISTRY[name] is not target:
            raise ValueError(f"Conflicting benchmark registration for '{name}': already registered.")

        BENCHMARK_REGISTRY[name] = target
        _BENCHMARK_METADATA[name] = {
            "name": name,
            "mode": mode,
            "hardware": hardware,
            "description": description or getattr(target, "__doc__", "") or "",
            "target": target,
        }
        # Annotate target attributes directly
        try:
            target._benchmark_name = name
            target._benchmark_mode = mode
            target._benchmark_hardware = hardware
        except (AttributeError, TypeError):
            pass
        return target

    return decorator


def get_benchmark(name: str) -> Optional[Any]:
    """Retrieve registered benchmark runner by name."""
    return BENCHMARK_REGISTRY.get(name)


def list_benchmarks(mode: Optional[str] = None, hardware: Optional[str] = None) -> List[str]:
    """List registered benchmark names, optionally filtered by mode or hardware."""
    results: List[str] = []
    for name, meta in _BENCHMARK_METADATA.items():
        if mode is not None and meta["mode"] not in (mode, "any", "all"):
            continue
        if hardware is not None and meta["hardware"] not in (hardware, "any", "all"):
            continue
        results.append(name)
    return sorted(results)


class BenchmarkRegistry:
    """Registry accessor for benchmarks discovery, validation, and operational modes."""

    @staticmethod
    def get_attention_modules() -> list[str]:
        """Return sorted list of all registered attention modules."""
        return sorted(list(ATTENTION_REGISTRY.keys()))

    @staticmethod
    def get_compressor_modules() -> list[str]:
        """Return sorted list of all registered compressor modules."""
        return sorted(list(COMPRESSOR_REGISTRY.keys()))

    @staticmethod
    def get_all_modules() -> list[str]:
        """Return combined list of all candidate modules."""
        return sorted(BenchmarkRegistry.get_attention_modules() + BenchmarkRegistry.get_compressor_modules())

    @staticmethod
    def get_supported_models() -> list[str]:
        """Return canonical benchmark model list."""
        return list(CANONICAL_BENCHMARK_MODELS)

    @staticmethod
    def get_supported_datasets() -> list[str]:
        """Return canonical benchmark dataset list."""
        return list(CANONICAL_BENCHMARK_DATASETS)

    @staticmethod
    def get_operational_modes() -> list[str]:
        """Return the 12 canonical operational benchmark modes."""
        return list(OPERATIONAL_BENCHMARK_MODES)

    @staticmethod
    def is_valid_mode(mode: str) -> bool:
        """Validate if a mode is one of the 12 operational benchmark modes."""
        return mode in OPERATIONAL_BENCHMARK_MODES

    @staticmethod
    def populate_canonical_benchmarks() -> int:
        """Populate BENCHMARK_REGISTRY and _BENCHMARK_METADATA with canonical inventory.

        Registers all 131 attention mechanisms, 72 compressor engines, and verified ops
        under strict canonical names ('attention:<name>', 'compressor:<name>', 'cuda:<name>', 'triton:<name>').
        """
        from neuralops.benchmarks.discovery import discover_all_modules
        from neuralops.benchmarks.ops_registry import OPS_REGISTRY

        count = 0
        modules = discover_all_modules()
        for m in modules:
            cat = getattr(m, "type", None) or getattr(m, "module_type", "attention")
            canon_name = f"{cat}:{m.name}"
            spec = BenchmarkSpec(
                canonical_name=canon_name,
                category=cat,
                target_name=m.name,
                requires_cuda=getattr(m, "requires_cuda", False),
                supports_causal=getattr(m, "supports_causal", True),
                supports_backward=True,
                supports_gqa=getattr(m, "supports_gqa", False),
                supports_rope=getattr(m, "supports_rope", False),
                default_timeout_s=getattr(m, "timeout_s", 1800),
            )
            BENCHMARK_REGISTRY[canon_name] = spec
            _BENCHMARK_METADATA[canon_name] = {
                "name": canon_name,
                "category": cat,
                "target_name": m.name,
                "mode": f"sweep-{cat}s" if cat in ("attention", "compressor") else "any",
                "hardware": "cuda" if spec.requires_cuda else "any",
                "description": f"Canonical {cat} benchmark for {m.name}",
                "spec": spec,
            }
            count += 1

        for raw_op_name, op_cfg in OPS_REGISTRY.items():
            cat = op_cfg.category
            canon_name = raw_op_name if ":" in raw_op_name else f"{cat}:{raw_op_name}"
            target_name = raw_op_name.split(":", 1)[1] if ":" in raw_op_name else raw_op_name
            if canon_name not in BENCHMARK_REGISTRY:
                spec = BenchmarkSpec(
                    canonical_name=canon_name,
                    category=cat,
                    target_name=target_name,
                    requires_cuda=(cat == "cuda"),
                    requires_triton=(cat == "triton"),
                    reference_backend="pytorch",
                )
                BENCHMARK_REGISTRY[canon_name] = spec
                _BENCHMARK_METADATA[canon_name] = {
                    "name": canon_name,
                    "category": cat,
                    "target_name": target_name,
                    "mode": "any",
                    "hardware": "cuda" if cat == "cuda" else "any",
                    "description": f"Verified {cat} kernel operation {target_name}",
                    "spec": spec,
                }
                count += 1

        return count


# Auto-populate canonical benchmarks on initial import
BenchmarkRegistry.populate_canonical_benchmarks()


__all__ = [
    "BaseBenchmark",
    "BenchmarkRegistry",
    "BenchmarkSpec",
    "BENCHMARK_REGISTRY",
    "register_benchmark",
    "get_benchmark",
    "list_benchmarks",
    "CANONICAL_BENCHMARK_MODELS",
    "CANONICAL_BENCHMARK_DATASETS",
    "OPERATIONAL_BENCHMARK_MODES",
]
