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
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

LOG = logging.getLogger(__name__)

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

    def decorator(target: Any) -> Any:
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


__all__ = [
    "BaseBenchmark",
    "BenchmarkRegistry",
    "BENCHMARK_REGISTRY",
    "register_benchmark",
    "get_benchmark",
    "list_benchmarks",
    "CANONICAL_BENCHMARK_MODELS",
    "CANONICAL_BENCHMARK_DATASETS",
    "OPERATIONAL_BENCHMARK_MODES",
]
