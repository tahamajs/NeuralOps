"""Module, Model, and Dataset registry indexing for benchmarks."""
from __future__ import annotations

import logging
from typing import Any
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


class BenchmarkRegistry:
    """Registry accessor for benchmarks discovery and validation."""

    @staticmethod
    def get_attention_modules() -> list[str]:
        """Return sorted list of all registered attention modules."""
        import neuralops.attentions  # Ensure registration
        return sorted(list(ATTENTION_REGISTRY.keys()))

    @staticmethod
    def get_compressor_modules() -> list[str]:
        """Return sorted list of all registered compressor modules."""
        import neuralops.compression_engine  # Ensure registration
        return sorted(list(COMPRESSOR_REGISTRY.keys()))

    @staticmethod
    def get_all_modules() -> list[str]:
        """Return combined list of all 232 candidate modules."""
        return sorted(BenchmarkRegistry.get_attention_modules() + BenchmarkRegistry.get_compressor_modules())

    @staticmethod
    def get_supported_models() -> list[str]:
        """Return canonical benchmark model list."""
        return list(CANONICAL_BENCHMARK_MODELS)

    @staticmethod
    def get_supported_datasets() -> list[str]:
        """Return canonical benchmark dataset list."""
        return list(CANONICAL_BENCHMARK_DATASETS)


__all__ = [
    "BenchmarkRegistry",
    "CANONICAL_BENCHMARK_MODELS",
    "CANONICAL_BENCHMARK_DATASETS",
]
