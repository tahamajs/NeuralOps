"""Backward-compatibility shim. Logic moved to neuralops.data.benchmark_datasets package."""

from __future__ import annotations

from neuralops.data.benchmark_datasets import (
    BenchmarkDatasetInfo,
    BenchmarkDatasetRegistry,
    BenchmarkTorchDataset,
    _generate_synthetic_arc,
    _generate_synthetic_c4,
    _generate_synthetic_glue,
    _generate_synthetic_gsm8k,
    _generate_synthetic_hellaswag,
    _generate_synthetic_humaneval,
    _generate_synthetic_lambada,
    _generate_synthetic_mmlu,
    _generate_synthetic_pg19,
    _generate_synthetic_pile,
    _generate_synthetic_ptb,
    _generate_synthetic_wikitext,
    get_synthetic_benchmark_sample,
    load_benchmark_dataset,
)

__all__ = [
    "BenchmarkDatasetInfo",
    "BenchmarkDatasetRegistry",
    "BenchmarkTorchDataset",
    "load_benchmark_dataset",
    "get_synthetic_benchmark_sample",
    "_generate_synthetic_wikitext",
    "_generate_synthetic_c4",
    "_generate_synthetic_ptb",
    "_generate_synthetic_pg19",
    "_generate_synthetic_gsm8k",
    "_generate_synthetic_humaneval",
    "_generate_synthetic_mmlu",
    "_generate_synthetic_pile",
    "_generate_synthetic_arc",
    "_generate_synthetic_hellaswag",
    "_generate_synthetic_lambada",
    "_generate_synthetic_glue",
]
