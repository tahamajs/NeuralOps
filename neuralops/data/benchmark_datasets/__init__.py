"""Unified Benchmark Dataset Loader & Registry for NLP / LLM Evaluation.

Exports metadata structures, registry, dataset loaders, and synthetic generation utilities.
"""

from __future__ import annotations

from neuralops.data.benchmark_datasets.loaders import (
    BenchmarkTorchDataset,
    get_synthetic_benchmark_sample,
    load_benchmark_dataset,
)
from neuralops.data.benchmark_datasets.manifest import (
    BenchmarkDatasetInfo,
    BenchmarkDatasetRegistry,
)
from neuralops.data.benchmark_datasets.synthetic import (
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

# Auto-wired package members
from neuralops.data.benchmark_datasets import loaders
from neuralops.data.benchmark_datasets import manifest
from neuralops.data.benchmark_datasets import synthetic

from neuralops.data.benchmark_datasets import loaders
from neuralops.data.benchmark_datasets import manifest
from neuralops.data.benchmark_datasets import synthetic
