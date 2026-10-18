"""Unit tests for BenchmarkDatasetRegistry and load_benchmark_dataset.
Tests offline synthetic fallback, registered benchmarks, and data formats.
"""

import pytest
from neuralops.data.benchmark_datasets import (
    BenchmarkDatasetInfo,
    BenchmarkDatasetRegistry,
    load_benchmark_dataset,
)


def test_registry_contains_standard_benchmarks():
    benchmarks = BenchmarkDatasetRegistry.list_benchmarks()
    required = [
        "wikitext2",
        "wikitext103",
        "c4",
        "ptb",
        "pg19",
        "gsm8k",
        "humaneval",
        "mmlu",
        "pile",
        "arc",
        "arc_challenge",
        "hellaswag",
        "lambada",
        "glue_sst2",
        "glue_mrpc",
        "glue_qqp",
        "glue_stsb",
        "glue_mnli",
        "glue_qnli",
        "glue_rte",
        "glue_wnli",
        "glue_cola",
    ]
    for req in required:
        assert req in benchmarks, f"Missing benchmark '{req}' in registry."


def test_synthetic_loading_all_benchmarks():
    benchmarks = [
        "wikitext2",
        "c4",
        "ptb",
        "pg19",
        "gsm8k",
        "humaneval",
        "mmlu",
        "pile",
        "arc",
        "arc_challenge",
        "hellaswag",
        "lambada",
        "glue/sst2",
        "glue/mrpc",
    ]
    for b in benchmarks:
        samples = load_benchmark_dataset(
            dataset_name=b,
            use_synthetic=True,
            max_samples=10,
            return_dataloader=False,
            seed=123,
        )
        assert isinstance(samples, list)
        assert len(samples) == 10, f"Expected 10 samples for {b}, got {len(samples)}"
        assert all(isinstance(s, dict) for s in samples)


def test_custom_registration():
    info = BenchmarkDatasetInfo(
        name="custom_math",
        aliases=["my_math"],
        category="reasoning",
        default_hf_path="custom/math",
        synthetic_generator=lambda n, s: [{"text": f"Math prob #{i}", "label": i} for i in range(n)],
    )
    BenchmarkDatasetRegistry.register(info)
    assert BenchmarkDatasetRegistry.is_registered("custom_math")
    assert BenchmarkDatasetRegistry.is_registered("my_math")

    res = load_benchmark_dataset("my_math", use_synthetic=True, max_samples=4, return_dataloader=False)
    assert len(res) == 4
    assert res[0]["text"] == "Math prob #0"


def test_manifest_and_torch_dataset():
    from neuralops.data.benchmark_datasets.manifest import BenchmarkDatasetRegistry
    from neuralops.data.benchmark_datasets.loaders import BenchmarkTorchDataset
    from neuralops.data.benchmark_datasets.synthetic import _generate_synthetic_wikitext
    
    info = BenchmarkDatasetRegistry.get("wikitext2")
    assert info is not None
    assert info.category == "language_modeling"

    samples = _generate_synthetic_wikitext(5)
    assert len(samples) == 5
    ds = BenchmarkTorchDataset(samples)
    assert len(ds) == 5
    item = ds[0]
    assert "text" in item


class TestBenchmarkRegistryExtended:
    """Extended test cases for benchmark registry queries and edge cases."""

    def test_registry_contains_glue(self):
        benchmarks = BenchmarkDatasetRegistry.list_benchmarks()
        glue_names = [d for d in benchmarks if "glue" in str(d).lower()]
        assert len(glue_names) > 0

    def test_load_unregistered_fallback(self):
        # Unregistered dataset safely falls back to synthetic generator
        res = load_benchmark_dataset("nonexistent_dataset_xyz", use_synthetic=True, max_samples=3, return_dataloader=False)
        assert len(res) == 3
        assert "text" in res[0]
