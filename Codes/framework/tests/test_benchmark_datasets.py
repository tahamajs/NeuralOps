"""
Unit tests for BenchmarkDatasetRegistry and load_benchmark_dataset.
Tests offline synthetic fallback, registered benchmarks, and data formats.
"""

import pytest
from framework.data.benchmark_datasets import (BenchmarkDatasetInfo,
                                               BenchmarkDatasetRegistry,
                                               load_benchmark_dataset)


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
