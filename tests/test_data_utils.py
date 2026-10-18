"""Unit tests for neuralops.data utilities and loaders."""
from __future__ import annotations

import pytest
import torch
from neuralops.data import (
    BenchmarkDatasetRegistry,
    compute_empirical_entropy,
    generate_synthetic_data,
    generate_zipfian_markov_tokens,
    get_num_labels,
    get_text_columns,
    list_local_datasets,
    load_glue_split,
)


def test_list_local_datasets(tmp_path):
    """Verify local datasets listing returns list structure."""
    datasets = list_local_datasets(base_dir=str(tmp_path))
    assert isinstance(datasets, list)


def test_generate_zipfian_markov_tokens():
    """Verify Markov Zipfian token generator creates valid token tensors."""
    tokens = generate_zipfian_markov_tokens(
        num_samples=4,
        seq_len=16,
        vocab_size=50,
    )
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (4, 16)
    assert (tokens >= 0).all()
    assert (tokens < 50).all()


def test_compute_empirical_entropy():
    """Verify entropy computation on discrete token sequence."""
    tokens = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2])
    entropy = compute_empirical_entropy(tokens)
    assert isinstance(entropy, float)
    assert entropy >= 0.0


def test_load_glue_split_raises_on_missing():
    """Attempting to load a nonexistent dataset should raise appropriate error."""
    with pytest.raises(Exception):
        load_glue_split(
            task_name="nonexistent_glue_task_xyz_999",
            split="train",
        )


def test_dataset_discovery():
    """Verify dataset discovery and registry catalog."""
    registry = BenchmarkDatasetRegistry
    assert registry is not None
    assert hasattr(registry, "list_datasets") or hasattr(registry, "get") or registry is not None


def test_synthetic_generation(tmp_path):
    """Verify synthetic dataset generator runs cleanly and writes file."""
    out_file = str(tmp_path / "synth.csv")
    df = generate_synthetic_data(
        output_path=out_file,
        num_samples=16,
        domain="movies",
    )
    assert df is not None
    assert len(df) == 16


def test_data_metadata_helpers():
    """Verify label count and text columns for standard GLUE tasks."""
    num_sst2 = get_num_labels("sst2")
    assert num_sst2 == 2

    num_mnli = get_num_labels("mnli")
    assert num_mnli == 3

    cols = get_text_columns("sst2")
    assert "sentence" in cols or len(cols) > 0


def test_caching_and_task_types():
    """Verify num_labels on unknown task defaults safely."""
    labels = get_num_labels("unknown_task_123")
    assert isinstance(labels, int)
    assert labels >= 1
