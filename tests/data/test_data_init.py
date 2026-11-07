#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying data package exports, GLUE split loader, and dataset registry (F-61).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from neuralops.data import BenchmarkDatasetRegistry, load_glue_split


def test_load_glue_split_or_absent(tmp_path):
    """Verify load_glue_split delegates properly and does not unconditionally raise FileNotFoundError."""
    mock_data = {
        "train": [{"sentence": "Hello world", "label": 1}],
        "validation": [{"sentence": "Test sentence", "label": 0}],
    }

    with patch("neuralops.data.glue_loader.load_glue_local", return_value=mock_data):
        train_split = load_glue_split("sst2", split="train")
        assert len(train_split) == 1
        assert train_split[0]["label"] == 1

        val_split = load_glue_split("sst2", split="validation")
        assert len(val_split) == 1

        with pytest.raises(KeyError):
            load_glue_split("sst2", split="nonexistent_split")


def test_registry_get_returns_real_or_raises():
    """Verify get() for an existing dataset returns a real object, and for missing raises KeyError."""
    ds = BenchmarkDatasetRegistry.get("sst2")
    assert ds is not None
    assert isinstance(ds, dict)
    assert ds["name"] == "sst2"

    with pytest.raises(KeyError):
        BenchmarkDatasetRegistry.get("completely_unknown_dataset_xyz")


def test_list_datasets_matches_registry():
    """Verify list_datasets() matches the registered keys of _datasets."""
    listed = BenchmarkDatasetRegistry.list_datasets()
    assert isinstance(listed, list)
    assert set(listed) == set(BenchmarkDatasetRegistry._datasets.keys())
    assert "sst2" in listed
    assert "wikitext2" in listed
