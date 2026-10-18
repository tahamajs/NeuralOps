"""Tests for neuralops.data.glue_loader module."""

import pytest
from unittest.mock import MagicMock, patch

from neuralops.data.glue_loader import (
    GLUE_TASK_KEYS,
    get_num_labels_and_is_regression,
    load_glue_local,
)


class TestGlueMetadata:
    """Test GLUE task labels, regression flags, and key mappings."""

    @pytest.mark.parametrize("task,expected_labels,expected_regression", [
        ("sst2", 2, False),
        ("mrpc", 2, False),
        ("qqp", 2, False),
        ("mnli", 3, False),
        ("qnli", 2, False),
        ("rte", 2, False),
        ("cola", 2, False),
        ("stsb", 1, True),
        ("wnli", 2, False),
    ])
    def test_get_labels(self, task, expected_labels, expected_regression):
        n, is_reg = get_num_labels_and_is_regression(task)
        assert n == expected_labels
        assert is_reg == expected_regression

    def test_unknown_task_defaults(self):
        n, is_reg = get_num_labels_and_is_regression("unknown_task")
        assert n == 2
        assert is_reg is False

    def test_task_keys_present(self):
        assert "sst2" in GLUE_TASK_KEYS
        assert GLUE_TASK_KEYS["sst2"] == ("sentence", None)
        assert GLUE_TASK_KEYS["mrpc"] == ("sentence1", "sentence2")


class TestGlueLocalLoader:
    """Test load_glue_local routing and fallback logic."""

    @patch("datasets.load_from_disk")
    def test_load_from_disk_if_exists(self, mock_load_from_disk, tmp_path, monkeypatch):
        # Patch GLUE_LOCAL_DIR to point to tmp_path
        task_dir = tmp_path / "sst2"
        task_dir.mkdir()
        monkeypatch.setattr("neuralops.data.glue_loader.GLUE_LOCAL_DIR", tmp_path)

        mock_load_from_disk.return_value = {"train": [], "validation": []}
        res = load_glue_local("sst2")
        mock_load_from_disk.assert_called_once_with(str(task_dir))
        assert "train" in res
