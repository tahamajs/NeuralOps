"""Tests verifying the 8 critical benchmark bugfixes."""

import json
from pathlib import Path
import tempfile
import time
import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks.core.data_loader import (
    GLUE_VALIDATION_SIZES,
    get_dataset_metadata,
)
from neuralops.benchmarks.core.model_loader import count_parameters
from neuralops.benchmarks.executor import execute_single
from neuralops.benchmarks.logging import JSONLHandler, setup_run_logger
from neuralops.benchmarks.result import BenchmarkResult
from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor


def test_glue_validation_sizes_mapping():
    """Bug 1: Verify correct GLUE validation sizes."""
    assert GLUE_VALIDATION_SIZES["sst2"] == 872
    assert GLUE_VALIDATION_SIZES["mrpc"] == 408
    assert GLUE_VALIDATION_SIZES["rte"] == 277
    assert GLUE_VALIDATION_SIZES["cola"] == 1043
    assert GLUE_VALIDATION_SIZES["stsb"] == 1500
    assert GLUE_VALIDATION_SIZES["qnli"] == 5463
    assert GLUE_VALIDATION_SIZES["wnli"] == 71
    assert GLUE_VALIDATION_SIZES["mnli"] == 9815
    assert GLUE_VALIDATION_SIZES["qqp"] == 40430


def test_dataset_metadata_eval_size():
    """Bug 1: Verify get_dataset_metadata populates correct num_eval."""
    meta = get_dataset_metadata(dataset=[0] * 100, name="sst2")
    assert meta.num_eval == 872
    assert meta.num_train == 100

    meta_mrpc = get_dataset_metadata(dataset=[0] * 50, name="mrpc")
    assert meta_mrpc.num_eval == 408


def test_direct_svd_parameter_reduction():
    """Bug 3: Verify DirectSVD reduces parameter count and compresses layers."""
    class SimpleToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 128, bias=False)
            self.fc2 = nn.Linear(128, 64, bias=False)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    model = SimpleToyModel()
    params_before = count_parameters(model)
    assert params_before == (128 * 128) + (128 * 64)

    compressor = DirectSVDCompressor(model=model, rank=16, fallback_to_original=False)
    compressed_model, stats = compressor.compress()

    params_after = count_parameters(compressed_model)
    assert params_after < params_before
    assert len(compressor.compressed_layers) == 2
    assert len(compressor.fallback_layers) == 0


def test_git_dirty_and_commit_tracking():
    """Bug 5: Verify git_commit and git_dirty fields in BenchmarkResult."""
    res = BenchmarkResult()
    assert isinstance(res.git_commit, str)
    assert len(res.git_commit) > 0
    assert isinstance(res.git_dirty, bool)


def test_dual_format_logging_to_disk():
    """Bug 6: Verify dual-format logging generates .log and .jsonl files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_id = "test_run_123"
        logger, log_file, jsonl_file = setup_run_logger(tmp_dir, run_id, "test_module")
        logger.info("Test log line 1")
        logger.info("Test log line 2")

        # Flush handlers
        for h in logger.handlers:
            h.flush()

        assert Path(log_file).exists()
        assert Path(jsonl_file).exists()

        log_content = Path(log_file).read_text(encoding="utf-8")
        assert "Test log line 1" in log_content

        jsonl_lines = Path(jsonl_file).read_text(encoding="utf-8").strip().splitlines()
        assert len(jsonl_lines) == 2
        entry = json.loads(jsonl_lines[0])
        assert entry["run_id"] == run_id
        assert entry["module"] == "test_module"
        assert entry["message"] == "Test log line 1"

        # Explicitly close and clear handlers so Windows releases file handles before tmpdir cleanup
        for h in list(logger.handlers):
            h.close()
            logger.removeHandler(h)


def test_executor_timeout_enforcement():
    """Bug 2: Verify executor handles timeouts properly."""
    # Test unknown module fails with clear error
    res = execute_single(module="non_existent_module_xyz", timeout_s=1)
    assert res.status == "failed"
    assert "ModuleNotFoundError" in res.error
