"""
Unit tests for MultiModeBenchmarkRunner across diverse inference modes.
"""

import os
import shutil
import tempfile
import unittest

from neuralops.core.multi_mode_benchmark import MultiModeBenchmarkRunner


class TestMultiModeBenchmark(unittest.TestCase):
    """Test suite verifying MultiModeBenchmarkRunner mode execution and metric reporting."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_list_supported_modes(self):
        modes = MultiModeBenchmarkRunner.list_supported_modes()
        self.assertGreaterEqual(len(modes), 10)
        self.assertIn("dense_fp16", modes)
        self.assertIn("svd_linear_lowrank", modes)
        self.assertIn("rotational_slicegpt", modes)
        self.assertIn("mamba_ssm_scan", modes)

    def test_benchmark_attention_module(self):
        runner = MultiModeBenchmarkRunner(
            model_name_or_path="distilbert-base-uncased",
            device="cpu",
            batch_size=2,
            seq_len=16,
            num_samples=8,
            output_dir=self.temp_dir,
        )
        res = runner.benchmark_attention_module(
            attention_type="svd_linear",
            hidden_size=32,
            num_heads=2,
            seq_len=16,
            batch_size=2,
            num_iters=3,
        )
        self.assertIn("latency_ms", res)
        self.assertIn("peak_vram_mb", res)
        self.assertGreater(res["latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()
