"""
Unit tests for ResultsSaver, metric exports, configuration archiving, and checkpoints.
"""

import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
from neuralops.results_saver.saver import ResultsSaver


class TestResultsSaver(unittest.TestCase):
    """Test suite verifying ResultsSaver functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.saver = ResultsSaver(run_id="test_run_001", base_dir=self.temp_dir)
        self.dummy_model = nn.Linear(32, 32)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_metrics(self):
        metrics = {"eval_loss": 0.42, "eval_accuracy": 0.94, "ppl": 11.2}
        p = self.saver.save_metrics(metrics)
        self.assertTrue(os.path.exists(p))
        with open(p, "r") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["eval_loss"], 0.42)

    def test_save_config(self):
        config = {"model": "mamba_svd_flash", "lr": 1e-4, "rank": 16}
        p = self.saver.save_config(config)
        self.assertTrue(os.path.exists(p))
        with open(p, "r") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["model"], "mamba_svd_flash")

    def test_save_model_weights(self):
        p = self.saver.save_model_weights(self.dummy_model, filename="weights.pt")
        self.assertTrue(os.path.exists(p))
        state = torch.load(p, map_location="cpu")
        self.assertIn("weight", state)


if __name__ == "__main__":
    unittest.main()
