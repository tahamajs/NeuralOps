"""
Unit tests for asynchronous telemetry callbacks and non-blocking .cpu().numpy() processing.
"""

import os
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from framework.callbacks.analysis import (ActivationStatsCallback,
                                          AttentionHeatmapCallback,
                                          LayerwiseAnalysisCallback,
                                          SVDAttentionCallback)
from framework.callbacks.base import AsyncTelemetryCallback


class DummyAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32)
        self.k_proj = nn.Linear(32, 32)
        self.v_proj = nn.Linear(32, 32)

    def forward(self, input_ids=None):
        return None


class DummyTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = DummyAttentionModule()
        self.fc = nn.Linear(32, 10)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, output_attentions=False, output_hidden_states=False):
        batch_size = 2
        seq_len = 16
        hidden_dim = 32
        hidden = torch.randn(batch_size, seq_len, hidden_dim)
        attentions = [torch.rand(batch_size, 4, seq_len, seq_len)] if output_attentions else None
        hidden_states = [hidden, hidden * 0.9] if output_hidden_states else None
        
        return SimpleNamespace(
            logits=torch.randn(batch_size, 10),
            attentions=attentions,
            hidden_states=hidden_states,
        )


class TestAsyncTelemetryCallbacks(unittest.TestCase):
    """Test suite verifying non-blocking async execution in telemetry callbacks."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model = DummyTransformerModel()
        self.sample_inputs = {"input_ids": torch.randint(0, 100, (2, 16))}

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_async_telemetry_base_executor(self):
        cb = AsyncTelemetryCallback(max_workers=2)
        future = cb.submit_async(lambda x: x * 2, 21)
        self.assertEqual(future.result(timeout=2.0), 42)
        cb.on_train_end(None, None, None)

    def test_attention_heatmap_async_generation(self):
        cb = AttentionHeatmapCallback(
            output_dir=self.temp_dir,
            sample_inputs=self.sample_inputs,
            log_every_steps=1,
            max_workers=2,
        )
        state = SimpleNamespace(global_step=1)
        control = SimpleNamespace()
        cb.on_evaluate(None, state, control, model=self.model)

        # Wait for background thread pool execution
        cb.on_train_end(None, None, None)

        expected_image = os.path.join(self.temp_dir, "attn_heatmap_step1.png")
        self.assertTrue(os.path.exists(expected_image))
        self.assertGreater(os.path.getsize(expected_image), 0)

    def test_svd_attention_callback_async(self):
        cb = SVDAttentionCallback(layer_interval=1, log_every_steps=1, max_workers=2)
        state = SimpleNamespace(global_step=1)
        control = SimpleNamespace()
        cb.on_step_end(None, state, control, model=self.model)
        cb.on_train_end(None, None, None)

    def test_layerwise_analysis_callback_async(self):
        cb = LayerwiseAnalysisCallback(
            log_every_steps=1,
            sample_inputs=self.sample_inputs,
            max_workers=2,
        )
        state = SimpleNamespace(global_step=1)
        control = SimpleNamespace()
        cb.on_evaluate(None, state, control, model=self.model)
        cb.on_train_end(None, None, None)

    def test_activation_stats_callback_async(self):
        cb = ActivationStatsCallback(sample_inputs=self.sample_inputs, max_workers=2)
        state = SimpleNamespace(global_step=1)
        control = SimpleNamespace()
        cb.on_evaluate(None, state, control, model=self.model)
        cb.on_train_end(None, None, None)
        self.assertGreater(len(cb.stats), 0)


if __name__ == "__main__":
    unittest.main()
