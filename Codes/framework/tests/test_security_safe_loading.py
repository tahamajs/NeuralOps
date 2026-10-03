"""
Unit tests for safe model deserialization and checkpoint loading utilities.
Verifies enforcement of weights_only=True and safetensors integration.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from framework.security.config import SecurityConfig
from framework.security.deserialization import (is_safe_checkpoint_format,
                                                safe_load_checkpoint,
                                                safe_load_model_weights,
                                                safe_save_checkpoint)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestSecuritySafeDeserialization(unittest.TestCase):
    """Test suite for safe checkpoint loading and weights_only enforcement."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model = DummyModel()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_safe_checkpoint_format_validation(self):
        self.assertTrue(is_safe_checkpoint_format("model.safetensors"))
        self.assertTrue(is_safe_checkpoint_format("checkpoint.pt"))
        self.assertTrue(is_safe_checkpoint_format("weights.pth"))
        self.assertTrue(is_safe_checkpoint_format("model.ckpt"))
        self.assertFalse(is_safe_checkpoint_format("malicious.exe"))
        self.assertFalse(is_safe_checkpoint_format("script.sh"))

    def test_save_and_load_safetensors(self):
        state = self.model.state_dict()
        save_path = os.path.join(self.temp_dir, "model.safetensors")
        out = safe_save_checkpoint(state, save_path, use_safetensors=True)
        self.assertTrue(os.path.exists(out))
        self.assertTrue(out.endswith(".safetensors"))

        loaded = safe_load_checkpoint(out, map_location="cpu")
        self.assertIn("fc1.weight", loaded)
        self.assertTrue(torch.allclose(state["fc1.weight"], loaded["fc1.weight"]))

    def test_save_and_load_torch_weights_only(self):
        state = self.model.state_dict()
        save_path = os.path.join(self.temp_dir, "model.pt")
        out = safe_save_checkpoint(state, save_path, use_safetensors=False)
        self.assertTrue(os.path.exists(out))

        loaded = safe_load_checkpoint(out, map_location="cpu", weights_only=True)
        self.assertIn("fc2.bias", loaded)
        self.assertTrue(torch.allclose(state["fc2.bias"], loaded["fc2.bias"]))

    def test_safe_load_model_weights_direct(self):
        new_model = DummyModel()
        save_path = os.path.join(self.temp_dir, "direct_weights.safetensors")
        safe_save_checkpoint(self.model.state_dict(), save_path, use_safetensors=True)

        loaded_model = safe_load_model_weights(new_model, save_path, strict=True)
        self.assertIs(loaded_model, new_model)
        for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_security_config_defaults(self):
        cfg = SecurityConfig()
        self.assertTrue(cfg.enforce_weights_only)
        self.assertTrue(cfg.prefer_safetensors)
        self.assertIn(".safetensors", cfg.allowed_extensions)


if __name__ == "__main__":
    unittest.main()
