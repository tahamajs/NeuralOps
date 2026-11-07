#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Test Hugging Face Hub self-contained export roundtrip and auto_map resolution.
"""

import tempfile
from pathlib import Path
import pytest
import torch
from transformers import AutoConfig, AutoModel, BertConfig, BertModel

import neuralops
from neuralops.models import NeuralOpsConfig, NeuralOpsModel, NeuralOpsForCausalLM


def test_export_model_safetensors_and_automap():
    config = NeuralOpsConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        attention_types=["linear", "linear"],
    )
    model = NeuralOpsModel(config)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "exported_neuralops"
        exported_dir = neuralops.export_model(model, out_path, format="safetensors")

        assert (exported_dir / "config.json").exists()
        assert (exported_dir / "model.safetensors").exists()
        assert (exported_dir / "configuration_neuralops.py").exists()
        assert (exported_dir / "modeling_neuralops.py").exists()
        assert (exported_dir / "modeling_block.py").exists()

        import json
        with open(exported_dir / "config.json", "r", encoding="utf-8") as f:
            c_dict = json.load(f)
        assert "auto_map" in c_dict
        assert "AutoConfig" in c_dict["auto_map"]
        assert "AutoModel" in c_dict["auto_map"]

        loaded_model = AutoModel.from_pretrained(str(exported_dir), trust_remote_code=True)
        assert loaded_model is not None

        x = torch.randint(0, 1000, (1, 8))
        with torch.no_grad():
            orig_out = model(x).last_hidden_state
            loaded_out = loaded_model(x).last_hidden_state
        assert torch.allclose(orig_out, loaded_out, atol=1e-5)


def test_push_to_hub_api_exposed():
    assert hasattr(neuralops, "push_to_hub")
    assert callable(neuralops.push_to_hub)
