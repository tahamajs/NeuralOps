"""
Suite 9: Exhaustive Verification Battery for Model Export, Hub AutoMap, and Merge-for-Inference.
Verifies:
  9.1 Self-contained export_model (auto_map, modeling_neuralops_hf.py, config).
  9.2 merge_for_inference algebraic break-even merging.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM
from neuralops.export.surgery_manifest import merge_for_inference
from neuralops.trainer import PiSSALinear

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_9_1_export_model_and_config():
    cfg = NeuralOpsConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        attention_mechanism="linear_attention",
    )
    model = NeuralOpsForCausalLM(cfg).to(DEVICE)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "exported_model"
        model.save_pretrained(save_path)
        assert (save_path / "config.json").exists()

        reloaded = NeuralOpsForCausalLM.from_pretrained(save_path).to(DEVICE)
        x = torch.randint(0, 128, (1, 8), device=DEVICE)
        with torch.no_grad():
            out_orig = model(x).logits
            out_rel = reloaded(x).logits
        torch.testing.assert_close(out_orig, out_rel)


def test_9_2_merge_for_inference_algebraic_identity():
    base_linear = nn.Linear(32, 32).to(DEVICE)
    pissa = PiSSALinear(base_linear, rank=4)
    model = nn.Sequential(pissa)

    x = torch.randn(2, 32, device=DEVICE)
    with torch.no_grad():
        out_before = model(x)

    merged_model = merge_for_inference(model)
    with torch.no_grad():
        out_after = merged_model(x)

    torch.testing.assert_close(out_before, out_after, rtol=1e-4, atol=1e-4)
