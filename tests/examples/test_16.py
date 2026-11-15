#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess test for examples/16_distributed_parallelism.py.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import torch

from examples._shared import TinyClassifier, tiny_inputs
import importlib

ex16 = importlib.import_module("examples.16_distributed_parallelism")
simulate_tensor_parallel_mlp = ex16.simulate_tensor_parallel_mlp
simulate_context_parallel_attention = ex16.simulate_context_parallel_attention
simulate_layer_parallel_transformer = ex16.simulate_layer_parallel_transformer
find_attention_modules = ex16.find_attention_modules
find_transformer_blocks = ex16.find_transformer_blocks


def test_16_runs_clean():
    """Verify example 16 executes cleanly and returns exit code 0."""
    proc = subprocess.run(
        [sys.executable, "examples/16_distributed_parallelism.py"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"Example 16 failed with stderr: {proc.stderr}"
    assert "[PASS] Overall Example 16 Status" in proc.stdout


def test_16_no_fragile_paths():
    """Assert source code does not contain hardcoded layer traversal paths."""
    src = Path("examples/16_distributed_parallelism.py").read_text(encoding="utf-8")
    assert "layer[0]" not in src, "Found fragile hardcoded layer[0] in examples/16_distributed_parallelism.py"
    assert "transformer.layer[" not in src, "Found fragile hardcoded transformer.layer[ in source"
    assert "torch.randn" not in src, "Found torch.randn in evaluation tokens/weights in source"


def test_16_tp_mathematical_parity():
    """Verify that split tensor parallel MLP matches dense MLP within 1e-4."""
    model = TinyClassifier().eval()
    ffn_mods = [m for _name, m in model.named_modules() if hasattr(m, "lin1") and hasattr(m, "lin2")]
    assert len(ffn_mods) > 0, "No FFN modules found in TinyClassifier"
    ffn = ffn_mods[0]
    w1 = ffn.lin1.weight.detach()
    w2 = ffn.lin2.weight.detach()

    tokens = tiny_inputs(batch_size=4, seq_len=16)
    emb_mod = [m for _name, m in model.named_modules() if hasattr(m, "word_embeddings")][0]
    with torch.no_grad():
        x = emb_mod(tokens["input_ids"]).detach()[:, 0, :]

    y_dense, y_tp, diff = simulate_tensor_parallel_mlp(x, w1, w2)
    assert diff < 1e-4, f"TP diff {diff} exceeds tolerance 1e-4"
    assert torch.allclose(y_dense, y_tp, atol=1e-4)


def test_16_cp_mathematical_parity():
    """Verify that simulated context parallel attention matches full attention within 1e-4."""
    model = TinyClassifier().eval()
    tokens = tiny_inputs(batch_size=2, seq_len=16)
    emb_mod = [m for _name, m in model.named_modules() if hasattr(m, "word_embeddings")][0]
    with torch.no_grad():
        hidden_states = emb_mod(tokens["input_ids"])
        attn_mods = find_attention_modules(model)
        assert len(attn_mods) > 0, "No attention modules found"
        attn = attn_mods[0][1]
        q = attn.q_lin(hidden_states)
        k = attn.k_lin(hidden_states)
        v = attn.v_lin(hidden_states)

    y_single, y_cp, diff = simulate_context_parallel_attention(q, k, v)
    assert diff < 1e-4, f"CP diff {diff} exceeds tolerance 1e-4"
    assert torch.allclose(y_single, y_cp, atol=1e-4)


def test_16_dynamic_discovery():
    """Verify dynamic discovery helpers correctly locate modules in model."""
    model = TinyClassifier().eval()
    attns = find_attention_modules(model)
    blocks = find_transformer_blocks(model)
    assert len(attns) >= 2, f"Expected at least 2 attention modules, got {len(attns)}"
    assert len(blocks) >= 2, f"Expected at least 2 transformer blocks, got {len(blocks)}"
