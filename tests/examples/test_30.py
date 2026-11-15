#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess and mathematical unit tests for Example 30: TTT-Online Test-Time Training.
"""

from __future__ import annotations

import subprocess
import sys
import torch
import torch.nn as nn

from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM
from neuralops.trainer.regimes.ttt_online import TTTOnlineAdapter


def test_30_runs_clean():
    """Verify example 30 executes cleanly end-to-end via subprocess."""
    proc = subprocess.run(
        [sys.executable, "examples/30_ttt_online_test_time_training.py"],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, f"Example 30 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 30 Status" in proc.stdout


def test_30_ttt_adaptation_and_exact_restoration():
    """Verify TTT parameter update during test-time adaptation and clean rollback."""
    torch.manual_seed(42)
    cfg = NeuralOpsConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
    )
    model = NeuralOpsForCausalLM(cfg)
    model.eval()

    inputs = {"input_ids": torch.randint(0, cfg.vocab_size, (2, 8))}

    with torch.no_grad():
        initial_out = model(**inputs).logits.clone()

    adapter = TTTOnlineAdapter(
        model=model,
        inner_lr=1e-2,
        inner_steps=2,
        target_substrings=["o_proj", "v_proj"],
    )

    target_params = adapter._get_target_parameters()
    assert len(target_params) > 0, "Expected target projection parameters for TTT"

    backups = [p.data.clone() for p in target_params]

    # Perform prompt adaptation
    res = adapter.adapt_on_prompt(inputs)
    assert "ttt_loss" in res
    assert res["ttt_loss"] > 0.0

    with torch.no_grad():
        adapted_out = model(**inputs).logits

    # Logits should have shifted due to parameter update
    shift = (adapted_out - initial_out).norm().item()
    assert shift > 0.0, "Expected model output to shift after TTT adaptation"

    # Roll back parameters
    adapter.reset_parameters(backups)

    with torch.no_grad():
        restored_out = model(**inputs).logits

    # Logits must match initial logits perfectly
    max_diff = (restored_out - initial_out).abs().max().item()
    assert max_diff < 1e-6, f"Expected clean rollback, got max diff {max_diff}"
