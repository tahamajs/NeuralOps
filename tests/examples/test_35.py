#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for examples/35_cuda_graph_decode_runner.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM
from neuralops.models.cuda_graph import CUDAGraphDecodeRunner

EXAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "35_cuda_graph_decode_runner.py"


def test_35_runs_clean_cpu():
    """Verify Example 35 runs cleanly via subprocess on CPU."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--device", "cpu", "--steps", "3"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 35 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 35 Status" in proc.stdout
    assert "Runner Initialized" in proc.stdout
    assert "Autoregressive Decode Sequence Executed" in proc.stdout


def test_35_decode_runner_contracts():
    """Verify CUDAGraphDecodeRunner initialization, batch shape verification, and step decoding."""
    cfg = NeuralOpsConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=32,
    )
    model = NeuralOpsForCausalLM(cfg)
    model.eval()

    device = torch.device("cpu")
    runner = CUDAGraphDecodeRunner(
        model=model,
        batch_size=1,
        device=device,
        warmup_steps=1,
    )

    assert runner.batch_size == 1
    assert runner.device == device

    runner.capture()
    assert not runner.is_captured  # CUDA Graph requires CUDA device

    # Test single-token decode step
    logits = runner.decode_step(token_id=7)
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert torch.isfinite(logits).all()
