#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Test suite for Example 10: Advanced Model Compression Zoo.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import torch

from examples._shared import TinyClassifier, tiny_inputs
from neuralops.compression.svd.hnc_svd_continuation import HessianNullSpaceSVDContinuation


def test_10_runs_clean():
    """Verify example 10 executes cleanly via subprocess and exits 0."""
    proc = subprocess.run(
        [sys.executable, "examples/10_advanced_compression_zoo.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 10 failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "[PASS] Overall Example 10 Status" in proc.stdout
    assert "All 4 advanced compression paradigms evaluated with spectral metrics" in proc.stdout


def test_10_no_fake_hnc_claims():
    """Assert source code does not claim Hessian Null Space SVD without invoking actual continuation."""
    example_path = Path("examples/10_advanced_compression_zoo.py")
    content = example_path.read_text()

    # Must import and execute HessianNullSpaceSVDContinuation
    assert "HessianNullSpaceSVDContinuation" in content, "Missing HessianNullSpaceSVDContinuation integration"
    assert "hnc.run_continuation" in content, "Missing invocation of continuation engine"
    # No fake torch.randn for data/evaluation
    assert "torch.randn" not in content, "Found forbidden torch.randn in example 10"


def test_10_hnc_svd_spectral_reconstruction():
    """Verify genuine HNC-SVD continuation reconstructs weight with bounded relative error."""
    model = TinyClassifier().to("cpu")
    target_layer = model.transformer.distilbert.transformer.layer[0].ffn.lin1
    weight = target_layer.weight.data.clone()
    d_out, d_in = weight.shape
    target_rank = 16

    calib_tokens = tiny_inputs(batch_size=8, seq_len=16, seed=42)
    activations: list[torch.Tensor] = []

    def hook(module: torch.nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        activations.append(inp[0].detach().reshape(-1, inp[0].shape[-1]))

    h = target_layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(**calib_tokens)
    h.remove()

    x_act = activations[0]
    hnc = HessianNullSpaceSVDContinuation(
        target_rank=target_rank,
        walk_steps=20,
        restore_steps_m=3,
        step_size_eta=0.01,
        device="cpu",
    )
    w_recon = hnc.run_continuation(weight, probe_inputs=x_act)
    assert w_recon.shape == weight.shape, f"Reconstructed shape {w_recon.shape} != {weight.shape}"

    frob_orig = torch.norm(weight, p="fro").item()
    diff = weight - w_recon
    rel_error = torch.norm(diff, p="fro").item() / (frob_orig + 1e-12)

    # Relative Frobenius error must be well-bounded (< 0.65)
    assert rel_error < 0.65, f"HNC reconstruction relative error {rel_error:.4f} exceeded threshold 0.65"
