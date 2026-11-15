#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess test for examples/03_lora_finetuning.py.
"""

from __future__ import annotations

import subprocess
import sys
import torch

from examples._shared import TinyClassifier, tiny_inputs
import importlib

lora_module = importlib.import_module("examples.03_lora_finetuning")
inject_multi_target_lora = lora_module.inject_multi_target_lora
from neuralops.core.peft.lora import LoRALinear


def test_03_runs_clean():
    proc = subprocess.run(
        [sys.executable, "examples/03_lora_finetuning.py"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"Example 03 failed with stderr: {proc.stderr}"
    assert "[PASS] Overall Example 03 Status" in proc.stdout


def test_03_lora_merge_numerical_parity():
    """Verify that merged linear layer matches adapter forward output within 1e-4."""
    model = TinyClassifier().to("cpu")
    inject_multi_target_lora(model, rank=4)
    model.eval()

    inputs = tiny_inputs()
    with torch.no_grad():
        out_unmerged = model(**inputs).logits

    lora_modules: list[LoRALinear] = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(lora_modules) > 0, "No LoRALinear modules found"
    for mod in lora_modules:
        mod.merge()

    with torch.no_grad():
        out_merged = model(**inputs).logits

    assert torch.allclose(out_merged, out_unmerged, atol=1e-4)
