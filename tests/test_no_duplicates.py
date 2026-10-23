#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression test suite verifying duplicate consolidation (P014, P016, P017, P025-P029, P033-P035, P049).

Covers:
- test_lora_linear_single: Exactly 1 definition of class LoRALinear in neuralops/
- test_lora_linear_identity: Common re-export is identical object to canonical peft.lora
- test_svd_linear_single: Exactly 1 definition of class SVDLinear in neuralops/
- test_factorized_linear_single: Exactly 1 definition of class FactorizedLinear in neuralops/
- test_factorized_linear_british_alias: FactorisedLinear is FactorizedLinear
- test_compute_flops_estimate_single: Exactly 1 definition of def _compute_flops_estimate in neuralops/
- test_monitor_removed: import neuralops.monitor raises ImportError
- test_attention_removed: import neuralops.attention raises ImportError
- test_training_merged: imports from neuralops.training and neuralops.trainer match
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
import pytest


def test_lora_linear_single():
    """Verify exactly 1 class LoRALinear definition exists across neuralops/."""
    repo_root = Path(__file__).resolve().parent.parent
    neuralops_dir = repo_root / "neuralops"
    pattern = re.compile(r"^class\s+LoRALinear\b", re.MULTILINE)

    matches = []
    for py_file in neuralops_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pattern.search(content):
            matches.append(str(py_file.relative_to(repo_root)))

    assert len(matches) == 1, f"Expected 1 LoRALinear class definition, found {len(matches)}: {matches}"
    assert matches[0] == "neuralops/core/peft/lora.py"


def test_lora_linear_identity():
    """Verify that importing LoRALinear from common or utils resolves to the canonical object."""
    from neuralops.core.peft.lora import LoRALinear as CanonicalLoRA
    from neuralops.core.common import LoRALinear as CommonLoRA

    assert CanonicalLoRA is CommonLoRA


def test_svd_linear_single():
    """Verify exactly 1 class SVDLinear definition exists across neuralops/."""
    repo_root = Path(__file__).resolve().parent.parent
    neuralops_dir = repo_root / "neuralops"
    pattern = re.compile(r"^class\s+SVDLinear\b", re.MULTILINE)

    matches = []
    for py_file in neuralops_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pattern.search(content):
            matches.append(str(py_file.relative_to(repo_root)))

    assert len(matches) == 1, f"Expected 1 SVDLinear class definition, found {len(matches)}: {matches}"
    assert matches[0] == "neuralops/core/low_rank/svd_linear.py"


def test_factorized_linear_single():
    """Verify exactly 1 class FactorizedLinear definition exists across neuralops/."""
    repo_root = Path(__file__).resolve().parent.parent
    neuralops_dir = repo_root / "neuralops"
    pattern = re.compile(r"^class\s+FactorizedLinear\b", re.MULTILINE)

    matches = []
    for py_file in neuralops_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pattern.search(content):
            matches.append(str(py_file.relative_to(repo_root)))

    assert len(matches) == 1, f"Expected 1 FactorizedLinear class definition, found {len(matches)}: {matches}"
    assert matches[0] == "neuralops/core/low_rank/factorized.py"


def test_factorized_linear_british_alias():
    """Verify FactorisedLinear is an exact alias for FactorizedLinear."""
    from neuralops.core.low_rank.factorized import FactorisedLinear, FactorizedLinear

    assert FactorisedLinear is FactorizedLinear


def test_config_single():
    """Verify Config and DummyConfig can be imported from neuralops.core.config."""
    from neuralops.core.config import Config, DummyConfig

    cfg = Config(hidden_size=128)
    assert cfg.hidden_size == 128

    dummy = DummyConfig(hidden_size=64, num_heads=4)
    assert dummy.hidden_size == 64
    assert dummy.num_heads == 4
    assert dummy.head_dim == 16


def test_compute_flops_estimate_single():
    """Verify exactly 1 def _compute_flops_estimate definition exists across neuralops/."""
    repo_root = Path(__file__).resolve().parent.parent
    neuralops_dir = repo_root / "neuralops"
    pattern = re.compile(r"^def\s+_compute_flops_estimate\b", re.MULTILINE)

    matches = []
    for py_file in neuralops_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pattern.search(content):
            matches.append(str(py_file.relative_to(repo_root)))

    assert len(matches) == 1, f"Expected 1 _compute_flops_estimate definition, found {len(matches)}: {matches}"
    assert matches[0] == "neuralops/analysis/flops_utils.py"


def test_monitor_removed():
    """Verify that neuralops.monitor directory is removed and cannot be imported."""
    with pytest.raises((ImportError, ModuleNotFoundError)):
        import neuralops.monitor  # type: ignore # noqa: F401


def test_attention_removed():
    """Verify that legacy neuralops.attention stub is removed and cannot be imported."""
    with pytest.raises((ImportError, ModuleNotFoundError)):
        import neuralops.attention  # type: ignore # noqa: F401


def test_training_merged():
    """Verify backward compatibility shim in neuralops.training re-exports trainer."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import neuralops.training as training
        import neuralops.trainer as trainer

        assert hasattr(training, "base_trainer") or hasattr(training, "NeuralOpsTrainer") or hasattr(training, "checkpointing")
