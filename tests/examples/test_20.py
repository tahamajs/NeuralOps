#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Verification tests for Example 20 deletion and audit verdict documentation.
"""

from __future__ import annotations

from pathlib import Path


def test_20_example_deleted():
    """Verify that fake Example 20 has been permanently removed."""
    example_path = Path("examples/20_long_context_niah_eval.py")
    assert not example_path.exists(), (
        f"{example_path} should be deleted because it fabricated NIAH via torch.randn."
    )


def test_20_audit_verdict_documented():
    """Verify that audit verdict documentation exists and contains expected rationale."""
    verdict_path = Path("audits/wave6/w6_14_verdict.md")
    assert verdict_path.exists(), "audits/wave6/w6_14_verdict.md must exist."

    content = verdict_path.read_text(encoding="utf-8")
    assert "DELETED" in content
    assert "torch.randn" in content
    assert "AutoModelForCausalLM" in content or "causal language model" in content


def test_20_deletions_file_documented():
    """Verify that audits/wave6/w6_14_deletions.md exists and documents deletion rationale."""
    deletions_path = Path("audits/wave6/w6_14_deletions.md")
    assert deletions_path.exists(), "audits/wave6/w6_14_deletions.md must exist."

    content = deletions_path.read_text(encoding="utf-8")
    assert "examples/20_long_context_niah_eval.py" in content
    assert "DELETED" in content
