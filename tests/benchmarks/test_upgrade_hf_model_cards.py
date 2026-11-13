#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for `scripts/benchmarks/upgrade_hf_model_cards.py`."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "benchmarks"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from upgrade_hf_model_cards import (  # noqa: E402
    MANDATORY_TAGS,
    audit_readme_compliance,
    enhance_readme_content,
    retry_hf_api_call,
)


def test_enhance_readme_content_adds_all_required_sections():
    raw_readme = """---
language:
  - en
license: mit
---
# Test Model Card
This is a test model card without mandatory disclosures.
"""
    repo_id = "Hooshaai/bert-test-sst2"
    enhanced = enhance_readme_content(
        original_readme=raw_readme,
        repo_id=repo_id,
        family="bert_attention",
        metrics={"post_kd_acc": 91.25, "post_kd_ppl": 12.4},
    )

    # 1. Universal loader
    assert "from neuralops.hub import load_surgeried_model" in enhanced
    assert f'load_surgeried_model("{repo_id}")' in enhanced

    # 2. 0 MB Teacher GPU VRAM disclosure per AGENTS.md §21.1
    assert "0 MB" in enhanced
    assert "Offline Teacher Representation Caching" in enhanced
    assert "Teacher GPU VRAM Allocation" in enhanced

    # 3. Hardware telemetry (NVIDIA RTX 3090, 24GB GDDR6X, CUDA 11.3)
    assert "NVIDIA RTX 3090" in enhanced
    assert "24GB GDDR6X" in enhanced
    assert "CUDA 11.3" in enhanced

    # 4. Benchmark evaluation metrics (SST-2 Accuracy, WikiText-2 PPL, ScienceQA Accuracy)
    assert "SST-2 Accuracy" in enhanced
    assert "WikiText-2 PPL" in enhanced
    assert "ScienceQA Accuracy" in enhanced

    # 5. Mandatory tags and MIT license
    assert "license: mit" in enhanced
    for tag in MANDATORY_TAGS:
        assert tag in enhanced


def test_audit_readme_compliance():
    # Deficient readme
    bad_readme = "# Incomplete Model\nNo disclosures."
    audit_bad = audit_readme_compliance(bad_readme)
    assert not audit_bad["fully_compliant"]
    assert not audit_bad["universal_loader"]
    assert not audit_bad["zero_mb_teacher_vram"]
    assert not audit_bad["hardware_telemetry"]
    assert not audit_bad["evaluation_metrics"]
    assert not audit_bad["mandatory_tags"]

    # Enhanced readme
    good_readme = enhance_readme_content(
        original_readme=bad_readme,
        repo_id="Hooshaai/bert-adaptive-span-attention-sst2",
        family="bert_attention",
        metrics={"post_kd_acc": 91.06},
    )
    audit_good = audit_readme_compliance(good_readme)
    assert audit_good["universal_loader"]
    assert audit_good["zero_mb_teacher_vram"]
    assert audit_good["hardware_telemetry"]
    assert audit_good["evaluation_metrics"]
    assert audit_good["mandatory_tags"]
    assert audit_good["mit_license"]
    assert audit_good["fully_compliant"]


class MockHTTPError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_retry_hf_api_call_rate_limit_and_timeout():
    mock_func = MagicMock()

    # Simulate 429 rate limit then success
    rate_limit_exc = MockHTTPError("429 Client Error: Too Many Requests", status_code=429)

    mock_func.side_effect = [rate_limit_exc, "upload_ok"]
    result = retry_hf_api_call(mock_func, max_retries=3, base_delay=0.01)
    assert result == "upload_ok"
    assert mock_func.call_count == 2

    # Simulate timeout error then success
    mock_func.reset_mock()
    timeout_exc = TimeoutError("Connection to huggingface.co timed out")
    mock_func.side_effect = [timeout_exc, "upload_ok_after_timeout"]
    result2 = retry_hf_api_call(mock_func, max_retries=3, base_delay=0.01)
    assert result2 == "upload_ok_after_timeout"
    assert mock_func.call_count == 2


def test_retry_hf_api_call_fatal_error_not_retried():
    mock_func = MagicMock()
    fatal_exc = ValueError("Invalid authentication token format")
    mock_func.side_effect = fatal_exc

    with pytest.raises(ValueError, match="Invalid authentication token format"):
        retry_hf_api_call(mock_func, max_retries=3, base_delay=0.01)

    assert mock_func.call_count == 1
