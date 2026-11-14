#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for Hugging Face Hub uploader backward compatibility and secret redaction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from neuralops.benchmarks.kd.zoo_orchestrator import ZooOrchestrator
from neuralops.hub.uploader import AsyncHFHubUploader, SECRET_PATTERNS, _redact_secrets


def test_uploader_canonical_signature(tmp_path: Path) -> None:
    """Verify submit_upload works with canonical signature (attn_key, staging_dir, metrics, repo_id, zoo_type)."""
    uploader = AsyncHFHubUploader(token=None)
    mock_worker = MagicMock(return_value={"status": "MOCK_OK"})

    with patch.object(uploader, "_worker_upload_and_cleanup", mock_worker):
        metrics = {"post_kd_acc": 92.5, "post_kd_ppl": 14.1}
        fut = uploader.submit_upload(
            attn_key="gated_delta_net",
            staging_dir=tmp_path,
            metrics=metrics,
            repo_id="Hooshaai/test-gated-delta-net",
            zoo_type="bert",
        )
        res = fut.result()
        assert res["status"] == "MOCK_OK"
        mock_worker.assert_called_once_with(
            "gated_delta_net",
            tmp_path,
            metrics,
            "Hooshaai/test-gated-delta-net",
            "bert",
        )
    uploader.shutdown(wait=False)


def test_uploader_legacy_signature_mechanism_kwarg(tmp_path: Path) -> None:
    """Verify submit_upload works with legacy caller (ckpt_dir, repo_id, mechanism_name=key)."""
    uploader = AsyncHFHubUploader(token=None)
    mock_worker = MagicMock(return_value={"status": "MOCK_OK"})

    with patch.object(uploader, "_worker_upload_and_cleanup", mock_worker):
        fut = uploader.submit_upload(
            tmp_path,
            "Hooshaai/test-legacy-repo",
            mechanism_name="linear_attention",
        )
        res = fut.result()
        assert res["status"] == "MOCK_OK"
        mock_worker.assert_called_once_with(
            "linear_attention",
            tmp_path,
            {},
            "Hooshaai/test-legacy-repo",
            None,
        )
    uploader.shutdown(wait=False)


def test_uploader_folder_path_kwargs(tmp_path: Path) -> None:
    """Verify submit_upload works with keyword folder_path, repo_id, mechanism_name, metrics."""
    uploader = AsyncHFHubUploader(token=None)
    mock_worker = MagicMock(return_value={"status": "MOCK_OK"})

    with patch.object(uploader, "_worker_upload_and_cleanup", mock_worker):
        custom_metrics = {"loss": 0.05}
        fut = uploader.submit_upload(
            folder_path=tmp_path,
            repo_id="Hooshaai/test-kwarg-repo",
            mechanism_name="rwkv6_attention",
            metrics=custom_metrics,
        )
        res = fut.result()
        assert res["status"] == "MOCK_OK"
        mock_worker.assert_called_once_with(
            "rwkv6_attention",
            tmp_path,
            custom_metrics,
            "Hooshaai/test-kwarg-repo",
            None,
        )
    uploader.shutdown(wait=False)


def test_uploader_path_positional_signature(tmp_path: Path) -> None:
    """Verify submit_upload works when passing Path object and repo_id positionally."""
    uploader = AsyncHFHubUploader(token=None)
    mock_worker = MagicMock(return_value={"status": "MOCK_OK"})

    with patch.object(uploader, "_worker_upload_and_cleanup", mock_worker):
        fut = uploader.submit_upload(tmp_path, "Hooshaai/direct-path-repo")
        res = fut.result()
        assert res["status"] == "MOCK_OK"
        mock_worker.assert_called_once_with(
            tmp_path.name,
            tmp_path,
            {},
            "Hooshaai/direct-path-repo",
            None,
        )
    uploader.shutdown(wait=False)


def test_uploader_secret_redaction_all_8_patterns() -> None:
    """Verify all 8 secret token patterns are present and redacted by _redact_secrets."""
    assert len(SECRET_PATTERNS) == 8, f"Expected 8 patterns, found {len(SECRET_PATTERNS)}"

    secrets = [
        ("hf_abcdefghijklmnopqrstuvwxyz1234567890", "Hugging Face"),
        ("sk-abcdefghijklmnopqrstuvwxyz1234567890", "OpenAI"),
        ("ghp_123456789012345678901234567890123456", "GitHub Personal Token"),
        ("ghs_123456789012345678901234567890123456", "GitHub Server Token"),
        ("AKIAIOSFODNN7EXAMPLE", "AWS Access Key"),
        ("AIzaSyD-xyz1234567890123456789012345678", "Google API Key"),
        ("pypi-AgEIcHlwaS5vcmc-example-token-that-is-at-least-fifty-chars-long-and-more", "PyPI Token"),
        ('api_key="1234567890abcdef123"', "Generic API Key"),
    ]

    for secret_val, name in secrets:
        # 1. Pattern matching check
        matched = any(p.search(secret_val) for p in SECRET_PATTERNS)
        assert matched, f"Pattern failed to match {name}: {secret_val}"

        # 2. Redaction check
        message = f"Error during request: failed with credentials {secret_val} on remote endpoint"
        redacted = _redact_secrets(message)
        assert secret_val not in redacted, f"Secret leaked for {name}: {redacted}"
        assert "[REDACTED_HF_TOKEN]" in redacted, f"Redaction marker missing for {name}: {redacted}"


def test_uploader_secret_redaction_env_and_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify env vars and explicit secrets are scrubbed."""
    env_token = "my_custom_env_token_secret_12345"
    explicit_token = "my_explicit_token_secret_67890"

    monkeypatch.setenv("HF_TOKEN", env_token)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub_token_xyz987")

    msg = f"Connecting with {env_token} and {explicit_token} and hub_token_xyz987"
    cleaned = _redact_secrets(msg, secret=explicit_token)

    assert env_token not in cleaned
    assert explicit_token not in cleaned
    assert "hub_token_xyz987" not in cleaned
    assert "[REDACTED_HF_TOKEN]" in cleaned


def test_zoo_orchestrator_live_uploader_shim_dispatch(tmp_path: Path) -> None:
    """Integration test: ZooOrchestrator calling submit_upload via the real AsyncHFHubUploader."""
    uploader = AsyncHFHubUploader(token=None, cleanup_weights=False)
    mock_worker = MagicMock(return_value={"status": "UPLOAD_SUCCESS", "repo_id": "Hooshaai/test-mech"})

    with patch.object(uploader, "_worker_upload_and_cleanup", mock_worker):
        orchestrator = ZooOrchestrator(
            zoo_type="attention",
            out_dir=tmp_path / "out",
            metrics_log=tmp_path / "metrics.jsonl",
            hf_org="Hooshaai",
            hf_upload=False,
        )
        orchestrator.uploader = uploader

        def train_fn(key: str, ckpt_dir: Path) -> dict[str, Any]:
            return {"metrics": {"accuracy": 0.96}, "repo_id": f"Hooshaai/{key}"}

        results = orchestrator.run_zoo(
            keys=["gated_delta_net"],
            train_fn=train_fn,
            key_field="attn_key",
        )

        assert len(results) == 1
        assert results[0]["status"] == "PASSED"
        assert mock_worker.called
    uploader.shutdown(wait=False)
