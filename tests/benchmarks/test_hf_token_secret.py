#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests verifying HF token secret redaction and CLI argument refusal."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from neuralops.benchmarks.cli import _build_parser
from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.hub.uploader import AsyncHFHubUploader, _redact_secrets, retry_hf_api_call


def test_to_dict_excludes_token():
    """Assert to_dict and repr do not expose plaintext HF token."""
    secret = "hf_xxx_secret"
    cfg = BenchmarkConfig(hf_token=secret)
    serialized = json.dumps(cfg.to_dict())
    assert secret not in serialized, f"Plaintext secret {secret!r} found in to_dict() serialization"
    assert secret not in repr(cfg), f"Plaintext secret {secret!r} found in repr(config)"


def test_save_excludes_token(tmp_path: Path):
    """Assert config.save() writes JSON without exposing plaintext HF token."""
    secret = "hf_xxx_secret"
    cfg = BenchmarkConfig(hf_token=secret)
    save_file = tmp_path / "benchmark_config.json"
    cfg.save(save_file)

    saved_text = save_file.read_text(encoding="utf-8")
    assert secret not in saved_text, f"Plaintext secret {secret!r} leaked to saved JSON config file"


def test_no_hf_token_cli_flag():
    """Assert --hf-token is either not in the parser or warns and refuses the value."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--hf-token", "hf_test_secret"])


def test_async_uploader_repr_excludes_token():
    """Assert AsyncHFHubUploader.__repr__ does not expose plaintext token."""
    secret = "hf_token_super_secret_12345"
    uploader = AsyncHFHubUploader(token=secret)
    assert secret not in repr(uploader)
    assert "token='set'" in repr(uploader)


def test_async_uploader_reads_hf_token_env(monkeypatch: pytest.MonkeyPatch):
    """Assert AsyncHFHubUploader prioritizes HF_TOKEN env var."""
    secret = "hf_env_secret_token_abcdef"
    monkeypatch.setenv("HF_TOKEN", secret)
    uploader = AsyncHFHubUploader()
    assert uploader.token == secret
    assert secret not in repr(uploader)


def test_async_uploader_redacts_token_on_failure(tmp_path: Path):
    """Assert _worker_upload_and_cleanup redacts secret token from error file."""
    secret = "hf_secret_token_do_not_leak_98765"
    uploader = AsyncHFHubUploader(token=secret)
    stage = tmp_path / "stage"
    stage.mkdir()

    # Trigger a failure containing the secret
    mock_api = MagicMock()
    mock_api.create_repo.side_effect = RuntimeError(f"Connection failed to api.huggingface.co with token {secret}")
    with patch("huggingface_hub.HfApi", return_value=mock_api):
        res = uploader._worker_upload_and_cleanup("test_key", stage, {})

    assert res["status"] == "UPLOAD_FAILED"
    assert secret not in res["error"]
    assert "[REDACTED_HF_TOKEN]" in res["error"]

    err_file = stage / "upload_error.json"
    assert err_file.exists()
    err_text = err_file.read_text(encoding="utf-8")
    assert secret not in err_text
    assert "[REDACTED_HF_TOKEN]" in err_text


def test_uploader_retry_hf_api_call_rate_limit():
    """Assert retry_hf_api_call handles HTTP 429 rate limit with backoff."""
    class Mock429(Exception):
        status_code = 429

    mock_func = MagicMock()
    mock_func.side_effect = [Mock429("Rate limit exceeded"), "success"]
    result = retry_hf_api_call(mock_func, max_retries=3, base_delay=0.01)
    assert result == "success"
    assert mock_func.call_count == 2


def test_redact_secrets_filters_patterns():
    """Assert _redact_secrets masks explicit tokens and regex patterns."""
    raw = "Failed with token hf_123456789012345678901234567890 in URL"
    clean = _redact_secrets(raw)
    assert "hf_1234567890" not in clean
    assert "[REDACTED_HF_TOKEN]" in clean


def test_upload_with_retry_requires_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Assert upload_with_retry raises ValueError if no token is available."""
    from neuralops.hub.uploader import upload_with_retry

    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    try:
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    except Exception:
        pass
    with pytest.raises(ValueError, match="HF_TOKEN environment variable is strictly required"):
        upload_with_retry("test/repo", tmp_path, token=None)


