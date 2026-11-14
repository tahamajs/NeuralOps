#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for AsyncHFHubUploader submit_upload signature compatibility shim (P0-13).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from neuralops.hub.uploader import AsyncHFHubUploader


def test_standard_signature_convention(tmp_path: Path) -> None:
    """Test calling submit_upload(attn_key, staging_dir, metrics)."""
    uploader = AsyncHFHubUploader(max_workers=1)
    uploader.token = None  # Force no-token mode to avoid network call in unit test
    staging_dir = tmp_path / "stage_standard"
    staging_dir.mkdir(parents=True, exist_ok=True)

    future = uploader.submit_upload("vanilla_mha", staging_dir, {"ppl_after": 3.14})
    result = future.result(timeout=5)

    uploader.shutdown(wait=True)

    assert result["status"] == "SKIPPED_NO_TOKEN"
    err_file = staging_dir / "upload_error.json"
    assert err_file.exists()


def test_caller_pattern_with_mechanism_name(tmp_path: Path) -> None:
    """Test calling submit_upload(ckpt_dir, repo_id, mechanism_name=key)."""
    uploader = AsyncHFHubUploader(max_workers=1)
    uploader.token = None
    ckpt_dir = tmp_path / "stage_caller"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    future = uploader.submit_upload(ckpt_dir, "Hooshaai/test-repo", mechanism_name="cosformer")
    result = future.result(timeout=5)

    uploader.shutdown(wait=True)

    assert result["status"] == "SKIPPED_NO_TOKEN"
    err_file = ckpt_dir / "upload_error.json"
    assert err_file.exists()


def test_caller_pattern_with_string_path(tmp_path: Path) -> None:
    """Test calling submit_upload(str_dir, repo_id, mechanism_name=key)."""
    uploader = AsyncHFHubUploader(max_workers=1)
    uploader.token = None
    ckpt_dir = tmp_path / "stage_str"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    future = uploader.submit_upload(str(ckpt_dir), "Hooshaai/test-repo", mechanism_name="linformer")
    result = future.result(timeout=5)

    uploader.shutdown(wait=True)

    assert result["status"] == "SKIPPED_NO_TOKEN"
    assert (ckpt_dir / "upload_error.json").exists()


def test_caller_pattern_without_mechanism_name(tmp_path: Path) -> None:
    """Test calling submit_upload(ckpt_dir, repo_id)."""
    uploader = AsyncHFHubUploader(max_workers=1)
    uploader.token = None
    ckpt_dir = tmp_path / "stage_inferred"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    future = uploader.submit_upload(ckpt_dir, "Hooshaai/test-repo-inferred")
    result = future.result(timeout=5)

    uploader.shutdown(wait=True)

    assert result["status"] == "SKIPPED_NO_TOKEN"
    assert (ckpt_dir / "upload_error.json").exists()


def test_mock_successful_upload_with_both_conventions(tmp_path: Path) -> None:
    """Verify that worker receives correct repo_id and attn_key under both conventions."""
    uploader = AsyncHFHubUploader(max_workers=1)

    with patch.object(uploader, "_worker_upload_and_cleanup") as mock_worker:
        mock_worker.return_value = {"status": "UPLOAD_SUCCESS"}

        dir_a = tmp_path / "stage_a"
        fut_a = uploader.submit_upload("key_a", dir_a, {"acc": 95.0}, repo_id="Hooshaai/repo_a")
        res_a = fut_a.result(timeout=5)
        assert res_a["status"] == "UPLOAD_SUCCESS"
        mock_worker.assert_called_with("key_a", dir_a, {"acc": 95.0}, "Hooshaai/repo_a", None)

        mock_worker.reset_mock()
        dir_b = tmp_path / "stage_b"
        fut_b = uploader.submit_upload(dir_b, "Hooshaai/repo_b", mechanism_name="key_b")
        res_b = fut_b.result(timeout=5)
        assert res_b["status"] == "UPLOAD_SUCCESS"
        mock_worker.assert_called_with("key_b", dir_b, {}, "Hooshaai/repo_b", None)

    uploader.shutdown(wait=True)
