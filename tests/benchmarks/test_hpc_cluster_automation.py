#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests verifying HPC cluster automation, shell syntax, license headers, and disk quota pruning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/hpc is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hpc.bert_zoo_upload_hf import upload_directory


def test_hpc_shell_scripts_mit_license_and_syntax():
    """Verify all scripts/hpc/*.sh files have MIT copyright headers and pass bash -n syntax check."""
    hpc_dir = REPO_ROOT / "scripts" / "hpc"
    sh_files = list(hpc_dir.glob("*.sh"))
    assert len(sh_files) >= 5, f"Expected at least 5 shell scripts in scripts/hpc, found {len(sh_files)}"

    for sh_path in sh_files:
        content = sh_path.read_text(encoding="utf-8")
        assert content.startswith("#!/"), f"{sh_path.name} missing shebang on line 1"
        assert "Copyright (c) 2024-2026, Taha Majlesi" in content, (
            f"{sh_path.name} missing MIT copyright header"
        )
        assert "This source code is licensed under the MIT license" in content, (
            f"{sh_path.name} missing license notice"
        )

        # Run bash -n syntax validation
        res = subprocess.run(["bash", "-n", str(sh_path)], capture_output=True, text=True)
        assert res.returncode == 0, f"bash -n failed on {sh_path.name}: {res.stderr}"


def test_bert_zoo_upload_cleanup_on_success(tmp_path: Path):
    """Verify cluster disk quota cleanup purges heavy weights upon verified upload success."""
    model_dir = tmp_path / "bert_test_model"
    model_dir.mkdir()

    # Create metadata files
    (model_dir / "training_metrics.json").write_text(json.dumps({"loss": 0.42, "accuracy": 92.5}))
    (model_dir / "config.json").write_text(json.dumps({"vocab_size": 30522}))
    (model_dir / "README.md").write_text("# Test Model Card")

    # Create heavy weight files
    (model_dir / "attention_weights.safetensors").write_bytes(b"x" * 1024)
    (model_dir / "pytorch_model.bin").write_bytes(b"y" * 2048)
    (model_dir / "optimizer.pt").write_bytes(b"z" * 512)

    with patch("huggingface_hub.HfApi") as MockHfApi:
        mock_api = MagicMock()
        MockHfApi.return_value = mock_api
        mock_api.whoami.return_value = {"name": "tahamajs", "orgs": []}
        mock_api.create_repo.return_value = "https://huggingface.co/Hooshaai/bert-test-model-sst2"
        mock_api.upload_file.return_value = "commit_123"

        success = upload_directory(
            target_dir=model_dir,
            hf_org="Hooshaai",
            cleanup_weights=True,
            token="mock_hf_token",
        )

    assert success is True
    assert (model_dir / "upload_success.json").exists()

    # Heavy weights must be pruned to protect cluster disk quotas
    assert not (model_dir / "attention_weights.safetensors").exists()
    assert not (model_dir / "pytorch_model.bin").exists()
    assert not (model_dir / "optimizer.pt").exists()

    # Metadata and config files must be preserved
    assert (model_dir / "training_metrics.json").exists()
    assert (model_dir / "config.json").exists()
    assert (model_dir / "README.md").exists()


def test_bert_zoo_upload_cleanup_disabled(tmp_path: Path):
    """Verify heavy weights are preserved when cleanup_weights is explicitly False."""
    model_dir = tmp_path / "qwen_test_model"
    model_dir.mkdir()

    (model_dir / "training_metrics.json").write_text(json.dumps({"model": "qwen2.5-0.5b", "ppl": 15.2}))
    (model_dir / "config.json").write_text(json.dumps({"hidden_size": 896}))
    (model_dir / "model.safetensors").write_bytes(b"weights" * 100)

    with patch("huggingface_hub.HfApi") as MockHfApi:
        mock_api = MagicMock()
        MockHfApi.return_value = mock_api
        mock_api.whoami.return_value = {"name": "tahamajs", "orgs": []}
        mock_api.create_repo.return_value = "https://huggingface.co/Hooshaai/qwen-2.5-0.5b-test-model-wikitext2"
        mock_api.upload_file.return_value = "commit_456"

        success = upload_directory(
            target_dir=model_dir,
            hf_org="Hooshaai",
            cleanup_weights=False,
            token="mock_hf_token",
        )

    assert success is True
    assert (model_dir / "upload_success.json").exists()
    # Weights preserved because cleanup_weights=False
    assert (model_dir / "model.safetensors").exists()


def test_bert_zoo_upload_failure_preserves_weights(tmp_path: Path):
    """Verify weights are NEVER purged if upload to Hugging Face fails."""
    model_dir = tmp_path / "failed_model"
    model_dir.mkdir()

    (model_dir / "training_metrics.json").write_text(json.dumps({"acc": 88.0}))
    (model_dir / "attention_weights.safetensors").write_bytes(b"critical_weights")

    with patch("huggingface_hub.HfApi") as MockHfApi:
        mock_api = MagicMock()
        MockHfApi.return_value = mock_api
        mock_api.whoami.return_value = {"name": "tahamajs"}
        mock_api.create_repo.return_value = "https://huggingface.co/test"
        mock_api.upload_file.side_effect = ConnectionError("Network unreachable on HPC")

        success = upload_directory(
            target_dir=model_dir,
            hf_org="Hooshaai",
            cleanup_weights=True,
            token="mock_hf_token",
        )

    assert success is False
    assert (model_dir / "upload_error.json").exists()
    assert not (model_dir / "upload_success.json").exists()
    # Weights must NOT be purged when upload fails
    assert (model_dir / "attention_weights.safetensors").exists()
