#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for HuggingFace and Git exporter utilities."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from neuralops.utils.hf_git_exporter import (
    sanitize_slug,
    generate_model_card,
    find_git_root,
    upload_to_huggingface,
    update_documentation_table,
)


def test_sanitize_slug():
    """Verify string slugification for repository identifiers."""
    assert sanitize_slug("Meta-Llama/Llama-2-7b") == "meta-llama-llama-2-7b"
    assert sanitize_slug("Direct SVD\\Layer") == "direct-svd-layer"


def test_generate_model_card():
    """Verify Markdown model card generation contains metrics and config."""
    card = generate_model_card(
        model_name="Llama-2-7b",
        module_name="svd_linear_attention",
        dataset_name="wikitext",
        perf_metrics={"accuracy_pct": 89.5, "f1_score": 0.88, "val_loss": 0.42},
        comp_metrics={"compression_rate": "1.50", "num_compressed_layers": 12},
        config_dict={"rank": 32, "target_ratio": 0.5},
    )
    assert "Llama-2-7b" in card
    assert "svd_linear_attention" in card
    assert "89.50%" in card
    assert '"rank": 32' in card


def test_find_git_root():
    """Verify repository root detection."""
    root = find_git_root()
    assert (root / "pyproject.toml").exists() or (root / ".git").exists()


def test_upload_to_huggingface_retry_and_success(tmp_path):
    """Verify upload_to_huggingface calls HfApi.upload_folder with retries."""
    export_dir = tmp_path / "model_export"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}", encoding="utf-8")

    mock_api = MagicMock()
    # First attempt fails, second attempt succeeds
    attempts = [0]

    def mock_upload_folder(*args, **kwargs):
        attempts[0] += 1
        if attempts[0] == 1:
            raise ConnectionError("Transient network failure")
        return "https://huggingface.co/Hooshaai/test-repo"

    mock_api.upload_folder.side_effect = mock_upload_folder

    mock_hf_hub = types.ModuleType("huggingface_hub")
    mock_hf_hub.HfApi = MagicMock(return_value=mock_api)

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
        url = upload_to_huggingface(
            export_dir=export_dir,
            repo_id="Hooshaai/test-repo",
            model_name="bert",
            module_name="svd",
            dataset_name="sst2",
            perf_metrics={"accuracy_pct": 90.0},
            comp_metrics={},
        )
        assert url == "https://huggingface.co/Hooshaai/test-repo"
        assert attempts[0] == 2
        assert mock_api.upload_folder.call_count == 2


def test_update_documentation_table(tmp_path):
    """Verify updating documentation tables with benchmark rows."""
    doc_path = tmp_path / "svd.md"
    doc_path.write_text(
        "# SVD Documentation\n\n| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |\n| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
        encoding="utf-8",
    )

    success = update_documentation_table(
        doc_path=doc_path,
        model_name="gpt2",
        module_name="svd",
        dataset_name="wikitext",
        status="PASS",
        baseline_acc=85.0,
        acc=84.5,
        f1=0.84,
        comp_rate="1.25",
        num_compressed=10,
        hf_repo_url="https://huggingface.co/Hooshaai/test",
    )
    assert success is True
    content = doc_path.read_text(encoding="utf-8")
    assert "**gpt2**" in content
    assert "84.50%" in content
