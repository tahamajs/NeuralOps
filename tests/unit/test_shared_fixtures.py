#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests verifying shared conftest fixtures."""

import torch
import pytest


@pytest.mark.fast
def test_fixture_tiny_gpt2(tiny_gpt2):
    """Verify tiny_gpt2 fixture produces a valid model."""
    assert tiny_gpt2 is not None
    assert hasattr(tiny_gpt2, "config")
    input_ids = torch.tensor([[1, 2, 3]])
    out = tiny_gpt2(input_ids)
    assert hasattr(out, "logits")
    assert out.logits.shape[:2] == (1, 3)


@pytest.mark.fast
def test_fixture_tiny_bert(tiny_bert):
    """Verify tiny_bert fixture produces a valid BERT model."""
    assert tiny_bert is not None
    assert hasattr(tiny_bert, "config")
    input_ids = torch.tensor([[1, 2, 3]])
    out = tiny_bert(input_ids)
    assert hasattr(out, "last_hidden_state")
    assert out.last_hidden_state.shape[:2] == (1, 3)


@pytest.mark.fast
def test_fixture_tiny_llama(tiny_llama):
    """Verify tiny_llama fixture produces a valid LLaMA model."""
    assert tiny_llama is not None
    assert hasattr(tiny_llama, "config")
    input_ids = torch.tensor([[1, 150, 200]])
    out = tiny_llama(input_ids)
    assert hasattr(out, "logits")
    assert out.logits.shape[:2] == (1, 3)


@pytest.mark.fast
def test_fixture_mock_cuda_device(mock_cuda_device):
    """Verify mock_cuda_device simulates CUDA environment."""
    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() >= 1
    props = torch.cuda.get_device_properties(mock_cuda_device)
    assert props.L2_cache_size > 0
    assert torch.cuda.memory_allocated(mock_cuda_device) >= 0


@pytest.mark.fast
def test_fixture_mock_hf_api(mock_hf_api):
    """Verify mock_hf_api intercepts network calls."""
    res = mock_hf_api.create_repo(repo_id="test/repo")
    assert "huggingface.co" in res
    upload = mock_hf_api.upload_file(path_or_fileobj="dummy", path_in_repo="file.txt", repo_id="test/repo")
    assert "huggingface.co" in upload


@pytest.mark.fast
def test_fixture_isolated_temp_ledger(isolated_temp_ledger):
    """Verify isolated_temp_ledger fixture operates in an isolated temp directory."""
    assert isolated_temp_ledger is not None
    assert isolated_temp_ledger.root_dir.exists()
    assert (isolated_temp_ledger.registry_dir / "master_ledger.jsonl").exists()
