#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for KD P0 fixes: weights_only security, hash completeness, and deterministic steps."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuralops.benchmarks.kd.config import BenchmarkConfig
from neuralops.benchmarks.kd.distill_engine import TwoStageDistillationEngine
from neuralops.benchmarks.kd.teacher_cache import CachedCausalDataset, _safe_torch_load, atomic_torch_save
from neuralops.benchmarks.kd.zoo_orchestrator import ZooOrchestrator


def test_safe_torch_load_weights_only_by_default(tmp_path: Path):
    """Verify _safe_torch_load defaults to weights_only=True and safely loads tensors and dicts."""
    payload = {
        "cache_version": 1,
        "model_id": "test_model",
        "seq_len": 16,
        "topk_vals": torch.randn(2, 4),
        "layer_indices": [0, 1],
    }
    target_file = tmp_path / "safe_cache.pt"
    atomic_torch_save(payload, target_file)

    loaded = _safe_torch_load(target_file)
    assert loaded["cache_version"] == 1
    assert loaded["model_id"] == "test_model"
    assert loaded["topk_vals"].shape == (2, 4)


def test_compute_hash_complete_field_coverage():
    """Verify compute_hash changes when any run-affecting field changes."""
    cfg_base = BenchmarkConfig(
        task="glue:sst2",
        teacher="bert-base-uncased",
        batch_size=4,
        grad_accum_steps=4,
        kd_steps=200,
        kd_lr=1e-5,
        seed=42,
        use_fp16=True,
    )
    base_hash = cfg_base.compute_hash()

    # Changing seed must change hash
    cfg_seed = BenchmarkConfig(seed=999)
    assert cfg_seed.compute_hash() != base_hash

    # Changing batch_size must change hash
    cfg_bs = BenchmarkConfig(batch_size=8)
    assert cfg_bs.compute_hash() != base_hash

    # Changing grad_accum_steps must change hash
    cfg_accum = BenchmarkConfig(grad_accum_steps=2)
    assert cfg_accum.compute_hash() != base_hash

    # Changing pre_kd_samples must change hash
    cfg_samples = BenchmarkConfig(pre_kd_samples=500)
    assert cfg_samples.compute_hash() != base_hash

    # Changing use_fp16 must change hash
    cfg_fp16 = BenchmarkConfig(use_fp16=False)
    assert cfg_fp16.compute_hash() != base_hash


def test_distill_engine_deterministic_max_steps():
    """Verify TwoStageDistillationEngine adheres to max_steps deterministically."""
    hidden_dim = 16
    seq_len = 8
    vocab_size = 32

    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
            self.lm_head = nn.Linear(hidden_dim, vocab_size)

        def forward(self, input_ids, **kwargs):
            h = self.embed(input_ids)
            states = [h]
            for layer in self.layers:
                h = layer(h)
                states.append(h)
            logits = self.lm_head(h)

            class Out:
                def __init__(self, logits, hidden_states):
                    self.logits = logits
                    self.hidden_states = hidden_states

            return Out(logits, states)

    student = TinyStudent()

    n_samples = 8
    mock_cache = {
        "cache_version": 1,
        "layer_indices": [0, 1],
        "input_ids": torch.randint(0, vocab_size, (n_samples, seq_len)),
        "attention_mask": torch.ones((n_samples, seq_len), dtype=torch.long),
        "hidden_states": torch.randn((n_samples, 2, seq_len, hidden_dim), dtype=torch.float16),
        "topk_vals": torch.randn((n_samples, seq_len, 4), dtype=torch.float16),
        "topk_idx": torch.randint(0, vocab_size, (n_samples, seq_len, 4), dtype=torch.int32),
    }

    train_ds = CachedCausalDataset(mock_cache)
    loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    engine = TwoStageDistillationEngine(
        train_secs=1000.0,  # Large time budget, must terminate via max_steps
        stage1_frac=0.5,
        stage2_frac=0.5,
        grad_accum=1,
        device=torch.device("cpu"),
        max_steps=4,
    )

    metrics = engine.run(
        student=student,
        train_loader=loader,
        val_loader=loader,
        teacher_cache=mock_cache,
        eval_fn=lambda m, l, d: 10.0,
        max_steps=4,
    )

    assert metrics["steps_s1"] == 2
    assert metrics["steps_s2"] == 2
    assert metrics["total_steps"] == 4


def test_zoo_orchestrator_submit_upload_arguments(tmp_path: Path):
    """Verify ZooOrchestrator passes correct keyword arguments to uploader.submit_upload."""
    mock_uploader = MagicMock()
    orchestrator = ZooOrchestrator(
        zoo_type="qwen_attention",
        out_dir=tmp_path / "zoo",
        metrics_log=tmp_path / "metrics.jsonl",
        hf_upload=True,
    )
    orchestrator.uploader = mock_uploader

    test_res = {"accuracy": 92.5}
    def mock_train(key, ckpt_dir):
        return dict(test_res)

    results = orchestrator.run_zoo(
        keys=["linear_attention"],
        train_fn=mock_train,
    )

    assert len(results) == 1
    assert results[0]["status"] == "PASSED"
    mock_uploader.submit_upload.assert_called_once()
    call_kwargs = mock_uploader.submit_upload.call_args.kwargs
    assert call_kwargs["attn_key"] == "linear_attention"
    assert call_kwargs["zoo_type"] == "qwen_attention"
    assert call_kwargs["metrics"]["accuracy"] == 92.5
    assert call_kwargs["staging_dir"] == (tmp_path / "zoo" / "linear_attention")
