#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression test suite for P0 core fixes:
- evaluation.py: sample_batch dict/tensor handling in measure_resources, evaluate_classification, evaluate_perplexity.
- training.py: trainable_params bound for all optimizers; dict batch support.
- kd.py: KDLoss projection fallback, 3D cross_entropy logits reshaping, dict batch handling in kd_train_step.
- model_loader.py: kwargs deduplication and safe forwarding in apply_attention_module and apply_compressor.
- data_loader.py: max_samples in local file loading and token padding for short texts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuralops.benchmarks.core.data_loader import load_dataset_by_name
from neuralops.benchmarks.core.evaluation import evaluate_classification, evaluate_perplexity, measure_resources
from neuralops.benchmarks.core.kd import KDLoss, kd_train_step
from neuralops.benchmarks.core.model_loader import apply_attention_module, apply_compressor
from neuralops.benchmarks.core.training import finetune_classification


class _TinyClassifier(nn.Module):
    def __init__(self, vocab_size: int = 100, dim: int = 16, num_labels: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.classifier = nn.Linear(dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        h = self.embed(input_ids).mean(dim=1)
        logits = self.classifier(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 100, dim: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=dim)
        self.embed = nn.Embedding(vocab_size, dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        past_key_values: object = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        pkv = ("kv_state",) if use_cache else None
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=(h,), past_key_values=pkv)


class _DictDataset(Dataset):
    def __init__(self, n: int = 4, seq_len: int = 8, vocab_size: int = 50) -> None:
        self.items = [
            {
                "input_ids": torch.randint(0, vocab_size, (seq_len,)),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": torch.tensor(i % 2, dtype=torch.long),
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


# =========================================================================
# 1. evaluation.py tests
# =========================================================================


def test_measure_resources_with_dict_batch() -> None:
    model = _TinyClassifier()
    dict_batch = {
        "input_ids": torch.randint(0, 50, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    stats = measure_resources(model, dict_batch, device="cpu", warmup_iters=1, measure_iters=2)
    assert "latency_ms_per_token" in stats
    assert "throughput_tok_per_sec" in stats
    assert stats["throughput_tok_per_sec"] > 0


def test_measure_resources_with_tensor_batch() -> None:
    model = _TinyClassifier()
    tensor_batch = torch.randint(0, 50, (2, 8))
    stats = measure_resources(model, tensor_batch, device="cpu", warmup_iters=1, measure_iters=2)
    assert stats["throughput_tok_per_sec"] > 0


def test_evaluate_classification_dict_loader() -> None:
    model = _TinyClassifier()
    loader = DataLoader(_DictDataset(4, 8), batch_size=2)
    res = evaluate_classification(model, loader, device="cpu")
    assert "accuracy" in res
    assert "loss" in res
    assert 0.0 <= res["accuracy"] <= 1.0


def test_evaluate_perplexity_dict_loader() -> None:
    model = _TinyLM()
    loader = DataLoader(_DictDataset(4, 8), batch_size=2)
    res = evaluate_perplexity(model, loader, device="cpu")
    assert "perplexity" in res
    assert res["perplexity"] > 0


# =========================================================================
# 2. training.py tests
# =========================================================================


def test_finetune_classification_trainable_params_muon_no_unbound_local() -> None:
    model = _TinyClassifier()
    loader = DataLoader(_DictDataset(4, 8), batch_size=2)
    mock_opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    with patch(
        "neuralops.trainer.optimizers.muon.create_muon_adamw_optimizer",
        return_value=mock_opt,
    ):
        result = finetune_classification(
            model=model,
            train_loader=loader,
            epochs=1,
            lr=1e-4,
            device="cpu",
            optimizer_name="muon",
            max_steps=1,
        )
    assert result["status"] == "success"
    assert result["steps_completed"] == 1


def test_finetune_classification_dict_batch_handling() -> None:
    model = _TinyClassifier()
    loader = DataLoader(_DictDataset(4, 8), batch_size=2)
    result = finetune_classification(
        model=model,
        train_loader=loader,
        epochs=1,
        lr=1e-4,
        device="cpu",
        max_steps=2,
    )
    assert result["status"] == "success"
    assert result["steps_completed"] == 2


# =========================================================================
# 3. kd.py tests
# =========================================================================


def test_kd_loss_unfused_projection_from_hidden_states() -> None:
    loss_fn = KDLoss(temperature=2.0, alpha=0.5, use_fused_linear=False)
    batch_size, seq_len, dim, vocab = 2, 4, 8, 20
    s_h = torch.randn(batch_size, seq_len, dim)
    t_h = torch.randn(batch_size, seq_len, dim)
    s_w = torch.randn(vocab, dim)
    t_w = torch.randn(vocab, dim)
    labels = torch.randint(0, vocab, (batch_size, seq_len))

    # Calling without precomputed logits: should project and compute loss without ValueError
    loss = loss_fn(
        student_hidden=s_h,
        student_weight=s_w,
        teacher_hidden=t_h,
        teacher_weight=t_w,
        labels=labels,
    )
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss).item()


def test_kd_loss_3d_cross_entropy_reshaping() -> None:
    loss_fn = KDLoss(temperature=2.0, alpha=0.5)
    batch_size, seq_len, vocab = 2, 4, 30
    s_logits = torch.randn(batch_size, seq_len, vocab)
    t_logits = torch.randn(batch_size, seq_len, vocab)
    labels = torch.randint(0, vocab, (batch_size, seq_len))

    loss = loss_fn(
        student_logits=s_logits,
        teacher_logits=t_logits,
        labels=labels,
    )
    assert torch.isfinite(loss).item()


def test_kd_train_step_dict_batch() -> None:
    student = _TinyLM()
    teacher = _TinyLM()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    loss_fn = KDLoss(temperature=2.0, alpha=0.5)

    dict_batch = {
        "input_ids": torch.randint(0, 50, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels": torch.randint(0, 50, (2, 8)),
    }
    step_loss = kd_train_step(
        student=student,
        teacher=teacher,
        batch=dict_batch,
        optimizer=optimizer,
        kd_loss_fn=loss_fn,
        device="cpu",
    )
    assert isinstance(step_loss, float)
    assert step_loss >= 0.0


# =========================================================================
# 4. model_loader.py tests
# =========================================================================


def test_apply_attention_module_kwargs_deduplication() -> None:
    class _MockSelfAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(32, 32)

    class _MockTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = _MockSelfAttn()

    model = _MockTransformer()

    # Pass kwargs containing duplicate arguments that would crash inject_attention
    extra_kwargs = {
        "rank": 16,
        "copy_weights": True,
        "attention_type": "cosformer",
        "attention": "cosformer",
        "batch_size": 4,
    }
    with patch("neuralops.core.attention_utils.inject_attention") as mock_inject:
        mock_inject.return_value = model
        result = apply_attention_module(
            model=model,
            module_name="cosformer",
            **extra_kwargs,
        )
        assert result is model
        mock_inject.assert_called_once()
        _, call_kwargs = mock_inject.call_args
        assert call_kwargs["rank"] == 16  # overriden by extra_kwargs safely
        assert call_kwargs["attention_type"] == "cosformer"
        assert "attention" not in call_kwargs
        assert call_kwargs["attention_type"] == "cosformer"
        # Duplicate keys must not be present in remaining kwargs
        assert "module_name" not in call_kwargs


def test_apply_compressor_kwargs_deduplication() -> None:
    model = nn.Linear(16, 16)
    with patch("neuralops.benchmarks.core.model_loader.COMPRESSOR_REGISTRY") as mock_reg:
        mock_cls = MagicMock()
        mock_reg.__contains__.return_value = True
        mock_reg.__getitem__.return_value = mock_cls
        mock_comp = MagicMock()
        mock_cls.return_value = mock_comp
        mock_comp.compress.return_value = model

        apply_compressor(
            model=model,
            module_name="magnitude_pruning",
            rank=16,
            module_name_duplicate="test",
        )
        mock_cls.assert_called_once()


# =========================================================================
# 5. data_loader.py tests
# =========================================================================


def test_load_dataset_by_name_local_max_samples(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "test_data.jsonl"
    jsonl_file.write_text(
        '{"sentence": "hello world", "label": 1}\n'
        '{"sentence": "foo bar", "label": 0}\n'
        '{"sentence": "baz qux", "label": 1}\n'
        '{"sentence": "spam eggs", "label": 0}\n'
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }

    # With max_samples=2, only 2 rows should be processed
    ds = load_dataset_by_name(str(jsonl_file), mock_tokenizer, max_length=4, max_samples=2)
    assert len(ds) == 2
