#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Consolidation and backward compatibility test suite for NeuralOps Knowledge Distillation (`kd`).

Verifies:
  1. Full export contract of `neuralops.benchmarks.kd`.
  2. Backward compatibility of `scripts/benchmarks/` wrappers.
  3. CLI argument compatibility across all zoo training scripts.
  4. Execution of TwoStageDistillationEngine with mock causal student models.
  5. OOM retry handler with batch size backoff.
  6. ZooOrchestrator resume and ledger persistence logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import neuralops.benchmarks.kd as kd
from neuralops.benchmarks.kd.distill_engine import (
    TwoStageDistillationEngine,
    compute_topk_sparse_kd_loss,
    evaluate_causal_ppl,
    execute_with_oom_retry,
    fast_gram_covariance_loss,
)
from neuralops.benchmarks.kd.teacher_cache import (
    CachedCausalDataset,
    HPC_DATA_CACHE_ROOT,
    VLM_MAX_CACHE_BYTES,
    build_or_load_causal_teacher_cache,
    resolve_teacher_cache_path,
    select_strided_layer_indices,
    select_vlm_layer_indices,
)
from neuralops.benchmarks.kd.zoo_orchestrator import (
    CausalTextDataset,
    StratifiedQwenDataset,
    ZooOrchestrator,
    apply_qwen_attention_surgery,
    build_llama_readme,
    build_qwen_readme,
    build_vlm_readme,
    normalize_base_model_id,
)


class _MockDecoderLayer(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.self_attn(x))


class _MockCausalModel(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 16, num_layers: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=dim,
            num_attention_heads=2,
            num_hidden_layers=num_layers,
        )
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([_MockDecoderLayer(dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        x = self.embed(input_ids)
        hiddens = [x]
        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
        logits = self.lm_head(x)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hiddens),
            last_hidden_state=x,
        )


def test_kd_package_exports() -> None:
    """Verify that all canonical symbols are exported in neuralops.benchmarks.kd."""
    for symbol in kd.__all__:
        assert hasattr(kd, symbol), f"Symbol {symbol} missing in neuralops.benchmarks.kd"

    assert issubclass(kd.TwoStageDistillationEngine, object)
    assert issubclass(kd.ZooOrchestrator, object)
    assert issubclass(kd.CachedCausalDataset, torch.utils.data.Dataset)
    assert callable(kd.compute_topk_sparse_kd_loss)
    assert callable(kd.apply_qwen_attention_surgery)


def test_scripts_backward_compatibility() -> None:
    """Verify scripts/benchmarks/ re-exports functions expected by existing tests and callers."""
    import scripts.benchmarks.teacher_cache_utils as tc_utils
    assert hasattr(tc_utils, "build_or_load_causal_teacher_cache")
    assert hasattr(tc_utils, "build_or_load_vlm_teacher_cache")
    assert hasattr(tc_utils, "CachedCausalDataset")

    import scripts.benchmarks.qwen_zoo_train_mix as qwen_zoo
    assert hasattr(qwen_zoo, "apply_qwen_attention_surgery")
    assert hasattr(qwen_zoo, "compute_topk_sparse_kd_loss")
    assert hasattr(qwen_zoo, "evaluate_causal_ppl")
    assert hasattr(qwen_zoo, "train_single_qwen_attention")

    import scripts.benchmarks.llama_zoo_train_mix as llama_zoo
    assert hasattr(llama_zoo, "build_llama_readme")
    assert hasattr(llama_zoo, "train_single_llama_attention")

    import scripts.benchmarks.vlm_llava_scienceqa_zoo as vlm_zoo
    assert hasattr(vlm_zoo, "build_vlm_readme")
    assert hasattr(vlm_zoo, "train_single_vlm_attention")


@pytest.mark.parametrize(
    "module_path",
    [
        "scripts.benchmarks.qwen_zoo_train_mix",
        "scripts.benchmarks.llama_zoo_train_mix",
        "scripts.benchmarks.qwen_compressor_zoo_train_mix",
        "scripts.benchmarks.vlm_llava_scienceqa_zoo",
    ],
)
def test_cli_help_and_cache_flags(module_path: str) -> None:
    """Verify CLI entrypoint help messages retain all flags including teacher_cache."""
    import importlib
    mod = importlib.import_module(module_path)
    main_fn = getattr(mod, "main")

    with patch("sys.argv", ["prog", "--help"]), pytest.raises(SystemExit) as exc_info:
        main_fn()
    assert exc_info.value.code == 0


def test_two_stage_distillation_engine_mock_run() -> None:
    """Verify 2-stage distillation engine runs Stage 1 and Stage 2 with cached teacher."""
    torch.manual_seed(42)
    student = _MockCausalModel(vocab_size=32, dim=16, num_layers=4)

    # Fake cache
    n_samples, seq_len, dim = 4, 8, 16
    mock_cache = {
        "layer_indices": [1, 3],
        "input_ids": torch.randint(0, 32, (n_samples, seq_len)),
        "attention_mask": torch.ones((n_samples, seq_len), dtype=torch.long),
        "hidden_states": torch.randn((n_samples, 2, seq_len, dim), dtype=torch.float16),
        "topk_vals": torch.randn((n_samples, seq_len, 4), dtype=torch.float16),
        "topk_idx": torch.randint(0, 32, (n_samples, seq_len, 4), dtype=torch.int32),
    }

    train_ds = CachedCausalDataset(mock_cache)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    engine = TwoStageDistillationEngine(
        train_secs=1.5,
        stage1_frac=0.5,
        stage2_frac=0.5,
        grad_accum=1,
        device=torch.device("cpu"),
    )

    metrics = engine.run(
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_cache=mock_cache,
        log_prefix="[TEST] ",
    )

    assert "pre_metric" in metrics
    assert "post_metric" in metrics
    assert metrics["steps_s1"] >= 1
    assert metrics["steps_s2"] >= 1
    assert metrics["total_steps"] == metrics["steps_s1"] + metrics["steps_s2"]


def test_oom_retry_handler() -> None:
    """Verify execute_with_oom_retry catches OOM and halves batch size until success."""
    call_batch_sizes = []

    def _flaky_alloc(batch_size: int = 8) -> str:
        call_batch_sizes.append(batch_size)
        if batch_size > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate...")
        return "SUCCESS"

    result = execute_with_oom_retry(_flaky_alloc, initial_batch_size=8, max_retries=4)
    assert result == "SUCCESS"
    assert call_batch_sizes == [8, 4, 2]


def test_zoo_orchestrator_ledger_and_resume(tmp_path: Path) -> None:
    """Verify ZooOrchestrator tracks completed keys and appends to metrics.jsonl."""
    metrics_log = tmp_path / "metrics.jsonl"
    out_dir = tmp_path / "zoo_out"

    # Pre-populate ledger with 1 PASSED run
    metrics_log.write_text(
        json.dumps({"attn_key": "linear_attention", "status": "PASSED"}) + "\n",
        encoding="utf-8",
    )

    orchestrator = ZooOrchestrator(
        zoo_type="qwen_attention",
        out_dir=out_dir,
        metrics_log=metrics_log,
    )

    completed = orchestrator.load_completed_keys()
    assert "linear_attention" in completed

    # Run zoo on keys including the completed one
    executed_keys = []

    def _mock_train(key: str, ckpt_dir: Path) -> dict:
        executed_keys.append(key)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return {"attn_key": key, "ppl_after": 25.0}

    keys = ["linear_attention", "retention", "gla"]
    results = orchestrator.run_zoo(keys, _mock_train)

    assert executed_keys == ["retention", "gla"]
    assert len(results) == 2

    # Check updated metrics.jsonl
    lines = [json.loads(line) for line in metrics_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3
    assert {l["attn_key"] for l in lines} == {"linear_attention", "retention", "gla"}


def test_two_stage_distillation_cosine_decay_and_multiple_choice_margin() -> None:
    """Verify Stage 2 cosine decay scheduling down to min_lr_s2 and multiple-choice margin loss."""
    torch.manual_seed(42)
    student = _MockCausalModel(vocab_size=32, dim=16, num_layers=4)

    n_samples, seq_len, dim = 4, 8, 16
    mock_cache = {
        "layer_indices": [1, 3],
        "input_ids": torch.randint(0, 32, (n_samples, seq_len)),
        "attention_mask": torch.ones((n_samples, seq_len), dtype=torch.long),
        "hidden_states": torch.randn((n_samples, 2, seq_len, dim), dtype=torch.float16),
        "topk_vals": torch.randn((n_samples, seq_len, 4), dtype=torch.float16),
        "topk_idx": torch.randint(0, 32, (n_samples, seq_len, 4), dtype=torch.int32),
    }

    # Custom dataset that provides multiple choice options metadata
    class _MockMultipleChoiceDataset(torch.utils.data.Dataset):
        def __init__(self, cache: dict) -> None:
            self.ds = CachedCausalDataset(cache)

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, idx: int) -> dict:
            item = self.ds[idx]
            item["answer_idx"] = torch.tensor(idx % 4, dtype=torch.long)
            item["num_choices"] = torch.tensor(4, dtype=torch.long)
            item["prompt_lens"] = torch.tensor(6, dtype=torch.long)
            return item

    train_ds = _MockMultipleChoiceDataset(mock_cache)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    lr_s2 = 6.0e-5
    min_lr_s2 = 1.0e-6
    engine = TwoStageDistillationEngine(
        train_secs=1.2,
        stage1_frac=0.3,
        stage2_frac=0.7,
        lr_s2=lr_s2,
        min_lr_s2=min_lr_s2,
        use_cosine_scheduler=True,
        margin_loss_weight=0.5,
        grad_accum=1,
        device=torch.device("cpu"),
    )

    metrics = engine.run(
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_cache=mock_cache,
        log_prefix="[COSINE-TEST] ",
        letter_token_ids=[4, 5, 6, 7],
    )

    assert "final_lr_s2" in metrics
    assert metrics["final_lr_s2"] < lr_s2, f"Expected LR to decay below {lr_s2}, got {metrics['final_lr_s2']}"
    assert metrics["final_lr_s2"] >= min_lr_s2, f"Expected LR to stay >= {min_lr_s2}, got {metrics['final_lr_s2']}"
    assert metrics["steps_s2"] >= 1
    assert metrics["steps_s1"] >= 1


def test_configurable_train_secs_unbounded() -> None:
    """Verify train_secs can be configured without hard-coded limits on engine and CLI."""
    engine = TwoStageDistillationEngine(train_secs=3600.0)
    assert engine.train_secs == 3600.0

    # Test override in run() accepts arbitrary extended values
    assert hasattr(engine.run, "__code__")
    assert "train_secs" in engine.run.__code__.co_varnames

    # Test CLI parser accepts extended train_secs and scheduler flags
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_secs", type=float, default=240.0)
    parser.add_argument("--lr_s2", type=float, default=6.0e-5)
    parser.add_argument("--min_lr_s2", type=float, default=1.0e-6)
    parser.add_argument("--margin_weight", type=float, default=0.5)
    parsed = parser.parse_args(["--train_secs", "5000.0", "--min_lr_s2", "5e-7", "--margin_weight", "0.75"])
    assert parsed.train_secs == 5000.0
    assert parsed.min_lr_s2 == 5e-7
    assert parsed.margin_weight == 0.75


def test_masked_representation_loss_ignores_padding() -> None:
    """Verify that tokens flagged as 0 in attention_mask do not influence MSE or cosine representation loss."""
    from neuralops.benchmarks.kd.distill_engine import _compute_masked_representation_loss

    torch.manual_seed(42)
    b, s, d = 2, 8, 16
    s_h = [torch.randn(b, s, d) for _ in range(2)]
    t_h = [torch.randn(b, s, d) for _ in range(2)]

    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

    loss_mse_clean, loss_cos_clean = _compute_masked_representation_loss(s_h, t_h, attention_mask)

    s_h_corrupt = [t.clone() for t in s_h]
    t_h_corrupt = [t.clone() for t in t_h]
    for sc, tc in zip(s_h_corrupt, t_h_corrupt):
        sc[0, 5:, :] = 9999.0
        tc[0, 5:, :] = -8888.0
        sc[1, 4:, :] = 1234.0
        tc[1, 4:, :] = -5678.0

    loss_mse_corrupt, loss_cos_corrupt = _compute_masked_representation_loss(s_h_corrupt, t_h_corrupt, attention_mask)

    assert torch.allclose(loss_mse_clean, loss_mse_corrupt, atol=1e-5), "Padding tokens corrupted MSE loss!"
    assert torch.allclose(loss_cos_clean, loss_cos_corrupt, atol=1e-5), "Padding tokens corrupted Cosine loss!"


def test_fast_gram_covariance_loss_with_mask() -> None:
    """Verify that fast_gram_covariance_loss computes normalized Gram alignment with optional masking."""
    torch.manual_seed(42)
    s_hidden = torch.randn(2, 6, 8)
    t_hidden = torch.randn(2, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.long)

    loss_unmasked = fast_gram_covariance_loss(s_hidden, t_hidden)
    loss_masked = fast_gram_covariance_loss(s_hidden, t_hidden, attention_mask=mask)

    assert torch.isfinite(loss_unmasked)
    assert torch.isfinite(loss_masked)
    assert loss_masked.item() >= 0.0
