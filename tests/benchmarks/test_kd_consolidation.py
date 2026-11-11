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
