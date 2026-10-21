#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for trainer, training, tracking, and evaluation contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from neuralops.evaluation import eval_pipeline
from neuralops.evaluation.eval_data import _PREPARED_DATASET_CACHE
from neuralops.evaluation.eval_patcher import build_model_instance
from neuralops.tracking.backends import LocalTracker
from neuralops.trainer.base_trainer import Trainer
from neuralops.trainer.results_manager import ResultsManager
from neuralops.training.checkpointing import SelectiveCheckpointModule
from neuralops.training.layer_distill import LayerAttentionDistillLoss


class _Classifier(nn.Module):
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        logits = torch.zeros((input_ids.shape[0], 2), device=input_ids.device)
        logits[:, 0] = 1.0
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_labels = None

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        logits = torch.zeros((*input_ids.shape, 16), device=input_ids.device)
        loss = None
        if labels is not None:
            self.last_labels = labels.detach().clone()
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def test_eval_uses_supplied_batch_when_prepared_dataset_cache_is_empty(monkeypatch):
    key = ("synthetic", "bert-base-uncased", 4)
    _PREPARED_DATASET_CACHE.pop(key, None)
    batch = {
        "input_ids": torch.ones((3, 4), dtype=torch.long),
        "attention_mask": torch.ones((3, 4), dtype=torch.long),
        "labels": torch.zeros(3, dtype=torch.long),
    }
    monkeypatch.setattr(eval_pipeline, "get_dataset_batch", lambda *args, **kwargs: batch)
    monkeypatch.setattr(
        eval_pipeline,
        "get_base_model_cached",
        lambda **kwargs: (_Classifier(), None),
    )

    metrics = eval_pipeline._evaluate_model_accuracy(
        model=_Classifier(),
        model_family="bert",
        dataset="synthetic",
        num_labels=2,
        seq_len=4,
        current_batch_size=3,
        is_lm=False,
        is_cuda=False,
        device=torch.device("cpu"),
        mem_phases={},
        run_log_lines=[],
    )

    assert metrics["accuracy_pct"] == 100.0
    assert metrics["fidelity_score"] > 0.0
    assert metrics["evaluation_data_source"] == "single_batch_fallback"


def test_eval_propagates_validation_errors_instead_of_returning_zero_metrics(monkeypatch):
    batch = {
        "input_ids": torch.ones((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.zeros(2, dtype=torch.long),
    }
    monkeypatch.setattr(eval_pipeline, "get_dataset_batch", lambda *args, **kwargs: batch)
    monkeypatch.setattr(
        eval_pipeline,
        "get_base_model_cached",
        lambda **kwargs: (_Classifier(), None),
    )
    monkeypatch.setattr(
        eval_pipeline,
        "_filter_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("invalid batch")),
    )

    with pytest.raises(RuntimeError, match="[Vv]alidation failed"):
        eval_pipeline._evaluate_model_accuracy(
            model=_Classifier(),
            model_family="bert",
            dataset="synthetic",
            num_labels=2,
            seq_len=4,
            current_batch_size=2,
            is_lm=False,
            is_cuda=False,
            device=torch.device("cpu"),
            mem_phases={},
            run_log_lines=[],
        )


def test_eval_does_not_silently_report_zero_f1_when_metric_computation_fails(monkeypatch):
    batch = {
        "input_ids": torch.ones((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.zeros(2, dtype=torch.long),
    }
    monkeypatch.setattr(eval_pipeline, "get_dataset_batch", lambda *args, **kwargs: batch)
    monkeypatch.setattr(
        eval_pipeline,
        "get_base_model_cached",
        lambda **kwargs: (_Classifier(), None),
    )
    monkeypatch.setattr(
        eval_pipeline,
        "f1_score",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad labels")),
    )

    with pytest.raises(RuntimeError, match="Failed to compute F1"):
        eval_pipeline._evaluate_model_accuracy(
            model=_Classifier(),
            model_family="bert",
            dataset="sst2",
            num_labels=2,
            seq_len=4,
            current_batch_size=2,
            is_lm=False,
            is_cuda=False,
            device=torch.device("cpu"),
            mem_phases={},
            run_log_lines=[],
        )


def test_language_model_fallback_batch_produces_measured_perplexity(monkeypatch):
    key = ("wikitext2", "gpt2", 4)
    _PREPARED_DATASET_CACHE.pop(key, None)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
    }
    monkeypatch.setattr(eval_pipeline, "get_dataset_batch", lambda *args, **kwargs: batch)
    monkeypatch.setattr(
        eval_pipeline,
        "get_base_model_cached",
        lambda **kwargs: (_CausalLM(), None),
    )
    model = _CausalLM()

    metrics = eval_pipeline._evaluate_model_accuracy(
        model=model,
        model_family="gpt2",
        dataset="wikitext2",
        num_labels=0,
        seq_len=4,
        current_batch_size=2,
        is_lm=True,
        is_cuda=False,
        device=torch.device("cpu"),
        mem_phases={},
        run_log_lines=[],
        allow_synthetic=True,
    )

    assert metrics["perplexity"] > 1.0
    assert metrics["perplexity"] == pytest.approx(16.0)
    assert torch.equal(model.last_labels, batch["input_ids"])
    assert metrics["evaluation_data_source"] == "single_batch_fallback"


def test_lm_model_factory_uses_causal_language_model_head():
    model, _ = build_model_instance(
        model_family="gpt2",
        num_labels=0,
        hidden_size=16,
        num_heads=2,
        num_layers=1,
        device=torch.device("cpu"),
    )

    assert model.__class__.__name__ == "GPT2LMHeadModel"


def test_lm_eval_environment_uses_language_model_head():
    env = eval_pipeline._setup_eval_environment(
        model_family="gpt2",
        dataset="wikitext2",
        seq_len=8,
        device=torch.device("cpu"),
    )

    assert env["num_labels"] == 0
    assert env["model"].__class__.__name__ == "GPT2LMHeadModel"


def test_language_model_eval_rejects_encoder_only_family():
    with pytest.raises(ValueError, match="not supported"):
        eval_pipeline._setup_eval_environment(
            model_family="bert",
            dataset="wikitext2",
            seq_len=8,
            device=torch.device("cpu"),
        )


def test_cpu_synthetic_benchmark_completes_with_truthful_source_metadata(tmp_path):
    result = eval_pipeline.evaluate_single_combination_opt(
        track="cpu-regression",
        module_name="baseline",
        module_type="attention",
        model_family="bert",
        dataset="synthetic",
        num_warmup=0,
        num_steps=1,
        batch_size=2,
        seq_len=8,
        device=torch.device("cpu"),
        results_dir=str(tmp_path),
        allow_synthetic=True,
        seed=23,
    )

    assert result.status == "PASS"
    assert result.evaluation_data_source == "single_batch_fallback"
    assert result.model_initialization == "random_config"
    artifact = tmp_path / "bert" / "synthetic" / "baseline" / "rank_16" / "config.json"
    assert json.loads(artifact.read_text())["seed"] == 23


def test_batch_probe_respects_cap_on_cpu():
    result = eval_pipeline.probe_optimal_batch_size(
        model=_Classifier(),
        model_family="bert",
        dataset="sst2",
        seq_len=4,
        device=torch.device("cpu"),
        max_batch_cap=8,
        min_batch_floor=2,
    )

    assert result == 8


def test_synthetic_batch_does_not_mutate_global_torch_rng():
    from neuralops.evaluation.eval_data import _generate_synthetic_batch

    torch.manual_seed(17)
    state_before = torch.random.get_rng_state()
    _generate_synthetic_batch("sst2", batch_size=2, seq_len=4, vocab_size=1000)
    state_after = torch.random.get_rng_state()

    assert torch.equal(state_before, state_after)


@pytest.mark.parametrize(
    ("students", "teachers"),
    [([], [torch.zeros(1, 2, 3)]), ([torch.zeros(1, 2, 3)], [])],
)
def test_layer_attention_distillation_rejects_empty_or_unpaired_layers(students, teachers):
    with pytest.raises(ValueError, match="same non-zero number"):
        LayerAttentionDistillLoss()(students, teachers)


def test_layer_attention_distillation_rejects_shape_mismatch():
    student = [torch.zeros(1, 2, 3)]
    teacher = [torch.zeros(1, 2, 4)]

    with pytest.raises(ValueError, match="matching shapes"):
        LayerAttentionDistillLoss()(student, teacher)


def test_selective_checkpoint_reads_sequence_length_from_keyword(monkeypatch):
    class _Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return hidden_states * 2

    module = _Spy()
    wrapped = SelectiveCheckpointModule(module, min_seqlen_threshold=4)
    checkpoint_calls = []

    def checkpoint_spy(function, *args, **kwargs):
        checkpoint_calls.append((function, args, kwargs))
        return function(*args)

    monkeypatch.setattr("neuralops.training.checkpointing.checkpoint", checkpoint_spy)
    hidden_states = torch.ones(1, 8, 2, requires_grad=True)
    output = wrapped(hidden_states=hidden_states)
    output.sum().backward()

    assert len(checkpoint_calls) == 1
    assert checkpoint_calls[0][1] == (hidden_states,)
    assert torch.equal(hidden_states.grad, torch.full_like(hidden_states, 2.0))


def test_unconfigured_trainer_does_not_claim_training_succeeded():
    trainer = Trainer(model=nn.Linear(2, 2))

    with pytest.raises(RuntimeError, match="train_dataset"):
        trainer.train()
    with pytest.raises(RuntimeError, match="eval_dataset"):
        trainer.evaluate()


def test_zero_accuracy_is_a_valid_completed_evaluation(tmp_path):
    result = eval_pipeline._finalize_benchmark_result(
        result=eval_pipeline.BenchmarkResult(
            track="cpu",
            module_name="test_attention",
            module_type="attention",
            model_family="bert",
            dataset="sst2",
            status="PASS",
            accuracy_pct=0.0,
        ),
        run_log_lines=[],
        t_run_start=0.0,
        module_name="test_attention",
        module_type="attention",
        model_family="bert",
        dataset="sst2",
        rank=4,
        seq_len=8,
        batch_size=2,
        current_batch_size=2,
        num_warmup=0,
        num_steps=1,
        allow_synthetic=False,
        device=torch.device("cpu"),
        results_dir=str(tmp_path),
        comp_stats={},
        mem_phases={},
        t_model_load=0.0,
        t_calib=0.0,
        t_comp=0.0,
        t_eval=0.0,
        baseline_logits_samples=[],
        compressed_logits_samples=[],
        sample_cosines=[],
        exc_traceback=None,
        exc_type=None,
        retry_count=0,
        success=True,
        seed=123,
    )

    assert result.status == "PASS"
    run_dir = tmp_path / "bert" / "sst2" / "test_attention" / "rank_4"
    assert json.loads((run_dir / "seed.json").read_text()) == {
        "random": 123,
        "numpy": 123,
        "torch": 123,
        "cuda": 123,
        "synthetic_data": 42,
    }


def test_results_manager_preserves_hf_trainer_step_and_log_signature(tmp_path):
    manager = ResultsManager("step_tracking", base_dir=str(tmp_path), log_to_file=False)

    class _FakeTrainer:
        state = SimpleNamespace(global_step=37)

        def __init__(self):
            self.calls = []

        def add_callback(self, callback):
            pass

        def log(self, logs, start_time=None):
            self.calls.append((logs, start_time))

    trainer = manager.wrap_trainer(_FakeTrainer())
    trainer.log({"loss": 0.25}, start_time=1.5)

    assert trainer.calls == [({"loss": 0.25}, 1.5)]
    assert manager.history[0]["step"] == 37


def test_results_manager_reports_ledger_write_failure(tmp_path, monkeypatch, caplog):
    from neuralops.core import results_ledger

    def _fail_to_log(**kwargs):
        raise OSError("ledger unavailable")

    monkeypatch.setattr(results_ledger, "log_experiment_result", _fail_to_log)
    manager = ResultsManager("ledger_failure", base_dir=str(tmp_path), log_to_file=False)

    manager.generate_summary_report({"accuracy": 0.9})

    assert "Could not record run summary in the experiment ledger" in caplog.text
    assert "ledger unavailable" in caplog.text
    assert (manager.run_dir / "run_summary.json").exists()


def test_local_tracker_merges_parameter_updates(tmp_path):
    tracker = LocalTracker(log_dir=str(tmp_path), experiment_name="merge")
    tracker.log_params({"lr": 0.001})
    tracker.log_params({"batch_size": 8})

    with open(tracker.params_file, encoding="utf-8") as stream:
        params = json.load(stream)

    assert params == {"lr": 0.001, "batch_size": 8}
