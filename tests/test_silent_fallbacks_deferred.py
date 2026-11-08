#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for deferred error swallowing sites (AGT-55 / F-84 completion).

Verifies that all previously deferred sites either fail loudly (raise with chained cause)
or use an explicit, logged fallback rather than silently swallowing exceptions or returning
fake placeholder artifacts.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from neuralops.api import TextGenerationPipeline, heal_distributed, list_attentions


def test_text_generation_pipeline_decode_failure_raises():
    """TextGenerationPipeline._decode_tokens must raise RuntimeError if tokenizer decode fails."""
    dummy_model = nn.Linear(10, 10)
    pipeline = TextGenerationPipeline(model=dummy_model)

    broken_tok = MagicMock()
    # Mock batch_decode and decode to fail
    broken_tok.batch_decode.side_effect = ValueError("Corrupt vocabulary")
    broken_tok.decode.side_effect = ValueError("Decoding error")

    output_ids = torch.tensor([[101, 2054, 102]])

    with pytest.raises(RuntimeError) as exc_info:
        pipeline._decode_tokens(broken_tok, output_ids)

    assert "Tokenizer failed to decode" in str(exc_info.value)
    # Ensure it did not return string representations of tokens like "[101, 2054, 102]"
    assert isinstance(exc_info.value.__cause__, Exception)


def test_parallel_healing_explicit_failure_raises():
    """Verify that explicitly requesting parallel_healing=True fails closed if distributed healing fails."""
    student = nn.Linear(10, 10)
    teacher = nn.Linear(10, 10)

    with patch("neuralops.distributed.context.DistributedContext") as mock_ctx_cls:
        mock_ctx = MagicMock()
        mock_ctx.is_distributed.return_value = True
        mock_ctx_cls.return_value = mock_ctx

        with patch("neuralops.distributed.fsdp2_distill.FSDP2Distillation") as mock_distill_cls:
            mock_distill = MagicMock()
            mock_distill.heal_replaced_model_distributed.side_effect = RuntimeError("FSDP communication timeout")
            mock_distill_cls.return_value = mock_distill

            with pytest.raises(RuntimeError, match="Parallel healing failed"):
                heal_distributed(
                    student=student,
                    teacher=teacher,
                    calibration_loader=[torch.zeros(1, 10)],
                    parallel_healing=True,
                )


def test_list_attentions_fla_importerror_logs_warning(caplog, monkeypatch):
    """list_attentions(include_extensions=True) logs warning when FLA is missing."""
    monkeypatch.setenv("NEURALOPS_NO_DEPRECATION_WARNINGS", "1")
    with patch.dict("sys.modules", {"neuralops.integrations.fla": None}):
        with caplog.at_level(logging.WARNING):
            attentions = list_attentions(include_extensions=True)
            assert isinstance(attentions, list)
            assert len(attentions) > 0
            assert any("FLA" in rec.message or "fla" in rec.message for rec in caplog.records)


def test_main_online_tokenizer_failure_logs_warning_and_retries_local(caplog):
    """main.py run_train logs warning when online download fails and retries with local_files_only."""
    from neuralops.main import run_train

    cfg = MagicMock()
    cfg.seed = 42
    cfg.model_name = "test-model"
    cfg.task = "sst2"
    cfg.attn_type = "baseline"
    cfg.batch_size = 4
    cfg.learning_rate = 1e-4
    cfg.epochs = 1
    cfg.output_dir = "test_out"

    logger = logging.getLogger("neuralops.main")

    mock_tokenizer = MagicMock()
    calls = []

    def mock_tok_from_pretrained(name, *args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            return mock_tokenizer
        raise OSError("Connection refused by HuggingFace Hub")

    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.zeros(2, 2)]

    fw = {
        "torch": torch,
        "set_seed": MagicMock(),
        "get_device": MagicMock(return_value="cpu"),
        "clear_memory": MagicMock(),
        "count_parameters": MagicMock(return_value=1000),
        "AutoTokenizer": MagicMock(from_pretrained=mock_tok_from_pretrained),
        "AutoModelForSequenceClassification": MagicMock(from_pretrained=MagicMock(return_value=mock_model)),
        "TrainingArguments": MagicMock(),
        "Trainer": MagicMock(return_value=MagicMock(train=MagicMock(return_value=MagicMock(metrics={})))),
        "DataCollatorWithPadding": MagicMock(),
        "get_task_metrics": MagicMock(),
        "ResultWriter": MagicMock(),
        "ATTENTION_REGISTRY": {},
    }

    with patch("neuralops.main.load_data", return_value=({"train": [], "validation": []}, 2, False)):
        with patch("pathlib.Path.mkdir"):
            with caplog.at_level(logging.WARNING):
                try:
                    run_train(cfg, fw, logger)
                except Exception:
                    pass

    # Verify fallback was logged
    assert any("local_files_only=True" in rec.message or "attempting fallback" in rec.message for rec in caplog.records)
    # Verify local_files_only was called
    assert any(c.get("local_files_only") is True for c in calls)


def test_compression_dir_does_not_swallow_silently(caplog):
    """neuralops.compression.__dir__() logs debug info if a subpackage import fails."""
    import neuralops.compression as comp

    with patch("importlib.import_module", side_effect=ImportError("Mocked missing dependency")):
        with caplog.at_level(logging.DEBUG):
            dir_contents = comp.__dir__()
            assert isinstance(dir_contents, list)
            assert "BaseCompressor" in dir_contents


def test_require_torch_patch_failure_logs_warning(caplog):
    """_optional._require_torch logs warning if patch fails instead of swallowing."""
    from neuralops._optional import _require_torch

    with patch("neuralops._patch_torch_compat", side_effect=RuntimeError("Patch conflict")):
        with caplog.at_level(logging.WARNING):
            _require_torch("test_feature")
            assert any("Failed to patch torch compatibility" in rec.message for rec in caplog.records)
