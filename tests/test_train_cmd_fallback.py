#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests verifying fail-closed model loading in training CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from neuralops.cli.train_cmd import _load_model, cmd_train


def test_load_model_failure_raises_not_random() -> None:
    """Mock model loading failure and assert _load_model raises RuntimeError instead of fallback."""
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=OSError("Checkpoint not found")),
        patch("transformers.AutoModel.from_pretrained", side_effect=OSError("Checkpoint not found")),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            _load_model("nonexistent-checkpoint-path", device="cpu")

        err_msg = str(exc_info.value)
        assert "nonexistent-checkpoint-path" in err_msg
        assert "will NOT continue on a random model" in err_msg


def test_cmd_train_fails_fast_on_missing_checkpoint(tmp_path: Path) -> None:
    """Assert cmd_train fails fast on missing checkpoint without creating checkpoints."""
    save_dir = tmp_path / "checkpoints"
    args = argparse.Namespace(
        model="nonexistent-checkpoint-xyz",
        dataset="wikitext",
        dataset_name=None,
        split="train",
        varlen=False,
        seq_len=128,
        context_len=128,
        batch_size=1,
        gradient_accumulation_steps=1,
        lr=1e-4,
        lr_scheduler="wsd",
        decay_ratio=0.2,
        warmup_steps=10,
        lr_min=1e-6,
        optimizer="adamw",
        epochs=1,
        steps=5,
        save_dir=str(save_dir),
        checkpoint_interval=10,
        eval_steps=10,
        compile=False,
        config=None,
    )

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=OSError("Missing weights")),
        patch("transformers.AutoModel.from_pretrained", side_effect=OSError("Missing weights")),
    ):
        with pytest.raises(RuntimeError):
            cmd_train(args)

    assert not save_dir.exists(), f"Save dir '{save_dir}' should not have been created on failure"


def test_error_message_names_model_and_refuses_random() -> None:
    """Bogus model name must fail closed with name and refusal statement in message."""
    bogus_name = "definitely-not-a-real-model-xyz"
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=ValueError("Model not recognized")),
        patch("transformers.AutoModel.from_pretrained", side_effect=ValueError("Model not recognized")),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            _load_model(bogus_name, device="cpu")

        err_msg = str(exc_info.value)
        assert bogus_name in err_msg
        assert "will NOT continue on a random model" in err_msg
