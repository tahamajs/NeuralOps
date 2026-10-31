#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive test suite for NeuralOps v1.7.0 high-level ergonomics and public API contracts.

Tests:
1. Canonical package version check: assert `neuralops.__version__ == "1.7.0"` and `neuralops.get_version() == "1.7.0"`.
2. Public symbol exposure in `neuralops`: assert that `transplant`, `export`, `compress`, `replace_attention`
   are directly importable from `neuralops` and listed in `__all__`.
3. `transplant` alias behavior delegating to `replace_attention`.
4. `export` alias behavior delegating to `export_model`.
5. End-to-end integration of high-level ergonomics on a toy PyTorch model.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

import neuralops
from neuralops import (
    compress,
    export,
    export_model,
    get_version,
    pipeline,
    replace_attention,
    transplant,
    transplant_attention,
)
from neuralops.core._attention_surgery import heal_replaced_model


class DummyAttention(nn.Module):
    """Toy attention layer for testing transplantation and replacement."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(scores, v)


class ToyTransformerBlock(nn.Module):
    """Toy transformer block containing self_attn and mlp."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.self_attn = DummyAttention(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.self_attn(x))


class ToyModel(nn.Module):
    """Toy model containing stacked blocks."""

    def __init__(self, num_layers: int = 2, dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyTransformerBlock(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ToyCausalLM(nn.Module):
    """Toy causal language model with embedding and lm_head for token generation."""

    def __init__(self, vocab_size: int = 100, dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.block = ToyTransformerBlock(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.block(x)
        return self.lm_head(x)


def test_canonical_package_version() -> None:
    """Verify that neuralops.__version__ and neuralops.get_version() report exactly '1.7.0'."""
    assert neuralops.__version__ == "1.7.0"
    assert get_version() == "1.7.0"
    assert neuralops.get_version() == "1.7.0"


def test_public_symbol_exposure() -> None:
    """Verify that transplant, export, compress, replace_attention, pipeline are exposed in __all__ and importable."""
    expected_symbols = ["transplant", "export", "compress", "replace_attention", "pipeline"]
    for sym in expected_symbols:
        assert sym in neuralops.__all__, f"Symbol '{sym}' is missing from neuralops.__all__"
        attr = getattr(neuralops, sym, None)
        assert attr is not None, f"neuralops has no attribute '{sym}'"
        assert callable(attr), f"neuralops.{sym} is not callable"

    # Verify that directly imported symbols match top-level module attributes
    assert transplant is neuralops.transplant
    assert export is neuralops.export
    assert compress is neuralops.compress
    assert replace_attention is neuralops.replace_attention
    assert pipeline is neuralops.pipeline


def test_transplant_alias_identity_and_delegation() -> None:
    """Verify that transplant is an alias for transplant_attention and delegates to replace_attention."""
    # Identity check
    assert transplant is transplant_attention
    assert neuralops.transplant is neuralops.transplant_attention

    # Delegation check with mock
    dummy_model = ToyModel()
    with patch("neuralops.api.replace_attention") as mock_replace:
        mock_replace.return_value = dummy_model

        result = transplant(
            model=dummy_model,
            attention="gla",
            rank=16,
            copy_weights=False,
            layers=[0],
            custom_arg="test_val",
        )

        mock_replace.assert_called_once_with(
            model=dummy_model,
            attention="gla",
            rank=16,
            copy_weights=False,
            layers=[0],
            custom_arg="test_val",
        )
        assert result is dummy_model

    # Check argument alias resolution (attention_type and target_attention)
    with patch("neuralops.api.replace_attention") as mock_replace:
        mock_replace.return_value = dummy_model

        transplant(model=dummy_model, attention_type="cosformer")
        mock_replace.assert_called_once_with(
            model=dummy_model,
            attention="cosformer",
            rank=None,
            copy_weights=True,
            layers=None,
        )

    with patch("neuralops.api.replace_attention") as mock_replace:
        mock_replace.return_value = dummy_model

        transplant(model=dummy_model, target_attention="retnet")
        mock_replace.assert_called_once_with(
            model=dummy_model,
            attention="retnet",
            rank=None,
            copy_weights=True,
            layers=None,
        )


def test_export_alias_identity_and_delegation() -> None:
    """Verify that export is an alias for export_model and delegates to _export_model."""
    # Identity check
    assert export is export_model
    assert neuralops.export is neuralops.export_model

    # Delegation check with mock
    dummy_model = ToyModel()
    with patch("neuralops.api._export_model") as mock_export_impl:
        mock_export_impl.return_value = Path("/tmp/mock_export_dir")

        out = export(
            model=dummy_model,
            output_dir="/tmp/mock_export_dir",
            format="safetensors",
            extra_meta="ok",
        )

        mock_export_impl.assert_called_once_with(
            model=dummy_model,
            output_dir="/tmp/mock_export_dir",
            format="safetensors",
            extra_meta="ok",
        )
        assert out == Path("/tmp/mock_export_dir")


def test_export_end_to_end_toy_model() -> None:
    """Verify that neuralops.export works end-to-end saving safetensors / neuralops_config."""
    model = ToyModel(num_layers=1, dim=16)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = export(model=model, output_dir=tmp_dir, format="safetensors")
        assert Path(out_path).exists()
        config_file = Path(tmp_dir) / "neuralops_config.json"
        assert config_file.exists(), f"Expected config file at {config_file}"


def test_compress_symbol_callable() -> None:
    """Verify that neuralops.compress is callable and delegates to api.compress."""
    dummy_model = MagicMock(spec=nn.Module)
    with patch("neuralops.api.compress") as mock_compress:
        mock_compress.return_value = dummy_model

        res = compress(model=dummy_model, method="svd", ratio=0.5)
        mock_compress.assert_called_once_with(model=dummy_model, method="svd", ratio=0.5)
        assert res is dummy_model


def test_heal_replaced_model_multi_batch_cycling() -> None:
    """Verify that heal_replaced_model cycles through multiple batches without starvation."""
    teacher = ToyModel(num_layers=1, dim=16)
    student = ToyModel(num_layers=1, dim=16)

    batches = [torch.randn(2, 4, 16) for _ in range(3)]
    result = heal_replaced_model(
        teacher,
        student,
        batches,
        steps=6,
        also_ridge_init=False,
        log_to_ledger=False,
    )
    assert result["num_healed_layers"] == 1
    assert "initial_mse" in result
    assert "final_mse" in result


def test_heal_replaced_model_nan_loss_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that a structured logger.warning is emitted when non-finite loss is encountered."""
    teacher = ToyModel(num_layers=1, dim=16)
    student = ToyModel(num_layers=1, dim=16)

    # Initialize student parameters to NaN to trigger non-finite loss immediately
    with torch.no_grad():
        for p in student.parameters():
            p.fill_(float("nan"))

    with caplog.at_level(logging.WARNING):
        result = heal_replaced_model(
            teacher,
            student,
            torch.randn(2, 4, 16),
            steps=3,
            also_ridge_init=False,
            log_to_ledger=False,
        )

    assert result["num_healed_layers"] == 1
    assert any("Non-finite loss" in record.message for record in caplog.records)


def test_cli_transplant_parser_ergonomics() -> None:
    """Verify that transplant parser cleanly handles --target-attention, --output, --dataset."""
    from neuralops.benchmarks.cli import _build_parser

    parser = _build_parser()

    # Test --target-attention and --output aliases
    args1 = parser.parse_args([
        "transplant",
        "--model", "gpt2",
        "--target-attention", "cosformer",
        "--output", "models/custom_out",
    ])
    assert args1.model == "gpt2"
    assert args1.attention == "cosformer"
    assert args1.output_dir == "models/custom_out"

    # Test --calibration-dataset alias
    args2 = parser.parse_args([
        "transplant",
        "--model", "gpt2",
        "--attention", "cosformer",
        "--calibration-dataset", "wikitext",
    ])
    assert args2.calibration_text == "wikitext"

    # Test --dataset alias
    args3 = parser.parse_args([
        "transplant",
        "--model", "gpt2",
        "--attention", "cosformer",
        "--dataset", "sst2",
    ])
    assert args3.calibration_text == "sst2"


def test_cli_compress_parser_ergonomics() -> None:
    """Verify that compress parser cleanly handles --output, --dataset, --calibration-dataset."""
    from neuralops.benchmarks.cli import _build_parser

    parser = _build_parser()

    # Test --output alias
    args1 = parser.parse_args([
        "compress",
        "--model", "gpt2",
        "--output", "results/compressed_gpt2",
    ])
    assert args1.model == "gpt2"
    assert args1.output_dir == "results/compressed_gpt2"

    # Test --dataset and --calibration-dataset
    args2 = parser.parse_args([
        "compress",
        "--model", "gpt2",
        "--dataset", "wikitext-2",
    ])
    assert args2.dataset == "wikitext-2"

    args3 = parser.parse_args([
        "compress",
        "--model", "gpt2",
        "--calibration-dataset", "c4",
    ])
    assert args3.dataset == "c4"


def test_cli_compose_parser_separation() -> None:
    """Verify that compose parser arguments do not contaminate compress subcommand."""
    from neuralops.benchmarks.cli import _build_parser

    parser = _build_parser()

    # compress should reject --projection
    with pytest.raises(SystemExit):
        parser.parse_args(["compress", "--model", "gpt2", "--projection", "svd"])

    # compose should accept --projection and --kernel-map
    args = parser.parse_args(["compose", "--projection", "svd", "--kernel-map", "cosformer"])
    assert args.projection == "svd"
    assert args.kernel_map == "cosformer"


def test_cli_architecture_mismatch_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that main() catches architecture mismatches and prints a diagnostic card via print_cli_error."""
    from neuralops.benchmarks import cli

    error_calls: list[dict[str, Any]] = []

    def mock_print_cli_error(msg: str, hint: str | None = None, title: str = "Execution Error") -> None:
        error_calls.append({"msg": msg, "hint": hint, "title": title})

    def mock_dispatch(_args: Any) -> int:
        raise ValueError("Could not locate encoder in model architecture for selected attention")

    monkeypatch.setattr(cli, "print_cli_error", mock_print_cli_error)
    monkeypatch.setattr(cli, "_dispatch", mock_dispatch)

    exit_code = cli.main(["transplant", "--model", "custom", "--attention", "cosformer"])
    assert exit_code == 1
    assert len(error_calls) == 1
    assert error_calls[0]["title"] == "Architecture Mismatch"
    assert "Architecture mismatch" in error_calls[0]["msg"]
    assert "supported and compatible" in error_calls[0]["hint"]


def test_pipeline_text_generation_ergonomics() -> None:
    """Verify that neuralops.pipeline instantiates and executes text-generation."""
    toy = ToyCausalLM(vocab_size=100, dim=16)

    class MockTokenizer:
        def __call__(self, text, return_tensors="pt", **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def batch_decode(self, token_ids, **kwargs):
            return ["Generated: Hello NeuralOps"]

        def decode(self, token_ids, **kwargs):
            return "Generated: Hello NeuralOps"

    pipe = neuralops.pipeline("text-generation", model=toy, tokenizer=MockTokenizer())
    out = pipe("Hello")
    assert isinstance(out, list)
    assert len(out) == 1
    assert "generated_text" in out[0]
    assert "Hello NeuralOps" in out[0]["generated_text"]


def test_pipeline_exposure_and_callable() -> None:
    """Verify that neuralops.pipeline is exposed in __all__, importable, and callable."""
    assert "pipeline" in neuralops.__all__, "pipeline is missing from neuralops.__all__"
    assert hasattr(neuralops, "pipeline"), "neuralops has no attribute 'pipeline'"
    assert callable(neuralops.pipeline), "neuralops.pipeline is not callable"
    assert pipeline is neuralops.pipeline


def test_pipeline_text_generation_end_to_end() -> None:
    """Verify neuralops.pipeline text-generation task with model and tokenizer."""
    mock_model = MagicMock()
    mock_model.generate.return_value = torch.tensor([[101, 2054, 2003, 102]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[101, 2054]])}
    mock_tokenizer.batch_decode.return_value = ["What is"]

    # 1. Pipeline creation
    pipe = pipeline("text-generation", model=mock_model, tokenizer=mock_tokenizer)
    assert pipe.task == "text-generation"
    assert callable(pipe)

    # 2. Execution with single prompt
    output = pipe("What is")
    assert isinstance(output, list)
    assert len(output) == 1
    assert "generated_text" in output[0]
    assert output[0]["generated_text"] == "What is"

    # 3. Execution with multiple prompts
    mock_tokenizer.batch_decode.return_value = ["What is", "How to"]
    batch_out = pipe(["Prompt 1", "Prompt 2"])
    assert isinstance(batch_out, list)
    assert len(batch_out) == 2
    assert batch_out[0][0]["generated_text"] == "What is"
    assert batch_out[1][0]["generated_text"] == "How to"

    # 4. Immediate execution via kwargs prompt
    mock_tokenizer.batch_decode.return_value = ["Immediate generation"]
    direct_out = pipeline("text-generation", model=mock_model, tokenizer=mock_tokenizer, prompt="Test prompt")
    assert isinstance(direct_out, list)
    assert direct_out[0]["generated_text"] == "Immediate generation"

    # 5. Unsupported task rejection
    with pytest.raises(ValueError, match="Unsupported pipeline task"):
        pipeline("image-segmentation", model=mock_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
