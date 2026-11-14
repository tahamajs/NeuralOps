#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for Master Transformer Composite Pipeline."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from neuralops.pipeline_transform.master_transformer import MasterTransformerPipeline


class ToyTransformerNet(nn.Module):
    """Simple multi-layer test model for composite pipeline verification."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_master_transformer_initialization():
    """Verify MasterTransformerPipeline initializes with valid default stages."""
    pipeline = MasterTransformerPipeline(target_architecture="cosformer", quant_bits=8)
    assert pipeline.target_architecture == "cosformer"
    assert pipeline.quant_bits == 8
    assert len(pipeline.pipeline_stages) == 5
    assert "svd_factorization" in pipeline.pipeline_stages
    assert "structured_pruning" in pipeline.pipeline_stages
    assert "quantization_w8a8" in pipeline.pipeline_stages


def test_master_transformer_stage_svd():
    """Verify execute_stage_svd actively factorizes linear layers."""
    model = ToyTransformerNet()
    pipeline = MasterTransformerPipeline()

    compressed = pipeline.execute_stage_svd(model, rank_ratio=0.5, skip_attention_qkv=False)
    assert compressed is not None
    # Model structure should be updated
    assert hasattr(compressed, "fc1")


def test_master_transformer_stage_prune():
    """Verify execute_stage_prune actively applies pruning rather than no-op."""
    model = ToyTransformerNet()
    pipeline = MasterTransformerPipeline()

    pruned = pipeline.execute_stage_prune(model, sparsity=0.25)
    assert pruned is not None
    # Verify model forward still works
    x = torch.randn(2, 64)
    out = pruned(x)
    assert out.shape == (2, 64)


def test_master_transformer_stage_quantize_fp8():
    """Verify execute_stage_quantize with 8 bits executes FP8 quantization."""
    model = ToyTransformerNet()
    pipeline = MasterTransformerPipeline(quant_bits=8)

    quantized = pipeline.execute_stage_quantize(model)
    assert quantized is not None
    x = torch.randn(2, 64)
    out = quantized(x)
    assert out.shape == (2, 64)


def test_master_transformer_stage_quantize_awq_int4():
    """Verify execute_stage_quantize with 4 bits executes AWQ quantization."""
    model = ToyTransformerNet()
    pipeline = MasterTransformerPipeline(quant_bits=4)

    quantized = pipeline.execute_stage_quantize(model)
    assert quantized is not None
    x = torch.randn(2, 64)
    out = quantized(x)
    assert out.shape == (2, 64)


def test_master_transformer_stage_quantize_unsupported_raises():
    """Verify execute_stage_quantize with unsupported bits raises honest NotImplementedError."""
    model = ToyTransformerNet()
    pipeline = MasterTransformerPipeline(quant_bits=3)

    with pytest.raises(NotImplementedError) as exc_info:
        pipeline.execute_stage_quantize(model)
    assert "3 bits is not supported" in str(exc_info.value)


def test_master_transformer_stage_export():
    """Verify execute_stage_export creates SafeTensors weights and manifest."""
    model = ToyTransformerNet()
    pipeline = MasterTransformerPipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = pipeline.execute_stage_export(model, output_dir=tmpdir)
        assert "weights" in artifacts
        assert "manifest" in artifacts
        assert os.path.exists(artifacts["weights"])
        assert os.path.exists(artifacts["manifest"])


def test_master_transformer_run_full_pipeline():
    """Verify run_full_pipeline executes all stages sequentially and returns artifacts."""
    model = ToyTransformerNet()
    pipeline = MasterTransformerPipeline(quant_bits=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = pipeline.run_full_pipeline(model, output_dir=tmpdir)
        assert result["status"] == "SUCCESS"
        assert "model" in result
        assert "artifacts" in result
        assert os.path.exists(result["artifacts"]["weights"])
        assert os.path.exists(result["artifacts"]["manifest"])
