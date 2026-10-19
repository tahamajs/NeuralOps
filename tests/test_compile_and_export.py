#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import os
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

import neuralops
from neuralops.export.exporter import compile_to_so, AOTICompiledModelRunner


class TinyModel(nn.Module):
    def __init__(self, d_in=32, d_out=32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TinyModelWithTupleOutput(nn.Module):
    def __init__(self, d_in=32, d_out=32):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        out = self.fc(x)
        return out, {"past_kv": torch.zeros(1)}


def test_compile_to_so_basic_and_runner():
    model = TinyModel()
    sample_input = torch.randn(2, 16, 32)
    with tempfile.TemporaryDirectory() as tmpdir:
        so_path = Path(tmpdir) / "tiny_model.so"
        runner = compile_to_so(
            model=model,
            sample_inputs=sample_input,
            output_path=so_path,
            dynamic_batch=True,
            dynamic_seq_len=True,
        )

        assert isinstance(runner, AOTICompiledModelRunner)
        out = runner(sample_input)
        ref = model(sample_input)
        assert out.shape == ref.shape
        diff = (out - ref).abs().max().item()
        assert diff < 1e-4, f"AOT output difference too large: {diff}"

        for s in [8, 24, 32]:
            var_input = torch.randn(2, s, 32)
            var_out = runner(var_input)
            var_ref = model(var_input)
            assert var_out.shape == (2, s, 32)
            assert (var_out - var_ref).abs().max().item() < 1e-4


def test_compile_to_so_tuple_output_wrapper():
    model = TinyModelWithTupleOutput()
    sample_input = torch.randn(1, 10, 32)
    with tempfile.TemporaryDirectory() as tmpdir:
        so_path = Path(tmpdir) / "tuple_model.so"
        runner = compile_to_so(
            model=model,
            sample_inputs=sample_input,
            output_path=so_path,
        )
        out = runner(sample_input)
        ref, _ = model(sample_input)
        assert out.shape == ref.shape
        assert (out - ref).abs().max().item() < 1e-4


def test_top_level_neuralops_compile_to_so_export():
    assert hasattr(neuralops, "compile_to_so")
    assert callable(neuralops.compile_to_so)
