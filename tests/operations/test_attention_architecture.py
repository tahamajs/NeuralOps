"""Tests for attention and architecture operations."""

import pytest
import torch
import torch.nn as nn

from neuralops.operations.attention.configure import AttentionConfigureOperation
from neuralops.operations.attention.replace import AttentionReplaceOperation
from neuralops.operations.architecture.modify import ArchitectureModifyOperation
from neuralops.operations.architecture.drop_layer import ArchitectureDropLayerOperation
from neuralops.operations.architecture.reduce_mlp import ArchitectureReduceMLPOperation


class DummyAttention(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.chunk_size = 32

    def forward(self, x):
        return x


class DummyMLP(nn.Module):
    def __init__(self, dim=64, mid_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, mid_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(mid_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.attn = DummyAttention(64)
        self.mlp = DummyMLP(64, 128)
        self.layers = nn.ModuleList([DummyAttention(64) for _ in range(3)])

    def forward(self, x):
        return self.mlp(self.attn(self.norm(x)))


def test_attention_configure():
    model = DummyModel()
    op = AttentionConfigureOperation()
    model, ev = op.apply(model, {"chunk_size": 64, "auto_tune": True})
    assert ev["configured_modules_count"] >= 1
    assert model.attn.chunk_size in (16, 32, 64, 128, 256)


def test_attention_replace():
    model = DummyModel()
    op = AttentionReplaceOperation()
    model, ev = op.apply(model, {"target_attention": "linear"})
    assert ev["replaced_count"] >= 1
    x = torch.randn(2, 8, 64)
    out = model(x)
    assert out.shape == (2, 8, 64)


def test_architecture_modify():
    model = DummyModel()
    op = ArchitectureModifyOperation()
    model, ev = op.apply(model, {"target_activation": "gelu", "norm_epsilon": 1e-5})
    assert isinstance(model.mlp.act, nn.GELU)
    assert model.norm.eps == 1e-5


def test_architecture_drop_layer():
    model = DummyModel()
    op = ArchitectureDropLayerOperation()
    model, ev = op.apply(model, {"layer_indices": [1]})
    assert ev["remaining_layer_count"] == 2
    assert len(model.layers) == 2


def test_architecture_reduce_mlp():
    model = DummyModel()
    op = ArchitectureReduceMLPOperation()
    model, ev = op.apply(model, {"reduction_factor": 0.5})
    assert ev["reduced_mlps_count"] == 1
    assert model.mlp.fc1.out_features == 64
