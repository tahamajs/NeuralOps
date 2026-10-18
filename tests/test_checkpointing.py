"""Tests for selective gradient checkpointing."""
import torch
import torch.nn as nn
from neuralops.training.checkpointing import checkpoint_selective, SelectiveCheckpointModule


def test_checkpoint_selective():
    lin = nn.Linear(8, 8)
    x = torch.randn(2, 8, requires_grad=True)

    # With checkpoint
    out_cp = checkpoint_selective(lin, x, should_checkpoint=True)
    out_cp.sum().backward()
    assert x.grad is not None

    # Without checkpoint
    x2 = torch.randn(2, 8, requires_grad=True)
    out_no = checkpoint_selective(lin, x2, should_checkpoint=False)
    out_no.sum().backward()
    assert x2.grad is not None


def test_selective_checkpoint_module():
    lin = nn.Linear(8, 8)
    wrapper = SelectiveCheckpointModule(lin, min_seqlen_threshold=16)

    # Short seqlen -> no checkpointing
    x_short = torch.randn(2, 8, 8, requires_grad=True)
    out1 = wrapper(x_short)
    assert out1.shape == (2, 8, 8)

    # Long seqlen -> triggers checkpointing
    x_long = torch.randn(2, 32, 8, requires_grad=True)
    out2 = wrapper(x_long)
    out2.sum().backward()
    assert x_long.grad is not None
