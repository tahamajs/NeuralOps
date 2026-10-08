"""Unit tests for sink attention."""
import torch
import pytest
from neuralops.attentions.adaptive.sink_attention import SinkAttention


def test_sink_attention_no_op():
    layer = SinkAttention(num_sink_tokens=4, window_size=64)
    x = torch.randn(2, 32, 16)
    out, _ = layer(x)
    assert out.shape == (2, 32, 16)
    assert torch.equal(out, x)


def test_sink_attention_clamping():
    layer = SinkAttention(num_sink_tokens=4, window_size=32)
    x = torch.randn(2, 64, 16)
    out, _ = layer(x)
    assert out.shape == (2, 32, 16)
    # Check sinks preserved
    assert torch.equal(out[:, :4, :], x[:, :4, :])
    # Check recent preserved
    assert torch.equal(out[:, 4:, :], x[:, -28:, :])
