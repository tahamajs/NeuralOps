"""Tests for NeuralOpsConfig validation and serialization."""

import pytest
import tempfile
from pathlib import Path
from neuralops.models.configuration import NeuralOpsConfig


def test_config_instantiation_and_defaults():
    config = NeuralOpsConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    assert config.model_type == "neuralops"
    assert config.vocab_size == 1000
    assert config.hidden_size == 64
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 2
    assert config.head_dim == 32


def test_config_validation_errors():
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        NeuralOpsConfig(hidden_size=-1)

    with pytest.raises(ValueError, match="num_hidden_layers must be positive"):
        NeuralOpsConfig(num_hidden_layers=0)

    with pytest.raises(ValueError, match="num_attention_heads must be divisible by num_key_value_heads"):
        NeuralOpsConfig(
            hidden_size=64,
            num_attention_heads=3,
            num_key_value_heads=2,
        )


def test_config_json_roundtrip():
    config = NeuralOpsConfig(
        vocab_size=500,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        attention_type="cosformer",
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        config.save_pretrained(tmp_dir)
        loaded = NeuralOpsConfig.from_pretrained(tmp_dir)
        assert loaded.vocab_size == 500
        assert loaded.hidden_size == 64
        assert loaded.attention_type == "cosformer"
