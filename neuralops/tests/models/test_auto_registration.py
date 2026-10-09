"""Tests for AutoConfig and AutoModel registration."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
import neuralops.models
from neuralops.models import NeuralOpsConfig, NeuralOpsModel, NeuralOpsForCausalLM


def test_auto_classes_registered():
    config = NeuralOpsConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        attention_type="cosformer",
    )
    model = AutoModelForCausalLM.from_config(config)
    assert isinstance(model, NeuralOpsForCausalLM)

    base_model = AutoModel.from_config(config)
    assert isinstance(base_model, NeuralOpsModel)
