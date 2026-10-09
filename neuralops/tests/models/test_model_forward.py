"""Tests for NeuralOps model creation, forward pass, loss computation, and generation."""

import pytest
import torch
import tempfile
from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM


def get_tiny_config(attn_type="cosformer"):
    return NeuralOpsConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        attention_type=attn_type,
    )


def test_tiny_model_creation():
    config = get_tiny_config()
    model = NeuralOpsForCausalLM(config)
    assert model is not None
    assert model.model.embed_tokens.num_embeddings == 128
    assert len(model.model.layers) == 2


def test_model_forward_with_and_without_labels():
    config = get_tiny_config()
    model = NeuralOpsForCausalLM(config)
    model.eval()

    inputs = torch.randint(0, 128, (2, 8))
    outputs = model(inputs)
    assert outputs.logits.shape == (2, 8, 128)
    assert outputs.loss is None

    # With labels
    labels = inputs.clone()
    outputs_with_loss = model(inputs, labels=labels)
    assert outputs_with_loss.loss is not None
    assert outputs_with_loss.loss.item() > 0


def test_model_generation():
    config = get_tiny_config()
    model = NeuralOpsForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, 128, (1, 4))
    generated = model.generate(input_ids, max_new_tokens=4, do_sample=False)
    assert generated.shape == (1, 8)


def test_model_save_and_reload():
    config = get_tiny_config()
    model = NeuralOpsForCausalLM(config)
    model.eval()

    inputs = torch.randint(0, 128, (1, 4))
    with torch.no_grad():
        out1 = model(inputs).logits

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        reloaded = NeuralOpsForCausalLM.from_pretrained(tmp_dir)
        reloaded.eval()
        with torch.no_grad():
            out2 = reloaded(inputs).logits

        assert torch.allclose(out1, out2, atol=1e-5)
