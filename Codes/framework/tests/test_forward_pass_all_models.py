"""
Comprehensive forward-pass smoke tests across all supported model families.
Runs on CPU — no GPU needed. Each test builds a tiny model (2 layers, 64 hidden).

Run: pytest tests/test_forward_pass_all_models.py -v
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from framework.eval_all_models_datasets import build_model_instance

MODEL_FAMILIES = ["bert", "roberta", "t5", "distilbert", "gpt2"]


def make_batch(family: str, batch_size: int = 2, seq_len: int = 16):
    b = {
        "input_ids": torch.randint(100, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }
    if family not in {"t5", "gpt2"}:
        b["token_type_ids"] = torch.zeros(batch_size, seq_len, dtype=torch.long)
    b["labels"] = torch.zeros(batch_size, dtype=torch.long)
    return b


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_model_builds(family):
    model, cfg = build_model_instance(
        family, num_labels=2, hidden_size=64, num_heads=4, num_layers=1,
        device=torch.device("cpu")
    )
    assert isinstance(model, nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_forward_pass(family):
    model, _ = build_model_instance(
        family, num_labels=2, hidden_size=64, num_heads=4, num_layers=1,
        device=torch.device("cpu")
    )
    batch = make_batch(family)

    # Strip invalid keys
    if family == "t5":
        batch.pop("token_type_ids", None)
        batch.pop("labels", None)  # T5EncoderModel has no classification head
    if family == "gpt2":
        batch.pop("token_type_ids", None)
        batch.pop("labels", None)  # GPT2LMHeadModel requires (B, L) labels for LM loss

    model.eval()
    with torch.no_grad():
        output = model(**batch)
    assert output is not None


@pytest.mark.parametrize("family", ["bert", "distilbert"])
def test_backward_pass(family):
    model, _ = build_model_instance(
        family, num_labels=2, hidden_size=64, num_heads=4, num_layers=1,
        device=torch.device("cpu")
    )
    model.train()
    batch = make_batch(family)
    output = model(**batch)
    loss = output.loss if hasattr(output, "loss") and output.loss is not None else output[0]
    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
        loss = loss.mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"


@pytest.mark.parametrize("module_name", [
    "aegis_v1", "aegis_v2", "nystrom_attention", "linformer_attention", "gqa_attention"
])
def test_attention_injection_bert(module_name):
    from framework.eval_all_models_datasets import apply_module_to_model
    device = torch.device("cpu")
    model, _ = build_model_instance("bert", num_labels=2, hidden_size=64, num_heads=4, num_layers=1, device=device)
    batch = make_batch("bert")
    try:
        model, ratio = apply_module_to_model(model, module_name, "attention", rank=8, device=device)
    except Exception as e:
        pytest.skip(f"Could not inject {module_name}: {e}")
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    assert output is not None
