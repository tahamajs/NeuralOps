"""
Public API Verification Tests for NeuralOps 1.3.0.

Verifies the 19 functions in `neuralops.api` and `neuralops`:
  - Model/Export: load_model, replace_attention, compress, export_model
  - Discovery: list_attentions, list_compressors, get_attention_class, get_compressor_class, list_families
  - Runners: train, distill, evaluate, benchmark, benchmark_all
  - Utils: info, available_backends, has_triton, validate, get_version
"""

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

import neuralops
from neuralops import (
    load_model,
    replace_attention,
    compress,
    export_model,
    list_attentions,
    list_compressors,
    get_attention_class,
    get_compressor_class,
    list_families,
    train,
    distill,
    evaluate,
    benchmark,
    benchmark_all,
    info,
    available_backends,
    has_triton,
    validate,
    get_version,
)


def test_public_api_symbols_and_callables():
    api_symbols = [
        "load_model",
        "replace_attention",
        "compress",
        "export_model",
        "list_attentions",
        "list_compressors",
        "get_attention_class",
        "get_compressor_class",
        "list_families",
        "train",
        "distill",
        "evaluate",
        "benchmark",
        "benchmark_all",
        "info",
        "available_backends",
        "has_triton",
        "validate",
        "get_version",
    ]
    for sym in api_symbols:
        assert hasattr(neuralops, sym), f"Missing {sym} from neuralops"
        assert callable(getattr(neuralops, sym)), f"{sym} is not callable"


def test_version_and_info():
    assert get_version() == "1.3.0"
    assert neuralops.__version__ == "1.3.0"

    inf = info()
    assert isinstance(inf, dict)
    assert inf["version"] == "1.3.0"
    assert inf["attentions_count"] >= 130
    assert inf["compressors_count"] >= 70
    assert isinstance(inf["backends"], list)
    assert "cpu" in inf["backends"]


def test_discovery_functions():
    attns = list_attentions()
    comps = list_compressors()
    assert len(attns) >= 130
    assert len(comps) >= 70
    assert "cosformer" in attns
    assert "direct_svd" in comps

    cls_attn = get_attention_class("cosformer")
    assert cls_attn is not None

    cls_comp = get_compressor_class("direct_svd")
    assert cls_comp is not None

    fams = list_families()
    assert isinstance(fams, dict)
    assert any(k.startswith("attention_") for k in fams)
    assert any(k.startswith("compression_") for k in fams)


def test_validation_and_backends():
    backends = available_backends()
    assert isinstance(backends, list)
    assert "cpu" in backends

    triton_avail = has_triton()
    assert isinstance(triton_avail, bool)

    # Validate on dummy model
    class SimpleToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 32)
            self.lin = nn.Linear(32, 100)

        def forward(self, input_ids, labels=None):
            x = self.emb(input_ids)
            logits = self.lin(x)
            loss = None
            if labels is not None:
                loss = logits.sum() * 0.0
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits, loss=loss)

    toy = SimpleToyModel()
    res = validate(toy, input_shape=(2, 16), device="cpu")
    assert res is True


def test_model_surgery_metadata_and_export_roundtrip(tmp_path):
    config = BertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )
    model = BertModel(config)

    # 1. Replace attention
    model = replace_attention(model, "cosformer")
    assert hasattr(model, "_neuralops_metadata")
    assert len(model._neuralops_metadata) == 1
    assert model._neuralops_metadata[0]["op"] == "replace"
    assert model._neuralops_metadata[0]["type"] == "cosformer"

    # 2. Compress
    model = compress(model, method="direct_svd", rank=8)
    assert len(model._neuralops_metadata) == 2
    assert model._neuralops_metadata[1]["op"] == "compress"
    assert model._neuralops_metadata[1]["type"] == "direct_svd"

    # 3. Export
    out_dir = tmp_path / "exported_model"
    export_model(model, out_dir, format="safetensors")
    assert (out_dir / "neuralops_config.json").exists()

    # 4. Load
    fresh_base = BertModel(config)
    loaded = load_model(out_dir, base_model=fresh_base)
    assert hasattr(loaded, "_neuralops_metadata")
    assert len(loaded._neuralops_metadata) == 2
