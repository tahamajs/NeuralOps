"""Tests for hybrid attention and compression specifications."""

import pytest
from neuralops.models.hybrid_spec import normalize_hybrid_spec, get_layer_spec
from neuralops.core.registry import ATTENTION_REGISTRY
import neuralops.attentions


def test_hybrid_spec_normalization_default():
    specs = normalize_hybrid_spec(None, num_layers=4, default_type="cosformer")
    assert len(specs) == 4
    for s in specs:
        assert s["type"] == "cosformer"


def test_hybrid_spec_single_dict():
    spec = {"layers": [1, 3], "type": "rpc_attention"}
    specs = normalize_hybrid_spec(spec, num_layers=4, default_type="cosformer")
    assert specs[0]["type"] == "cosformer"
    assert specs[1]["type"] == "rpc_attention"
    assert specs[2]["type"] == "cosformer"
    assert specs[3]["type"] == "rpc_attention"


def test_hybrid_spec_list_of_dicts():
    spec = [
        {"layers": [0, 2], "type": "galerkin_attention"},
        {"layers": [1, 3], "type": "cosformer"},
    ]
    specs = normalize_hybrid_spec(spec, num_layers=4, default_type="cosformer")
    assert specs[0]["type"] == "galerkin_attention"
    assert specs[1]["type"] == "cosformer"
    assert specs[2]["type"] == "galerkin_attention"
    assert specs[3]["type"] == "cosformer"


def test_hybrid_spec_out_of_bounds_and_duplicates():
    with pytest.raises(ValueError, match="out of range"):
        normalize_hybrid_spec({"layers": [5], "type": "cosformer"}, num_layers=4)

    with pytest.raises(ValueError, match="specified more than once"):
        normalize_hybrid_spec(
            [{"layers": [1], "type": "cosformer"}, {"layers": [1], "type": "rpc_attention"}],
            num_layers=4,
        )
