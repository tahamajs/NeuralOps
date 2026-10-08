"""Test coverage for each attention family."""
import pytest
import torch
from neuralops.core.registry import ATTENTION_REGISTRY
import neuralops.attentions


FAMILIES = {
    "svd": ["svd_linear_attention", "adaptive_svd_attention"],
    "linear": ["cosformer", "performer"],
    "ssm": ["mamba_block", "selective_ssm", "simple_ssm"],
    "hybrid": ["hybrid_attention", "fusion_attn"],
    "sparse": ["adaptive_sparse_attention", "cluster_attention"],
    "gated": ["gated_delta_net", "gated_linear_attention"],
    "adaptive": ["adaptive_span_attention", "token_compressed_attention"],
    "modern": ["gqa_attention", "mla_attention"],
}


@pytest.mark.smoke
@pytest.mark.parametrize("family,modules", FAMILIES.items())
def test_family_registered(family, modules):
    """Each family's modules should be registered."""
    for mod in modules:
        cls = ATTENTION_REGISTRY.get(mod)
        assert cls is not None, f"{family}/{mod} not registered"
