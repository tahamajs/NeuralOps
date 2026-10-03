"""Small, dependency-light tests for the multi-architecture contract."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from framework.utils.compat import (FamilyModuleWrapper,
                                    causal_mask_from_attention_mask,
                                    is_decoder_projection_name,
                                    maybe_apply_rotary, module_family_support,
                                    repeat_kv, should_compress_linear,
                                    wrap_for_family)
from framework.utils.model_family import ModelAdapter, detect_family
from framework.utils.qwen_compat import QwenIncompatibleError


def cfg(**kw):
    base = dict(hidden_size=896, num_attention_heads=14, num_key_value_heads=2,
                head_dim=64, num_hidden_layers=24, model_type="qwen2")
    base.update(kw)
    return SimpleNamespace(**base)


def test_family_detection():
    expected = {"distilbert": "bert", "bert": "bert", "roberta": "roberta",
                "gpt2": "gpt", "qwen2": "qwen", "t5": "t5", "mamba": "ssm",
                "made_up": "unknown"}
    for model_type, family in expected.items():
        assert detect_family(SimpleNamespace(model_type=model_type)) == family


def test_adapter_geometry():
    q = ModelAdapter(cfg(), "qwen")
    assert (q.hidden_size, q.num_heads, q.num_key_value_heads, q.head_dim) == (896, 14, 2, 64)
    assert (q.q_proj_out, q.kv_proj_out, q.kv_repeat) == (896, 128, 7)
    b = ModelAdapter(cfg(model_type="bert", hidden_size=768, num_attention_heads=12,
                         num_key_value_heads=None, head_dim=64, num_hidden_layers=12), "bert")
    assert b.num_key_value_heads == 12
    assert not b.causal and b.position_embedding == "absolute"


def test_repeat_kv():
    x = torch.randn(2, 2, 3, 64)
    y = repeat_kv(x, 7)
    assert y.shape == (2, 14, 3, 64)
    assert torch.equal(y[:, 0], x[:, 0]) and torch.equal(y[:, 7], x[:, 1])


def test_rotary_is_required_for_qwen():
    q = torch.randn(1, 2, 3, 64)
    k = torch.randn_like(q)
    with pytest.raises(QwenIncompatibleError):
        maybe_apply_rotary(q, k, ModelAdapter(cfg(), "qwen"), None, None)
    b = ModelAdapter(cfg(model_type="bert"), "bert")
    q2, k2 = maybe_apply_rotary(q, k, b, None, None)
    assert q2 is q and k2 is k


def test_causal_and_padding_masks():
    q = ModelAdapter(cfg(), "qwen")
    additive = torch.zeros(2, 1, 3, 3)
    additive[:, :, :, 0] = -torch.inf
    assert causal_mask_from_attention_mask(additive, q) is additive
    b = ModelAdapter(cfg(model_type="bert"), "bert")
    padding = causal_mask_from_attention_mask(torch.tensor([[1, 1, 0]]), b)
    assert padding.dtype == torch.bool and padding.shape == (1, 1, 1, 3)


def test_alias_and_wrapper_contract():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.o_proj = nn.Linear(8, 8)

        def forward(self, x, **kwargs):
            return self.o_proj(self.v_proj(x))

    m = Dummy()
    assert wrap_for_family(m, "bert") is m
    wrapped = wrap_for_family(m, "qwen")
    assert isinstance(wrapped, FamilyModuleWrapper)
    assert wrapped.module.q_lin is wrapped.module.q_proj
    # A module without a native projection is not given a fabricated alias.
    plain = nn.Linear(8, 8)
    wrap_for_family(plain, "bert")
    assert not hasattr(plain, "q_lin")


def test_family_support_and_decoder_compression_targeting():
    assert module_family_support("mamba")["qwen"] == "incompatible"
    assert module_family_support("mamba")["bert"] == "encoder_compatible"
    assert module_family_support("faq")["qwen"] == "adapted"
    assert is_decoder_projection_name("model.layers.0.self_attn.q_proj.weight")
    assert not is_decoder_projection_name("model.layers.0.mlp.gate_proj.weight")
    model = SimpleNamespace(config=cfg())
    assert should_compress_linear(model, "q_proj")
    assert not should_compress_linear(model, "gate_proj")
    assert should_compress_linear(SimpleNamespace(config=cfg(model_type="bert")), "fc1")


def test_common_attention_constructors_use_config_geometry():
    # Constructors are intentionally tested without a pretrained source. This
    # catches hard-coded 12-head/768-dimensional projection assumptions while
    # remaining independent of CUDA and model downloads.
    from framework.attentions.adaptive_sparse_attention import \
        AdaptiveSparseAttention
    from framework.attentions.adaptive_svd_attention import \
        AdaptiveSVDAttention
    from framework.attentions.asvd_attention import ASVDAttention
    from framework.attentions.cluster_attention import ClusterAttention
    from framework.attentions.flash_attention3 import FlashAttention3

    classes = [FlashAttention3, ClusterAttention, AdaptiveSparseAttention,
               AdaptiveSVDAttention, ASVDAttention]
    for cls in classes:
        module = cls(cfg())
        q = next((getattr(module, n) for n in ("q_proj", "query", "W_q") if hasattr(module, n)), None)
        assert q is not None
        assert q.in_features == 896
        assert q.out_features == 896
