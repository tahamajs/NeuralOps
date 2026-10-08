"""Unit tests for neuralops.utils.compat package."""

import pytest
import torch
import torch.nn as nn

from neuralops.utils.compat import (
    ALLOWED_FFN_PATTERNS,
    PROTECTED_KEYWORDS,
    SENSITIVE_PATTERNS,
    FamilyModuleWrapper,
    _is_protected_param,
    get_dynamic_compression_rank,
    get_recovery_params,
    init_output_scale,
    install_projection_aliases,
    is_decoder_family,
    is_decoder_projection_name,
    repeat_kv,
    should_compress_linear,
    should_compress_qwen3_5,
)


def test_repeat_kv():
    # [B, Hkv, L, D] -> [B, H, L, D]
    x = torch.randn(2, 4, 8, 16)
    out = repeat_kv(x, n_rep=2)
    assert out.shape == (2, 8, 8, 16)

    # n_rep = 1 should be identity
    out_ident = repeat_kv(x, n_rep=1)
    assert out_ident is x


def test_init_output_scale():
    mod = nn.Linear(10, 10)
    assert not hasattr(mod, "output_scale")
    init_output_scale(mod)
    assert hasattr(mod, "output_scale")
    assert isinstance(mod.output_scale, nn.Parameter)
    assert mod.output_scale.item() == 1.0


def test_decoder_checks():
    assert is_decoder_family("gpt2")
    assert is_decoder_family("llama")
    assert not is_decoder_family("bert")

    assert is_decoder_projection_name("model.layers.0.self_attn.q_proj")
    assert is_decoder_projection_name("transformer.h.0.attn.c_attn")
    assert not is_decoder_projection_name("model.layers.0.mlp.gate_proj")


def test_should_compress():
    assert should_compress_qwen3_5("model.layers.0.mlp.down_proj")
    assert not should_compress_qwen3_5("model.layers.0.self_attn.q_proj")

    dummy_model = nn.Linear(10, 10)
    assert should_compress_linear(dummy_model, "transformer.h.0.attn.c_attn", family="gpt2")
    assert not should_compress_linear(dummy_model, "lm_head.weight", family="gpt2")


def test_dynamic_compression_rank():
    assert get_dynamic_compression_rank(1024, 1024, model_or_family="gpt2", default_rank=16) == 16
    # Qwen3.5: target = min(1024, 1024) // 4 = 256
    assert get_dynamic_compression_rank(1024, 1024, model_or_family="qwen3_5") == 256


def test_install_projection_aliases():
    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16)
            self.k_proj = nn.Linear(16, 16)
            self.v_proj = nn.Linear(16, 16)
            self.out_proj = nn.Linear(16, 16)

    mod = DummyAttention()
    install_projection_aliases(mod)
    # CANONICAL_PROJECTION_ALIASES maps query <-> q_proj, key <-> k_proj, etc.
    assert hasattr(mod, "query")
    assert mod.query is mod.q_proj
    assert hasattr(mod, "key")
    assert mod.key is mod.k_proj


def test_get_recovery_params():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 32)
            self.lora_a = nn.Parameter(torch.randn(32, 4))
            self.lora_b = nn.Parameter(torch.randn(4, 32))

    model = DummyModel()
    params = get_recovery_params(model)
    param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert "lora_a" in param_names
    assert "lora_b" in param_names
    assert "fc1.weight" not in param_names
