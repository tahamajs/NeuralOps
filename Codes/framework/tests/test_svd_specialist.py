import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

dummy_pkg = types.ModuleType("framework.attentions")
dummy_pkg.__path__ = [os.path.abspath(os.path.join(os.path.dirname(__file__), "../attentions"))]
sys.modules["framework.attentions"] = dummy_pkg

from framework.core.registry import ATTENTION_REGISTRY

target_files = [
    "fused_svd_kernel.py",
    "svd_linear_attention.py",
    "asvd_attention.py",
    "adaptive_svd_attention.py",
    "minimal_svd.py",
    "minimal_svd_v2.py",
    "minimal_aurora.py",
    "minimal_nova.py",
    "minimal_prism.py",
    "linformer_attention.py",
    "lola_attention.py",
    "svd_projection_attention.py",
    "SVD-Mamba.py",
    "svd_compressed.py",
    "svd_llm_v2_attention.py",
]

att_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../attentions"))

for f in target_files:
    path = os.path.join(att_dir, f)
    mod_name = f.replace(".py", "").replace("-", "_")
    full_name = f"framework.attentions.{mod_name}"
    loader = importlib.machinery.SourceFileLoader(full_name, path)
    spec = importlib.util.spec_from_loader(full_name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    setattr(dummy_pkg, mod_name, mod)
    loader.exec_module(mod)


@pytest.fixture
def dummy_config():
    return types.SimpleNamespace(
        hidden_size=768,
        num_attention_heads=12,
        dropout=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        rank=64,
        svd_rank=64,
        residual_rank=8,
    )


class DummyOldAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lin = nn.Linear(768, 768)
        self.k_lin = nn.Linear(768, 768)
        self.v_lin = nn.Linear(768, 768)
        self.out_lin = nn.Linear(768, 768)
        self.query = self.q_lin
        self.key = self.k_lin
        self.value = self.v_lin
        self.out_proj = self.out_lin


@pytest.mark.parametrize("key", [
    "svd_linear", "svd_linear_attn", "svd_linear_decay", "projection_factorized_attn",
    "asvd", "adaptive_svd", "adaptive_svd_attn",
    "minimal_svd", "minimal_svd_v2", "minimal_aurora", "minimal_nova", "minimal_prism",
    "linformer", "linformer_inject", "lola_attn",
    "svd_proj_attn", "svd_compressed_module", "svd_llm_v2", "svd_mamba"
])
def test_svd_module_forward_backward(key, dummy_config):
    fn = ATTENTION_REGISTRY.get(key)
    assert fn is not None, f"Key {key} not found in ATTENTION_REGISTRY"
    
    dummy_src = DummyOldAttn()
    mod = None
    for attempt in [
        lambda: fn(dummy_config),
        lambda: fn(dummy_config, pretrained_attn=dummy_src),
        lambda: fn(dummy_src, dummy_config, None, 0, 6, dummy_src),
        lambda: fn(dummy_config, ranks={"q": 64, "k": 64, "v": 64, "o": 64}, pretrained_attn=dummy_src),
        lambda: fn(config=dummy_config, d_model=768),
    ]:
        try:
            mod = attempt()
            if isinstance(mod, nn.Module):
                break
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
            pass

    assert mod is not None, f"Failed to instantiate {key}"

    if hasattr(mod, "copy_weights"):
        try:
            mod.copy_weights(dummy_src)
        except Exception as e:
            pytest.fail(f"copy_weights failed for {key}: {e}")

    x = torch.randn(2, 128, 768, requires_grad=True)
    mask = torch.ones(2, 128)
    mask[:, 100:] = 0

    out = mod(x, attention_mask=mask)
    out_tensor = out[0] if isinstance(out, tuple) else out

    assert out_tensor.shape == (2, 128, 768), f"Unexpected output shape: {out_tensor.shape}"
    assert not torch.isnan(out_tensor).any(), f"NaN detected in forward output of {key}"
    assert not torch.isinf(out_tensor).any(), f"Inf detected in forward output of {key}"

    loss = out_tensor.sum()
    loss.backward()

    assert x.grad is not None, f"x.grad is None after backward on {key}"
    assert not torch.isnan(x.grad).any(), f"NaN detected in input gradient of {key}"
