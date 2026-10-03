"""Integration Matrix: All Families x Representative Modules (including Qwen3.5)."""
import pytest
import torch
import torch.nn as nn
from framework.compression_engine.aa_svd_compressor import AASVDCompressor
from framework.core.attention_utils import inject_attention
from framework.core.registry import is_module_declared_compatible
from transformers import (GPT2Config, GPT2ForSequenceClassification,
                          RobertaConfig, RobertaForSequenceClassification,
                          T5Config, T5EncoderModel)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)

@pytest.mark.parametrize("family,model_cls,cfg_cls,cfg_kwargs", [
    ("roberta", RobertaForSequenceClassification, RobertaConfig, {"num_hidden_layers": 1, "num_attention_heads": 2, "hidden_size": 32, "intermediate_size": 64, "vocab_size": 100}),
    ("gpt2", GPT2ForSequenceClassification, GPT2Config, {"n_layer": 1, "n_head": 2, "n_embd": 32, "vocab_size": 100, "pad_token_id": 0}),
    ("t5", T5EncoderModel, T5Config, {"num_layers": 1, "num_decoder_layers": 1, "num_heads": 2, "d_model": 32, "d_kv": 16, "d_ff": 64, "vocab_size": 100}),
])
def test_matrix_adaptive_svd(family, model_cls, cfg_cls, cfg_kwargs):
    cfg = cfg_cls(**cfg_kwargs)
    m = model_cls(cfg)
    m = inject_attention(m, "adaptive_svd", rank=8, copy_weights=True)
    x = torch.randint(0, 100, (2, 8))
    out = m(x)
    res = out.logits if hasattr(out, "logits") else out.last_hidden_state
    assert torch.isfinite(res).all()

def test_qwen3_5_synthetic_hybrid_architecture():
    """Verify Qwen3.5 hybrid architecture (Gated DeltaNet + Gated Attention) is supported."""
    class GatedDeltaNetBlock(nn.Module):
        def __init__(self, hidden=32, num_v_heads=2, num_qk_heads=2, head_dim=16):
            super().__init__()
            self.qk_proj = nn.Linear(hidden, num_qk_heads * head_dim)
            self.v_proj = nn.Linear(hidden, num_v_heads * head_dim)
            self.out_proj = nn.Linear(num_v_heads * head_dim, hidden)
        def forward(self, x, **kwargs):
            return self.out_proj(self.v_proj(x)), None

    class GatedAttentionBlock(nn.Module):
        def __init__(self, hidden=32, num_q_heads=2, num_kv_heads=1, head_dim=16):
            super().__init__()
            self.q_proj = nn.Linear(hidden, num_q_heads * head_dim)
            self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim)
            self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim)
            self.out_proj = nn.Linear(num_q_heads * head_dim, hidden)
        def forward(self, x, **kwargs):
            return self.out_proj(self.q_proj(x)), None

    class MockQwen3_5Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.deltanet = GatedDeltaNetBlock()
            self.attn = GatedAttentionBlock()
            self.classifier = nn.Linear(32, 2)
        def forward(self, x):
            h, _ = self.deltanet(x)
            h, _ = self.attn(h)
            return self.classifier(h)

    m = MockQwen3_5Model()
    from framework.core.architecture import get_attention_modules
    modules = get_attention_modules(m)
    assert len(modules) == 2
    assert any("deltanet" in name for name, _ in modules)
    assert any("attn" in name for name, _ in modules)

    x = torch.randn(2, 4, 32)
    out = m(x)
    assert out.shape == (2, 4, 2)
    assert torch.isfinite(out).all()
