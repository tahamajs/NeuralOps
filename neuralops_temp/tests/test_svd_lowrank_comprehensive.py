import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODES = _HERE.parent.parent
sys.path.insert(0, str(_CODES))
sys.path.insert(0, str(_HERE.parent))

import torch
import torch.nn as nn
from neuralops.attentions.adaptive_svd_attention import AdaptiveSVDAttention
from neuralops.compression_engine.cur_compressor import cur_compressor_factory
from neuralops.compression_engine.kronecker_tt_lora import \
    kronecker_tt_lora_factory


class DummyConfig:
    num_attention_heads = 12
    hidden_size = 768
    num_heads = 12
    d_model = 768
    svd_rank = 32
    residual_rank = 8
    dropout = 0.1
    energy_threshold = 0.92

def test_adaptive_svd():
    config = DummyConfig()
    model = AdaptiveSVDAttention(config)
    x = torch.randn(2, 128, 768, requires_grad=True)
    out, _ = model(x)
    assert out.shape == (2, 128, 768)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("AdaptiveSVDAttention passed!")

def test_kronecker_tt_lora():
    model = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
    config = DummyConfig()
    compressed = kronecker_tt_lora_factory(model, comp_config=config)
    x = torch.randn(2, 128, 768, requires_grad=True)
    out = compressed(x)
    assert out.shape == (2, 128, 768)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("KroneckerTTLoRA passed!")

def test_cur():
    model = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
    config = DummyConfig()
    compressed = cur_compressor_factory(model, comp_config=config, rank=64)
    x = torch.randn(2, 128, 768, requires_grad=True)
    out = compressed(x)
    assert out.shape == (2, 128, 768)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("CUR passed!")

if __name__ == "__main__":
    test_adaptive_svd()
    test_kronecker_tt_lora()
    test_cur()
