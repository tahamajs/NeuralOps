"""End-to-end integration and regression tests."""
import pytest
import torch
import torch.nn as nn
from neuralops.core.attention_utils import inject_attention
from neuralops.inference.engine import NeuralOpsInferenceEngine
from neuralops.compression_engine.quantization.awq import quantize_model_awq


def test_full_pipeline_injection_and_inference():
    class MiniTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, 16)
            self.self_attn = nn.Linear(16, 16)
            self.head = nn.Linear(16, 32)

        def forward(self, input_ids):
            h = self.emb(input_ids)
            h = h + self.self_attn(h)
            return self.head(h)

    model = MiniTransformer()
    # Inject attention
    injected = inject_attention(model, attention_type="full_attention")
    assert injected is not None

    # Run inference engine
    engine = NeuralOpsInferenceEngine(injected)
    ids = torch.tensor([[1, 2, 3]])
    out = engine.generate(ids, max_new_tokens=4)
    assert out.shape == (1, 7)


def test_regression_quantization_preserves_weights_dtype():
    model = nn.Sequential(nn.Linear(16, 16))
    calib = [torch.randn(4, 16)]
    quantize_model_awq(model, calib, bits=4)
    assert model[0].weight.dtype == torch.float32
    assert not torch.isnan(model[0].weight).any()
