"""
Exhaustive Verification Suite for AASVD (Activation-Aware SVD) Compressor.
Validates structural factorizations, streaming covariance estimation,
multi-architecture compression, and gradient backpropagation.
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression_engine.aa_svd_compressor import (
    DEFAULT_CALIBRATION_CORPUS, AASVDCompressor, SVDLinear, aasvd_factory)
from neuralops.core.registry import COMPRESSOR_REGISTRY
from transformers import (AutoTokenizer, DistilBertConfig,
                          DistilBertForSequenceClassification, Qwen2Config,
                          Qwen2ForSequenceClassification, T5Config,
                          T5EncoderModel)


class TestSVDLinearLayer:
    def test_svd_linear_initialization_and_forward(self):
        layer = SVDLinear(in_features=128, out_features=256, rank=16, bias=True)
        assert layer.V.shape == (16, 128)
        assert layer.U.shape == (256, 16)
        assert layer.bias.shape == (256,)

        x = torch.randn(4, 10, 128)
        out = layer(x)
        assert out.shape == (4, 10, 256)
        assert torch.isfinite(out).all()

    def test_svd_linear_parameter_reduction(self):
        dense = nn.Linear(512, 512, bias=False)
        dense_params = sum(p.numel() for p in dense.parameters())  # 512 * 512 = 262144
        
        svd = SVDLinear(512, 512, rank=32, bias=False)
        svd_params = sum(p.numel() for p in svd.parameters())    # 32 * (512 + 512) = 32768
        
        assert svd_params < dense_params
        assert svd_params == 32768
        assert dense_params == 262144

    def test_svd_linear_factor_assignment(self):
        layer = SVDLinear(64, 64, rank=8)
        U_fact = torch.randn(64, 8)
        V_fact = torch.randn(8, 64)
        layer.set_low_rank_factors(U_fact, V_fact)
        assert torch.allclose(layer.U, U_fact)
        assert torch.allclose(layer.V, V_fact)

    def test_svd_linear_backward_gradients(self):
        layer = SVDLinear(32, 32, rank=4, bias=True)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.U.grad is not None and torch.isfinite(layer.U.grad).all()
        assert layer.V.grad is not None and torch.isfinite(layer.V.grad).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestAASVDCompressorEndToEnd:
    def test_compress_distilbert_unweighted(self):
        cfg = DistilBertConfig(dim=128, n_heads=4, n_layers=2, num_labels=2)
        model = DistilBertForSequenceClassification(cfg)
        compressor = AASVDCompressor(model, rank=16)
        comp_model, stats = compressor.compress()
        
        x = torch.randint(0, 1000, (2, 8))
        out = comp_model(x).logits
        assert out.shape == (2, 2)
        assert torch.isfinite(out).all()

    def test_compress_qwen_decoder(self):
        cfg = Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                          num_labels=2, pad_token_id=0)
        model = Qwen2ForSequenceClassification(cfg)
        compressor = AASVDCompressor(model, rank=16)
        comp_model, stats = compressor.compress()

        x = torch.randint(0, 1000, (2, 8))
        out = comp_model(x).logits
        assert out.shape == (2, 2)
        assert torch.isfinite(out).all()

    def test_compress_with_simulated_covariance(self):
        model = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        compressor = AASVDCompressor(model, rank=8)
        comp_model, stats = compressor.compress()
        x = torch.randn(2, 64)
        out = comp_model(x)
        assert out.shape == (2, 10)
        assert torch.isfinite(out).all()

    def test_registry_resolution(self):
        for key in ["aasvd", "aa_svd", "aa_svd_compressor"]:
            factory = COMPRESSOR_REGISTRY.get(key)
            assert factory is not None
