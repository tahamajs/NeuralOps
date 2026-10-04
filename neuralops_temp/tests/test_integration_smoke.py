"""Integration smoke tests verifying end-to-end injection and forward passes per model family."""

import pytest
import torch
import torch.nn as nn
from neuralops.compression_engine.aa_svd_compressor import AASVDCompressor
from neuralops.core.attention_utils import inject_attention
from neuralops.core.registry import is_module_declared_compatible
from transformers import (BertConfig, BertForSequenceClassification,
                          DistilBertConfig,
                          DistilBertForSequenceClassification, GPT2Config,
                          GPT2ForSequenceClassification, RobertaConfig,
                          RobertaForSequenceClassification, T5Config,
                          T5EncoderModel)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


class TestIntegrationSmokePerFamily:
    """Verify end-to-end forward and backward passes across model families."""

    def test_roberta_adaptive_svd_smoke(self):
        cfg = RobertaConfig(num_hidden_layers=1, num_attention_heads=2, hidden_size=32, intermediate_size=64, vocab_size=100)
        model = RobertaForSequenceClassification(cfg)

        model = inject_attention(model, "adaptive_svd", rank=8, copy_weights=True)
        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        assert out.logits.shape == (2, 2)
        assert torch.isfinite(out.logits).all()

        loss = out.logits.sum()
        loss.backward()
        # Verify gradients reached parameters
        trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(trainable) > 0

    def test_bert_rpc_attention_smoke(self):
        cfg = BertConfig(num_hidden_layers=1, num_attention_heads=2, hidden_size=32, intermediate_size=64, vocab_size=100)
        model = BertForSequenceClassification(cfg)

        model = inject_attention(model, "rpc", copy_weights=True)
        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        assert out.logits.shape == (2, 2)
        assert torch.isfinite(out.logits).all()

    def test_distilbert_adaptive_svd_smoke(self):
        cfg = DistilBertConfig(n_layers=1, n_heads=2, dim=32, hidden_dim=64, vocab_size=100)
        model = DistilBertForSequenceClassification(cfg)

        model = inject_attention(model, "adaptive_svd", rank=8, copy_weights=True)
        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        assert out.logits.shape == (2, 2)
        assert torch.isfinite(out.logits).all()

    def test_gpt2_adaptive_svd_smoke(self):
        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=32, vocab_size=100)
        cfg.pad_token_id = 0
        model = GPT2ForSequenceClassification(cfg)

        model = inject_attention(model, "adaptive_svd", rank=8, copy_weights=True)
        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        assert out.logits.shape == (2, 2)
        assert torch.isfinite(out.logits).all()

    def test_t5_encoder_adaptive_svd_smoke(self):
        cfg = T5Config(num_layers=1, num_decoder_layers=1, num_heads=2, d_model=32, d_kv=16, d_ff=64, vocab_size=100)
        model = T5EncoderModel(cfg)

        model = inject_attention(model, "adaptive_svd", rank=8, copy_weights=True)
        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        assert out.last_hidden_state.shape == (2, 8, 32)
        assert torch.isfinite(out.last_hidden_state).all()

    def test_roberta_compressor_smoke(self):
        cfg = RobertaConfig(num_hidden_layers=1, num_attention_heads=2, hidden_size=32, intermediate_size=64, vocab_size=100)
        model = RobertaForSequenceClassification(cfg)

        comp = AASVDCompressor(model, rank=8)
        model, stats = comp.compress()
        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        assert out.logits.shape == (2, 2)
        assert torch.isfinite(out.logits).all()

    def test_incompatible_causal_skip_smoke(self):
        """Incompatible causal module is cleanly identified for RoBERTa without crashing model."""
        is_compat, reason = is_module_declared_compatible("adaptive_span", "roberta")
        assert not is_compat
        assert "causal_only" in reason
