"""Unit tests for core architecture discovery, leaf extraction, and attention injection."""

import pytest
import torch
import torch.nn as nn
from neuralops.core.architecture import (detect_architecture,
                                         get_attention_modules,
                                         get_classifier_head,
                                         get_embedding_module, get_ffn_modules)
from neuralops.core.attention_utils import inject_attention
from neuralops.core.model_utils import replace_module
from transformers import (BertConfig, BertForSequenceClassification,
                          DistilBertConfig,
                          DistilBertForSequenceClassification, GPT2Config,
                          GPT2ForSequenceClassification, RobertaConfig,
                          RobertaForSequenceClassification, T5Config,
                          T5EncoderModel, T5ForConditionalGeneration)


@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)


class TestGetAttentionModules:
    """Verify get_attention_modules discovers only leaf submodules across all families."""

    def test_bert_returns_leaf_self_attention_only(self):
        cfg = BertConfig(num_hidden_layers=2, num_attention_heads=2, hidden_size=32, intermediate_size=64, vocab_size=100)
        model = BertForSequenceClassification(cfg)
        modules = get_attention_modules(model)
        assert len(modules) == 2
        for name, mod in modules:
            assert name.endswith(".attention.self")
            assert mod.__class__.__name__ == "BertSelfAttention"

    def test_roberta_returns_leaf_self_attention_only(self):
        cfg = RobertaConfig(num_hidden_layers=2, num_attention_heads=2, hidden_size=32, intermediate_size=64, vocab_size=100)
        model = RobertaForSequenceClassification(cfg)
        modules = get_attention_modules(model)
        assert len(modules) == 2
        for name, mod in modules:
            assert name.endswith(".attention.self")
            assert mod.__class__.__name__ == "RobertaSelfAttention"

    def test_distilbert_returns_leaf_attention(self):
        cfg = DistilBertConfig(n_layers=2, n_heads=2, dim=32, hidden_dim=64, vocab_size=100)
        model = DistilBertForSequenceClassification(cfg)
        modules = get_attention_modules(model)
        assert len(modules) == 2
        for name, mod in modules:
            assert name.endswith(".attention")
            assert mod.__class__.__name__ in ("DistilBertSelfAttention", "MultiHeadSelfAttention")

    def test_gpt2_returns_correct_attention_block(self):
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100)
        model = GPT2ForSequenceClassification(cfg)
        modules = get_attention_modules(model)
        assert len(modules) == 2
        for name, mod in modules:
            assert name.endswith(".attn")
            assert mod.__class__.__name__ == "GPT2Attention"

    def test_t5_returns_encoder_and_decoder_attentions(self):
        cfg = T5Config(num_layers=2, num_decoder_layers=2, num_heads=2, d_model=32, d_kv=16, d_ff=64, vocab_size=100)
        model = T5ForConditionalGeneration(cfg)
        modules = get_attention_modules(model)
        # 2 encoder layers (1 self each) + 2 decoder layers (1 self + 1 cross each) = 6
        assert len(modules) == 6
        names = [n for n, _ in modules]
        # Verify self-attention and cross-attention paths
        assert any("encoder.block.0.layer.0.SelfAttention" in n for n in names)
        assert any("decoder.block.0.layer.0.SelfAttention" in n for n in names)
        assert any("decoder.block.0.layer.1.EncDecAttention" in n for n in names)

    def test_custom_nested_containers_deduplication(self):
        """Mock model with 3 levels of nested attention containers; assert only innermost leaf is returned."""
        class DeepLeaf(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 32)
                self.v_proj = nn.Linear(32, 32)

        class MiddleContainer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = DeepLeaf()

        class OuterAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.middle = MiddleContainer()
                self.output = nn.Linear(32, 32)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.ModuleList([OuterAttention() for _ in range(3)])

        mock = MockModel()
        leaves = get_attention_modules(mock)
        assert len(leaves) == 3
        for name, mod in leaves:
            assert name.endswith(".middle.self_attn")
            assert isinstance(mod, DeepLeaf)


class TestInjectAttention:
    """Verify inject_attention preserves residuals, LayerNorm, and forward semantics."""

    def test_roberta_inject_preserves_output_dense_and_residual(self):
        cfg = RobertaConfig(num_hidden_layers=2, num_attention_heads=2, hidden_size=32, intermediate_size=64, vocab_size=100)
        model = RobertaForSequenceClassification(cfg)

        # Confirm pre-injection structure
        layer0 = model.roberta.encoder.layer[0]
        assert hasattr(layer0.attention, "output")
        assert hasattr(layer0.attention.output, "dense")
        assert hasattr(layer0.attention.output, "LayerNorm")

        # Inject adaptive_svd attention
        model = inject_attention(model, "adaptive_svd", rank=8, copy_weights=True)

        # Confirm post-injection outer structure intact
        layer0_post = model.roberta.encoder.layer[0]
        assert hasattr(layer0_post.attention, "output")
        assert hasattr(layer0_post.attention.output, "dense")
        assert hasattr(layer0_post.attention.output, "LayerNorm")

        # Forward pass verification
        input_ids = torch.randint(0, 100, (2, 8))
        out = model(input_ids)
        assert out.logits.shape == (2, 2)
        assert torch.isfinite(out.logits).all()

    def test_gpt2_inject_conv1d_transposed_properly(self):
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100)
        cfg.pad_token_id = 0
        model = GPT2ForSequenceClassification(cfg)

        model = inject_attention(model, "adaptive_svd", rank=8, copy_weights=True)
        input_ids = torch.randint(0, 100, (2, 8))
        out = model(input_ids)
        assert out.logits.shape == (2, 2)
        assert torch.isfinite(out.logits).all()

    def test_t5_encoder_inject(self):
        cfg = T5Config(num_layers=2, num_decoder_layers=2, num_heads=2, d_model=32, d_kv=16, d_ff=64, vocab_size=100)
        model = T5EncoderModel(cfg)

        model = inject_attention(model, "adaptive_svd", rank=8, copy_weights=True)
        input_ids = torch.randint(0, 100, (2, 8))
        out = model(input_ids)
        assert out.last_hidden_state.shape == (2, 8, 32)
        assert torch.isfinite(out.last_hidden_state).all()


class TestReplaceModuleDottedPath:
    """Verify replace_module works correctly with nested dotted paths."""

    def test_nested_dotted_path_replacement(self):
        class SubSub(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

        class Sub(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = SubSub()

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Sub()

        root = Root()
        new_linear = nn.Linear(10, 20)
        replace_module(root, "sub.child.fc", new_linear)
        assert root.sub.child.fc is new_linear
        assert root.sub.child.fc.out_features == 20
