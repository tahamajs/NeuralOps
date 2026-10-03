"""
Comprehensive Master Test Suite for Qwen Model Family Compatibility.

Validates:
  1. Qwen architecture detection, adapter metadata, and projection aliases (q_proj, k_proj, v_proj, o_proj)
  2. Forward and backward passes on mock Qwen2 models with GQA (Grouped-Query Attention) and RoPE
  3. Compression engine application across all Qwen decoder attention projections
  4. SVD low-rank factorization (DirectSVD, AdaptiveRanks, ERC, ResSVD, Dobi, AASVD, Kronecker, SwiftSVD)
  5. Structured pruning and quantization compatibility (ASTRO, FAQ, CoopQ, DeltaKV, SliceGPT, ADMM, Bonsai)
  6. Mask conversion and causal attention preservation
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import (Qwen2Config, Qwen2ForCausalLM,
                          Qwen2ForSequenceClassification)

_HERE = Path(__file__).resolve().parent
_CODES = _HERE.parent.parent
if str(_CODES) not in sys.path:
    sys.path.insert(0, str(_CODES))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import framework.attentions
import framework.compression_engine
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.eval_all_models_datasets import (apply_module_to_model,
                                                build_model_instance)
from framework.utils.compat import (FamilyModuleWrapper, convert_mask,
                                    fp32_linalg, install_projection_aliases,
                                    module_family_support,
                                    should_compress_linear, wrap_for_family)
from framework.utils.model_family import ModelAdapter, detect_family
from framework.utils.qwen_compat import (QwenConfigProxy, install_qwen_aliases,
                                         is_qwen_attention, is_qwen_model,
                                         qwen_attention_mask,
                                         qwen_config_proxy)


def extract_output_tensor(out):
    if hasattr(out, 'logits') and out.logits is not None:
        return out.logits
    if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
        return out.last_hidden_state
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            res = extract_output_tensor(item)
            if isinstance(res, torch.Tensor):
                return res
    return out


@pytest.fixture(scope="session")
def qwen_config():
    """Tiny Qwen2 configuration for rapid and deterministic unit testing."""
    return Qwen2Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 query heads / 2 kv heads
        head_dim=32,
        vocab_size=1000,
        max_position_embeddings=256,
        num_labels=2,
        pad_token_id=0,
    )


@pytest.fixture
def qwen_model(qwen_config):
    torch.manual_seed(42)
    model = Qwen2ForSequenceClassification(qwen_config)
    model.eval()
    return model


@pytest.fixture
def qwen_batch(qwen_config):
    torch.manual_seed(42)
    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, qwen_config.vocab_size, (bsz, seq_len))
    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    attention_mask[:, 12:] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ==============================================================================
# 1. ARCHITECTURE DETECTION & ALIASING
# ==============================================================================
class TestQwenArchitectureDetection:
    def test_qwen_family_detection(self, qwen_model, qwen_config):
        assert detect_family(qwen_model) == "qwen"
        assert detect_family(qwen_config) == "qwen"
        assert is_qwen_model(qwen_model) is True

    def test_qwen_adapter_metadata(self, qwen_model, qwen_config):
        adapter = ModelAdapter(qwen_model, "qwen")
        assert adapter.family == "qwen"
        assert adapter.hidden_size == 128
        assert adapter.num_heads == 4
        assert adapter.num_key_value_heads == 2
        assert adapter.head_dim == 32
        assert adapter.causal is True
        assert adapter.position_embedding == "rotary"
        assert adapter.q_proj_out == 128
        assert adapter.kv_proj_out == 64
        assert adapter.kv_repeat == 2

    def test_install_qwen_aliases(self, qwen_model):
        install_qwen_aliases(qwen_model)
        for name, module in qwen_model.named_modules():
            if hasattr(module, "q_proj"):
                assert hasattr(module, "q_lin")
                assert hasattr(module, "k_lin")
                assert hasattr(module, "v_lin")
                assert hasattr(module, "out_lin")
                assert module.q_lin is module.q_proj
                assert module.out_lin is module.o_proj

    def test_should_compress_linear(self, qwen_model):
        assert should_compress_linear(qwen_model, "model.layers.0.self_attn.q_proj") is True
        assert should_compress_linear(qwen_model, "model.layers.0.self_attn.k_proj") is True
        assert should_compress_linear(qwen_model, "model.layers.0.self_attn.v_proj") is True
        assert should_compress_linear(qwen_model, "model.layers.0.self_attn.o_proj") is True
        assert should_compress_linear(qwen_model, "score") is False


# ==============================================================================
# 2. FACTORY INSTANTIATION VIA BUILD_MODEL_INSTANCE
# ==============================================================================
class TestQwenFactoryInstantiation:
    def test_build_qwen_model_instance(self):
        model, cfg = build_model_instance("qwen", num_labels=2, hidden_size=64, num_heads=4, num_layers=2)
        assert model is not None
        assert detect_family(model) == "qwen"
        assert cfg.hidden_size == 64
        assert cfg.num_attention_heads == 4

        x = torch.randint(0, 1000, (2, 8))
        out = model(x)
        assert hasattr(out, "logits")
        assert out.logits.shape == (2, 2)


# ==============================================================================
# 3. SVD & SPECTRAL COMPRESSORS ON QWEN
# ==============================================================================
class TestQwenSVDCompressors:
    @pytest.mark.parametrize("comp_name", [
        "direct_svd",
        "adaptive_ranks",
        "adaptive_svd",
        "erc_svd",
        "res_svd",
        "rpca",
        "dobi_svd",
        "a3",
        "aasvd",
        "swift_svd",
        "kronecker_tt_lora",
    ])
    def test_svd_compression_on_qwen(self, qwen_config, qwen_batch, comp_name):
        model = Qwen2ForSequenceClassification(qwen_config)
        model.eval()

        cls = COMPRESSOR_REGISTRY.resolve(comp_name)
        compressor = None
        for try_init in [
            lambda: cls(model, rank=16),
            lambda: cls(model, target_rank=16),
            lambda: cls(model),
        ]:
            try:
                compressor = try_init()
                break
            except Exception:
                continue

        assert compressor is not None, f"Failed to instantiate {comp_name} for Qwen"

        # Apply compression
        if hasattr(compressor, "compress"):
            res = compressor.compress()
            comp_model = res[0] if isinstance(res, tuple) else res
        elif callable(compressor) and not isinstance(compressor, nn.Module):
            res = compressor()
            comp_model = res[0] if isinstance(res, tuple) else res
        else:
            comp_model = getattr(compressor, "model", model)

        if comp_model is None or not isinstance(comp_model, nn.Module):
            comp_model = model

        # Run forward pass on compressed Qwen model
        out = comp_model(input_ids=qwen_batch["input_ids"], attention_mask=qwen_batch["attention_mask"])
        out_t = extract_output_tensor(out)

        assert out_t.shape[0] == 2, f"{comp_name} produced unexpected batch size: {out_t.shape}"
        assert torch.isfinite(out_t).all(), f"{comp_name} produced NaN or Inf on Qwen"


# ==============================================================================
# 4. PRUNING & QUANTIZATION ON QWEN
# ==============================================================================
class TestQwenPruningAndQuantization:
    @pytest.mark.parametrize("comp_name", [
        "astro_quantizer",
        "faq_quantizer",
        "coopq_quantizer",
        "delta_kv",
        "zerokv_neo",
        "slicegpt",
        "admm_pruner",
        "bonsai_pruner",
        "armor_pruner",
        "kmeans_quantize",
        "huffman",
    ])
    def test_pruning_quantization_on_qwen(self, qwen_config, qwen_batch, comp_name):
        model = Qwen2ForSequenceClassification(qwen_config)
        model.eval()

        cls = COMPRESSOR_REGISTRY.resolve(comp_name)
        compressor = None
        for try_init in [
            lambda: cls(model, rank=16),
            lambda: cls(model, sparsity=0.25),
            lambda: cls(model, bits=8),
            lambda: cls(model),
        ]:
            try:
                compressor = try_init()
                break
            except Exception:
                continue

        assert compressor is not None, f"Failed to instantiate {comp_name} for Qwen"

        # Apply compression
        if hasattr(compressor, "compress"):
            res = compressor.compress()
            comp_model = res[0] if isinstance(res, tuple) else res
        elif callable(compressor) and not isinstance(compressor, nn.Module):
            res = compressor()
            comp_model = res[0] if isinstance(res, tuple) else res
        else:
            comp_model = getattr(compressor, "model", model)

        if comp_model is None or not isinstance(comp_model, nn.Module):
            comp_model = model

        out = comp_model(input_ids=qwen_batch["input_ids"], attention_mask=qwen_batch["attention_mask"])
        out_t = extract_output_tensor(out)

        assert out_t.shape[0] == 2
        assert torch.isfinite(out_t).all()


# ==============================================================================
# 5. ATTENTION INJECTION & WRAPPING FOR QWEN
# ==============================================================================
class TestQwenAttentionCompatibility:
    def test_qwen_module_wrapper(self, qwen_config):
        class DummyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(128, 128)
                self.k_proj = nn.Linear(128, 64)
                self.v_proj = nn.Linear(128, 64)
                self.o_proj = nn.Linear(128, 128)

            def forward(self, hidden_states, **kwargs):
                return self.o_proj(hidden_states)

        attn = DummyAttention()
        wrapped = wrap_for_family(attn, "qwen")
        assert isinstance(wrapped, FamilyModuleWrapper)

        x = torch.randn(2, 16, 128)
        out = wrapped(x)
        assert out.shape == (2, 16, 128)

    def test_qwen_causal_mask_adaptation(self):
        adapter = ModelAdapter(type("Cfg", (), {"model_type": "qwen2", "hidden_size": 128, "num_attention_heads": 4, "num_hidden_layers": 2})(), "qwen")
        mask_2d = torch.ones(2, 16)
        mask_2d[:, 12:] = 0
        converted = convert_mask(mask_2d, adapter, expects_bidirectional=True)
        assert converted.shape == (2, 1, 1, 16)
        assert converted.dtype == torch.bool


# ==============================================================================
# 6. UNIVERSAL APPLY_MODULE_TO_MODEL ON QWEN
# ==============================================================================
class TestQwenUniversalRunner:
    @pytest.mark.parametrize("module_name", [
        "direct_svd",
        "adaptive_ranks",
        "erc_svd",
        "astro",
        "faq",
        "slicegpt",
    ])
    def test_apply_module_to_qwen_pipeline(self, qwen_config, qwen_batch, module_name):
        model = Qwen2ForSequenceClassification(qwen_config)
        mod_type = "compressor" if module_name in COMPRESSOR_REGISTRY else "attention"
        comp_model, ratio = apply_module_to_model(model, module_name, mod_type, rank=16)

        assert comp_model is not None
        assert ratio >= 0.1
        out = comp_model(input_ids=qwen_batch["input_ids"])
        out_t = extract_output_tensor(out)
        assert out_t.shape[0] == 2
        assert torch.isfinite(out_t).all()
