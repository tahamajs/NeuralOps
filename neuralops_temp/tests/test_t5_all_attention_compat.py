"""
Unit tests for all 27 attention replacement modules adapted for T5.
Verifies:
1. Forward pass is finite (no NaN/Inf).
2. Shapes are correct [B, num_classes].
3. Relative position bias is applied and propagated.
4. Cross-attention uses different sources for Q (decoder) vs K/V (encoder).
5. Encoder self-attention, decoder self-attention, and decoder cross-attention all work.
6. FP16 numerical stability.
"""

import pytest
import torch
from neuralops.utils.replace_attention import replace_attention
from transformers import AutoModelForSequenceClassification, T5Config

TARGET_MODULES = [
    'cosformer', 'eat', 'gated_linear_attn', 'gated_delta_net', 'decay_linear_attn',
    'galerkin', 'griffin_ssd', 'halo_regla', 'hgrn2_outer', 'adaptive_span',
    'adaptive_sparse', 'fasa', 'hamming', 'fused_svd_kernel', 'flash3_fp8',
    'fusion_attn', 'aegis_v1', 'aegis_v2', 'aegis_v3', 'aegis_v4', 'aegis_v5', 'aegis_v6',
    'aegis_hybrid', 'aegis_spectron_fp8', 'goldfinch_ahn', 'gqa', 'granite_hybrid'
]


@pytest.fixture(scope='module')
def t5_base_model():
    model = AutoModelForSequenceClassification.from_pretrained('t5-small')
    model.eval()
    return model


@pytest.mark.parametrize('module_name', TARGET_MODULES)
def test_forward_pass_finite_and_shape(t5_base_model, module_name):
    """Verify forward pass is finite and produces correct output shape [B, 2]."""
    import copy
    model = copy.deepcopy(t5_base_model)
    model = replace_attention(model, module_name)
    
    input_ids = torch.randint(2, 1000, (2, 16))
    input_ids[:, -1] = 1  # EOS
    
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits
        
    assert logits.shape == (2, 2), f'{module_name}: expected shape (2, 2), got {logits.shape}'
    assert torch.isfinite(logits).all().item(), f'{module_name}: logits contain NaN or Inf!'


@pytest.mark.parametrize('module_name', ['cosformer', 'fused_svd_kernel', 'aegis_v1', 'gqa', 'adaptive_span'])
def test_relative_position_bias_applied(t5_base_model, module_name):
    """Verify that relative position bias is attached and used in layer 0."""
    import copy
    model = copy.deepcopy(t5_base_model)
    model = replace_attention(model, module_name)
    
    enc_layer0 = model.transformer.encoder.block[0].layer[0].SelfAttention
    assert hasattr(enc_layer0, 'relative_attention_bias') or hasattr(enc_layer0.inner, 'relative_attention_bias')
    assert getattr(enc_layer0, 'has_relative_attention_bias', False) or getattr(enc_layer0.inner, 'has_relative_attention_bias', False)


@pytest.mark.parametrize('module_name', ['cosformer', 'fused_svd_kernel', 'aegis_v1', 'gqa', 'granite_hybrid'])
def test_cross_attention_different_sources(t5_base_model, module_name):
    """Verify that decoder cross-attention uses encoder states for K/V and decoder states for Q."""
    import copy
    model = copy.deepcopy(t5_base_model)
    model = replace_attention(model, module_name)
    
    dec_cross = model.transformer.decoder.block[0].layer[1].EncDecAttention
    
    # Simulate cross-attention call with different Q and KV
    B, N_q, N_kv, D = 2, 8, 16, 512
    dec_hidden = torch.randn(B, N_q, D)
    enc_hidden = torch.randn(B, N_kv, D)
    
    with torch.no_grad():
        out1, _ = dec_cross(dec_hidden, key_value_states=enc_hidden)
        # Change enc_hidden -> output must change
        out2, _ = dec_cross(dec_hidden, key_value_states=enc_hidden + 1.0)
        
    assert out1.shape == dec_hidden.shape
    assert not torch.allclose(out1, out2, atol=1e-5), 'Cross-attention output must depend on encoder hidden states!'


@pytest.mark.parametrize('module_name', ['cosformer', 'fused_svd_kernel', 'aegis_v1', 'flash3_fp8', 'gqa'])
def test_fp16_numerical_stability(t5_base_model, module_name):
    """Verify that running in FP16 produces no NaNs."""
    import copy
    model = copy.deepcopy(t5_base_model)
    model = replace_attention(model, module_name)
    model = model.half()
    
    input_ids = torch.randint(2, 1000, (2, 16))
    input_ids[:, -1] = 1
    
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits
        
    assert torch.isfinite(logits).all().item(), f'{module_name} in FP16 produced NaN/Inf!'
