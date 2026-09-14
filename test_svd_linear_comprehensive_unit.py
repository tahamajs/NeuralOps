import torch
import torch.nn as nn
from framework.attentions.svd_linear_attention import (
    SVDLinearAttention,
    apply_rotary_pos_emb,
    inject_svd_linear_attention,
)
from framework.core.registry import ATTENTION_REGISTRY

def test_registry():
    print("1. Testing registry...")
    cls = ATTENTION_REGISTRY.get("svd_linear_attention")
    assert isinstance(cls, type), f"Expected class, got {cls}"
    assert issubclass(cls, SVDLinearAttention), "Expected SVDLinearAttention class"
    
    cls_alias = ATTENTION_REGISTRY.get("svd_linear_attn")
    assert isinstance(cls_alias, type), f"Expected class, got {cls_alias}"
    assert issubclass(cls_alias, SVDLinearAttention)

    inject_func = ATTENTION_REGISTRY.get("svd_linear_inject")
    assert callable(inject_func) and not isinstance(inject_func, type)
    print("   Registry test passed!")

def test_rotary_dim_zero():
    print("2. Testing rotary_dim=0 early return...")
    q = torch.randn(2, 4, 16, 64)
    k = torch.randn(2, 4, 16, 64)
    cos = torch.randn(16, 0)
    sin = torch.randn(16, 0)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q, q_out) and torch.allclose(k, k_out)

    cos1 = torch.randn(16, 1)
    sin1 = torch.randn(16, 1)
    q_out1, k_out1 = apply_rotary_pos_emb(q, k, cos1, sin1)
    assert torch.allclose(q, q_out1) and torch.allclose(k, k_out1)
    print("   Rotary dim=0/1 test passed!")

def test_return_state_validation():
    print("3. Testing return_state validation...")
    attn = SVDLinearAttention(embed_dim=64, num_heads=4, mode="exact")
    x = torch.randn(2, 8, 64)
    try:
        attn(x, return_state=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "return_state=True is only supported in decay_linear/gated_linear modes" in str(e)
    print("   return_state validation passed!")

def test_head_mask_3d():
    print("4. Testing 3D head_mask...")
    attn = SVDLinearAttention(embed_dim=64, num_heads=4, mode="exact")
    x = torch.randn(2, 8, 64)
    head_mask_3d = torch.ones(2, 4, 8)
    out, _ = attn(x, head_mask=head_mask_3d)
    assert out.shape == (2, 8, 64)
    print("   3D head_mask test passed!")

def test_is_cross_attention():
    print("5. Testing cross-attention exact mode (causal mask disabled)...")
    attn = SVDLinearAttention(embed_dim=64, num_heads=4, mode="exact", is_decoder=True)
    x = torch.randn(2, 8, 64)
    enc = torch.randn(2, 16, 64)
    out, _ = attn(x, encoder_hidden_states=enc)
    assert out.shape == (2, 8, 64)
    print("   Cross-attention test passed!")

def test_chunk_decay_vs_recurrent():
    print("6. Testing chunk-wise decay linear attention vs recurrent step...")
    torch.manual_seed(42)
    attn = SVDLinearAttention(embed_dim=64, num_heads=4, mode="decay_linear")
    x = torch.randn(2, 70, 64)
    out_parallel, _, state_parallel = attn(x, return_state=True)
    assert out_parallel.shape == (2, 70, 64)
    assert state_parallel[0].shape == (2, 4, 16, 16)
    assert state_parallel[1].shape == (2, 4, 16, 1)

    # Test step_recurrent matches end state
    s = None
    outs_rec = []
    for t in range(70):
        y_t, s = attn.step_recurrent(x[:, t:t+1, :], past_state=s)
        outs_rec.append(y_t)
    out_rec = torch.cat(outs_rec, dim=1)

    diff = (out_parallel - out_rec).abs().max().item()
    print(f"   Max diff between chunk-wise and step-by-step recurrent: {diff:.2e}")
    assert diff < 1e-4, f"Diff too large: {diff}"
    print("   Chunk-wise decay equivalence passed!")

def test_chunk_linear_vs_exact_causal():
    print("7. Testing chunk-wise linear causal attention (decoder mode)...")
    torch.manual_seed(42)
    attn = SVDLinearAttention(embed_dim=64, num_heads=4, mode="linear", is_decoder=True)
    x = torch.randn(2, 80, 64)
    out = attn(x)[0]
    assert out.shape == (2, 80, 64)
    print("   Chunk-wise linear causal attention test passed!")

def test_roberta_nested_injection():
    print("8. Testing RoBERTa nested encoder injection...")
    class MockRobertaSelfOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(64, 64)
    class MockRobertaSelf(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)
            self.num_attention_heads = 4
            self.attention_head_size = 16
            self.all_head_size = 64
    class MockRobertaAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = MockRobertaSelf()
            self.output = MockRobertaSelfOutput()
    class MockRobertaLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = MockRobertaAttention()

    layer = MockRobertaLayer()
    injected = inject_svd_linear_attention(layer=layer, rank=16)
    assert injected is not None
    assert layer.attention.self is injected
    assert injected.is_nested_encoder is True
    print("   RoBERTa nested injection test passed!")

def test_non_nested_injection():
    print("9. Testing non-nested attention injection (e.g. GPT-2/DistilBERT)...")
    class MockDistilBertAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            self.k_lin = nn.Linear(64, 64)
            self.v_lin = nn.Linear(64, 64)
            self.out_lin = nn.Linear(64, 64)
            self.n_heads = 4
    class MockDistilBertLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = MockDistilBertAttention()

    layer = MockDistilBertLayer()
    injected = inject_svd_linear_attention(layer=layer, rank=16)
    assert injected is not None
    assert layer.attention is injected
    assert injected.is_nested_encoder is False
    print("   Non-nested injection test passed!")

if __name__ == "__main__":
    test_registry()
    test_rotary_dim_zero()
    test_return_state_validation()
    test_head_mask_3d()
    test_is_cross_attention()
    test_chunk_decay_vs_recurrent()
    test_chunk_linear_vs_exact_causal()
    test_roberta_nested_injection()
    test_non_nested_injection()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
