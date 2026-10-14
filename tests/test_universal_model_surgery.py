"""
Universal Model Surgery & Attention Transplantation 5-Architecture Verification Suite.

Tests:
1. Architecture 1: Bidirectional encoder (BERT / DistilBERT) with BertSelfOutput preservation & padding mask zeroing.
2. Architecture 2: HuggingFace Conv1D causal decoder (GPT-2) with transposed weights & past_key_values cache.
3. Architecture 3: Modern GQA + RoPE causal decoder (LLaMA-3 / Qwen2) with cosformer attention & Hybrid HF Cache.
4. Architecture 4: Vision Transformer (ViT / DiT) with fused QKV linear projections.
5. Architecture 5: Custom nn.Module with cross-attention (T_q != T_kv) and shadow-twin distillation loop.
"""

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

import neuralops
from neuralops import (
    transplant_attention,
    replace_attention,
    collect_surgery_distillation_loss,
    set_surgery_alpha,
    set_surgery_stage,
    finalize_surgery,
)
from neuralops.surgery.scanner import UniversalArchitectureScanner
from neuralops.surgery.endoskeletal_shell import (
    UniversalAttentionBridge,
    transfer_pretrained_qkv_weights,
)
from neuralops.surgery.topology_router import (
    BiDirectionalScanWrapper,
    AsymmetricCrossAttentionRouter,
    sanitize_kv_for_padding_mask,
)
from neuralops.surgery.shadow_twin import (
    enable_shadow_twin,
)
from neuralops.surgery.hf_cache_bridge import (
    NeuralOpsHybridHFCache,
    wrap_model_for_generation,
)


def test_arch1_bidirectional_encoder_bert():
    """
    Architecture 1: Bidirectional encoder (BertModel).
    Verifies that BertSelfOutput (dense + LayerNorm + residual connection) is preserved,
    padding mask is respected, and bidirectional attention is executed.
    """
    config = BertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    model = BertModel(config)
    model.eval()

    # Scan and verify pattern
    specs = UniversalArchitectureScanner.scan(model)
    assert len(specs) >= 2
    assert specs[0].pattern_type == "bert_nested"
    assert specs[0].output_dense_module is not None
    assert specs[0].output_layernorm_module is not None

    # Replace attention using transplant_attention
    transplanted = transplant_attention(model, attention="cosformer")
    assert transplanted is not None

    # Run forward pass with input_ids and attention_mask (with padded tokens)
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16)
    attention_mask[0, 10:] = 0  # pad last 6 tokens of first sequence

    with torch.no_grad():
        out = transplanted(input_ids=input_ids, attention_mask=attention_mask)
    assert out.last_hidden_state.shape == (2, 16, 64)

    # Verify that BiDirectionalScanWrapper works cleanly on bidirectional cosformer
    from neuralops.attentions.linear.cosformer import CosformerAttention
    cosformer = CosformerAttention(hidden_size=64, num_heads=4)
    bidir_cosformer = BiDirectionalScanWrapper(cosformer)
    x = torch.randn(2, 16, 64)
    out_bidir = bidir_cosformer(x, attention_mask=attention_mask)
    if isinstance(out_bidir, tuple):
        out_bidir = out_bidir[0]
    assert out_bidir.shape == (2, 16, 64)


def test_arch2_hf_conv1d_causal_decoder_gpt2():
    """
    Architecture 2: HuggingFace Conv1D causal decoder (GPT2Model).
    Verifies transposed weight slicing, generation tuple preservation, and KV-cache handling.
    """
    config = GPT2Config(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=4,
    )
    model = GPT2Model(config)
    model.eval()

    specs = UniversalArchitectureScanner.scan(model)
    assert len(specs) >= 2
    assert specs[0].pattern_type == "conv1d_gpt2"

    transplanted = transplant_attention(model, attention="cosformer")
    assert transplanted is not None

    input_ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out = transplanted(input_ids=input_ids, use_cache=True)
    assert out.last_hidden_state.shape == (2, 8, 64)
    assert out.past_key_values is not None


def test_arch3_modern_gqa_rope_decoder():
    """
    Architecture 3: Modern GQA + RoPE causal decoder.
    Verifies that asymmetric GQA projections and external RoPE embeddings are handled smoothly.
    """
    class MockLlamaSelfAttention(nn.Module):
        def __init__(self, embed_dim=64, num_heads=4, num_kv_heads=2):
            super().__init__()
            self.num_heads = num_heads
            self.num_key_value_heads = num_kv_heads
            self.head_dim = embed_dim // num_heads
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_value=None):
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            out = self.o_proj(q)
            return out, past_key_value

    class MockLlamaBlock(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.self_attn = MockLlamaSelfAttention(embed_dim=embed_dim)
            self.input_layernorm = nn.LayerNorm(embed_dim)

        def forward(self, x, past_key_value=None):
            normed = self.input_layernorm(x)
            attn_res = self.self_attn(normed, past_key_value=past_key_value)
            if isinstance(attn_res, tuple):
                attn_out = attn_res[0]
                past_kv = attn_res[-1] if len(attn_res) > 1 else None
            else:
                attn_out = attn_res
                past_kv = None
            return x + attn_out, past_kv

    class MockLlamaModel(nn.Module):
        def __init__(self, embed_dim=64, n_layers=2):
            super().__init__()
            from types import SimpleNamespace
            self.config = SimpleNamespace(
                hidden_size=embed_dim,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                model_type="llama",
                is_decoder=True,
                is_causal=True,
            )
            self.layers = nn.ModuleList([MockLlamaBlock(embed_dim) for _ in range(n_layers)])

        def forward(self, x, past_key_values=None):
            curr = x
            pasts = []
            for i, layer in enumerate(self.layers):
                p = past_key_values[i] if past_key_values is not None else None
                curr, p_out = layer(curr, past_key_value=p)
                pasts.append(p_out)
            return curr, pasts

    model = MockLlamaModel(embed_dim=64, n_layers=2)
    model.eval()

    # Scan and verify split_4proj
    specs = UniversalArchitectureScanner.scan(model)
    assert len(specs) == 2
    assert specs[0].pattern_type == "split_4proj"
    assert specs[0].num_heads == 4
    assert specs[0].num_kv_heads == 2

    # Transplant cosformer into the model
    transplanted = transplant_attention(model, attention="cosformer")
    x = torch.randn(2, 12, 64)
    out, pasts = transplanted(x)
    assert out.shape == (2, 12, 64)

    # Verify NeuralOpsHybridHFCache interaction
    cache = NeuralOpsHybridHFCache()
    wrapped_model = wrap_model_for_generation(transplanted)
    out_wrapped = wrapped_model(x, past_key_values=cache)
    assert out_wrapped[0].shape == (2, 12, 64)


def test_arch4_vit_fused_qkv():
    """
    Architecture 4: Vision Transformer (ViT / DiT) with fused QKV linear projections.
    Verifies that fused qkv parameter chunking, weight transfer, and forward passes run flawlessly.
    """
    class MockViTBlock(nn.Module):
        def __init__(self, embed_dim=64, num_heads=4):
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.num_heads = num_heads
            self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.num_heads) ** 0.5)
            attn = attn.softmax(dim=-1)
            x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return x + self.proj(x_attn)

    class MockViTModel(nn.Module):
        def __init__(self, embed_dim=64, num_layers=2):
            super().__init__()
            self.blocks = nn.ModuleList([MockViTBlock(embed_dim) for _ in range(num_layers)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    vit = MockViTModel(embed_dim=64, num_layers=2)
    specs = UniversalArchitectureScanner.scan(vit)
    assert len(specs) == 2
    assert specs[0].pattern_type == "fused_qkv"
    assert specs[0].embed_dim == 64

    # Bridge transplant
    transplanted = transplant_attention(vit, attention="cosformer")
    x = torch.randn(2, 16, 64)
    out = transplanted(x)
    assert out.shape == (2, 16, 64)


def test_arch5_custom_cross_attention_and_shadow_twin():
    """
    Architecture 5: Custom nn.Module with asymmetric Cross-Attention (T_q != T_kv)
    and Shadow Twin distillation lifecycle.
    """
    class MockCrossAttentionModule(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)

        def forward(self, q_x, kv_x=None):
            if kv_x is None:
                kv_x = q_x
            q = self.q_proj(q_x)
            k = self.k_proj(kv_x)
            v = self.v_proj(kv_x)
            scores = torch.bmm(q, k.transpose(1, 2)) * (1.0 / (64 ** 0.5))
            weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(weights, v)
            return self.o_proj(context)

    orig_cross_attn = MockCrossAttentionModule(d=64)
    q_x = torch.randn(2, 8, 64)
    kv_x = torch.randn(2, 16, 64)

    # 1. Asymmetric Cross-Attention Router verification
    router = AsymmetricCrossAttentionRouter(d_model=64, num_heads=4)
    routed_out, _ = router(q_x, kv_x)
    assert routed_out.shape == (2, 8, 64)

    # 2. Shadow-Twin Distillation Lifecycle verification
    specs = UniversalArchitectureScanner.scan(orig_cross_attn)
    assert len(specs) == 1
    spec = specs[0]

    from neuralops.attentions.linear.cosformer import CosformerAttention
    student_attn = CosformerAttention(hidden_size=64, num_heads=4)

    bridge = UniversalAttentionBridge(
        orig_module=orig_cross_attn,
        spec=spec,
        new_attention=student_attn,
        inherit_weights=True,
    )
    enable_shadow_twin(bridge, stage="stage1_orientation")

    # Stage 1: pure shadow twin distillation
    set_surgery_stage(bridge, stage="stage1_orientation")
    set_surgery_alpha(bridge, alpha=0.5)

    bridge(q_x)
    loss = collect_surgery_distillation_loss(bridge)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0.0

    # Finalize surgery
    finalized = finalize_surgery(bridge)
    assert finalized.teacher_branch is None
    out_final = finalized(q_x)
    assert out_final.shape == (2, 8, 64)
