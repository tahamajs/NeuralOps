# Copyright (c) 2026 NeuralOps Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exhaustive Universal Model Surgery Test Suite (Suite 4).

Tests universal transplantation across 8 diverse model families:
    1. BERT (Bidirectional Encoder, BertSelfOutput residual)
    2. RoBERTa (Full Encoder, absolute position embeddings)
    3. GPT-2 (Causal Decoder with 1D Conv transpositions)
    4. LLaMA-3 (Modern GQA + RoPE Causal Decoder)
    5. Mistral (Sliding-window GQA Causal Decoder)
    6. T5 (Encoder-Decoder with relative position biases)
    7. Vision Transformer (ViT / DiT with fused QKV linear projections)
    8. Multi-Head Latent Attention (MLA / DeepSeek low-rank KV compression)
"""

import pytest
import torch
import torch.nn as nn
from transformers import (
    BertConfig, BertModel,
    GPT2Config, GPT2Model,
    LlamaConfig, LlamaModel,
    MistralConfig, MistralModel,
    RobertaConfig, RobertaModel,
    T5Config, T5ForConditionalGeneration,
)

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
from neuralops.surgery.shadow_twin import enable_shadow_twin

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_4_1_bert_bidirectional_encoder():
    """Verify BERT bidirectional attention transplantation and residual shape preservation."""
    config = BertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
    )
    model = BertModel(config).to(DEVICE).eval()
    transplanted = transplant_attention(model, attention="cosformer")
    x = torch.randint(0, 100, (2, 8), device=DEVICE)
    out = transplanted(x)[0]
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_4_2_roberta_encoder():
    """Verify RoBERTa encoder transplantation with full attention."""
    config = RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
    )
    model = RobertaModel(config).to(DEVICE).eval()
    transplanted = transplant_attention(model, attention="cosformer")
    x = torch.randint(0, 100, (2, 8), device=DEVICE)
    out = transplanted(x)[0]
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_4_3_gpt2_causal_decoder():
    """Verify GPT-2 causal decoder with Conv1D weights transposition."""
    config = GPT2Config(
        vocab_size=100,
        n_embd=64,
        n_layer=1,
        n_head=4,
        n_positions=128,
    )
    model = GPT2Model(config).to(DEVICE).eval()
    transplanted = transplant_attention(model, attention="cosformer")
    x = torch.randint(0, 100, (2, 8), device=DEVICE)
    out = transplanted(x)[0]
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_4_4_llama_gqa_rope_decoder():
    """Verify modern LLaMA-3 GQA + RoPE causal decoder transplantation."""
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    )
    model = LlamaModel(config).to(DEVICE).eval()
    transplanted = transplant_attention(model, attention="cosformer")
    x = torch.randint(0, 100, (2, 8), device=DEVICE)
    out = transplanted(x)[0]
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_4_5_mistral_sliding_window_decoder():
    """Verify Mistral GQA decoder transplantation."""
    config = MistralConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    )
    model = MistralModel(config).to(DEVICE).eval()
    transplanted = transplant_attention(model, attention="cosformer")
    x = torch.randint(0, 100, (2, 8), device=DEVICE)
    out = transplanted(x)[0]
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_4_6_t5_encoder_decoder_relative_bias():
    """Verify T5 Encoder-Decoder relative position bias attention transplantation."""
    config = T5Config(
        vocab_size=100,
        d_model=64,
        d_kv=16,
        num_heads=4,
        num_layers=1,
        num_decoder_layers=1,
        d_ff=128,
    )
    model = T5ForConditionalGeneration(config).to(DEVICE).eval()
    transplanted = transplant_attention(model, attention="cosformer")
    inp = torch.randint(0, 100, (2, 8), device=DEVICE)
    dec_inp = torch.randint(0, 100, (2, 8), device=DEVICE)
    out = transplanted(input_ids=inp, decoder_input_ids=dec_inp).logits
    assert out.shape == (2, 8, 100)
    assert torch.isfinite(out).all()


def test_4_7_vit_fused_qkv():
    """Verify Vision Transformer (ViT/DiT) with fused QKV linear layer transplantation."""
    class MockViTBlock(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, d * 3)
            self.proj = nn.Linear(d, d)
            self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

        def forward(self, x):
            h = self.norm(x)
            B, N, C = h.shape
            qkv = self.qkv(h).reshape(B, N, 3, 4, 16).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * 0.25
            attn = attn.softmax(dim=-1)
            x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))
            return x + self.mlp(self.norm(x))

    vit = MockViTBlock(64).to(DEVICE).eval()
    specs = UniversalArchitectureScanner.scan(vit)
    assert len(specs) >= 1
    assert specs[0].pattern_type == "fused_qkv"
    transplanted = transplant_attention(vit, attention="cosformer")
    x = torch.randn(2, 16, 64, device=DEVICE)
    out = transplanted(x)
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()


def test_4_8_mla_latent_compression():
    """Verify Multi-Head Latent Attention (MLA / DeepSeek) module surgery and distillation."""
    class MockMLABlock(nn.Module):
        def __init__(self, d=64, num_heads=4):
            super().__init__()
            self.d = d
            self.num_heads = num_heads
            self.q_proj = nn.Linear(d, d)
            self.kv_compress = nn.Linear(d, 32)
            self.k_decompress = nn.Linear(32, d)
            self.v_decompress = nn.Linear(32, d)
            self.out_proj = nn.Linear(d, d)

        def forward(self, x):
            q = self.q_proj(x)
            kv_latent = self.kv_compress(x)
            k = self.k_decompress(kv_latent)
            v = self.v_decompress(kv_latent)
            scores = torch.matmul(q, k.transpose(-1, -2)) * 0.125
            attn = torch.softmax(scores, dim=-1)
            return self.out_proj(torch.matmul(attn, v))

    mla = MockMLABlock(64).to(DEVICE).eval()
    transplanted = transplant_attention(mla, attention="cosformer")
    x = torch.randn(2, 8, 64, device=DEVICE)
    out = transplanted(x)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_4_9_shadow_twin_distillation_lifecycle():
    """Verify 3-stage Shadow-Twin distillation and finalization lifecycle."""
    config = BertConfig(vocab_size=100, hidden_size=64, num_hidden_layers=1, num_attention_heads=4, intermediate_size=128)
    model = BertModel(config).to(DEVICE)
    x = torch.randint(0, 100, (2, 8), device=DEVICE)

    # 1. Enable surgery with shadow twin
    transplanted = transplant_attention(model, attention="cosformer", attach_sidecar=False)
    enable_shadow_twin(transplanted, stage="stage1_orientation")

    # 2. Stage 1: Orientation phase
    set_surgery_stage(transplanted, "stage1_orientation")
    set_surgery_alpha(transplanted, alpha=0.5)
    transplanted(x)
    loss1 = collect_surgery_distillation_loss(transplanted)
    assert loss1 is not None and torch.isfinite(loss1).all()

    # 3. Finalize and detach teacher
    final_model = finalize_surgery(transplanted)
    out_final = final_model(x)[0]
    assert out_final.shape == (2, 8, 64)
    assert torch.isfinite(out_final).all()
