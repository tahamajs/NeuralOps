#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Head Compression Healing & Shape-Adaptive Weight Copying.

Validates that Multi-Query Attention (MQA), Compressed MHA, and Block Transformer HIBA
faithfully copy weights from standard multi-head teacher attention (e.g. BertSelfAttention),
preserve representation variance, maintain high cosine similarity (>0.70), and pass autograd checks.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention

from neuralops.api import replace_attention
from neuralops.core.model_utils import copy_attention_weights
from neuralops.layers.modern.mqa import MultiQueryAttention
from neuralops.layers.svd.compressed_mha_attention import CompressedDistilBertAttention
from neuralops.layers.adaptive.block_transformer_hiba import BlockTransformerHIBA


@pytest.fixture
def bert_config():
    return BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_hidden_layers=2,
    )


@pytest.fixture
def bert_self(bert_config):
    torch.manual_seed(42)
    layer = BertSelfAttention(bert_config)
    layer.eval()
    return layer


def test_mqa_weight_copy_without_shape_mismatch(bert_config, bert_self):
    """Verify weight copy from BertSelfAttention into MQA adapts 12 KV heads to 1 without shape mismatch."""
    mqa = MultiQueryAttention(config=bert_config)
    copy_attention_weights(bert_self, mqa)

    assert mqa.q_proj.weight.shape == (768, 768), "Q projection shape mismatch"
    assert mqa.k_proj.weight.shape == (64, 768), "K projection shape mismatch (expected 1 head)"
    assert mqa.v_proj.weight.shape == (64, 768), "V projection shape mismatch (expected 1 head)"
    assert mqa.out_proj.weight.shape == (768, 768), "Out projection shape mismatch"

    # Query should directly match teacher query
    torch.testing.assert_close(mqa.q_proj.weight, bert_self.query.weight)

    # K projection should match scaled head-average of teacher key
    teacher_k_mean = bert_self.key.weight.view(12, 64, 768).mean(dim=0) * math.sqrt(12)
    torch.testing.assert_close(mqa.k_proj.weight, teacher_k_mean, atol=1e-5, rtol=1e-5)

    # V projection should match scaled head-average of teacher value
    teacher_v_mean = bert_self.value.weight.view(12, 64, 768).mean(dim=0) * math.sqrt(12)
    torch.testing.assert_close(mqa.v_proj.weight, teacher_v_mean, atol=1e-5, rtol=1e-5)


def test_compressed_mha_weight_copy_without_shape_mismatch(bert_config, bert_self):
    """Verify weight copy from BertSelfAttention into Compressed MHA without shape mismatch."""
    comp = CompressedDistilBertAttention(config=bert_config)
    copy_attention_weights(bert_self, comp)

    # Effective weights reconstructed from SVD factors should be valid and non-zero
    assert comp.query.weight.shape == (768, 768)
    assert comp.key.weight.shape == (768, 768)
    assert comp.value.weight.shape == (768, 768)
    assert comp.out_proj.weight.shape == (768, 768)

    assert comp.U_qkv.norm().item() > 0.0
    assert comp.Vh_qkv.norm().item() > 0.0
    assert comp.S_qkv.norm().item() > 0.0


def test_block_transformer_hiba_weight_copy(bert_config, bert_self):
    """Verify weight copy from BertSelfAttention into Block Transformer HIBA without error."""
    hiba = BlockTransformerHIBA(config=bert_config)
    copy_attention_weights(bert_self, hiba)

    assert hiba.W_q.weight.shape == (768, 768)
    assert hiba.W_k.weight.shape == (768, 768)
    assert hiba.W_v.weight.shape == (768, 768)
    assert hiba.out_proj.weight.shape == (768, 768)
    torch.testing.assert_close(hiba.W_q.weight, bert_self.query.weight)


def test_output_variance_and_cosine_similarity(bert_config):
    """Verify that output variance is preserved and cosine similarity to standard attention is >0.70."""
    torch.manual_seed(42)
    base_model = BertModel(bert_config)
    base_model.eval()

    x = torch.randn(2, 16, 768)
    with torch.no_grad():
        base_out = base_model(inputs_embeds=x).last_hidden_state
        base_var = base_out.var().item()

    mechanisms = ["mqa", "compressed_mha_attention", "block_transformer_hiba"]

    for mech in mechanisms:
        torch.manual_seed(42)
        student = BertModel(bert_config)
        student.load_state_dict(base_model.state_dict())
        student = replace_attention(student, attention=mech, copy_weights=True)
        student.eval()

        with torch.no_grad():
            student_out = student(inputs_embeds=x).last_hidden_state
            student_var = student_out.var().item()

        cos_sim = F.cosine_similarity(base_out.flatten(), student_out.flatten(), dim=0).item()
        var_ratio = student_var / base_var

        # Cosine similarity must exceed 0.70
        assert cos_sim > 0.70, f"[{mech}] Cosine similarity {cos_sim:.4f} is not > 0.70"

        # Variance ratio must be preserved within [0.80, 1.25]
        assert 0.80 <= var_ratio <= 1.25, f"[{mech}] Output variance ratio {var_ratio:.4f} is outside [0.80, 1.25]"


def test_direct_mqa_variance_preservation(bert_config, bert_self):
    """Verify that MQA directly preserves activation variance relative to teacher BertSelfAttention."""
    mqa = MultiQueryAttention(config=bert_config)
    copy_attention_weights(bert_self, mqa)
    mqa.eval()

    x = torch.randn(2, 16, 768)
    with torch.no_grad():
        y_teacher = bert_self(x)[0]
        y_mqa = mqa(x)[0]

    var_teacher = y_teacher.var().item()
    var_mqa = y_mqa.var().item()
    var_ratio = var_mqa / var_teacher

    # Direct MQA variance ratio should be close to 1.0 (between 0.85 and 1.15)
    assert 0.85 <= var_ratio <= 1.15, f"MQA direct variance ratio {var_ratio:.4f} not within [0.85, 1.15]"


def test_direct_compressed_mha_cosine_similarity(bert_config, bert_self):
    """Verify that Compressed MHA directly achieves >0.70 cosine similarity to teacher BertSelfAttention."""
    comp = CompressedDistilBertAttention(config=bert_config)
    copy_attention_weights(bert_self, comp)
    comp.eval()

    x = torch.randn(2, 16, 768)
    with torch.no_grad():
        y_teacher = bert_self(x)[0]
        y_comp = comp(x)[0]

    cos_sim = F.cosine_similarity(y_teacher.flatten(), y_comp.flatten(), dim=0).item()
    assert cos_sim > 0.70, f"Compressed MHA direct cosine similarity {cos_sim:.4f} is not > 0.70"


@pytest.mark.parametrize("mech_cls", [
    MultiQueryAttention,
    CompressedDistilBertAttention,
    BlockTransformerHIBA,
])
def test_forward_backward_autograd_check(bert_config, bert_self, mech_cls):
    """Verify forward and backward autograd pass cleanly with finite gradients on all parameters."""
    torch.manual_seed(42)
    layer = mech_cls(config=bert_config)
    copy_attention_weights(bert_self, layer)
    layer.train()

    # Ensure parameters require grad for training
    for p in layer.parameters():
        p.requires_grad = True

    x = torch.randn(2, 8, 768, requires_grad=True)
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]

    loss = out.sum()
    loss.backward()

    assert x.grad is not None, f"Input gradient is None for {mech_cls.__name__}"
    assert torch.isfinite(x.grad).all(), f"Input gradient has non-finite values for {mech_cls.__name__}"
    assert x.grad.norm().item() > 0.0, f"Input gradient is zero for {mech_cls.__name__}"

    # Verify at least one parameter gradient is non-zero and finite
    param_grads = [p.grad for p in layer.parameters() if p.grad is not None]
    assert len(param_grads) > 0, f"No parameter gradients computed for {mech_cls.__name__}"
    for g in param_grads:
        assert torch.isfinite(g).all(), f"Parameter gradient has non-finite values for {mech_cls.__name__}"
