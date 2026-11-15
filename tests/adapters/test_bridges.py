#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive Verification Test Battery for PEFT and Hugging Face Bi-directional Bridges.

Covers:
1. PEFT LoraConfig <-> NeuralOpsLoraConfig 100% roundtrip fidelity.
2. Parameter & State Dict translation with exact key mapping.
3. LoRA layer injection and weight copy parity.
4. Numerical output parity (diff == 0.0) between PEFT and NeuralOps forward passes.
5. Removable hook injection lifecycle.
6. HFTransformersAdapter architecture inspection and target discovery.
7. Edge cases: RSLoRA, rank patterns, modules_to_save, and task types.
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, RobertaConfig, RobertaForSequenceClassification

from neuralops.adapters.hf_transformers import HFTransformersAdapter
from neuralops.adapters.peft_adapter import (
    NeuralOpsLoraConfig,
    attach_lora_hooks,
    inject_lora_adapter,
    neuralops_to_peft_config,
    neuralops_to_peft_state_dict,
    peft_to_neuralops_config,
    peft_to_neuralops_state_dict,
)
from neuralops.adapters.registry import find_adapter
from neuralops.core.peft.lora import LoRALinear

try:
    from peft import LoraConfig, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config Roundtrip Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_neuralops_lora_config_defaults():
    """Verify default initialization of NeuralOpsLoraConfig."""
    cfg = NeuralOpsLoraConfig()
    assert cfg.r == 8
    assert cfg.lora_alpha == 16.0
    assert cfg.scaling == 2.0
    assert cfg.lora_dropout == 0.0
    assert cfg.bias == "none"
    assert not cfg.use_rslora
    assert not cfg.inference_mode


def test_neuralops_lora_config_rslora_scaling():
    """Verify RSLoRA scaling factor calculation (alpha / sqrt(r))."""
    cfg = NeuralOpsLoraConfig(r=16, lora_alpha=32.0, use_rslora=True)
    assert cfg.scaling == 32.0 / 4.0  # sqrt(16) = 4 -> 8.0


def test_neuralops_lora_config_dict_roundtrip():
    """Verify dictionary serialization and deserialization roundtrip."""
    original = NeuralOpsLoraConfig(
        r=16,
        lora_alpha=64.0,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="lora_only",
        use_rslora=True,
        modules_to_save=["score"],
        init_lora_weights=False,
        rank_pattern={"model.layers.0.q_proj": 32},
        alpha_pattern={"model.layers.0.q_proj": 128.0},
        task_type="CAUSAL_LM",
        inference_mode=True,
        custom_metadata="test_meta",
    )
    raw = original.to_dict()
    reconstructed = NeuralOpsLoraConfig.from_dict(raw)
    assert original == reconstructed
    assert reconstructed.extra_kwargs.get("custom_metadata") == "test_meta"


@pytest.mark.skipif(not HAS_PEFT, reason="PEFT is not installed")
def test_peft_to_neuralops_and_back_roundtrip():
    """Verify 100% roundtrip fidelity between PEFT LoraConfig and NeuralOpsLoraConfig."""
    peft_orig = LoraConfig(
        r=8,
        lora_alpha=32.0,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        use_rslora=False,
        task_type="SEQ_CLS",
        modules_to_save=["classifier"],
    )

    # PEFT -> NeuralOps
    nops_cfg = peft_to_neuralops_config(peft_orig)
    assert nops_cfg.r == 8
    assert nops_cfg.lora_alpha == 32.0
    assert nops_cfg.target_modules == {"query", "value"}
    assert abs(nops_cfg.lora_dropout - 0.05) < 1e-6
    assert nops_cfg.task_type == "SEQ_CLS"
    assert nops_cfg.modules_to_save == ["classifier"]

    # NeuralOps -> PEFT
    peft_roundtrip = neuralops_to_peft_config(nops_cfg)
    assert peft_roundtrip.r == peft_orig.r
    assert peft_roundtrip.lora_alpha == peft_orig.lora_alpha
    assert set(peft_roundtrip.target_modules) == set(peft_orig.target_modules)
    assert abs(peft_roundtrip.lora_dropout - peft_orig.lora_dropout) < 1e-6
    assert peft_roundtrip.bias == peft_orig.bias
    assert peft_roundtrip.modules_to_save == peft_orig.modules_to_save


# ─────────────────────────────────────────────────────────────────────────────
# 2. State Dict Translation Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_state_dict_bidirectional_translation():
    """Verify bidirectional state dict translation between PEFT and NeuralOps formats."""
    # Synthetic PEFT state dict
    peft_sd = {
        "base_model.model.encoder.layer.0.attention.self.query.base_layer.weight": torch.randn(16, 16),
        "base_model.model.encoder.layer.0.attention.self.query.base_layer.bias": torch.randn(16),
        "base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight": torch.randn(4, 16),
        "base_model.model.encoder.layer.0.attention.self.query.lora_B.default.weight": torch.randn(16, 4),
    }

    # Translate to NeuralOps
    nops_sd = peft_to_neuralops_state_dict(peft_sd, adapter_name="default")
    assert "encoder.layer.0.attention.self.query.base_layer.weight" in nops_sd
    assert "encoder.layer.0.attention.self.query.lora_A" in nops_sd
    assert "encoder.layer.0.attention.self.query.lora_B" in nops_sd
    assert torch.equal(
        nops_sd["encoder.layer.0.attention.self.query.lora_A"],
        peft_sd["base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight"],
    )

    # Translate back to PEFT
    peft_roundtrip_sd = neuralops_to_peft_state_dict(nops_sd, adapter_name="default", include_base_prefix=True)
    assert set(peft_roundtrip_sd.keys()) == set(peft_sd.keys())
    for k in peft_sd:
        assert torch.equal(peft_roundtrip_sd[k], peft_sd[k])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Injection and Numerical Parity Tests
# ─────────────────────────────────────────────────────────────────────────────


class MiniMLP(nn.Module):
    """Simple test module with two linear projections."""

    def __init__(self, in_dim: int = 8, hidden_dim: int = 16, out_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_inject_lora_adapter_structure():
    """Verify inject_lora_adapter replaces targeted linear layers with LoRALinear."""
    model = MiniMLP(8, 16, 8)
    config = NeuralOpsLoraConfig(r=4, lora_alpha=8.0, target_modules=["fc1", "fc2"])

    adapted_model, count = inject_lora_adapter(model, config)
    assert count == 2
    assert isinstance(adapted_model.fc1, LoRALinear)
    assert isinstance(adapted_model.fc2, LoRALinear)
    assert adapted_model.fc1.r == 4
    assert adapted_model.fc1.alpha == 8.0


@pytest.mark.skipif(not HAS_PEFT, reason="PEFT is not installed")
def test_peft_and_neuralops_numerical_parity():
    """Verify bit-exact output parity between PEFT model and NeuralOps LoRA adapted model."""
    torch.manual_seed(1337)
    base_model = MiniMLP(8, 16, 8).eval()
    x = torch.randn(2, 8)

    peft_cfg = LoraConfig(r=4, lora_alpha=8.0, target_modules=["fc1", "fc2"], lora_dropout=0.0)
    peft_model = get_peft_model(base_model, peft_cfg).eval()

    # Create NeuralOps copy sharing base layers
    nops_model = copy.deepcopy(base_model)
    nops_model.fc1 = LoRALinear(peft_model.base_model.model.fc1.base_layer, r=4, alpha=8.0, dropout=0.0).eval()
    nops_model.fc2 = LoRALinear(peft_model.base_model.model.fc2.base_layer, r=4, alpha=8.0, dropout=0.0).eval()

    # Copy weights from PEFT to NeuralOps
    nops_model.fc1.lora_A.data.copy_(peft_model.base_model.model.fc1.lora_A.default.weight.data)
    nops_model.fc1.lora_B.data.copy_(peft_model.base_model.model.fc1.lora_B.default.weight.data)
    nops_model.fc2.lora_A.data.copy_(peft_model.base_model.model.fc2.lora_A.default.weight.data)
    nops_model.fc2.lora_B.data.copy_(peft_model.base_model.model.fc2.lora_B.default.weight.data)

    with torch.no_grad():
        out_peft = peft_model(x)
        out_nops = nops_model(x)

    diff = (out_peft - out_nops).abs().max().item()
    assert diff == 0.0, f"Expected 0.0 difference between PEFT and NeuralOps, got {diff}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lifecycle Hooks Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_attach_lora_hooks_lifecycle():
    """Verify attach_lora_hooks registers and properly removes hooks from LoRALinear layers."""
    model = MiniMLP(8, 16, 8)
    config = NeuralOpsLoraConfig(r=2, lora_alpha=4.0, target_modules=["fc1"])
    inject_lora_adapter(model, config)

    pre_hook_called = False
    post_hook_called = False

    def pre_hook(mod, inp):
        nonlocal pre_hook_called
        pre_hook_called = True

    def post_hook(mod, inp, out):
        nonlocal post_hook_called
        post_hook_called = True

    handles = attach_lora_hooks(model, pre_hook_fn=pre_hook, post_hook_fn=post_hook)
    assert len(handles) == 2  # 1 pre-hook + 1 post-hook on fc1

    x = torch.randn(1, 8)
    model(x)
    assert pre_hook_called
    assert post_hook_called

    # Cleanup handles
    for h in handles:
        h.remove()

    pre_hook_called = False
    post_hook_called = False
    model(x)
    assert not pre_hook_called
    assert not post_hook_called


# ─────────────────────────────────────────────────────────────────────────────
# 5. HFTransformersAdapter Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_hf_transformers_adapter_gpt2_introspection():
    """Verify HFTransformersAdapter can inspect GPT-2 architecture and discover targets."""
    cfg = GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg).eval()

    adapter = find_adapter(model)
    assert adapter is not None

    hf_adapter = HFTransformersAdapter()
    assert hf_adapter.can_handle(model)
    family = hf_adapter.detect_family(model)
    assert family == "gpt2"

    info = hf_adapter.inspect(model)
    assert info["family"] == "gpt2"
    assert info["hidden_size"] == 32
    assert info["num_layers"] == 2
    assert info["num_heads"] == 2
    assert "c_attn" in info["linear_leaf_modules"]
    assert "c_proj" in info["linear_leaf_modules"]

    suggested = hf_adapter.get_suggested_lora_targets(model)
    assert "c_attn" in suggested
    assert "c_proj" in suggested


def test_hf_transformers_adapter_roberta_introspection():
    """Verify HFTransformersAdapter can inspect RoBERTa architecture and discover targets."""
    cfg = RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
    model = RobertaForSequenceClassification(cfg).eval()

    hf_adapter = HFTransformersAdapter()
    assert hf_adapter.can_handle(model)
    family = hf_adapter.detect_family(model)
    assert family == "roberta"

    info = hf_adapter.inspect(model)
    assert info["family"] == "roberta"
    assert info["hidden_size"] == 32
    assert info["num_layers"] == 2

    suggested = hf_adapter.get_suggested_lora_targets(model)
    assert "query" in suggested
    assert "key" in suggested
    assert "value" in suggested


def test_hf_transformers_adapter_extract_projection_weights():
    """Verify projection extraction for Q, K, V, and O from attention submodule."""
    cfg = RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
    model = RobertaForSequenceClassification(cfg).eval()

    hf_adapter = HFTransformersAdapter()
    attn_mod = model.roberta.encoder.layer[0].attention

    wq, bq = hf_adapter.extract_projection_weights(attn_mod, "q")
    assert wq.shape == (32, 32)
    assert bq is not None and bq.shape == (32,)

    wk, bk = hf_adapter.extract_projection_weights(attn_mod, "k")
    assert wk.shape == (32, 32)

    wv, bv = hf_adapter.extract_projection_weights(attn_mod, "v")
    assert wv.shape == (32, 32)

    wo, bo = hf_adapter.extract_projection_weights(attn_mod, "o")
    assert wo.shape == (32, 32)
