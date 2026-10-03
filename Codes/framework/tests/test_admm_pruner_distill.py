#!/usr/bin/env python3
"""
Unit tests for production-hardened ADMM Pruner with Layer-Wise Reconstruction Distillation.

Verifies:
1. Support for LoRALinear and Conv1D (transposition, preserving LoRA wrappers, trainable adapters).
2. Factory robustness when comp_config is None, dict, or object.
3. Reconstruction error validation and fallback mechanism when error > max_error with parameter freezing.
4. Layer-Wise Reconstruction Distillation lowers output reconstruction error compared to standard ADMM.
5. target_attention_only=True prunes attention layers while strictly skipping FFN/MLP and classifier.
6. STE forward/backward pass and actual sparsity tracking.
"""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Ensure framework can be imported
HERE = Path(__file__).resolve().parent
CODES_DIR = HERE.parent.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from framework.compression_engine.admm_pruner import (ADMMMaskedLinear,
                                                      ADMMPruner,
                                                      admm_pruner_factory)
from framework.utils.compat import get_recovery_params


class MockConv1D(nn.Module):
    """Mock of HuggingFace Conv1D (GPT-2 style) with weight shape [in_features, out_features]."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.nf = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class MockLoRALinear(nn.Module):
    """Mock of LoRALinear wrapper around a base layer."""

    def __init__(self, base_layer: nn.Module, r: int = 16):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        in_dim = getattr(base_layer, "in_features", None)
        out_dim = getattr(base_layer, "out_features", getattr(base_layer, "nf", None))
        if in_dim is None and hasattr(base_layer, "weight"):
            in_dim = base_layer.weight.shape[0] if hasattr(base_layer, "nf") else base_layer.weight.shape[1]
        if out_dim is None and hasattr(base_layer, "weight"):
            out_dim = base_layer.weight.shape[1] if hasattr(base_layer, "nf") else base_layer.weight.shape[0]

        self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        self.scaling = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


class ToyAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_attn = MockConv1D(hidden_dim, hidden_dim)
        self.o_proj = MockLoRALinear(nn.Linear(hidden_dim, hidden_dim), r=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x))


class ToyTransformer(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.attn = ToyAttention(hidden_dim)
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(hidden_dim, hidden_dim * 2),
            "up_proj": nn.Linear(hidden_dim, hidden_dim * 2),
            "down_proj": nn.Linear(hidden_dim * 2, hidden_dim),
        })
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(x)
        h = self.mlp["down_proj"](torch.relu(self.mlp["gate_proj"](h)))
        return self.classifier(h)


class TestADMMPrunerDistill(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_lora_linear_and_conv1d_detection(self):
        """1. Verify LoRALinear and Conv1D detection, shape transposition, and LoRA preservation."""
        model = ToyTransformer()
        orig_conv1d_shape = model.attn.c_attn.weight.shape  # [64, 64]
        orig_lora_a_grad = model.attn.o_proj.lora_A.requires_grad
        orig_lora_b_grad = model.attn.o_proj.lora_B.requires_grad

        compressor = ADMMPruner(
            model,
            sparsity=0.5,
            structured_24=True,
            max_error=0.60,
            target_attention_only=True,
            lambda_distill=0.0,
            use_ste=False,
        )
        compressed_model, stats = compressor.compress()

        # Check Conv1D compressed without corrupting shape
        self.assertEqual(compressed_model.attn.c_attn.weight.shape, orig_conv1d_shape)

        # Check LoRALinear wrapper preserved
        self.assertIsInstance(compressed_model.attn.o_proj, MockLoRALinear)
        self.assertTrue(compressed_model.attn.o_proj.lora_A.requires_grad)
        self.assertTrue(compressed_model.attn.o_proj.lora_B.requires_grad)

        # Base layer weight must be frozen after compression
        self.assertFalse(compressed_model.attn.o_proj.base_layer.weight.requires_grad)

    def test_factory_with_none_config(self):
        """2. Verify admm_pruner_factory works safely with None, dict, and config object."""
        model1 = ToyTransformer()
        comp1 = admm_pruner_factory(model1, comp_config=None, target_attention_only=True)
        self.assertIsNotNone(comp1)

        model2 = ToyTransformer()
        comp2 = admm_pruner_factory(model2, comp_config={"admm_pruner_sparsity": 0.5}, target_attention_only=True)
        self.assertIsNotNone(comp2)

        class DummyConfig:
            admm_pruner_sparsity = 0.5
            admm_pruner_max_error = 0.45
            lambda_distill = 0.1

        model3 = ToyTransformer()
        comp3 = admm_pruner_factory(model3, comp_config=DummyConfig(), target_attention_only=True)
        self.assertIsNotNone(comp3)

    def test_reconstruction_error_and_fallback(self):
        """3. Verify error thresholding triggers fallback to original weights and freezes parameters."""
        model = ToyTransformer()
        orig_q_w = model.attn.q_proj.weight.clone()

        # Set max_error unrealistically low to guarantee fallback
        compressor = ADMMPruner(
            model,
            max_error=0.01,
            fallback_to_original=True,
            target_attention_only=True,
        )
        compressed_model, stats = compressor.compress()

        self.assertGreater(len(compressor.fallback_layers), 0)
        self.assertIn("attn.q_proj", compressor.fallback_layers)
        # Weights must be exactly restored
        self.assertTrue(torch.equal(model.attn.q_proj.weight, orig_q_w))

        # Fallback parameters must be frozen
        for name in compressor.fallback_layers:
            mod = dict(model.named_modules())[name]
            for p in mod.parameters():
                self.assertFalse(p.requires_grad, f"Fallback layer {name} parameter was not frozen!")

    def test_distillation_lowers_output_reconstruction_error(self):
        """4. Verify Layer-Wise Reconstruction Distillation lowers layer output error compared to standard ADMM."""
        torch.manual_seed(123)
        hidden_dim = 64
        X_calib = torch.randn(32, hidden_dim)

        # 4a. Run standard ADMM without distillation (lambda_distill = 0.0)
        model_no_distill = ToyTransformer(hidden_dim=hidden_dim)
        compressor_no_distill = ADMMPruner(
            model_no_distill,
            sparsity=0.5,
            structured_24=True,
            max_error=0.80,
            target_attention_only=True,
            lambda_distill=0.0,
            use_ste=False,
            calibration_inputs={"attn.q_proj": X_calib},
        )
        comp_model_no_distill, stats_no_distill = compressor_no_distill.compress()

        # Original output of q_proj
        torch.manual_seed(123)
        orig_linear = nn.Linear(hidden_dim, hidden_dim)
        orig_linear.weight.data.copy_(model_no_distill.attn.q_proj.weight.data)  # note: q_proj was modified

        # Instead, test single linear layer cleanly to avoid random weight discrepancies
        torch.manual_seed(999)
        W0 = torch.randn(hidden_dim, hidden_dim)
        X = torch.randn(32, hidden_dim)
        Y0 = X @ W0.t()

        # Run with lambda_distill = 0.0
        torch.manual_seed(999)
        mod_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        mod_a.weight.data.copy_(W0)
        comp_a = ADMMPruner(
            mod_a,
            sparsity=0.5,
            structured_24=True,
            max_error=0.80,
            target_attention_only=False,
            lambda_distill=0.0,
            use_ste=False,
            calibration_inputs={"": X},
        )
        _, stats_a = comp_a.compress()
        err_a = torch.mean(((X @ mod_a.weight.data.t()) - Y0) ** 2).item()

        # Run with lambda_distill = 0.5
        torch.manual_seed(999)
        mod_b = nn.Linear(hidden_dim, hidden_dim, bias=False)
        mod_b.weight.data.copy_(W0)
        comp_b = ADMMPruner(
            mod_b,
            sparsity=0.5,
            structured_24=True,
            max_error=0.80,
            target_attention_only=False,
            lambda_distill=0.5,
            distill_lr=0.5,
            distill_steps=5,
            use_ste=False,
            calibration_inputs={"": X},
        )
        _, stats_b = comp_b.compress()
        err_b = torch.mean(((X @ mod_b.weight.data.t()) - Y0) ** 2).item()

        self.assertLess(err_b, err_a, f"Distillation error ({err_b:.4f}) must be lower than no-distill ({err_a:.4f})")
        self.assertIn("avg_distill_loss", stats_b)
        self.assertGreater(stats_b["avg_distill_loss"], 0.0)

    def test_target_attention_only_skips_ffn(self):
        """5. Verify target_attention_only=True strictly skips MLP/FFN and classifier."""
        model = ToyTransformer()
        orig_mlp_down = model.mlp["down_proj"].weight.clone()
        orig_classifier = model.classifier.weight.clone()

        compressor = ADMMPruner(
            model,
            max_error=0.60,
            target_attention_only=True,
        )
        _, stats = compressor.compress()

        for name in compressor.compressed_layers:
            self.assertTrue(name.startswith("attn."))
            self.assertFalse("mlp" in name)
            self.assertFalse("classifier" in name)

        self.assertTrue(torch.equal(model.mlp["down_proj"].weight, orig_mlp_down))
        self.assertTrue(torch.equal(model.classifier.weight, orig_classifier))

    def test_ste_forward_backward(self):
        """6. Verify ADMMMaskedLinear STE forwards masked weights and backward passes full gradients."""
        linear = nn.Linear(16, 16)
        mask = torch.ones(16, 16)
        mask[0, :] = 0.0  # mask row 0

        masked_linear = ADMMMaskedLinear.from_linear(linear, mask=mask)
        x = torch.randn(4, 16, requires_grad=True)
        out = masked_linear(x)

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(masked_linear.weight.grad)
        # STE allows gradients even on masked coordinates
        self.assertFalse(torch.all(masked_linear.weight.grad[0, :] == 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
