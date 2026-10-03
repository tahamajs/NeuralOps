"""
Unit tests for strict parameter freezing on hybrid recurrent attention modules
and trainable parameter count thresholds.
"""
import unittest

import torch
import torch.nn as nn
from neuralops.attentions.gated_delta_net import GatedDeltaNet, LoRALinear
from neuralops.attentions.hgrn2_gated_deltanet import HGRN2GatedDeltaNet
from neuralops.attentions.hgrn2_layer import HGRN2Layer
from neuralops.utils.compat import get_recovery_params


class DummyRecurrentModule(nn.Module):
    def __init__(self, d_model: int = 64, lora_rank: int = 16):
        super().__init__()
        self.d_model = d_model
        self.lora_rank = lora_rank
        self.q_proj = LoRALinear(d_model, d_model, lora_rank=lora_rank, freeze_base=True)
        self.k_proj = LoRALinear(d_model, d_model, lora_rank=lora_rank, freeze_base=True)
        self.v_proj = LoRALinear(d_model, d_model, lora_rank=lora_rank, freeze_base=True)
        self.out_proj = LoRALinear(d_model, d_model, lora_rank=lora_rank, freeze_base=True)
        self.beta = nn.Linear(d_model, 4)
        self.alpha = nn.Linear(d_model, 4)


class Dummy12LayerHybridModel(nn.Module):
    def __init__(self, d_model: int = 256, lora_rank: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            DummyRecurrentModule(d_model=d_model, lora_rank=lora_rank)
            for _ in range(12)
        ])
        self.classifier = nn.Linear(d_model, 2)


class TestHybridFreeze(unittest.TestCase):
    def test_freeze_hybrid_only_lora(self):
        """Verify that in recurrent/hybrid modules only LoRA adapters (and classifier heads) are trainable."""
        model = Dummy12LayerHybridModel(d_model=256, lora_rank=16)
        trainable_params = get_recovery_params(model)

        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        for name in trainable_names:
            if "classifier" in name:
                continue
            self.assertTrue(
                "lora_a" in name.lower() or "lora_b" in name.lower(),
                f"Parameter '{name}' should be frozen but is trainable!"
            )

        total_trainable = sum(p.numel() for p in trainable_params)
        self.assertLess(total_trainable, 3_000_000, f"Expected < 3M trainable params, got {total_trainable}")

    def test_oom_warning_on_large_params(self):
        """Simulate a model with > 5M trainable parameters and verify safety reduction."""
        # Create modules with large linear layers so total trainable params exceeds 5,000,000
        # 10 layers * (4096*64 + 4096*64) = 10 * 524,288 = 5,242,880 params
        layers = [LoRALinear(4096, 4096, lora_rank=64, freeze_base=False) for _ in range(10)]
        model = nn.Sequential(*layers)
        params = get_recovery_params(model)
        # Verify that lora_rank was automatically reduced to 8
        for m in model.modules():
            if hasattr(m, "lora_rank") and isinstance(m, LoRALinear):
                self.assertLessEqual(m.lora_rank, 8)


if __name__ == "__main__":
    unittest.main()
