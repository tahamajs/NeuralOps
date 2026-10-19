#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Verify 17 Accelerated Attention Mechanisms

Verification script for forward, backward, gradient flow, and registry preservation
across the 17 target attention operators.
"""

import sys
import torch

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

TARGET_MODULES = [
    "aegis_hybrid",
    "aegis_v1",
    "aegis_v2",
    "aegis_v3",
    "aegis_v4",
    "aegis_v5",
    "aegis_spectron_fp8",
    "hybrid_local_kernel",
    "split_head_hybrid",
    "tri_branch_hybrid",
    "bam_umoe",
    "expert_choice_moe",
    "reap_moe",
    "chela_slay",
    "chela_slay_attention",
    "lola_attention",
    "adasplash_attention",
]


class MockConfig:
    """Mockconfig definition."""

    def __init__(self, hidden_size=256, num_heads=4, head_dim=64):
        """Execute init routine."""
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = hidden_size
        self.model_type = "llama"
        self.is_decoder = True
        self.architectures = ["LlamaForCausalLM"]


def test_module(name: str):
    """Execute test module routine."""
    print(f"\n{'=' * 20} Testing {name} {'=' * 20}")
    if name not in ATTENTION_REGISTRY:
        print(f"FAILED: {name} not found in ATTENTION_REGISTRY!")
        return False

    cls = ATTENTION_REGISTRY[name]
    config = MockConfig(hidden_size=256, num_heads=4, head_dim=64)

    try:
        mod = cls(config=config, hidden_size=256, num_heads=4, head_dim=64).cuda().float()
    except Exception:
        try:
            mod = cls(config=config).cuda().float()
        except Exception as e2:
            print(f"FAILED to instantiate {name}: {e2}")
            return False

    B, L, D = 2, 64, 256
    x = torch.randn(B, L, D, device="cuda", requires_grad=True)

    try:
        mod.zero_grad(set_to_none=True)
        if hasattr(mod, "is_decoder"):
            mod.is_decoder = True
        if hasattr(mod, "is_causal"):
            mod.is_causal = True

        out = mod(x)
        if isinstance(out, tuple):
            out_tensor = out[0]
        else:
            out_tensor = out

        assert out_tensor.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out_tensor.shape}"
        assert torch.isfinite(out_tensor).all(), "Output contains NaN or Inf"

        loss = out_tensor.sum()
        loss.backward()

        assert x.grad is not None, "Gradient on input x is None"
        assert torch.isfinite(x.grad).all(), "Gradient on x contains NaN or Inf"
        grad_norm = x.grad.norm().item()
        print(f"  [PASS] Causal Fwd/Bwd OK (out norm: {out_tensor.norm().item():.4f}, x grad norm: {grad_norm:.4f})")
    except Exception as e:
        print(f"  [FAIL] Causal Fwd/Bwd failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        x_bi = torch.randn(B, L, D, device="cuda", requires_grad=True)
        mod.zero_grad(set_to_none=True)
        if hasattr(mod, "is_decoder"):
            mod.is_decoder = False
        if hasattr(mod, "is_causal"):
            mod.is_causal = False

        out_bi = mod(x_bi)
        if isinstance(out_bi, tuple):
            out_bi_tensor = out_bi[0]
        else:
            out_bi_tensor = out_bi

        assert out_bi_tensor.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out_bi_tensor.shape}"
        assert torch.isfinite(out_bi_tensor).all(), "Bidirectional output contains NaN or Inf"

        loss_bi = out_bi_tensor.sum()
        loss_bi.backward()

        assert x_bi.grad is not None, "Gradient on input x_bi is None"
        assert torch.isfinite(x_bi.grad).all(), "Gradient on x_bi contains NaN or Inf"
        print(
            f"  [PASS] Bidirectional Fwd/Bwd OK (out norm: {out_bi_tensor.norm().item():.4f}, x grad norm: {x_bi.grad.norm().item():.4f})"
        )
    except Exception as e:
        print(f"  [FAIL] Bidirectional Fwd/Bwd failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    """Execute main routine."""
    print(f"Testing {len(TARGET_MODULES)} modules on {torch.cuda.get_device_name(0)}...")
    print(f"Current ATTENTION_REGISTRY count: {len(ATTENTION_REGISTRY)}")
    print(f"Current COMPRESSOR_REGISTRY count: {len(COMPRESSOR_REGISTRY)}")

    results = {}
    for name in TARGET_MODULES:
        results[name] = test_module(name)

    print("\n" + "=" * 50)
    print("SUMMARY RESULTS:")
    print("=" * 50)
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:25s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    assert len(ATTENTION_REGISTRY) == 131, f"Expected 131 attentions, got {len(ATTENTION_REGISTRY)}"
    assert len(COMPRESSOR_REGISTRY) == 72, f"Expected 72 compressors, got {len(COMPRESSOR_REGISTRY)}"
    print("Registry verification: 131 Attentions and 72 Compressors strictly preserved!")

    if all_passed:
        print("\nALL 17 MODULES PASSED HARDWARE ACCELERATION VERIFICATION SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSOME MODULES FAILED VERIFICATION!")
        sys.exit(1)


if __name__ == "__main__":
    main()
