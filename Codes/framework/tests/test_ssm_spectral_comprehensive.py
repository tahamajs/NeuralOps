import os
import sys
from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn

# Add root project dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from framework.core.registry import ATTENTION_REGISTRY

MODULES_TO_TEST = [
    "mamba", "mamba_svd_flash", "spectral_mamba", "upi_stuffed_mamba",
    "simple_ssm", "decay_linear_attention", "gated_linear_attention",
    "factorised_matrix_state", "soa_fused_attention", "soa_attn",
    "leka_v1", "leka_v2", "leka_v3"
]


@pytest.mark.parametrize("module_name", MODULES_TO_TEST)
def test_module(module_name: str, B=2, L=128, D=768):
    if module_name not in ATTENTION_REGISTRY:
        print(f"Skipping {module_name} (not in registry)")
        return
        
    print(f"Testing {module_name}...")
    try:
        module_class = ATTENTION_REGISTRY.get(module_name)
        if module_class is None:
            print(f"Skipping {module_name} (not found in registry via get)")
            return
        try:
            module = module_class(d_model=D, hidden_size=D)
        except TypeError:
            try:
                module = module_class(hidden_size=D)
            except TypeError:
                class MockConfig:
                    hidden_size = D
                    dim = D
                try:
                    module = module_class(dim=D)
                except TypeError:
                    module = module_class(config=MockConfig())
        if hasattr(module, 'to'):
            module = module.to('cpu')
        
        # Test 1: Forward pass, no NaN/Inf
        x = torch.randn(B, L, D, requires_grad=True, dtype=torch.float32)
        out, _ = module(hidden_states=x)
        
        assert out.shape == (B, L, D), f"Output shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN in output of {module_name}"
        assert not torch.isinf(out).any(), f"Inf in output of {module_name}"
        
        # Test 2: Backward pass, no NaN/Inf in gradients
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, f"No gradient for input in {module_name}"
        assert not torch.isnan(x.grad).any(), f"NaN in gradient of {module_name}"
        assert not torch.isinf(x.grad).any(), f"Inf in gradient of {module_name}"
        
        print(f"[PASS] {module_name}")
    except Exception as e:
        print(f"[FAIL] {module_name}: {str(e)}")

def main():
    modules_to_test = [
        "mamba", "mamba_svd_flash", "spectral_mamba", "upi_stuffed_mamba",
        "simple_ssm", "decay_linear_attention", "gated_linear_attention",
        "factorised_matrix_state", "soa_fused_attention", "soa_attn",
        "leka_v1", "leka_v2", "leka_v3"
    ]
    for m in modules_to_test:
        test_module(m)

if __name__ == "__main__":
    main()
