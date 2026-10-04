import torch
from neuralops.core.registry import ATTENTION_REGISTRY

D = 64

def test_module(module_name):
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
                module = module_class()
        
        x = torch.randn(2, 8, D)
        out, _ = module(x)
        assert out.shape == (2, 8, D)
    except Exception as e:
        print(f"{module_name} failed: {e}")

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
