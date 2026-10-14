import pytest
import torch
import torch.nn as nn
from neuralops.core.triton_kernels._common import (
    pad_head_dim_for_triton,
    unpad_head_dim,
    pad_to_triton_power_of_2,
    unpad_from_triton_power_of_2,
)
from neuralops.compression_engine.svd.aa_svd_compressor import SVDLinear


@pytest.mark.parametrize("orig_dim", [48, 80, 112, 160])
def test_tensor_core_head_dim_auto_padding(orig_dim):
    # Tests non-power-of-2 dimensions common in Qwen, Falcon, etc.
    B, H, N = 2, 4, 16
    q = torch.randn(B, H, N, orig_dim)
    k = torch.randn(B, H, N, orig_dim)
    v = torch.randn(B, H, N, orig_dim)

    (q_pad, k_pad, v_pad), d_orig = pad_head_dim_for_triton(q, k, v)

    # Padded dim must be power of 2 >= 16
    padded_dim = q_pad.shape[-1]
    assert (padded_dim & (padded_dim - 1)) == 0, f"{padded_dim} is not a power of 2!"
    assert padded_dim >= orig_dim

    # Slicing unpad must perfectly restore original shape and tensor values
    q_restored = unpad_head_dim(q_pad, d_orig)
    assert q_restored.shape == q.shape
    assert torch.equal(q_restored, q)


def test_svd_linear_rank_tensor_core_alignment():
    # Tests that ranks can be freely initialized or aligned without dimension crashes
    in_features, out_features = 128, 256
    layer = SVDLinear(in_features, out_features, rank=16, bias=True)
    x = torch.randn(2, 8, in_features)
    out = layer(x)
    assert out.shape == (2, 8, out_features)
    assert layer.rank % 8 == 0 or layer.rank == 16


def test_all_131_attentions_direct_dim_256():
    """Verify all 131 registered attentions support direct class instantiation cls(dim=256, num_heads=4)."""
    from neuralops.core.registry import ATTENTION_REGISTRY
    names = ATTENTION_REGISTRY.list_registered()
    assert len(names) >= 131, f"Expected >= 131 registered attentions, got {len(names)}"
    
    for name in names:
        cls = ATTENTION_REGISTRY.get(name)
        assert cls is not None, f"Attention class {name} could not be retrieved"
        
        # Instantiate directly with dim=256, num_heads=4
        inst = None
        for kwargs in [
            {"dim": 256, "num_heads": 4},
            {"hidden_size": 256, "num_heads": 4},
            {"d_model": 256, "num_heads": 4},
            {},
        ]:
            try:
                inst = cls(**kwargs)
                break
            except (TypeError, ValueError):
                continue
                
        assert inst is not None, f"Failed to instantiate {name} directly"
        
        # Verify forward pass with (2, 16, 256)
        x = torch.randn(2, 16, 256)
        try:
            out = inst(x)
            out_tensor = out[0] if isinstance(out, tuple) else out
            assert out_tensor.shape == (2, 16, 256), f"Shape mismatch in {name}: expected (2, 16, 256), got {out_tensor.shape}"
        except Exception as e:
            pytest.fail(f"Direct instantiation forward failed for {name}: {e}")

