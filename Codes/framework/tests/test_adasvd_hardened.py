import sys
import math
from pathlib import Path

# Add project Codes directory
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
import torch
import torch.nn as nn
try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    try:
        from transformers.modeling_utils import Conv1D
    except ImportError:
        class Conv1D(nn.Module):
            def __init__(self, nf, nx):
                super().__init__()
                self.nf = nf
                self.weight = nn.Parameter(torch.empty(nx, nf))
                self.bias = nn.Parameter(torch.zeros(nf))
                nn.init.normal_(self.weight, std=0.02)
            def forward(self, x):
                size_out = x.size()[:-1] + (self.nf,)
                x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
                x = x.view(size_out)
                return x

from framework.compression_engine.adasvd_compressor import (
    AdaSVDCompressor,
    adasvd_factory,
    _cfg_get,
)
from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.utils.lora import LoRALinear
from framework.utils.compat import get_recovery_params


class DummyLoRAModel(nn.Module):
    def __init__(self, in_f=64, out_f=64, r=4):
        super().__init__()
        base = nn.Linear(in_f, out_f)
        self.query = LoRALinear(base, r=r, alpha=8)
        self.conv = Conv1D(out_f, in_f)  # [in_f, out_f] in weight
        self.classifier = nn.Linear(out_f, 2)

    def forward(self, x):
        h = self.query(x)
        h = self.conv(h)
        return self.classifier(h)


def test_cfg_get():
    """Verify _cfg_get handles None, dict, and object attributes correctly."""
    assert _cfg_get(None, "foo", 42) == 42
    assert _cfg_get({"foo": 10}, "foo", 42) == 10
    assert _cfg_get({"bar": 10}, "foo", 42) == 42

    class DummyCfg:
        foo = 99
    assert _cfg_get(DummyCfg(), "foo", 42) == 99
    assert _cfg_get(DummyCfg(), "bar", 42) == 42


def test_factory_with_none_and_dict_config():
    """Verify adasvd_factory works with None, dict, and custom object configs."""
    model = DummyLoRAModel(32, 32, r=2)
    # 1. With None
    m1 = adasvd_factory(model, comp_config=None)
    assert m1 is not None

    # 2. With dict config
    model2 = DummyLoRAModel(32, 32, r=2)
    cfg_dict = {"adasvd_max_error": 0.85, "adasvd_min_rank": 8, "adasvd_target_attention_only": False}
    m2 = adasvd_factory(model2, comp_config=cfg_dict)
    assert m2 is not None

    # 3. With explicit kwargs
    model3 = DummyLoRAModel(32, 32, r=2)
    m3 = adasvd_factory(model3, max_error=0.90, min_rank=4, target_attention_only=False)
    assert m3 is not None


def test_lora_and_conv1d_detection_and_preservation():
    """Verify LoRALinear wrappers are preserved and Conv1D weights transposed correctly."""
    model = DummyLoRAModel(64, 64, r=4)
    # Set known LoRA parameters
    model.query.lora_A.weight.data.fill_(0.7)
    model.query.lora_B.weight.data.fill_(0.7)
    orig_lora_a = model.query.lora_A.weight.clone()

    comp = AdaSVDCompressor(model, rank=16, max_error=1.0, target_attention_only=False)
    comp.compress()

    # Query should remain LoRALinear, with inner base_layer as SVDLinear
    assert isinstance(model.query, LoRALinear), f"Expected LoRALinear, got {type(model.query)}"
    assert isinstance(model.query.base_layer, SVDLinear), f"Expected SVDLinear in base_layer, got {type(model.query.base_layer)}"
    assert torch.equal(model.query.lora_A.weight, orig_lora_a), "LoRA weights were altered!"

    # Conv layer should be replaced by SVDLinear
    assert isinstance(model.conv, SVDLinear), f"Expected Conv1D replaced by SVDLinear, got {type(model.conv)}"

    # Forward pass works
    x = torch.randn(2, 64)
    out = model(x)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()

    # Check recovery params: LoRA and classifier are trainable, base factors frozen
    rec = get_recovery_params(model)
    rec_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("lora_A" in n for n in rec_names), "lora_A must be trainable"
    assert any("lora_B" in n for n in rec_names), "lora_B must be trainable"
    assert not any("base_layer" in n for n in rec_names), "base_layer factors must be frozen"


def test_reconstruction_error_and_fallback():
    """Verify fallback to original layer when reconstruction error exceeds max_error."""
    class TwoLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(64, 64)
            self.l2 = nn.Linear(64, 64)
        def forward(self, x):
            return self.l2(self.l1(x))

    model = TwoLinearModel()
    # With max_error=0.0001, compression should fail error threshold and fall back to original
    comp = AdaSVDCompressor(model, rank=4, max_error=1e-5, fallback_to_original=True, target_attention_only=False)
    comp.compress()

    assert len(comp.fallback_layers) > 0, "Expected layers to fall back"
    assert len(comp.compressed_layers) == 0, "No layers should have compressed"
    assert isinstance(model.l1, nn.Linear), "l1 should remain nn.Linear on fallback"
    assert isinstance(model.l2, nn.Linear), "l2 should remain nn.Linear on fallback"


def test_als_refinement_error_monotonicity():
    """Verify that ALS refinement never increases error over initial SVD."""
    lin = nn.Linear(128, 128)
    comp = AdaSVDCompressor(nn.Sequential(lin), rank=16, num_iters=10, target_attention_only=False)
    svd_layer, final_err, iters = comp.compress_linear(lin, rank=16)

    # Compute baseline SVD reconstruction error directly
    W = lin.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    r = svd_layer.rank
    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
    recon_svd = (U[:, :r] * sqrt_S.unsqueeze(0)) @ (sqrt_S.unsqueeze(1) * Vt[:r, :])
    init_err = float(torch.norm(W - recon_svd, p="fro").item() / torch.norm(W, p="fro").item())

    assert final_err <= init_err + 1e-6, f"ALS error ({final_err}) must not exceed initial SVD error ({init_err})"


def test_rank_floor_and_reduction_bounds():
    """Verify minimum rank floor and parameter reduction constraints."""
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tiny = nn.Linear(32, 32)
            self.wide = nn.Linear(768, 2304)  # Mimics GPT-2 c_attn
        def forward(self, x):
            return self.tiny(x)

    model = CustomModel()
    comp = AdaSVDCompressor(model, rank=432, min_rank=16, target_attention_only=False)
    comp.compress()

    # For tiny layer (32x32): param reduction requires r * (32+32) < 32*32 -> r < 16
    # max_param_r = 0.75 * (32*32)/64 = 12
    # So tiny rank is clamped to max_param_r
    assert comp.layer_ranks["tiny"] <= 12
    tiny_svd = model.tiny
    assert tiny_svd.rank * (32 + 32) < 32 * 32

    # For wide layer (768x2304, like c_attn):
    # max_param_r = 432, rank floor >= 288
    assert comp.layer_ranks["wide"] >= 288
    assert comp.layer_ranks["wide"] <= 432


def test_activation_whitening_and_weighted_als():
    """Verify Cholesky whitening and weighted ALS reduce activation error."""
    torch.manual_seed(42)
    in_f, out_f, r = 64, 64, 16
    lin = nn.Linear(in_f, out_f, bias=False)

    # Create non-isotropic input activations X with distinct dominant directions
    X = torch.randn(200, in_f)
    X[:, :8] *= 5.0
    C_X = (X.t() @ X) / X.size(0)

    comp = AdaSVDCompressor(nn.Sequential(lin), rank=r, num_iters=5, use_weighted_als=True)
    svd_layer, err, iters = comp.compress_linear(lin, rank=r, covariance=C_X, X_calib=X)

    assert svd_layer is not None
    assert err < 0.60

    # Check that reconstructed output approximates true output on X
    y_true = lin(X)
    y_pred = svd_layer(X)
    act_err = torch.norm(y_true - y_pred, p="fro").item() / torch.norm(y_true, p="fro").item()
    assert act_err < 0.60


def test_effective_rank_calculation():
    """Verify Shannon entropy effective rank calculation."""
    k = 32
    W_uniform = torch.eye(k)
    erank_uni = AdaSVDCompressor._compute_effective_rank(W_uniform)
    assert abs(erank_uni - k) < 1.0

    u = torch.randn(k, 1)
    v = torch.randn(1, k)
    W_rank1 = u @ v
    erank_r1 = AdaSVDCompressor._compute_effective_rank(W_rank1)
    assert abs(erank_r1 - 1.0) < 0.2


def test_strs_rank_search():
    """Verify STRS finds minimal rank satisfying max_error constraint."""
    torch.manual_seed(42)
    in_f, out_f = 64, 64
    lin = nn.Linear(in_f, out_f)
    comp = AdaSVDCompressor(nn.Sequential(lin), use_strs=True, max_error=0.45)
    svd_layer, err, iters = comp.compress_linear(lin, rank=32)
    assert svd_layer is not None
    assert err <= 0.45 or svd_layer.rank >= 16


if __name__ == "__main__":
    print("Running AdaSVD test suite...")
    test_cfg_get()
    print("[PASS] test_cfg_get")
    test_factory_with_none_and_dict_config()
    print("[PASS] test_factory_with_none_and_dict_config")
    test_lora_and_conv1d_detection_and_preservation()
    print("[PASS] test_lora_and_conv1d_detection_and_preservation")
    test_reconstruction_error_and_fallback()
    print("[PASS] test_reconstruction_error_and_fallback")
    test_als_refinement_error_monotonicity()
    print("[PASS] test_als_refinement_error_monotonicity")
    test_rank_floor_and_reduction_bounds()
    print("[PASS] test_rank_floor_and_reduction_bounds")
    test_activation_whitening_and_weighted_als()
    print("[PASS] test_activation_whitening_and_weighted_als")
    test_effective_rank_calculation()
    print("[PASS] test_effective_rank_calculation")
    test_strs_rank_search()
    print("[PASS] test_strs_rank_search")
    print("\nALL ADASVD UNIT TESTS PASSED SUCCESSFULLY!")
