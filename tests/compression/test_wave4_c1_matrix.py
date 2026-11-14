import pytest
import torch
import torch.nn as nn
from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor


OWNED_COMPRESSOR_KEYS = [
    # Low-Rank Family
    "come_compressor",
    "compot_compressor",
    "cr_compressor",
    "cur_moe_compressor",
    "drone_compressor",
    "enlighten_compressor",
    "frandi_compressor",
    "grecodec_compressor",
    "loha",
    "lokr",
    "lsc_compressor",
    "mlorq_compressor",
    "modegpt_compressor",
    "oats_compressor",
    "salr",
    "sdrm_compressor",
    "slim_compressor",
    "soka_compressor",
    "tomoe_compressor",
    "uniql_compressor",
    # Experimental Family
    "flat_llm",
    "hashed_net_compressor",
    "lillama_compressor",
    "roast_hashed_linear",
    # KV Cache Family
    "aatc_xkv_compressor",
    "cachenotes_compressor",
    "delta_kv",
    "hybridkv_compressor",
    "kvtc_compressor",
    "zerokv_neo_compressor",
    # Pruning Family
    "admm_compressor",
    "admm_pruner",
    "armor_pruner",
    "ast_pruner",
    "bonsai_pruner",
    "flap",
    "mamba_shedder",
    "proxsparse",
    "sparsegpt",
    "wanda",
    # Quantization Family
    "astro_quantizer",
    "awq",
    "coopq_quantizer",
    "faq_quantizer",
    "fp8",
    "gptq",
    "huffman_coder",
    "quarot",
    # Tensor Decomposition
    "tt_compressor",
]


class BenchmarkTwoLayer(nn.Module):
    def __init__(self, in_features: int = 768, out_features: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _build_and_compress(key: str, model: nn.Module, rank: int = 32):
    cls = COMPRESSOR_REGISTRY.get(key)
    kwargs = {
        "rank": rank,
        "target_ratio": 0.5,
        "sparsity": 0.5,
        "bits": 4,
        "fallback_to_original": False,
        "max_error": 2.0,
    }
    comp = None
    for fn in [
        lambda: create_compressor(key, model=model, **kwargs),
        lambda: cls(model, **kwargs),
        lambda: cls(model, rank=rank),
        lambda: cls(model, target_ratio=0.5),
        lambda: cls(model, sparsity=0.5),
        lambda: cls(model),
    ]:
        try:
            comp = fn()
            break
        except Exception:
            continue

    if comp is None:
        raise RuntimeError(f"Could not build compressor for {key}")

    if hasattr(comp, "compress") and callable(comp.compress):
        res = comp.compress()
        comp_model = res[0] if isinstance(res, tuple) else res
    else:
        comp_model = comp

    return comp_model


@pytest.mark.parametrize("key", OWNED_COMPRESSOR_KEYS)
def test_owned_compressor_registration(key: str):
    """Verify registration contract for every owned compressor."""
    assert key in COMPRESSOR_REGISTRY, f"Compressor '{key}' missing from COMPRESSOR_REGISTRY"
    cls = COMPRESSOR_REGISTRY.get(key)
    assert cls is not None, f"Compressor '{key}' resolved to None"


@pytest.mark.parametrize("key", OWNED_COMPRESSOR_KEYS)
def test_owned_compressor_forward_and_gradient(key: str):
    """Verify compression, output shape (2, 64, 768), finiteness, and gradient propagation."""
    model = BenchmarkTwoLayer(768, 768)
    comp_model = _build_and_compress(key, model, rank=32)

    assert isinstance(comp_model, nn.Module), f"Compressed output for {key} is not an nn.Module"

    x = torch.randn(2, 64, 768, requires_grad=True)
    out = comp_model(x)
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state

    assert out.shape == (2, 64, 768), f"Shape mismatch for {key}: {out.shape}"
    assert torch.isfinite(out).all(), f"Output of {key} contains NaN/Inf"

    # Verify autograd backward
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, f"Input gradient missing for {key}"
    assert torch.isfinite(x.grad).all(), f"Input gradient contains NaN/Inf for {key}"
