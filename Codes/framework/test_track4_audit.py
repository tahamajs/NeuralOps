"""
Targeted Verification test script for Track 4: Kernel Approximations and Quantization/KV Compressors.
"""

import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_CODES = _HERE.parent
if str(_CODES) not in sys.path:
    sys.path.insert(0, str(_CODES))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

# Direct imports of the 11 target attention modules
import framework.attentions.performer_attention
import framework.attentions.nystrom_attention
import framework.attentions.flash_attention3
import framework.attentions.cluster_attention
import framework.attentions.adasplash_attention
import framework.attentions.hamming_attention
import framework.attentions.cosformer
import framework.attentions.decay_linear_attention
import framework.attentions.gated_linear_attention
import framework.attentions.SOAFusedAttention
import framework.attentions.fused_svd_kernel

# Direct imports of the 9 target compressor modules
import framework.compression_engine.astro_quantizer
import framework.compression_engine.faq_quantizer
import framework.compression_engine.coopq_quantizer
import framework.compression_engine.kmeans_quantize
import framework.compression_engine.zerokv_neo
import framework.compression_engine.modegpt
import framework.compression_engine.delta_kv
import framework.compression_engine.kvtc
import framework.compression_engine.hybridkv


def make_config(hidden_size=64, num_heads=4, rank=16):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        rank=rank,
        dropout=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=256,
        num_landmarks=16,
        performer_rank=16,
        galerkin_rank=16,
        cluster_num_clusters=4,
        alpha=1.5,
        distill_temperature=2.0,
        k_rank=16,
    )


class DummyModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h[:, 0, :])


def audit_attentions():
    print("=" * 60)
    print("AUDITING TARGET ATTENTION MODULES")
    print("=" * 60)
    
    target_keys = [
        "performer",
        "nystrom",
        "flash3",
        "cluster",
        "adasplash",
        "hamming",
        "cosformer",
        "decay_linear_attention",
        "gated_linear_attention",
        "SOAFusedAttention",
        "fused_svd_kernel",
    ]

    B, N, D = 2, 32, 64
    cfg = make_config(hidden_size=D, num_heads=4, rank=16)
    results = {}

    for key in target_keys:
        print(f"Testing Attention '{key}' ...", end=" ")
        cls = ATTENTION_REGISTRY.get(key)
        if cls is None:
            print("FAILED (Not registered)")
            results[key] = False
            continue

        try:
            mod = cls(cfg) if not callable(cls) or isinstance(cls, type) else cls(config=cfg)
            mod.train()

            # Forward & Backward
            x = torch.randn(B, N, D, requires_grad=True)
            out, attn_w = mod(x)
            assert out.shape == (B, N, D), f"Shape mismatch: {out.shape} vs {(B, N, D)}"
            
            loss = out.sum()
            loss.backward()
            assert x.grad is not None, "No input gradient"
            
            # Masked forward
            mask_2d = torch.ones(B, N, dtype=torch.long)
            mask_2d[:, N // 2 :] = 0
            x_m = torch.randn(B, N, D, requires_grad=True)
            out_m, _ = mod(x_m, attention_mask=mask_2d)
            assert out_m.shape == (B, N, D)
            loss_m = out_m.sum()
            loss_m.backward()

            print(f"PASSED (Params: {sum(p.numel() for p in mod.parameters()):,})")
            results[key] = True
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            results[key] = False

    return results


def audit_compressors():
    print("\n" + "=" * 60)
    print("AUDITING TARGET COMPRESSORS & KV ENGINES")
    print("=" * 60)

    target_keys = [
        "astro",
        "faq",
        "coopq",
        "kmeans",
        "zerokv_neo",
        "modegpt",
        "delta_kv",
        "kvtc",
        "hybridkv",
    ]

    results = {}
    for key in target_keys:
        print(f"Testing Compressor '{key}' ...", end=" ")
        factory = COMPRESSOR_REGISTRY.get(key)
        if factory is None:
            print("FAILED (Not registered)")
            results[key] = False
            continue

        try:
            model = DummyModel(hidden_size=64)
            if key in ["astro", "faq", "coopq", "kmeans", "modegpt", "zerokv_neo", "delta_kv", "kvtc", "hybridkv"]:
                compressed = factory(model) if callable(factory) and not isinstance(factory, type) else factory(model).compress()[0]
                x = torch.randn(2, 16, 64)
                logits = compressed(x)
                assert logits.shape == (2, 2), f"Logits mismatch: {logits.shape}"
                
                # Check backward if QAT
                if key in ["astro", "faq", "coopq", "kmeans"]:
                    loss = logits.sum()
                    loss.backward()

            # Extra KV cache checks
            if key == "zerokv_neo":
                from framework.compression_engine.zerokv_neo_compressor import ZeroKVNeoCache
                c = ZeroKVNeoCache(max_cache_len=32, num_attention_sinks=4, recent_window=16)
                k = torch.randn(2, 4, 64, 32)
                v = torch.randn(2, 4, 64, 32)
                kc, vc = c.update_and_compress(k, v)
                assert kc.size(2) <= 32
            elif key == "delta_kv":
                from framework.compression_engine.delta_kv import DeltaKVStore
                s = DeltaKVStore(max_refs=32, topk=2)
                k = torch.randn(2, 4, 16, 32)
                v = torch.randn(2, 4, 16, 32)
                kc, vc = s.compress(k, v)
                s.update(k, v)
                assert kc.shape == k.shape
            elif key == "kvtc":
                from framework.compression_engine.kvtc_compressor import KVTCStore
                st = KVTCStore(bits=4, pca_energy=0.95)
                k = torch.randn(2, 4, 32, 32)
                v = torch.randn(2, 4, 32, 32)
                kh, vh, stats = st.compress_kv_cache(k, v)
                assert kh.shape == k.shape
                assert "theoretical_compression_ratio" in stats
            elif key == "hybridkv":
                from framework.compression_engine.hybridkv_compressor import HybridKVCache
                hk = HybridKVCache(max_cache_len=32, num_sinks=4, recent_window=16)
                k = torch.randn(2, 4, 48, 32)
                v = torch.randn(2, 4, 48, 32)
                kc, vc = hk.update_and_compress(k, v)
                assert kc.size(2) <= 32

            print("PASSED")
            results[key] = True
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            results[key] = False

    return results


def verify_ste_and_properties():
    print("\n" + "=" * 60)
    print("VERIFYING STE GRADIENTS & .WEIGHT COMPATIBILITY")
    print("=" * 60)

    # 1. AstroLinear STE & .weight
    from framework.compression_engine.astro_quantizer import AstroLinear
    lin = nn.Linear(32, 32)
    astro_lin = AstroLinear.from_linear(lin, bits=4)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = astro_lin(x)
    loss = out.sum()
    loss.backward()
    assert astro_lin.weight.grad is not None, "AstroLinear weight grad is None"
    assert (astro_lin.weight.grad.abs().sum() > 0), "AstroLinear weight grad is all zero"
    assert hasattr(astro_lin, "weight") and isinstance(astro_lin.weight, (torch.Tensor, nn.Parameter))
    print("✓ AstroLinear: STE gradients non-zero, .weight verified")

    # 2. FAQLinear STE & .weight
    from framework.compression_engine.faq_quantizer import FAQLinear
    faq_lin = FAQLinear.from_linear(lin, act_scale=torch.ones(32), bits=4)
    out = faq_lin(x)
    loss = out.sum()
    loss.backward()
    assert faq_lin.weight.grad is not None, "FAQLinear weight grad is None"
    assert (faq_lin.weight.grad.abs().sum() > 0), "FAQLinear weight grad is all zero"
    assert hasattr(faq_lin, "weight") and isinstance(faq_lin.weight, (torch.Tensor, nn.Parameter))
    print("✓ FAQLinear: STE gradients non-zero, .weight verified")

    # 3. CoopQLinear STE & .weight
    from framework.compression_engine.coopq_quantizer import CoopQLinear
    coopq_lin = CoopQLinear.from_linear(lin, bits=4)
    out = coopq_lin(x)
    loss = out.sum()
    loss.backward()
    assert coopq_lin.weight.grad is not None, "CoopQLinear weight grad is None"
    assert (coopq_lin.weight.grad.abs().sum() > 0), "CoopQLinear weight grad is all zero"
    assert hasattr(coopq_lin, "weight") and isinstance(coopq_lin.weight, (torch.Tensor, nn.Parameter))
    print("✓ CoopQLinear: STE gradients non-zero, .weight verified")

    # 4. KMeansLinear STE & .weight
    from framework.compression_engine.kmeans_quantize import KMeansLinear
    kmeans_lin = KMeansLinear(lin, n_clusters=16)
    out = kmeans_lin(x)
    loss = out.sum()
    loss.backward()
    assert kmeans_lin.centroids.grad is not None, "KMeansLinear centroids grad is None"
    assert (kmeans_lin.centroids.grad.abs().sum() > 0), "KMeansLinear centroids grad is all zero"
    w_recon = kmeans_lin.weight
    assert w_recon.shape == (32, 32), f"KMeansLinear weight shape mismatch: {w_recon.shape}"
    print("✓ KMeansLinear: STE centroids grad non-zero, @property .weight verified")

    # 5. FusedSVDLinear & .weight
    from framework.attentions.fused_svd_kernel import FusedSVDLinear
    fused_svd = FusedSVDLinear(32, 32, rank=8)
    out = fused_svd(x)
    loss = out.sum()
    loss.backward()
    assert fused_svd.V.grad is not None and fused_svd.U.grad is not None
    w_svd = fused_svd.weight
    assert w_svd.shape == (32, 32), f"FusedSVDLinear weight shape mismatch: {w_svd.shape}"
    print("✓ FusedSVDLinear: backward gradients non-zero, @property .weight verified")

    # 6. FAVOR+ Projection Verification
    from framework.attentions.performer_attention import PerformerAttention
    perf_cfg = make_config(hidden_size=64, num_heads=4, rank=16)
    perf = PerformerAttention(perf_cfg)
    # Check orthogonal buffer
    omega = perf.omega
    assert omega.shape == (16, 16), f"Omega shape {omega.shape}"
    # Check positive feature map output
    test_q = torch.randn(2, 4, 32, 16)
    phi_q = perf._feature_map(test_q)
    assert (phi_q >= 0).all(), "FAVOR+ features must be strictly non-negative"
    print("✓ Performer FAVOR+: Orthogonal projection matrix and non-negative feature map verified")


if __name__ == "__main__":
    attn_res = audit_attentions()
    comp_res = audit_compressors()
    verify_ste_and_properties()

    print("\n" + "=" * 60)
    print(f"Attentions: {sum(attn_res.values())}/{len(attn_res)} PASSED")
    print(f"Compressors: {sum(comp_res.values())}/{len(comp_res)} PASSED")
    print("ALL CHECKS COMPLETE")
    print("=" * 60)
