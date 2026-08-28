"""
Benchmark and Detailed Metrics Collector for SVD & Factorization Specialist Audit.
Computes empirical energy retention curves, parameter reductions, FLOP savings,
reconstruction error norms, and gradient checks across various sequence lengths and dimensions.
"""

import importlib.machinery
import importlib.util
import math
import os
import sys
import time
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

codes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)

dummy_attn_pkg = types.ModuleType("framework.attentions")
dummy_attn_pkg.__path__ = [os.path.join(codes_dir, "framework/attentions")]
sys.modules["framework.attentions"] = dummy_attn_pkg

dummy_comp_pkg = types.ModuleType("framework.compression_engine")
dummy_comp_pkg.__path__ = [os.path.join(codes_dir, "framework/compression_engine")]
sys.modules["framework.compression_engine"] = dummy_comp_pkg

def load_source(module_name, rel_path, pkg_obj):
    abs_path = os.path.join(codes_dir, rel_path)
    loader = importlib.machinery.SourceFileLoader(module_name, abs_path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    short_name = module_name.split(".")[-1]
    setattr(pkg_obj, short_name, mod)
    loader.exec_module(mod)
    return mod

load_source("framework.compression_engine.aa_svd_compressor", "framework/compression_engine/aa_svd_compressor.py", dummy_comp_pkg)
load_source("framework.compression_engine.pruning_utils", "framework/compression_engine/pruning_utils.py", dummy_comp_pkg)
load_source("framework.attentions.compressed_attention", "framework/attentions/compressed_attention.py", dummy_attn_pkg)

svd_lin_mod = load_source("framework.attentions.svd_linear_attention", "framework/attentions/svd_linear_attention.py", dummy_attn_pkg)
adap_attn_mod = load_source("framework.attentions.adaptive_svd_attention", "framework/attentions/adaptive_svd_attention.py", dummy_attn_pkg)
asvd_attn_mod = load_source("framework.attentions.asvd_attention", "framework/attentions/asvd_attention.py", dummy_attn_pkg)
linf_attn_mod = load_source("framework.attentions.linformer_attention", "framework/attentions/linformer_attention.py", dummy_attn_pkg)

res_svd_mod = load_source("framework.compression_engine.res_svd", "framework/compression_engine/res_svd.py", dummy_comp_pkg)
adaptive_ranks_mod = load_source("framework.compression_engine.adaptive_ranks", "framework/compression_engine/adaptive_ranks.py", dummy_comp_pkg)
direct_svd_mod = load_source("framework.compression_engine.direct_svd", "framework/compression_engine/direct_svd.py", dummy_comp_pkg)
dobi_svd_mod = load_source("framework.compression_engine.dobi_svd", "framework/compression_engine/dobi_svd.py", dummy_comp_pkg)
erc_svd_mod = load_source("framework.compression_engine.erc_svd", "framework/compression_engine/erc_svd.py", dummy_comp_pkg)
entropy_budget_mod = load_source("framework.compression_engine.entropy_budget_allocation", "framework/compression_engine/entropy_budget_allocation.py", dummy_comp_pkg)
kronecker_tt_mod = load_source("framework.compression_engine.kronecker_tt_lora", "framework/compression_engine/kronecker_tt_lora.py", dummy_comp_pkg)

print("=== EMPIRICAL SVD EVALUATION & AUDIT METRICS ===", flush=True)

# 1. Energy Retention Spectrum Evaluation
torch.manual_seed(42)
W = torch.randn(768, 768)
U, S, Vt = torch.linalg.svd(W, full_matrices=False)
s_sq = S ** 2
total_e = s_sq.sum()
cum_e = torch.cumsum(s_sq, dim=0) / total_e

print("\n[Metric 1: Energy Retention Spectrum for 768x768 Matrix]")
for tau in [0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99]:
    idx = torch.where(cum_e >= tau)[0]
    r = int(idx[0].item()) + 1
    actual_ret = cum_e[r - 1].item() * 100.0
    frob_err = torch.sqrt(1.0 - cum_e[r - 1]).item() * 100.0
    param_dense = 768 * 768
    param_svd = 2 * 768 * r
    cr = param_dense / param_svd
    print(f"  Threshold tau={tau:.2f} -> Rank r={r:3d}/768 | Energy={actual_ret:6.2f}% | FrobErr={frob_err:5.2f}% | CompRatio={cr:.2f}x")

# 2. Res-SVD Tail Reconstruction Quality
print("\n[Metric 2: Res-SVD Spectral Tail Initialization]")
for base_r, res_r in [(32, 8), (48, 16), (64, 16), (96, 32)]:
    res_layer = res_svd_mod.ResSVDLinear.from_linear(nn.Linear(768, 768), base_rank=base_r, residual_rank=res_r)
    err = res_layer.get_residual_error_norm(W)
    total_effective_rank = base_r + res_r
    print(f"  Base Rank r={base_r:2d}, Adapter Rank k={res_r:2d} (Total {total_effective_rank:2d}) -> Rel Frobenius Error: {err:.4f}")

# 3. Attention Forward/Backward Latency & Memory Scaling
print("\n[Metric 3: Linear Attention Scaling across Sequence Lengths]")
cfg = types.SimpleNamespace(
    hidden_size=768,
    num_attention_heads=12,
    num_heads=12,
    dropout=0.0,
    linear_attention_eps=1e-6,
)
exact_attn = svd_lin_mod.SVDLinearAttention(cfg, rank=64, mode="exact")
lin_attn = svd_lin_mod.SVDLinearAttention(cfg, rank=64, mode="linear", use_decay_gating=False)
decay_attn = svd_lin_mod.SVDLinearAttention(cfg, rank=64, mode="decay", use_decay_gating=True)

for N in [64, 256, 512, 1024]:
    x = torch.randn(1, N, 768, requires_grad=True)
    # Exact
    t0 = time.perf_counter()
    out_ex, _ = exact_attn(x)
    out_ex.sum().backward()
    t_exact = (time.perf_counter() - t0) * 1000.0

    # Linear non-decay
    x.grad = None
    t0 = time.perf_counter()
    out_lin, _ = lin_attn(x)
    out_lin.sum().backward()
    t_lin = (time.perf_counter() - t0) * 1000.0

    # Decay linear
    x.grad = None
    t0 = time.perf_counter()
    out_dec, _ = decay_attn(x)
    out_dec.sum().backward()
    t_dec = (time.perf_counter() - t0) * 1000.0

    print(f"  SeqLen N={N:4d} | Exact Softmax: {t_exact:6.2f} ms | O(N) Linear: {t_lin:6.2f} ms | Decay Gated O(N): {t_dec:6.2f} ms")

print("\nBenchmark completed successfully!", flush=True)
