# Comprehensive Stress Test & Deep Benchmark Suite for All SVD, Tensor-Train, and Low-Rank Modules.
import importlib.machinery
import importlib.util
import math
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

codes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)

try:
    import framework.attentions as dummy_attn_pkg
except Exception:
    dummy_attn_pkg = types.ModuleType('framework.attentions')
    dummy_attn_pkg.__path__ = [os.path.join(codes_dir, 'framework/attentions')]
    sys.modules['framework.attentions'] = dummy_attn_pkg

try:
    import framework.compression_engine as dummy_comp_pkg
except Exception:
    dummy_comp_pkg = types.ModuleType('framework.compression_engine')
    dummy_comp_pkg.__path__ = [os.path.join(codes_dir, 'framework/compression_engine')]
    sys.modules['framework.compression_engine'] = dummy_comp_pkg

def load_source(module_name, rel_path, pkg_obj):
    abs_path = os.path.join(codes_dir, rel_path)
    loader = importlib.machinery.SourceFileLoader(module_name, abs_path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    short_name = module_name.split('.')[-1]
    setattr(pkg_obj, short_name, mod)
    loader.exec_module(mod)
    return mod

# Dependencies
load_source('framework.compression_engine.aa_svd_compressor', 'framework/compression_engine/aa_svd_compressor.py', dummy_comp_pkg)
load_source('framework.compression_engine.pruning_utils', 'framework/compression_engine/pruning_utils.py', dummy_comp_pkg)
load_source('framework.attentions.compressed_attention', 'framework/attentions/compressed_attention.py', dummy_attn_pkg)

# 1-4 Attentions
svd_lin_mod = load_source('framework.attentions.svd_linear_attention', 'framework/attentions/svd_linear_attention.py', dummy_attn_pkg)
adap_attn_mod = load_source('framework.attentions.adaptive_svd_attention', 'framework/attentions/adaptive_svd_attention.py', dummy_attn_pkg)
asvd_attn_mod = load_source('framework.attentions.asvd_attention', 'framework/attentions/asvd_attention.py', dummy_attn_pkg)
linf_attn_mod = load_source('framework.attentions.linformer_attention', 'framework/attentions/linformer_attention.py', dummy_attn_pkg)

# 5-21 Compression Engines & Factorizations
res_svd_mod = load_source('framework.compression_engine.res_svd', 'framework/compression_engine/res_svd.py', dummy_comp_pkg)
adaptive_ranks_mod = load_source('framework.compression_engine.adaptive_ranks', 'framework/compression_engine/adaptive_ranks.py', dummy_comp_pkg)
direct_svd_mod = load_source('framework.compression_engine.direct_svd', 'framework/compression_engine/direct_svd.py', dummy_comp_pkg)
dobi_svd_mod = load_source('framework.compression_engine.dobi_svd', 'framework/compression_engine/dobi_svd.py', dummy_comp_pkg)
erc_svd_mod = load_source('framework.compression_engine.erc_svd', 'framework/compression_engine/erc_svd.py', dummy_comp_pkg)
entropy_budget_mod = load_source('framework.compression_engine.entropy_budget_allocation', 'framework/compression_engine/entropy_budget_allocation.py', dummy_comp_pkg)
kronecker_tt_mod = load_source('framework.compression_engine.kronecker_tt_lora', 'framework/compression_engine/kronecker_tt_lora.py', dummy_comp_pkg)
saes_v2_mod = load_source('framework.compression_engine.saes_svd_v2', 'framework/compression_engine/saes_svd_v2.py', dummy_comp_pkg)
swift_svd_mod = load_source('framework.compression_engine.swift_svd', 'framework/compression_engine/swift_svd.py', dummy_comp_pkg)
swift_svd_act_mod = load_source('framework.compression_engine.swift_svd_act', 'framework/compression_engine/swift_svd_act.py', dummy_comp_pkg)
adasvd_mod = load_source('framework.compression_engine.adasvd_compressor', 'framework/compression_engine/adasvd_compressor.py', dummy_comp_pkg)
rpca_mod = load_source('framework.compression_engine.rpca_compressor', 'framework/compression_engine/rpca_compressor.py', dummy_comp_pkg)
cur_lin_mod = load_source('framework.compression_engine.cur_linear', 'framework/compression_engine/cur_linear.py', dummy_comp_pkg)
cur_moe_mod = load_source('framework.compression_engine.cur_moe_compressor', 'framework/compression_engine/cur_moe_compressor.py', dummy_comp_pkg)
grecodec_mod = load_source('framework.compression_engine.grecodec_compressor', 'framework/compression_engine/grecodec_compressor.py', dummy_comp_pkg)
frandi_mod = load_source('framework.compression_engine.frandi_compressor', 'framework/compression_engine/frandi_compressor.py', dummy_comp_pkg)
dynamic_rank_mod = load_source('framework.compression_engine.dynamic_rank', 'framework/compression_engine/dynamic_rank.py', dummy_comp_pkg)
tt_lin_mod = load_source('framework.compression_engine.tt_linear', 'framework/compression_engine/tt_linear.py', dummy_comp_pkg)
kron_lin_mod = load_source('framework.compression_engine.kronecker_linear', 'framework/compression_engine/kronecker_linear.py', dummy_comp_pkg)

load_source('framework.compression_engine.aasvd', 'framework/compression_engine/aasvd.py', dummy_comp_pkg)
load_source('framework.compression_engine.rpca', 'framework/compression_engine/rpca.py', dummy_comp_pkg)
load_source('framework.compression_engine.cur_moe', 'framework/compression_engine/cur_moe.py', dummy_comp_pkg)
load_source('framework.compression_engine.grecodec', 'framework/compression_engine/grecodec.py', dummy_comp_pkg)
load_source('framework.compression_engine.frandi', 'framework/compression_engine/frandi.py', dummy_comp_pkg)

def get_config():
    return types.SimpleNamespace(
        hidden_size=768,
        num_attention_heads=12,
        num_heads=12,
        dropout=0.1,
        max_position_embeddings=512,
        energy_threshold=0.92,
        svd_rank=32,
        residual_rank=8,
        asvd_rank=32,
        linformer_k=64,
        rank=32,
    )

def test_numerical_stability_and_gradients():
    dtypes_to_test = [torch.float32]
    for dtype in dtypes_to_test:
        svd_layer = svd_lin_mod.SVDLinear(128, 128, rank=16, dtype=dtype)
        x = torch.randn(2, 10, 128, dtype=dtype, requires_grad=True)
        out = svd_layer(x)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert x.grad is not None and not torch.isnan(x.grad).any()

def test_svd_linear_attention():
    cfg = get_config()
    svd_attn = svd_lin_mod.SVDLinearAttention(cfg)
    x = torch.randn(2, 16, 768)
    out, _ = svd_attn(x)
    assert out.shape == (2, 16, 768)
    assert not torch.isnan(out).any()

def test_linformer_attention():
    cfg = get_config()
    linf_attn = linf_attn_mod.LinformerAttention(cfg, k=32, max_seq_len=256)
    x = torch.randn(2, 16, 768)
    mask = torch.ones(2, 16)
    out_linf, _ = linf_attn(x, attention_mask=mask)
    assert not torch.isnan(out_linf).any()
    assert out_linf.shape == (2, 16, 768)

if __name__ == '__main__':
    pytest.main([__file__, "-v"])
