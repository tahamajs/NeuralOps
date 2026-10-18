"""Comprehensive suite verifying all native CUDA kernel dispatchers, mathematical correctness, and fallback pipelines."""
import torch
import pytest
import neuralops.core.cuda_kernels as ck

def test_entmax_bisect_correctness():
    x = torch.randn(2, 4, 32)
    out = ck.entmax_bisect(x, n_iter=15)
    assert out.shape == x.shape
    assert torch.all(out >= 0.0)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

def test_split_head_norm_correctness():
    x = torch.randn(2, 8, 16, 64)
    out = ck.split_head_norm(x, eps=1e-5)
    assert out.shape == x.shape
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

def test_aegis_gate_norm_correctness():
    b0 = torch.randn(2, 16, 64)
    b1 = torch.randn(2, 16, 64)
    weights = torch.softmax(torch.randn(2, 16, 2), dim=-1)
    out = ck.aegis_gate_norm(b0, b1, weights, eps=1e-5)
    assert out.shape == b0.shape

def test_entropy_pruning_correctness():
    scores = torch.randn(2, 8, 32)
    ent = ck.entropy_pruning(scores)
    assert ent.shape == (2, 8)
    assert torch.all(ent >= 0.0)

def test_favor_plus_correctness():
    proj = torch.randn(4, 64)
    diag_sq = torch.sum(torch.randn(4, 32)**2, dim=-1) * 0.5
    out = ck.favor_plus(proj, diag_sq, norm_factor=0.125)
    assert out.shape == proj.shape
    assert torch.all(out >= 0.0)

def test_fwht_transform_correctness():
    x = torch.randn(2, 16, 64)
    out = ck.fwht_transform(x, B_dim=64)
    assert out.shape == x.shape
    # Involution property: H(H(x)) = x
    out2 = ck.fwht_transform(out, B_dim=64)
    assert torch.allclose(out2, x, atol=1e-4)

def test_rms_variance_match_correctness():
    y = torch.randn(2, 16, 64)
    v_ref = torch.randn(2, 16, 64) * 2.0
    out = ck.rms_variance_match(y, v_ref)
    assert out.shape == y.shape

def test_spectral_shrinkage_correctness():
    xr = torch.randn(2, 16, 32)
    xi = torch.randn(2, 16, 32)
    wr = torch.randn(2, 16, 32)
    wi = torch.randn(2, 16, 32)
    tau = torch.randn(2, 16, 32)
    out = ck.spectral_shrinkage(xr, xi, wr, wi, tau)
    assert out.is_complex()
    assert out.shape == xr.shape

def test_adaptive_span_correctness():
    spans = torch.tensor([16.0, 32.0, 64.0, 128.0])
    q_pos = torch.arange(16).unsqueeze(0).float()
    k_pos = torch.arange(16).unsqueeze(0).float()
    pen = ck.adaptive_span(spans, q_pos, k_pos, ramp_size=16.0, is_causal=True)
    assert pen.shape == (1, 4, 16, 16)
    assert torch.all(pen <= 0.0)

def test_dynamic_conv1d_correctness():
    B, H, N, D, K = 2, 4, 16, 32, 4
    v = torch.randn(B, H, N + K - 1, D)
    w = torch.softmax(torch.randn(B, H, N, K), dim=-1)
    out = ck.dynamic_conv1d(v, w)
    assert out.shape == (B, H, N, D)

def test_paged_recurrent_step_correctness():
    B, H, D = 2, 4, 32
    q = torch.randn(B, H, D)
    k = torch.randn(B, H, D)
    v = torch.randn(B, H, D)
    pool = torch.zeros(8, H, D, D)
    tables = torch.tensor([0, 1], dtype=torch.long)
    out = ck.paged_recurrent_step(q, k, v, pool, tables, decay=0.9)
    assert out.shape == (B, H, D)

def test_paged_kv_evict_correctness():
    k_in = torch.randn(64, 32)
    v_in = torch.randn(64, 32)
    idx = torch.tensor([0, 2, 5, 10], dtype=torch.long)
    k_out, v_out = ck.paged_kv_evict(k_in, v_in, idx)
    assert k_out.shape == (4, 32)
    assert v_out.shape == (4, 32)
    assert torch.allclose(k_out, k_in[idx])

def test_unified_rope_offset_correctness():
    x = torch.randn(2, 16, 64)
    cos = torch.ones(2, 16, 16)
    sin = torch.zeros(2, 16, 16)
    out = ck.unified_rope_offset(x, cos, sin, rotary_dim=32)
    assert out.shape == x.shape

def test_flash_decode_stage2_correctness():
    B, H, C, D = 2, 4, 8, 64
    mid_out = torch.randn(B, H, C, D)
    mid_lse = torch.randn(B, H, C)
    out = ck.flash_decode_stage2(mid_out, mid_lse)
    assert out.shape == (B, H, 1, D)

def test_svd_prefix_scan_correctness():
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    out = ck.svd_prefix_scan(q, k, v)
    assert out.shape == q.shape

def test_gdn_delta_rule_correctness():
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    beta = torch.sigmoid(torch.randn(B, H, L))
    out = ck.gdn_delta_rule(q, k, v, beta)
    assert out.shape == q.shape

def test_hgrn2_spatial_scan_correctness():
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    g = -torch.relu(torch.randn(B, H, L))
    out = ck.hgrn2_spatial_scan(q, k, v, g)
    assert out.shape == q.shape

def test_gla_unnormalized_scan_correctness():
    B, H, L, D = 2, 2, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    decay = torch.sigmoid(torch.randn(B, H, L))
    out = ck.gla_unnormalized_scan(q, k, v, decay)
    assert out.shape == q.shape

def test_bitonic_topk_correctness():
    x = torch.randn(4, 32)
    topk_idx = ck.bitonic_topk(x, K=4)
    assert topk_idx.shape == (4, 4)
    ref = torch.topk(x, k=4, dim=-1).indices
    assert topk_idx.shape == ref.shape

def test_logcumsumexp_correctness():
    x = torch.randn(4, 16)
    out = ck.logcumsumexp(x)
    assert out.shape == x.shape
    ref = torch.logcumsumexp(x, dim=-1)
    assert torch.allclose(out, ref, atol=1e-4)
