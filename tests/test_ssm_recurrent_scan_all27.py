import pytest
import torch
import time
import neuralops.core.cuda_kernels as ck
from neuralops.core.cuda_kernels import force_kernel_backend

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def assert_metrics(cuda_out, ref_out, name='', tol=1e-4):
    if isinstance(cuda_out, (tuple, list)):
        for i, (c, r) in enumerate(zip(cuda_out, ref_out)):
            assert not torch.isnan(c).any().item(), f'[{name}_{i}] NaNs detected in CUDA output!'
            assert not torch.isinf(c).any().item(), f'[{name}_{i}] Infs detected in CUDA output!'
            diff = (c.float() - r.float()).abs().max().item()
            assert diff < tol, f'[{name}_{i}] Diff too high: {diff:.6e} >= {tol}'
    else:
        assert not torch.isnan(cuda_out).any().item(), f'[{name}] NaNs detected in CUDA output!'
        assert not torch.isinf(cuda_out).any().item(), f'[{name}] Infs detected in CUDA output!'
        diff = (cuda_out.float() - ref_out.float()).abs().max().item()
        assert diff < tol, f'[{name}] Diff too high: {diff:.6e} >= {tol}'

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_01_selective_ssm():
    B, N, D = 4, 128, 64
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    dt = torch.randn(B, N, D, device=device, dtype=torch.float32).abs()
    A = -torch.rand(D, device=device, dtype=torch.float32)
    B_vec = torch.randn(B, N, D, device=device, dtype=torch.float32)
    C_vec = torch.randn(B, N, D, device=device, dtype=torch.float32)
    out_ref = ck.selective_ssm_ref(x, dt, A, B_vec, C_vec)
    out_cuda = ck.selective_ssm(x, dt, A, B_vec, C_vec)
    assert_metrics(out_cuda, out_ref, 'selective_ssm')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_02_selective_scan():
    from neuralops.core.cuda_kernels.selective_scan import selective_scan, selective_scan_pytorch_ref
    B, D, L, N = 2, 64, 128, 16
    u = torch.randn(B, D, L, device=device, dtype=torch.float32)
    delta = torch.randn(B, D, L, device=device, dtype=torch.float32).abs()
    A = -torch.rand(D, N, device=device, dtype=torch.float32)
    B_t = torch.randn(B, N, L, device=device, dtype=torch.float32)
    C_t = torch.randn(B, N, L, device=device, dtype=torch.float32)
    D_val = torch.randn(D, device=device, dtype=torch.float32)
    out_ref = selective_scan_pytorch_ref(u, delta, A, B_t, C_t, D_val)
    out_cuda = selective_scan(u, delta, A, B_t, C_t, D_val)
    assert_metrics(out_cuda, out_ref, 'selective_scan')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_03_simple_ssm():
    B, N, D = 4, 128, 64
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    decay = torch.rand(B, N, D, device=device, dtype=torch.float32) * 0.9
    scale = torch.randn(D, device=device, dtype=torch.float32)
    out_ref = ck.simple_ssm_ref(x, decay, scale)
    out_cuda = ck.simple_ssm(x, decay, scale)
    assert_metrics(out_cuda, out_ref, 'simple_ssm')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_04_linear_rnn_attention():
    B, N, D = 2, 64, 32
    q = torch.randn(B, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, N, D, device=device, dtype=torch.float32)
    out_ref = ck.linear_rnn_attention_ref(q, k, v)
    out_cuda = ck.linear_rnn_attention(q, k, v)
    assert_metrics(out_cuda, out_ref, 'linear_rnn_attention')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_05_mamba_block():
    B, D, S = 2, 64, 16
    u = torch.randn(B, D, device=device, dtype=torch.float32)
    delta = torch.randn(B, D, device=device, dtype=torch.float32).abs()
    A = -torch.rand(D, S, device=device, dtype=torch.float32)
    B_mat = torch.randn(B, S, device=device, dtype=torch.float32)
    C_mat = torch.randn(B, S, device=device, dtype=torch.float32)
    prev = torch.randn(B, D, S, device=device, dtype=torch.float32)
    out_ref = ck.mamba_block_ref(u, delta, A, B_mat, C_mat, prev)
    out_cuda = ck.mamba_block(u, delta, A, B_mat, C_mat, prev)
    assert_metrics(out_cuda, out_ref, 'mamba_block', tol=2e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_06_mamba_svd_flash():
    B, D, R = 32, 256, 32
    u = torch.randn(B, D, device=device, dtype=torch.float32)
    svd_u = torch.randn(D, R, device=device, dtype=torch.float32)
    svd_vt = torch.randn(R, D, device=device, dtype=torch.float32)
    decay = torch.rand(D, device=device, dtype=torch.float32) * 0.9
    out_ref = ck.mamba_svd_flash_ref(u, svd_u, svd_vt, decay)
    with force_kernel_backend('cuda'):
        out_cuda = ck.mamba_svd_flash(u, svd_u, svd_vt, decay)
    assert_metrics(out_cuda, out_ref, 'mamba_svd_flash', tol=2e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_07_hgrn2_scan():
    B, C, S, D = 2, 4, 32, 16
    x = torch.randn(B, C, S, D, device=device, dtype=torch.float32)
    g = torch.sigmoid(torch.randn(B, C, S, D, device=device, dtype=torch.float32))
    out_ref = ck.hgrn2_fused_bwd_scan_pytorch_ref(x, g)
    out_cuda = ck.hgrn2_fused_bwd_scan(x, g)
    assert_metrics(out_cuda, out_ref, 'hgrn2_scan')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_08_hgrn2_spatial_scan():
    B, H, W, C = 2, 16, 16, 32
    x = torch.randn(B, H, W, C, device=device, dtype=torch.float32)
    g = torch.sigmoid(torch.randn(B, H, W, C, device=device, dtype=torch.float32))
    out_ref = ck.hgrn2_spatial_scan_pytorch_ref(x, g)
    out_cuda = ck.hgrn2_spatial_scan(x, g)
    assert_metrics(out_cuda, out_ref, 'hgrn2_spatial_scan')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_09_hgrn2_gated_deltanet():
    class DummyConfig:
        hidden_size = 64
        num_attention_heads = 4
        lora_rank = 0
    mod = ck.hgrn2_gated_deltanet(config=DummyConfig(), d_model=64, num_heads=4).to(device)
    x = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
    out, _ = mod(x)
    assert not torch.isnan(out).any().item(), 'NaNs detected in hgrn2_gated_deltanet output!'
    assert out.shape == (2, 32, 64)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_10_gdn_scan():
    from neuralops.core.cuda_kernels.gdn_scan import chunked_delta_gate_scan, chunked_delta_gate_scan_pytorch_ref
    B, N, H, D = 2, 64, 4, 32
    q = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, N, H, device=device, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, N, H, device=device, dtype=torch.float32))
    out_ref = chunked_delta_gate_scan_pytorch_ref(q, k, v, alpha, beta)
    out_cuda = chunked_delta_gate_scan(q, k, v, alpha, beta)
    assert_metrics(out_cuda, out_ref, 'gdn_scan', tol=5e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_11_gdn_delta_rule():
    B, N, H, D = 2, 64, 4, 32
    q = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, N, H, device=device, dtype=torch.float32))
    out_ref = ck.gdn_delta_rule_pytorch_ref(q, k, v, beta)
    out_cuda = ck.gdn_delta_rule(q, k, v, beta)
    assert_metrics(out_cuda, out_ref, 'gdn_delta_rule', tol=2e-3)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_12_gla_scan():
    B, N, H, D = 2, 64, 4, 32
    q = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, N, H, device=device, dtype=torch.float32))
    out_ref = ck.gla_scan_pytorch_ref(q, k, v, alpha)
    out_cuda = ck.gla_scan(q, k, v, alpha)
    assert_metrics(out_cuda, out_ref, 'gla_scan')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_13_gla_unnormalized_scan():
    B, N, H, D = 2, 64, 4, 32
    q = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, N, H, D, device=device, dtype=torch.float32)
    decay = torch.sigmoid(torch.randn(B, N, H, D, device=device, dtype=torch.float32))
    out_ref = ck.gla_unnormalized_scan_pytorch_ref(q, k, v, decay)
    out_cuda = ck.gla_unnormalized_scan(q, k, v, decay)
    assert_metrics(out_cuda, out_ref, 'gla_unnormalized_scan')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_14_recurrent_gla_step():
    B, H, D = 2, 4, 32
    q = torch.randn(B, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, D, device=device, dtype=torch.float32)
    state = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, H, device=device, dtype=torch.float32))
    out_ref = ck.recurrent_gla_step_pytorch_ref(q, k, v, state, alpha)
    out_cuda = ck.recurrent_gla_step(q, k, v, state, alpha)
    assert_metrics(out_cuda, out_ref, 'recurrent_gla_step')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_15_state_update():
    from neuralops.core.cuda_kernels.state_update import recurrent_linear_step, recurrent_linear_step_pytorch_ref
    B, H, D = 2, 4, 32
    q = torch.randn(B, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, D, device=device, dtype=torch.float32)
    kv_state = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
    z_state = torch.zeros(B, H, D, device=device, dtype=torch.float32)
    out_ref = recurrent_linear_step_pytorch_ref(q, k, v, kv_state, z_state)
    out_cuda = recurrent_linear_step(q, k, v, kv_state, z_state)
    assert_metrics(out_cuda, out_ref, 'state_update')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_16_orion():
    B, H, N, D = 2, 4, 64, 32
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    out_ref = ck.orion_ref(q, k, v)
    out_cuda = ck.orion(q, k, v)
    assert_metrics(out_cuda, out_ref, 'orion')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_17_decay_linear_attention():
    B, H, N, D = 2, 4, 64, 32
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    gamma = torch.sigmoid(torch.randn(H, D, device=device, dtype=torch.float32))
    out_ref = ck.decay_linear_attention_ref(q, k, v, gamma)
    out_cuda = ck.decay_linear_attention(q, k, v, gamma)
    assert_metrics(out_cuda, out_ref, 'decay_linear_attention')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_18_rwkv6():
    B, H, N, D = 2, 4, 64, 32
    r = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    w = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    u = torch.randn(H, D, device=device, dtype=torch.float32)
    out_ref = ck.rwkv6_ref(r, k, v, w, u)
    out_cuda = ck.rwkv6(r, k, v, w, u)
    assert_metrics(out_cuda, out_ref, 'rwkv6')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_19_spectral_mamba():
    B, N, D = 4, 64, 32
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    freq = torch.randn(D, device=device, dtype=torch.float32)
    out_ref = ck.spectral_mamba_ref(x, freq)
    out_cuda = ck.spectral_mamba(x, freq)
    assert_metrics(out_cuda, out_ref, 'spectral_mamba')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_20_griffin_ssd():
    B, N, D = 4, 64, 32
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    r = torch.randn(B, N, D, device=device, dtype=torch.float32)
    decay = torch.rand(B, N, D, device=device, dtype=torch.float32) * 0.9
    out_ref = ck.griffin_ssd_ref(x, r, decay)
    out_cuda = ck.griffin_ssd(x, r, decay)
    assert_metrics(out_cuda, out_ref, 'griffin_ssd')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_21_retnet_retention():
    B, H, L, D = 2, 4, 64, 32
    q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    gammas = torch.linspace(0.8, 0.99, H, device=device, dtype=torch.float32)
    out_ref = ck.retnet_retention_pytorch_ref(q, k, v, gammas)
    with force_kernel_backend('cuda'):
        out_cuda = ck.retnet_retention(q, k, v, gammas)
    assert_metrics(out_cuda, out_ref, 'retnet_retention')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_22_gated_deltanet_block():
    class DummyConfig:
        hidden_size = 64
        num_attention_heads = 4
        lora_rank = 0
    block = ck.gated_deltanet_block(config=DummyConfig(), d_model=64, num_heads=4).to(device)
    x = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
    out, _ = block(x)
    assert not torch.isnan(out).any().item(), 'NaNs detected in gated_deltanet_block output!'
    assert out.shape == (2, 32, 64)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_23_hgrn2_layer():
    class DummyConfig:
        hidden_size = 64
        num_attention_heads = 4
        lora_rank = 0
    layer = ck.hgrn2_layer(config=DummyConfig(), d_model=64, is_decoder=True).to(device)
    x = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
    out, _ = layer(x)
    assert not torch.isnan(out).any().item(), 'NaNs detected in hgrn2_layer output!'
    assert out.shape == (2, 32, 64)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_24_hgrn2_gated_outer():
    class DummyConfig:
        hidden_size = 64
        num_attention_heads = 4
        lora_rank = 0
    outer = ck.hgrn2_gated_outer(config=DummyConfig(), d_model=64, num_heads=4, is_decoder=True).to(device)
    x = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
    out, _ = outer(x)
    assert not torch.isnan(out).any().item(), 'NaNs detected in hgrn2_gated_outer output!'
    assert out.shape == (2, 32, 64)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_25_mamba2_block():
    B, D, L, N = 2, 64, 128, 16
    u = torch.randn(B, D, L, device=device, dtype=torch.float32)
    delta = torch.randn(B, D, L, device=device, dtype=torch.float32).abs()
    A = torch.randn(D, device=device, dtype=torch.float32)
    B_mat = torch.randn(B, N, L, device=device, dtype=torch.float32)
    C_mat = torch.randn(B, N, L, device=device, dtype=torch.float32)
    D_val = torch.randn(D, device=device, dtype=torch.float32)
    out_ref = ck.mamba2_block_ref(u, delta, A, B_mat, C_mat, D_val)
    with force_kernel_backend('cuda'):
        out_cuda = ck.mamba2_block(u, delta, A, B_mat, C_mat, D_val)
    assert_metrics(out_cuda, out_ref, 'mamba2_block', tol=2e-4)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_26_liquid_attention():
    B, L, D = 2, 128, 64
    x = torch.randn(B, L, D, device=device, dtype=torch.float32)
    delta = torch.randn(B, L, D, device=device, dtype=torch.float32).abs()
    A = torch.randn(D, device=device, dtype=torch.float32)
    out_ref = ck.liquid_attention_ref(x, delta, A)
    with force_kernel_backend('cuda'):
        out_cuda = ck.liquid_attention(x, delta, A)
    assert_metrics(out_cuda, out_ref, 'liquid_attention')

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_27_recurrent_hgrn2_step():
    B, D = 4, 64
    x = torch.randn(B, D, device=device, dtype=torch.float32)
    g = torch.sigmoid(torch.randn(B, D, device=device, dtype=torch.float32))
    alpha = torch.sigmoid(torch.randn(B, D, device=device, dtype=torch.float32))
    prev = torch.randn(B, D, device=device, dtype=torch.float32)
    out_ref = ck.recurrent_hgrn2_step_pytorch_ref(x, g, alpha, prev)
    out_cuda = ck.recurrent_hgrn2_step(x, g, alpha, prev)
    assert_metrics(out_cuda, out_ref, 'recurrent_hgrn2_step')
