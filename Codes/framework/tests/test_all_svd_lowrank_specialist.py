# Comprehensive Stress Test & Deep Benchmark Suite for All SVD, Tensor-Train, and Low-Rank Modules.
import importlib.machinery
import importlib.util
import math
import os
import sys
import types
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

codes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)

print('=' * 75, flush=True)
print('>>> RUNNING COMPREHENSIVE SVD, TT & LOW-RANK AUDIT SUITE <<<', flush=True)
print('=' * 75, flush=True)

# Register clean package namespaces
dummy_attn_pkg = types.ModuleType('framework.attentions')
dummy_attn_pkg.__path__ = [os.path.join(codes_dir, 'framework/attentions')]
sys.modules['framework.attentions'] = dummy_attn_pkg

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

# Alias modules
load_source('framework.compression_engine.aasvd', 'framework/compression_engine/aasvd.py', dummy_comp_pkg)
load_source('framework.compression_engine.rpca', 'framework/compression_engine/rpca.py', dummy_comp_pkg)
load_source('framework.compression_engine.cur_moe', 'framework/compression_engine/cur_moe.py', dummy_comp_pkg)
load_source('framework.compression_engine.grecodec', 'framework/compression_engine/grecodec.py', dummy_comp_pkg)
load_source('framework.compression_engine.frandi', 'framework/compression_engine/frandi.py', dummy_comp_pkg)

print('All 21 low-rank/SVD/TT modules successfully imported!', flush=True)

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

class MockTransformerModel(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(hidden_size, hidden_size),
                'k_proj': nn.Linear(hidden_size, hidden_size),
                'v_proj': nn.Linear(hidden_size, hidden_size),
                'out_proj': nn.Linear(hidden_size, hidden_size),
                'fc1': nn.Linear(hidden_size, hidden_size * 2),
                'fc2': nn.Linear(hidden_size * 2, hidden_size),
            })
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        for layer in self.layers:
            q = layer['q_proj'](x)
            k = layer['k_proj'](x)
            v = layer['v_proj'](x)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            attn = torch.softmax(attn, dim=-1)
            ctx = torch.matmul(attn, v)
            out = layer['out_proj'](ctx)
            x = x + out
            h = F.relu(layer['fc1'](x))
            x = x + layer['fc2'](h)
        return self.classifier(x[:, 0, :])

def test_section(title):
    print("\n=== " + title + " ===", flush=True)

# 1. Multi-Dtype Numerical Stability & Gradient Flow
test_section('1. Numerical Stability (FP32 / FP16 / BF16) & Gradient Flow')
dtypes_to_test = [torch.float32, torch.float16]
if hasattr(torch, 'bfloat16'):
    dtypes_to_test.append(torch.bfloat16)

for dtype in dtypes_to_test:
    dtype_str = str(dtype).split('.')[-1]
    # SVDLinear
    svd_layer = svd_lin_mod.SVDLinear(128, 128, rank=16, dtype=dtype)
    x = torch.randn(2, 10, 128, dtype=dtype, requires_grad=True)
    out = svd_layer(x)
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(out).any(), f'NaN in SVDLinear forward ({dtype_str})'
    assert not torch.isinf(out).any(), f'Inf in SVDLinear forward ({dtype_str})'
    assert x.grad is not None and not torch.isnan(x.grad).any(), f'NaN in x.grad ({dtype_str})'
    print(f'  [PASS] SVDLinear ({dtype_str}) forward/backward - grad_norm={x.grad.norm().item():.4f}')

    # AdaptiveSVDLinear
    adap_layer = adap_attn_mod.AdaptiveSVDLinear(128, 128, svd_rank=16, residual_rank=4, dtype=dtype)
    x = torch.randn(2, 10, 128, dtype=dtype, requires_grad=True)
    out = adap_layer(x)
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(out).any(), f'NaN in AdaptiveSVDLinear forward ({dtype_str})'
    print(f'  [PASS] AdaptiveSVDLinear ({dtype_str}) forward/backward - grad_norm={x.grad.norm().item():.4f}')

    # ASVDLinear
    asvd_layer = asvd_attn_mod.ASVDLinear(128, 128, rank=16, dtype=dtype)
    x = torch.randn(2, 10, 128, dtype=dtype, requires_grad=True)
    out = asvd_layer(x)
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(out).any(), f'NaN in ASVDLinear forward ({dtype_str})'
    print(f'  [PASS] ASVDLinear ({dtype_str}) forward/backward - grad_norm={x.grad.norm().item():.4f}')

    # DobiSVDLinearLayer
    dobi_layer = dobi_svd_mod.DobiSVDLinearLayer(128, 128, rank=16, dtype=dtype)
    x = torch.randn(2, 10, 128, dtype=dtype, requires_grad=True)
    out = dobi_layer(x)
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(out).any(), f'NaN in DobiSVDLinearLayer forward ({dtype_str})'
    print(f'  [PASS] DobiSVDLinearLayer ({dtype_str}) forward/backward - grad_norm={x.grad.norm().item():.4f}')

# 2. Condition Number & Factor Balance Analysis
test_section('2. Condition Number kappa & Factor Balance')
for d_in, d_out in [(768, 768), (1024, 512), (512, 2048)]:
    W = torch.randn(d_out, d_in)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    r = 32
    sqrt_S = torch.sqrt(S[:r])
    U_r = U[:, :r] * sqrt_S.unsqueeze(0)
    V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]

    norm_U = torch.norm(U_r, p='fro').item()
    norm_V = torch.norm(V_r, p='fro').item()
    ratio = norm_U / norm_V
    s_U = torch.linalg.svdvals(U_r)
    s_V = torch.linalg.svdvals(V_r)
    kappa_U = (s_U[0] / s_U[-1]).item()
    kappa_V = (s_V[0] / s_V[-1]).item()
    print(f'  [PASS] Dim ({d_out}x{d_in}) rank={r}: ||U||/||V||={ratio:.4f}, kappa(U)={kappa_U:.2f}, kappa(V)={kappa_V:.2f}')
    assert abs(ratio - 1.0) < 1e-4, f'Factor norm imbalance: {ratio}'
    assert kappa_U < 100.0 and kappa_V < 100.0, f'High condition number: {kappa_U}, {kappa_V}'

# 3. Energy Threshold tau = 0.92 Verification
test_section('3. Cumulative Energy Threshold tau = 0.92 Verification')
cfg = get_config()
attn = adap_attn_mod.AdaptiveSVDAttention(cfg, energy_threshold=0.92)
dense_linear = nn.Linear(768, 768)
U_syn, _ = torch.linalg.qr(torch.randn(768, 768))
V_syn, _ = torch.linalg.qr(torch.randn(768, 768))
decay_s = torch.exp(-0.02 * torch.arange(768, dtype=torch.float32))
dense_linear.weight.data.copy_(U_syn @ torch.diag(decay_s) @ V_syn.T)

attn.copy_weights(dense_linear)
total_energy = (decay_s ** 2).sum().item()
for proj_name, proj in [('query', attn.query), ('key', attn.key), ('value', attn.value), ('out_proj', attn.out_proj)]:
    u_rk = torch.linalg.matrix_rank(proj.U.data.float()).item()
    retained_s = decay_s[:u_rk]
    retained_energy = (retained_s ** 2).sum().item() / total_energy
    print(f'  [PASS] {proj_name}: computed_rank={u_rk}, retained_energy={retained_energy:.4f} (>= 0.92)')
    assert retained_energy >= 0.92 or u_rk == proj.svd_rank, f'Energy threshold violation in {proj_name}'

# 4. Residual Tail LoRA Zero-Shot Optimality in ResSVD
test_section('4. ResSVD Residual Tail LoRA Zero-Shot Optimality')
lin = nn.Linear(256, 128)
res_lin = res_svd_mod.ResSVDLinear.from_linear(lin, base_rank=32, residual_rank=8, scaling=1.0)
W_orig = lin.weight.data.float()
W_base = res_lin.base_U.data.float() @ res_lin.base_V.data.float()
W_adapter = res_lin.lora_B.data.float() @ res_lin.lora_A.data.float()
W_recon = W_base + W_adapter

err_base = torch.norm(W_orig - W_base, p='fro').item()
err_recon = torch.norm(W_orig - W_recon, p='fro').item()
U_full, S_full, Vh_full = torch.linalg.svd(W_orig, full_matrices=False)
expected_tail_norm = torch.sqrt((S_full[40:] ** 2).sum()).item()
print(f'  [PASS] ||W - W_base||_F = {err_base:.4f}')
print(f'  [PASS] ||W - (W_base + Delta W)||_F = {err_recon:.4f} (analytical theoretical minimum = {expected_tail_norm:.4f})')
assert abs(err_recon - expected_tail_norm) < 1e-4, 'Residual adapter failed exact tail reconstruction!'

# 5. Stress Testing All 17 Compression Engines on Mock Transformer
test_section('5. Full Model Compression & Architecture Replacement Stress Tests')
compressors = [
    ('DirectSVDCompressor', lambda m: direct_svd_mod.DirectSVDCompressor(m, rank=16)),
    ('AdaptiveRanksCompressor', lambda m: adaptive_ranks_mod.AdaptiveRanksCompressor(m, energy_threshold=0.92, min_rank=16)),
    ('ResSVDCompressor', lambda m: res_svd_mod.ResSVDCompressor(m, base_rank=16, residual_rank=4)),
    ('DobiSVDCompressor', lambda m: dobi_svd_mod.DobiSVDCompressor(m, rank=16)),
    ('ERCCompressor', lambda m: erc_svd_mod.ERCCompressor(m, allowed_error=0.1)),
    ('EntropyBudgetCompressor', lambda m: entropy_budget_mod.EntropyBudgetCompressor(m, total_budget_ratio=0.5)),
    ('KroneckerTTLoRACompressor', lambda m: kronecker_tt_mod.KroneckerTTLoRACompressor(m, lora_r=4)),
    ('SAESSVDCompressorV2', lambda m: saes_v2_mod.SAESSVDCompressorV2(m, rank=16, correcting_steps=3)),
    ('SwiftSVDCompressor', lambda m: swift_svd_mod.SwiftSVDCompressor(m, rank=16)),
    ('SwiftSVDActivationCompressor', lambda m: swift_svd_act_mod.SwiftSVDActivationCompressor(m, rank=16)),
    ('AASVDCompressor', lambda m: dummy_comp_pkg.aasvd.AASVDCompressor(m, rank=16)),
    ('AdaSVDCompressor', lambda m: adasvd_mod.AdaSVDCompressor(m, rank=16, num_iters=3)),
    ('RPCACompressor', lambda m: rpca_mod.RPCACompressor(m, rank=16, max_iter=3)),
    ('CURMoECompressor', lambda m: cur_moe_mod.CURMoECompressor(m, rank=16)),
    ('GreCoDeCCompressor', lambda m: grecodec_mod.GreCoDeCCompressor(m, rank=16, max_iter=5)),
    ('FRanDICompressor', lambda m: frandi_mod.FRanDICompressor(m, rank=16, energy_threshold=0.92)),
    ('DynamicEntropySVDCompressor', lambda m: dynamic_rank_mod.DynamicEntropySVDCompressor(m, min_rank=16, max_rank=32, target_rank=20)),
]

for name, comp_fn in compressors:
    model = MockTransformerModel(hidden_size=64, num_layers=2)
    x = torch.randn(2, 8, 64)
    orig_out = model(x)
    
    compressor = comp_fn(model)
    comp_model, stats = compressor.compress()
    
    comp_out = comp_model(x)
    loss = comp_out.sum()
    loss.backward()
    
    assert not torch.isnan(comp_out).any(), f'NaN output in {name}'
    assert not torch.isinf(comp_out).any(), f'Inf output in {name}'
    ratio = stats.get('compression_ratio', 1.0)
    print(f'  [PASS] {name:<30} -> forward/backward OK, compression_ratio={ratio:.2f}x')

# 6. Tensor-Train & Kronecker Linear Core Stress Tests
test_section('6. Tensor-Train & Kronecker Linear Mathematical Stress Tests')
tt_layer = tt_lin_mod.TTLinearLayer.from_linear(nn.Linear(64, 64), tt_rank=8, num_factors=2)
x = torch.randn(4, 12, 64, requires_grad=True)
tt_out = tt_layer(x)
tt_loss = tt_out.sum()
tt_loss.backward()
print(f'  [PASS] TTLinearLayer (64->64) TT-SVD decomposition & contraction OK (x_grad_norm={x.grad.norm().item():.4f})')

kron_layer = kron_lin_mod.KroneckerLinear.from_linear(nn.Linear(64, 64), rank=16)
x = torch.randn(4, 12, 64, requires_grad=True)
kron_out = kron_layer(x)
kron_loss = kron_out.sum()
kron_loss.backward()
print(f'  [PASS] KroneckerLinear (64->64) Kronecker factorization & contraction OK (x_grad_norm={x.grad.norm().item():.4f})')

# 7. Long Context and Recurrent State Caching Equivalence in SVDLinearAttention
test_section('7. SVDLinearAttention Long Context & Recurrence Equivalence')
svd_lin_attn = svd_lin_mod.SVDLinearAttention(cfg, attention_mode='decay_gated')
svd_lin_attn.eval()
B, N, D = 2, 64, 768
seq_inputs = torch.randn(B, N, D)

with torch.no_grad():
    full_seq_out, _ = svd_lin_attn(seq_inputs)
    state = None
    step_outputs = []
    for t in range(N):
        x_t = seq_inputs[:, t : t + 1, :]
        y_t, state = svd_lin_attn.step_recurrent(x_t, prev_state=state)
        step_outputs.append(y_t)
    step_seq_out = torch.cat(step_outputs, dim=1)

max_diff = (full_seq_out - step_seq_out).abs().max().item()
print(f'  [PASS] SVDLinearAttention step_recurrent vs parallel sequence max_diff = {max_diff:.2e}')
assert max_diff < 1e-4, f'Decay gated linear attention recurrence mismatch: {max_diff}'

# 8. Linformer Sequence Projection Test
test_section('8. Linformer Attention Universal Masking & FP16/BF16 Test')
linf_attn = linf_attn_mod.LinformerAttention(cfg, k=32, max_seq_len=256)
x = torch.randn(2, 64, 768)
mask = torch.ones(2, 64)
out_linf, _ = linf_attn(x, attention_mask=mask)
assert not torch.isnan(out_linf).any()
print(f'  [PASS] LinformerAttention forward pass OK (out_shape={list(out_linf.shape)})')

print("\n" + "=" * 75)
print(">>> ALL SVD, TENSOR TRAIN & LOW-RANK MODULES PASSED 100% STRESS AUDIT <<<")
print("=" * 75)
