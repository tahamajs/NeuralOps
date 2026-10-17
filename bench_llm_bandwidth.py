import sys
import os
import math
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")
torch.cuda.set_device(device)

PEAK_BW_GBPS = 256.0

def time_fn(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

from neuralops.core.cuda_kernels.fused_dropout_add.dropout_add_autograd import _get_module as get_fda, fused_dropout_add_pytorch_ref
from neuralops.core.cuda_kernels.fused_norm_linear.norm_linear_autograd import _get_module as get_fnl, fused_norm_linear_pytorch_ref
from neuralops.core.cuda_kernels.rms_variance_match.rms_variance_match_autograd import _get_module as get_rvm, rms_variance_match_pytorch_ref
from neuralops.core.cuda_kernels.aegis_gate_norm.aegis_gate_norm_autograd import _get_module as get_agn, aegis_gate_norm_pytorch_ref

mod_fda = get_fda()
mod_fnl = get_fnl()
mod_rvm = get_rvm()
mod_agn = get_agn()

configs = [
    (4, 512, 2048),
    (8, 512, 2048),
    (4, 1024, 2048),
    (8, 1024, 2048),
    (4, 512, 4096),
    (8, 512, 4096),
    (16, 512, 2048),
    (4, 1024, 4096),
]

print("="*125)
print("Realistic LLM Training Batch Configurations Benchmark (RTX 4070 Ada Lovelace SM 8.9)")
print(f"Hardware Peak Memory Bandwidth: {PEAK_BW_GBPS:.1f} GB/s")
print("="*125)

results = []

for B, S, H in configs:
    N = B * S
    dtype = torch.float16
    x = torch.randn(N, H, device=device, dtype=dtype)
    res = torch.randn(N, H, device=device, dtype=dtype)
    bytes_fda = 3 * N * H * 2
    
    t_fda_c = time_fn(lambda: mod_fda.fused_dropout_add_fwd(x, res, 0.1, False, 42))
    t_fda_p = time_fn(lambda: fused_dropout_add_pytorch_ref(x, res, 0.1, False))
    bw_fda = (bytes_fda / (t_fda_c * 1e-3)) / 1e9
    sat_fda = (bw_fda / PEAK_BW_GBPS) * 100.0
    spd_fda = t_fda_p / t_fda_c
    
    results.append({
        'kernel': 'fused_dropout_add',
        'cfg': f"B={B}, S={S}, H={H}",
        'cuda_ms': t_fda_c,
        'ref_ms': t_fda_p,
        'speedup': spd_fda,
        'bw_gbps': bw_fda,
        'sat_pct': sat_fda
    })

    y = torch.randn(N, H, device=device, dtype=dtype)
    v = torch.randn(N, H, device=device, dtype=dtype)
    bytes_rvm = 3 * N * H * 2
    
    t_rvm_c = time_fn(lambda: mod_rvm.rms_variance_match_fwd(y, v, 0.25, 4.0))
    t_rvm_p = time_fn(lambda: rms_variance_match_pytorch_ref(y, v, 0.25, 4.0))
    bw_rvm = (bytes_rvm / (t_rvm_c * 1e-3)) / 1e9
    sat_rvm = (bw_rvm / PEAK_BW_GBPS) * 100.0
    spd_rvm = t_rvm_p / t_rvm_c

    results.append({
        'kernel': 'rms_variance_match',
        'cfg': f"B={B}, S={S}, H={H}",
        'cuda_ms': t_rvm_c,
        'ref_ms': t_rvm_p,
        'speedup': spd_rvm,
        'bw_gbps': bw_rvm,
        'sat_pct': sat_rvm
    })

    b0 = torch.randn(N, H, device=device, dtype=dtype)
    b1 = torch.randn(N, H, device=device, dtype=dtype)
    w_gate = torch.randn(N, 2, device=device, dtype=torch.float32)
    bytes_agn = (3 * N * H * 2) + (N * 2 * 4)
    
    t_agn_c = time_fn(lambda: mod_agn.aegis_gate_norm_2branch_fwd(b0, b1, w_gate, 1e-5))
    t_agn_p = time_fn(lambda: aegis_gate_norm_pytorch_ref(b0, b1, w_gate, 1e-5))
    bw_agn = (bytes_agn / (t_agn_c * 1e-3)) / 1e9
    sat_agn = (bw_agn / PEAK_BW_GBPS) * 100.0
    spd_agn = t_agn_p / t_agn_c

    results.append({
        'kernel': 'aegis_gate_norm',
        'cfg': f"B={B}, S={S}, H={H}",
        'cuda_ms': t_agn_c,
        'ref_ms': t_agn_p,
        'speedup': spd_agn,
        'bw_gbps': bw_agn,
        'sat_pct': sat_agn
    })

    w_lin = torch.randn(H, H, device=device, dtype=dtype)
    bytes_fnl = (2 * N * H + H * H) * 2
    empty_b = torch.empty(0, device=device, dtype=dtype)
    
    t_fnl_c = time_fn(lambda: mod_fnl.fused_norm_linear_fwd(x, w_lin, empty_b, 1e-6)[0])
    t_fnl_p = time_fn(lambda: fused_norm_linear_pytorch_ref(x, w_lin, None, 1e-6))
    bw_fnl = (bytes_fnl / (t_fnl_c * 1e-3)) / 1e9
    sat_fnl = (bw_fnl / PEAK_BW_GBPS) * 100.0
    spd_fnl = t_fnl_p / t_fnl_c

    results.append({
        'kernel': 'fused_norm_linear',
        'cfg': f"B={B}, S={S}, H={H}",
        'cuda_ms': t_fnl_c,
        'ref_ms': t_fnl_p,
        'speedup': spd_fnl,
        'bw_gbps': bw_fnl,
        'sat_pct': sat_fnl
    })

    print(f"Config B={B}, S={S}, H={H} completed.")

h_k = "Kernel Name"
h_cfg = "Configuration"
h_c = "CUDA (ms)"
h_p = "PyTorch (ms)"
h_s = "Speedup"
h_bw = "Bandwidth"
h_sat = "Saturation %"

print("\n" + "="*125)
print(f"| {h_k:20} | {h_cfg:18} | {h_c:10} | {h_p:12} | {h_s:8} | {h_bw:13} | {h_sat:12} |")
print("|" + "-"*22 + "|" + "-"*20 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*10 + "|" + "-"*15 + "|" + "-"*14 + "|")
for r in results:
    k = r['kernel']
    cfg = r['cfg']
    c_ms = r['cuda_ms']
    p_ms = r['ref_ms']
    spd = r['speedup']
    bw = r['bw_gbps']
    sat = r['sat_pct']
    print(f"| {k:20} | {cfg:18} | {c_ms:8.4f}ms | {p_ms:10.4f}ms | {spd:7.2f}x | {bw:8.2f} GB/s | {sat:10.1f}% |")
print("="*125)
