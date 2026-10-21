#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import torch
import pytest
import torch.nn.functional as F
from neuralops.core.cuda_kernels import (
    mamba_block, mamba_svd_flash, spectral_mamba, svd_prefix_scan,
    gdn_scan, gdn_delta_rule, orion, selective_ssm, selective_scan,
    paged_recurrent_step, state_update
)

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for the forward/backward kernel audit", allow_module_level=True)

device = "cuda"
torch.manual_seed(42)

results = []

def verify_module(name, run_cuda, run_ref, tensors_cuda, tensors_ref, tol=0.01):
    out_c = run_cuda(*tensors_cuda)
    if isinstance(out_c, (tuple, list)):
        loss_c = sum(o.sum() for o in out_c if isinstance(o, torch.Tensor))
    else:
        loss_c = out_c.sum()
    loss_c.backward()

    out_r = run_ref(*tensors_ref)
    if isinstance(out_r, (tuple, list)):
        loss_r = sum(o.sum() for o in out_r if isinstance(o, torch.Tensor))
    else:
        loss_r = out_r.sum()
    loss_r.backward()

    if isinstance(out_c, (tuple, list)):
        fwd_diff = max((c.float() - r.float()).abs().max().item() for c, r in zip(out_c, out_r) if isinstance(c, torch.Tensor))
    else:
        fwd_diff = (out_c.float() - out_r.float()).abs().max().item()

    grad_diffs = []
    has_nan = False
    has_inf = False
    for tc, tr in zip(tensors_cuda, tensors_ref):
        if tc.requires_grad:
            assert tc.grad is not None, "tc.grad is None!"
            if torch.isnan(tc.grad).any().item():
                has_nan = True
            if torch.isinf(tc.grad).any().item():
                has_inf = True
            diff = (tc.grad.float() - tr.grad.float()).abs().max().item()
            grad_diffs.append(diff)
    max_bwd_diff = max(grad_diffs) if grad_diffs else 0.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for t in tensors_cuda: t.grad = None
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        oc = run_cuda(*tensors_cuda)
        lc = sum(o.sum() for o in oc if isinstance(o, torch.Tensor)) if isinstance(oc, (tuple, list)) else oc.sum()
        lc.backward()
        for t in tensors_cuda: t.grad = None
    end.record()
    torch.cuda.synchronize()
    cuda_time = start.elapsed_time(end) / 10

    for t in tensors_ref: t.grad = None
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        orr = run_ref(*tensors_ref)
        lr = sum(o.sum() for o in orr if isinstance(o, torch.Tensor)) if isinstance(orr, (tuple, list)) else orr.sum()
        lr.backward()
        for t in tensors_ref: t.grad = None
    end.record()
    torch.cuda.synchronize()
    ref_time = start.elapsed_time(end) / 10

    speedup = ref_time / cuda_time if cuda_time > 0 else 1.0
    status = "PASS" if (max_bwd_diff < tol and not has_nan and not has_inf and speedup >= 0.95) else "WARN"

    print(f"{name:22s} | FwdDiff: {fwd_diff:8.2e} | BwdDiff: {max_bwd_diff:8.2e} | NaNs: {has_nan} | Infs: {has_inf} | CUDA: {cuda_time:6.3f}ms | Ref: {ref_time:6.3f}ms | Speedup: {speedup:5.2f}x | {status}")
    return {
        "module": name,
        "fwd_diff": fwd_diff,
        "bwd_diff": max_bwd_diff,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "cuda_ms": cuda_time,
        "ref_ms": ref_time,
        "speedup": speedup,
        "status": status
    }

print("="*110)
print(f"{'MODULE':22s} | {'FWD DIFF':10s} | {'BWD DIFF':10s} | {'NANs':5s} | {'INFs':5s} | {'CUDA (ms)':9s} | {'REF (ms)':9s} | {'SPEEDUP':7s} | STATUS")
print("="*110)

B, D, S = 2, 64, 16
u = torch.randn(B, D, device=device, requires_grad=True)
delta = torch.randn(B, D, device=device).clamp(-5, 5).requires_grad_(True)
A = torch.randn(D, S, device=device).clamp(-5, 0).requires_grad_(True)
B_mat = torch.randn(B, S, device=device, requires_grad=True)
C_mat = torch.randn(B, S, device=device, requires_grad=True)
prev = torch.randn(B, D, S, device=device, requires_grad=True)
tc = [u, delta, A, B_mat, C_mat, prev]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("mamba_block", mamba_block.mamba_block, mamba_block.mamba_block_ref, tc, tr))

B, D, R = 32, 256, 32
u = torch.randn(B, D, device=device, requires_grad=True)
svd_u = torch.randn(D, R, device=device, requires_grad=True)
svd_vt = torch.randn(R, D, device=device, requires_grad=True)
decay = torch.rand(D, device=device, requires_grad=True) * 0.9
tc = [u, svd_u, svd_vt, decay]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("mamba_svd_flash", mamba_svd_flash.mamba_svd_flash, mamba_svd_flash.mamba_svd_flash_ref, tc, tr))

B, N, D = 4, 64, 32
x = torch.randn(B, N, D, device=device, requires_grad=True)
freq = torch.randn(D, device=device, requires_grad=True)
tc = [x, freq]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("spectral_mamba", spectral_mamba.spectral_mamba, spectral_mamba.spectral_mamba_ref, tc, tr))

B, H, L, D = 2, 4, 128, 32
q = torch.randn(B, H, L, D, device=device, requires_grad=True)
k = torch.randn(B, H, L, D, device=device, requires_grad=True)
v = torch.randn(B, H, L, D, device=device, requires_grad=True)
tc = [q, k, v]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("svd_prefix_scan", svd_prefix_scan.svd_prefix_scan, svd_prefix_scan.svd_prefix_scan_pytorch_ref, tc, tr))

B, N, H, D = 2, 64, 4, 32
q = torch.randn(B, N, H, D, device=device, requires_grad=True)
k = torch.randn(B, N, H, D, device=device, requires_grad=True)
v = torch.randn(B, N, H, D, device=device, requires_grad=True)
alpha = torch.sigmoid(torch.randn(B, N, H, device=device)).requires_grad_(True)
beta = torch.sigmoid(torch.randn(B, N, H, device=device)).requires_grad_(True)
tc = [q, k, v, alpha, beta]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("gdn_scan", gdn_scan.chunked_delta_gate_scan, gdn_scan.chunked_delta_gate_scan_pytorch_ref, tc, tr))

B, N, H, D = 2, 64, 4, 32
q = torch.randn(B, N, H, D, device=device, requires_grad=True)
k = torch.randn(B, N, H, D, device=device, requires_grad=True)
v = torch.randn(B, N, H, D, device=device, requires_grad=True)
beta = torch.sigmoid(torch.randn(B, N, H, device=device)).requires_grad_(True)
tc = [q, k, v, beta]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("gdn_delta_rule", gdn_delta_rule.gdn_delta_rule, gdn_delta_rule.gdn_delta_rule_pytorch_ref, tc, tr))

B, H, N, D = 2, 4, 64, 32
q = torch.randn(B, H, N, D, device=device, requires_grad=True)
k = torch.randn(B, H, N, D, device=device, requires_grad=True)
v = torch.randn(B, H, N, D, device=device, requires_grad=True)
tc = [q, k, v]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("orion", orion.orion, orion.orion_ref, tc, tr))

B, N, D = 4, 128, 64
x = torch.randn(B, N, D, device=device, requires_grad=True)
dt = torch.randn(B, N, D, device=device).abs().requires_grad_(True)
A = (-torch.rand(D, device=device)).requires_grad_(True)
B_vec = torch.randn(B, N, D, device=device, requires_grad=True)
C_vec = torch.randn(B, N, D, device=device, requires_grad=True)
tc = [x, dt, A, B_vec, C_vec]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("selective_ssm", selective_ssm.selective_ssm, selective_ssm.selective_ssm_ref, tc, tr))

B, D, L, N = 2, 64, 128, 16
u = torch.randn(B, D, L, device=device, requires_grad=True)
delta = torch.randn(B, D, L, device=device).sigmoid().mul(0.1).requires_grad_(True)
A = (-torch.rand(D, N, device=device) - 0.5).requires_grad_(True)
B_t = torch.randn(B, N, L, device=device, requires_grad=True)
C_t = torch.randn(B, N, L, device=device, requires_grad=True)
D_val = torch.randn(D, device=device, requires_grad=True)
tc = [u, delta, A, B_t, C_t, D_val]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("selective_scan", selective_scan.selective_scan, selective_scan.selective_scan_pytorch_ref, tc, tr))

B, H, D = 2, 4, 32
q = torch.randn(B, H, D, device=device, requires_grad=True)
k = torch.randn(B, H, D, device=device, requires_grad=True)
v = torch.randn(B, H, D, device=device, requires_grad=True)
pool = torch.randn(16, D, D, device=device, requires_grad=True)
tables = torch.zeros(B, dtype=torch.long, device=device)
tc = [q, k, v, pool, tables]
tr = [q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True), pool.detach().clone().requires_grad_(True), tables]
results.append(verify_module("paged_recurrent_step", paged_recurrent_step.paged_recurrent_step, paged_recurrent_step.paged_recurrent_step_pytorch_ref, tc, tr))

B, H, D = 2, 4, 32
q = torch.sigmoid(torch.randn(B, H, D, device=device)).requires_grad_(True)
k = torch.sigmoid(torch.randn(B, H, D, device=device)).requires_grad_(True)
v = torch.randn(B, H, D, device=device, requires_grad=True)
kv_state = torch.randn(B, H, D, D, device=device, requires_grad=True)
z_state = torch.randn(B, H, D, device=device, requires_grad=True)
tc = [q, k, v, kv_state, z_state]
tr = [t.detach().clone().requires_grad_(True) for t in tc]
results.append(verify_module("state_update", state_update.recurrent_linear_step, state_update.recurrent_linear_step_pytorch_ref, tc, tr))
print("="*110)
