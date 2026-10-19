#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Verify my 5 modules module.
"""

import sys
import time
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neuralops.core.cuda_kernels._fallback import force_kernel_backend

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Executing benchmarks on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")


def time_op(fn, warmup=5, iters=20):
    """Execute time op routine."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


results = {}

print("\n" + "=" * 60)
print("1. Benchmarking & Verifying MODULE gdn_scan")
print("=" * 60)
from neuralops.core.cuda_kernels.gdn_scan import (
    chunked_delta_gate_scan,
    chunked_delta_gate_scan_pytorch_ref,
)

B, H, L, D = 4, 8, 128, 64
torch.manual_seed(42)
q = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1
k = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1
v = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1
alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32)) * 0.95
beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32)) * 0.95

with force_kernel_backend("cuda"):
    out_cuda = chunked_delta_gate_scan(q, k, v, alpha, beta)
out_ref = chunked_delta_gate_scan_pytorch_ref(q, k, v, alpha, beta)
diff_gdn = (out_cuda - out_ref).abs().max().item()
assert diff_gdn < 1e-3, f"gdn_scan diff too large: {diff_gdn}"

with force_kernel_backend("cuda"):
    cuda_ms = time_op(lambda: chunked_delta_gate_scan(q, k, v, alpha, beta))
with force_kernel_backend("pytorch"):
    ref_ms = time_op(lambda: chunked_delta_gate_scan_pytorch_ref(q, k, v, alpha, beta))
speedup = ref_ms / max(cuda_ms, 1e-5)

q_large = torch.randn(B, H, 16, 128, device=device, dtype=torch.float32) * 0.1
k_large = torch.randn(B, H, 16, 128, device=device, dtype=torch.float32) * 0.1
v_large = torch.randn(B, H, 16, 128, device=device, dtype=torch.float32) * 0.1
alpha_l = torch.sigmoid(torch.randn(B, H, 16, device=device, dtype=torch.float32)) * 0.95
beta_l = torch.sigmoid(torch.randn(B, H, 16, device=device, dtype=torch.float32)) * 0.95
out_fallback = chunked_delta_gate_scan(q_large, k_large, v_large, alpha_l, beta_l)
ref_large = chunked_delta_gate_scan_pytorch_ref(q_large, k_large, v_large, alpha_l, beta_l)
diff_fb = (out_fallback - ref_large).abs().max().item()
assert diff_fb < 1e-3, f"gdn_scan fallback diff too large: {diff_fb}"

results["gdn_scan"] = {
    "status": "PASS",
    "cuda_ms": cuda_ms,
    "ref_ms": ref_ms,
    "speedup": speedup,
    "max_diff": diff_gdn,
}
print(
    f"gdn_scan: CUDA={cuda_ms:.3f} ms, Ref={ref_ms:.3f} ms, Speedup={speedup:.2f}x, MaxDiff={diff_gdn:.2e}, Fallback=OK"
)

print("\n" + "=" * 60)
print("2. Benchmarking & Verifying MODULE linear_attention")
print("=" * 60)
from neuralops.core.cuda_kernels.linear_attention import (
    linear_attention,
    linear_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.linear_attention.linear_attn_autograd import _get_module as get_linear_mod

B, H, L, D = 4, 8, 128, 64
q = torch.nn.functional.elu(torch.randn(B, H, L, D, device=device, dtype=torch.float32)) + 1.0
k = torch.nn.functional.elu(torch.randn(B, H, L, D, device=device, dtype=torch.float32)) + 1.0
v = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1

with force_kernel_backend("cuda"):
    out_cuda = linear_attention(q, k, v)
out_ref = linear_attention_pytorch_ref(q, k, v)
diff_lin = (out_cuda - out_ref).abs().max().item()
assert diff_lin < 1e-3, f"linear_attention diff too large: {diff_lin}"

with force_kernel_backend("cuda"):
    cuda_ms = time_op(lambda: linear_attention(q, k, v))
with force_kernel_backend("pytorch"):
    ref_ms = time_op(lambda: linear_attention_pytorch_ref(q, k, v))
speedup = ref_ms / max(cuda_ms, 1e-5)

q_large = torch.nn.functional.elu(torch.randn(B, H, 16, 128, device=device, dtype=torch.float32)) + 1.0
k_large = torch.nn.functional.elu(torch.randn(B, H, 16, 128, device=device, dtype=torch.float32)) + 1.0
v_large = torch.randn(B, H, 16, 128, device=device, dtype=torch.float32) * 0.1
out_fallback = linear_attention(q_large, k_large, v_large)
ref_large = linear_attention_pytorch_ref(q_large, k_large, v_large)
diff_fb = (out_fallback - ref_large).abs().max().item()
assert diff_fb < 1e-3

try:
    mod_lin = get_linear_mod()
    mod_lin.linear_attention_fwd(q_large, k_large, v_large)
    raise RuntimeError("linear_attention_fwd did not trigger TORCH_CHECK on D_K > 64")
except RuntimeError as e:
    assert "exceeds max 64" in str(e)

results["linear_attention"] = {
    "status": "PASS",
    "cuda_ms": cuda_ms,
    "ref_ms": ref_ms,
    "speedup": speedup,
    "max_diff": diff_lin,
}
print(
    f"linear_attention: CUDA={cuda_ms:.3f} ms, Ref={ref_ms:.3f} ms, Speedup={speedup:.2f}x, MaxDiff={diff_lin:.2e}, Fallback=OK, TORCH_CHECK=OK"
)

print("\n" + "=" * 60)
print("3. Benchmarking & Verifying MODULE gdn_delta_rule")
print("=" * 60)
from neuralops.core.cuda_kernels.gdn_delta_rule import (
    gdn_delta_rule,
    gdn_delta_rule_pytorch_ref,
)
from neuralops.core.cuda_kernels.gdn_delta_rule.gdn_delta_rule_autograd import _get_module as get_gdn_delta_mod

B, H, L, D = 4, 8, 64, 64
q = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1
k = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1
v = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1
beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))

with force_kernel_backend("cuda"):
    out_cuda = gdn_delta_rule(q, k, v, beta)
out_ref = gdn_delta_rule_pytorch_ref(q, k, v, beta)
diff_delta = (out_cuda - out_ref).abs().max().item()
assert diff_delta < 1e-3, f"gdn_delta_rule diff too large: {diff_delta}"

with force_kernel_backend("cuda"):
    cuda_ms = time_op(lambda: gdn_delta_rule(q, k, v, beta))
with force_kernel_backend("pytorch"):
    ref_ms = time_op(lambda: gdn_delta_rule_pytorch_ref(q, k, v, beta))
speedup = ref_ms / max(cuda_ms, 1e-5)

D_large = 128
q_128 = torch.randn(2, 4, 16, D_large, device=device, dtype=torch.float32) * 0.1
k_128 = torch.randn(2, 4, 16, D_large, device=device, dtype=torch.float32) * 0.1
v_128 = torch.randn(2, 4, 16, D_large, device=device, dtype=torch.float32) * 0.1
beta_128 = torch.sigmoid(torch.randn(2, 4, 16, device=device, dtype=torch.float32))
with force_kernel_backend("cuda"):
    out_128 = gdn_delta_rule(q_128, k_128, v_128, beta_128)
ref_128 = gdn_delta_rule_pytorch_ref(q_128, k_128, v_128, beta_128)
diff_128 = (out_128 - ref_128).abs().max().item()
assert diff_128 < 1e-3, f"D=128 diff too large: {diff_128}"

try:
    mod_delta = get_gdn_delta_mod()
    q_huge = torch.randn(1, 1, 1, 160, device=device, dtype=torch.float32)
    k_huge = torch.randn(1, 1, 1, 160, device=device, dtype=torch.float32)
    v_huge = torch.randn(1, 1, 1, 160, device=device, dtype=torch.float32)
    beta_huge = torch.sigmoid(torch.randn(1, 1, 1, device=device, dtype=torch.float32))
    mod_delta.gdn_delta_rule_fwd(q_huge, k_huge, v_huge, beta_huge)
    raise RuntimeError("gdn_delta_rule_fwd did not trigger TORCH_CHECK on >96KB")
except RuntimeError as e:
    assert "max 96 KB" in str(e)

results["gdn_delta_rule"] = {
    "status": "PASS",
    "cuda_ms": cuda_ms,
    "ref_ms": ref_ms,
    "speedup": speedup,
    "max_diff": diff_delta,
}
print(
    f"gdn_delta_rule: CUDA={cuda_ms:.3f} ms, Ref={ref_ms:.3f} ms, Speedup={speedup:.2f}x, MaxDiff={diff_delta:.2e}, Shmem_128=OK, TORCH_CHECK=OK"
)

print("\n" + "=" * 60)
print("4. Benchmarking & Verifying MODULE svd_prefix_scan")
print("=" * 60)
from neuralops.core.cuda_kernels.svd_prefix_scan import (
    svd_prefix_scan,
    svd_prefix_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.svd_prefix_scan.svd_prefix_scan_autograd import _get_module as get_svd_mod

B, H, L, D = 4, 8, 64, 64
q = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1
k = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1
v = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.1

with force_kernel_backend("cuda"):
    out_cuda = svd_prefix_scan(q, k, v)
out_ref = svd_prefix_scan_pytorch_ref(q, k, v)
diff_svd = (out_cuda - out_ref).abs().max().item()
assert diff_svd < 1e-3, f"svd_prefix_scan diff too large: {diff_svd}"

with force_kernel_backend("cuda"):
    cuda_ms = time_op(lambda: svd_prefix_scan(q, k, v))
with force_kernel_backend("pytorch"):
    ref_ms = time_op(lambda: svd_prefix_scan_pytorch_ref(q, k, v))
speedup = ref_ms / max(cuda_ms, 1e-5)

D_large = 128
q_128 = torch.randn(2, 4, 16, D_large, device=device, dtype=torch.float32) * 0.1
k_128 = torch.randn(2, 4, 16, D_large, device=device, dtype=torch.float32) * 0.1
v_128 = torch.randn(2, 4, 16, D_large, device=device, dtype=torch.float32) * 0.1
with force_kernel_backend("cuda"):
    out_128 = svd_prefix_scan(q_128, k_128, v_128)
ref_128 = svd_prefix_scan_pytorch_ref(q_128, k_128, v_128)
diff_128 = (out_128 - ref_128).abs().max().item()
assert diff_128 < 1e-3, f"D=128 diff too large: {diff_128}"

try:
    mod_svd = get_svd_mod()
    q_huge = torch.randn(1, 1, 1, 160, device=device, dtype=torch.float32)
    k_huge = torch.randn(1, 1, 1, 160, device=device, dtype=torch.float32)
    v_huge = torch.randn(1, 1, 1, 160, device=device, dtype=torch.float32)
    mod_svd.svd_prefix_scan_fwd(q_huge, k_huge, v_huge, 1e-4)
    raise RuntimeError("svd_prefix_scan_fwd did not trigger TORCH_CHECK on >96KB")
except RuntimeError as e:
    assert "max 96 KB" in str(e)

results["svd_prefix_scan"] = {
    "status": "PASS",
    "cuda_ms": cuda_ms,
    "ref_ms": ref_ms,
    "speedup": speedup,
    "max_diff": diff_svd,
}
print(
    f"svd_prefix_scan: CUDA={cuda_ms:.3f} ms, Ref={ref_ms:.3f} ms, Speedup={speedup:.2f}x, MaxDiff={diff_svd:.2e}, Shmem_128=OK, TORCH_CHECK=OK"
)

print("\n" + "=" * 60)
print("5. Benchmarking & Verifying MODULE mamba_block")
print("=" * 60)
from neuralops.core.cuda_kernels.mamba_block import (
    mamba_block,
    mamba_block_ref,
)

B, D, S = 4, 128, 16
u = torch.randn(B, D, device=device, dtype=torch.float32) * 0.1
delta = torch.randn(B, D, device=device, dtype=torch.float32) * 0.1
A = torch.randn(D, S, device=device, dtype=torch.float32) * 0.1
B_mat = torch.randn(B, S, device=device, dtype=torch.float32) * 0.1
C_mat = torch.randn(B, S, device=device, dtype=torch.float32) * 0.1
prev_state = torch.randn(B, D, S, device=device, dtype=torch.float32) * 0.1

with force_kernel_backend("cuda"):
    ret_cuda = mamba_block(u, delta, A, B_mat, C_mat, prev_state)
assert isinstance(ret_cuda, (tuple, list)) and len(ret_cuda) == 2, f"Expected (out, next_state), got {type(ret_cuda)}"
out_cuda, next_state_cuda = ret_cuda

ret_ref = mamba_block_ref(u, delta, A, B_mat, C_mat, prev_state)
out_ref, next_state_ref = ret_ref

diff_out = (out_cuda - out_ref).abs().max().item()
diff_state = (next_state_cuda - next_state_ref).abs().max().item()
assert diff_out < 1e-3, f"mamba_block out diff too large: {diff_out}"
assert diff_state < 1e-3, f"mamba_block next_state diff too large: {diff_state}"
assert not torch.allclose(next_state_cuda, prev_state), "next_state was not updated!"
assert not next_state_cuda.isnan().any(), "next_state contains NaNs!"

with force_kernel_backend("cuda"):
    cuda_ms = time_op(lambda: mamba_block(u, delta, A, B_mat, C_mat, prev_state))
with force_kernel_backend("pytorch"):
    ref_ms = time_op(lambda: mamba_block_ref(u, delta, A, B_mat, C_mat, prev_state))
speedup = ref_ms / max(cuda_ms, 1e-5)

results["mamba_block"] = {
    "status": "PASS",
    "cuda_ms": cuda_ms,
    "ref_ms": ref_ms,
    "speedup": speedup,
    "max_diff": max(diff_out, diff_state),
}
print(
    f"mamba_block: CUDA={cuda_ms:.3f} ms, Ref={ref_ms:.3f} ms, Speedup={speedup:.2f}x, MaxDiff={max(diff_out, diff_state):.2e}, NextStateUpdated=OK"
)

print("\n" + "=" * 70)
print("SUMMARY OF BENCHMARKS & VERIFICATION ON NVIDIA RTX 4070:")
print(f"{'Module':<20} | {'Status':<6} | {'CUDA (ms)':<10} | {'Ref (ms)':<10} | {'Speedup':<8} | {'Max Diff':<10}")
print("-" * 70)
for mod, info in results.items():
    print(
        f"{mod:<20} | {info['status']:<6} | {info['cuda_ms']:<10.4f} | {info['ref_ms']:<10.4f} | {info['speedup']:<7.2f}x | {info['max_diff']:<10.2e}"
    )
print("=" * 70)
