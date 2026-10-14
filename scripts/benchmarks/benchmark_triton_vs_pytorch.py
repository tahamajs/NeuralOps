import time
import torch

from neuralops.attentions.hybrid.positive_kernel_linear_attention import (
    PositiveKernelLinearAttention,
)
from neuralops.attentions.modern.alibi import ALiBiAttention
import neuralops.core.triton_kernels as tk
import neuralops.core.triton_kernels.elementwise as elem
from neuralops.core.triton_kernels.elementwise import (
    fused_alibi_bias_pytorch,
    fused_alibi_bias_triton,
    fused_dynamic_conv1d_pytorch,
    fused_dynamic_conv1d_triton,
)
import neuralops.core.triton_kernels.selective_scan as tss
from neuralops.core.triton_kernels.sparse_attn import (
    _pytorch_block_sparse_reference,
    triton_block_sparse_attention,
)
from neuralops.core.selective_scan import selective_scan_chunked

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True

print("=" * 80)
print("  NEURALOPS TRITON vs PYTORCH COMPREHENSIVE BENCHMARK & NUMERICAL QA")
print(f"  Device: {torch.cuda.get_device_name(0)}")
print("=" * 80 + "\n")


def time_fn(fn, warmups=10, iters=50):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


results = []

# --- 1. ALiBi Relative Position Distance Bias ---
print("--- 1. ALiBi Relative Position Bias (H=16, N_q=2048, N_k=2048) ---")
slopes = torch.linspace(0.1, 0.9, 16, device=device)
N_q, N_k = 2048, 2048

t_pt = time_fn(lambda: fused_alibi_bias_pytorch(slopes, N_q, N_k, past_len=0, is_causal=True))
t_tri = time_fn(lambda: fused_alibi_bias_triton(slopes, N_q, N_k, past_len=0, is_causal=True))
out_pt = fused_alibi_bias_pytorch(slopes, N_q, N_k, past_len=0, is_causal=True)
out_tri = fused_alibi_bias_triton(slopes, N_q, N_k, past_len=0, is_causal=True)
diff = (out_pt - out_tri).abs().max().item()
speedup = t_pt / t_tri
print(f"PyTorch Latency : {t_pt:.3f} ms")
print(f"Triton Latency  : {t_tri:.3f} ms")
print(f"Speedup         : {speedup:.2f}x")
print(f"Max Abs Diff    : {diff:.6e}")
assert diff < 1e-5
results.append(("ALiBi Relative Bias", t_pt, t_tri, speedup, diff))

# --- 2. Selective Scan Kernel (Mamba State-Space Model) ---
print("\n--- 2. Selective Scan Kernel (B=4, L=1024, D=64, N=16, FP16) ---")
B, L, D, N = 4, 1024, 64, 16
u = torch.randn(B, D, L, device=device, dtype=torch.float16)
delta = torch.randn(B, D, L, device=device, dtype=torch.float16)
A = -torch.rand(D, N, device=device, dtype=torch.float32)
B_t = torch.randn(B, N, L, device=device, dtype=torch.float16)
C_t = torch.randn(B, N, L, device=device, dtype=torch.float16)
D_p = torch.randn(D, device=device, dtype=torch.float32)

t_pt = time_fn(lambda: selective_scan_chunked(u, delta, A, B_t, C_t, D=D_p, delta_softplus=True))
t_tri = time_fn(lambda: tss.selective_scan_fn(u, delta, A, B_t, C_t, D=D_p, delta_softplus=True))
out_pt = selective_scan_chunked(u, delta, A, B_t, C_t, D=D_p, delta_softplus=True)
out_tri = tss.selective_scan_fn(u, delta, A, B_t, C_t, D=D_p, delta_softplus=True)
diff = (out_pt - out_tri).abs().max().item()
speedup = t_pt / t_tri
print(f"PyTorch Latency : {t_pt:.3f} ms")
print(f"Triton Latency  : {t_tri:.3f} ms")
print(f"Speedup         : {speedup:.2f}x")
print(f"Max Abs Diff    : {diff:.6e}")
assert diff < 1e-2
results.append(("Selective Scan (Mamba SSM)", t_pt, t_tri, speedup, diff))

# --- 3. Positive Kernel Linear Attention ---
print("\n--- 3. Positive Kernel Linear Attention (B=4, L=512, D=256, H=8, FP16) ---")
pk_attn = PositiveKernelLinearAttention(hidden_size=256, num_heads=8).to(
    device=device, dtype=torch.float16
)
x_pk = torch.randn(4, 512, 256, device=device, dtype=torch.float16)

# Time Triton
t_tri = time_fn(lambda: pk_attn(x_pk))
out_tri, _ = pk_attn(x_pk)

# Force PyTorch fallback
orig_td = tk.triton_dispatch


def mock_td(*args, **kwargs):
    raise RuntimeError("Triton disabled")


tk.triton_dispatch = mock_td
t_pt = time_fn(lambda: pk_attn(x_pk))
out_pt, _ = pk_attn(x_pk)
tk.triton_dispatch = orig_td

diff = (out_tri - out_pt).abs().max().item()
speedup = t_pt / t_tri
print(f"PyTorch Latency : {t_pt:.3f} ms")
print(f"Triton Latency  : {t_tri:.3f} ms")
print(f"Speedup         : {speedup:.2f}x")
print(f"Max Abs Diff    : {diff:.6e}")
results.append(("Positive Kernel Linear Attn", t_pt, t_tri, speedup, diff))

# --- 4. Fast Block Sparse Attention ---
print("\n--- 4. Fast Block Sparse Attention Kernel (B=4, H=8, N=512, D=64, FP16) ---")
q_sp = torch.randn(4, 8, 512, 64, device=device, dtype=torch.float16)
k_sp = torch.randn(4, 8, 512, 64, device=device, dtype=torch.float16)
v_sp = torch.randn(4, 8, 512, 64, device=device, dtype=torch.float16)
n_blocks = 512 // 64
b_mask = torch.tril(torch.ones((n_blocks, n_blocks), device=device, dtype=torch.int32))

t_pt = time_fn(
    lambda: _pytorch_block_sparse_reference(
        q_sp, k_sp, v_sp, b_mask, block_size=64, is_causal=True
    )
)
t_tri = time_fn(
    lambda: triton_block_sparse_attention(
        q_sp, k_sp, v_sp, b_mask, block_size=64, is_causal=True
    )
)
out_pt = _pytorch_block_sparse_reference(
    q_sp, k_sp, v_sp, b_mask, block_size=64, is_causal=True
)
out_tri = triton_block_sparse_attention(
    q_sp, k_sp, v_sp, b_mask, block_size=64, is_causal=True
)
diff = (out_pt - out_tri).abs().max().item()
speedup = t_pt / t_tri
print(f"PyTorch Latency : {t_pt:.3f} ms")
print(f"Triton Latency  : {t_tri:.3f} ms")
print(f"Speedup         : {speedup:.2f}x")
print(f"Max Abs Diff    : {diff:.6e}")
assert diff < 1e-2
results.append(("Block Sparse Attention", t_pt, t_tri, speedup, diff))

# --- 5. Fused Dynamic Conv1D ---
print("\n--- 5. Fused Dynamic Conv1D Kernel (B=4, L=1024, D=128, K=4, FP16) ---")
x_conv = torch.randn(4, 1024, 128, device=device, dtype=torch.float16)
w_conv = torch.randn(128, 4, device=device, dtype=torch.float16)
b_conv = torch.randn(128, device=device, dtype=torch.float16)

t_pt = time_fn(
    lambda: fused_dynamic_conv1d_pytorch(x_conv, w_conv, b_conv, is_causal=True)
)
t_tri = time_fn(
    lambda: fused_dynamic_conv1d_triton(x_conv, w_conv, b_conv, is_causal=True)
)
out_pt = fused_dynamic_conv1d_pytorch(x_conv, w_conv, b_conv, is_causal=True)
out_tri = fused_dynamic_conv1d_triton(x_conv, w_conv, b_conv, is_causal=True)
diff = (out_pt - out_tri).abs().max().item()
speedup = t_pt / t_tri
print(f"PyTorch Latency : {t_pt:.3f} ms")
print(f"Triton Latency  : {t_tri:.3f} ms")
print(f"Speedup         : {speedup:.2f}x")
print(f"Max Abs Diff    : {diff:.6e}")
assert diff < 1e-4
results.append(("Fused Dynamic Conv1D", t_pt, t_tri, speedup, diff))

# Summary table
print("\n" + "=" * 80)
header = f"{'Module / Kernel':<30} | {'PyTorch (ms)':<14} | {'Triton (ms)':<14} | {'Speedup':<9} | {'Max Diff':<10}"
print(header)
print("-" * 80)
for name, pt, tri, sp, df in results:
    row = f"{name:<30} | {pt:>11.3f} ms | {tri:>11.3f} ms | {sp:>7.2f}x | {df:>10.2e}"
    print(row)
print("=" * 80)
