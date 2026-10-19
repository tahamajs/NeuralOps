#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Benchmark spec 9modules module.
"""

import sys
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neuralops.core.cuda_kernels import force_kernel_backend

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "CUDA required for GPU benchmark"


def benchmark(fn, warmup=10, iters=50):
    """Execute benchmark routine."""
    assert callable(fn), "fn must be callable"
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iters


rows = []

from neuralops.core.cuda_kernels.fwht_transform import fwht_transform, fwht_transform_pytorch_ref

x_fwht = torch.randn(64, 512, device=device, dtype=torch.float16)
with force_kernel_backend("cuda"):
    out_cuda_1 = fwht_transform(x_fwht, B_dim=512)
    cuda_ms_1 = benchmark(lambda: fwht_transform(x_fwht, B_dim=512))
with force_kernel_backend("pytorch"):
    out_ref_1 = fwht_transform(x_fwht, B_dim=512)
    ref_ms_1 = benchmark(lambda: fwht_transform_pytorch_ref(x_fwht, B_dim=512))
diff_1 = (out_cuda_1.float() - out_ref_1.float()).abs().max().item()
sp_1 = ref_ms_1 / cuda_ms_1 if cuda_ms_1 > 0 else 0
rows.append(("fwht_transform", cuda_ms_1, ref_ms_1, sp_1, diff_1))

from neuralops.core.cuda_kernels.unified_rope_offset import unified_rope_offset

B, H, N, D = 8, 16, 128, 64
x_rope = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
cos_rope = torch.cos(torch.randn(N + 16, D // 2, device=device, dtype=torch.float32))
sin_rope = torch.sin(torch.randn(N + 16, D // 2, device=device, dtype=torch.float32))
with force_kernel_backend("cuda"):
    out_cuda_2 = unified_rope_offset(x_rope, cos_rope, sin_rope, rotary_dim=64, offset=0)
    cuda_ms_2 = benchmark(lambda: unified_rope_offset(x_rope, cos_rope, sin_rope, rotary_dim=64, offset=0))
with force_kernel_backend("pytorch"):
    out_ref_2 = unified_rope_offset(x_rope, cos_rope, sin_rope, rotary_dim=64, offset=0)
    ref_ms_2 = benchmark(lambda: unified_rope_offset(x_rope, cos_rope, sin_rope, rotary_dim=64, offset=0))
diff_2 = (out_cuda_2 - out_ref_2).abs().max().item()
sp_2 = ref_ms_2 / cuda_ms_2 if cuda_ms_2 > 0 else 0
rows.append(("unified_rope_offset", cuda_ms_2, ref_ms_2, sp_2, diff_2))

from neuralops.core.cuda_kernels.moe_routing import moe_dispatch_gather

B_moe, D_moe = 128, 256
tokens_moe = torch.randn(B_moe, D_moe, device=device, dtype=torch.float16)
gate_logits_moe = torch.randn(B_moe, 8, device=device, dtype=torch.float16)
expert_moe = torch.nn.Linear(D_moe, D_moe, device=device, dtype=torch.float16)
with force_kernel_backend("cuda"):
    out_cuda_3 = moe_dispatch_gather(tokens_moe, gate_logits_moe, expert_moe, top_k=2)
    cuda_ms_3 = benchmark(lambda: moe_dispatch_gather(tokens_moe, gate_logits_moe, expert_moe, top_k=2))
with force_kernel_backend("pytorch"):
    out_ref_3 = moe_dispatch_gather(tokens_moe, gate_logits_moe, expert_moe, top_k=2)
    ref_ms_3 = benchmark(lambda: moe_dispatch_gather(tokens_moe, gate_logits_moe, expert_moe, top_k=2))
diff_3 = (out_cuda_3.float() - out_ref_3.float()).abs().max().item()
sp_3 = ref_ms_3 / cuda_ms_3 if cuda_ms_3 > 0 else 0
rows.append(("moe_routing", cuda_ms_3, ref_ms_3, sp_3, diff_3))

from neuralops.core.cuda_kernels.split_head_norm import split_head_norm, split_head_norm_pytorch_ref

x_shn = torch.randn(256, 128, device=device, dtype=torch.float32)
w_shn = torch.randn(128, device=device, dtype=torch.float32)
with force_kernel_backend("cuda"):
    out_cuda_4 = split_head_norm(x_shn, weight=w_shn)
    cuda_ms_4 = benchmark(lambda: split_head_norm(x_shn, weight=w_shn))
with force_kernel_backend("pytorch"):
    out_ref_4 = split_head_norm_pytorch_ref(x_shn, weight=w_shn)
    ref_ms_4 = benchmark(lambda: split_head_norm_pytorch_ref(x_shn, weight=w_shn))
diff_4 = (out_cuda_4 - out_ref_4).abs().max().item()
sp_4 = ref_ms_4 / cuda_ms_4 if cuda_ms_4 > 0 else 0
rows.append(("split_head_norm", cuda_ms_4, ref_ms_4, sp_4, diff_4))

from neuralops.core.cuda_kernels.paged_attention import paged_attention, paged_attention_pytorch_reference

num_seqs, num_heads, num_kv_heads, head_dim, block_sz, max_blocks = 16, 16, 4, 64, 16, 16
q_pa = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=torch.float32)
num_total_blocks = num_seqs * max_blocks
k_pa = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_sz, device=device, dtype=torch.float32)
v_pa = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_sz, device=device, dtype=torch.float32)
context_lens_pa = torch.randint(16, max_blocks * block_sz, (num_seqs,), device=device, dtype=torch.int32)
block_tables_pa = torch.zeros(num_seqs, max_blocks, device=device, dtype=torch.int32)
for i in range(num_seqs):
    nb = int(context_lens_pa[i].item() + block_sz - 1) // block_sz
    block_tables_pa[i, :nb] = torch.arange(i * max_blocks, i * max_blocks + nb, device=device)
with force_kernel_backend("cuda"):
    out_cuda_5 = paged_attention(q_pa, k_pa, v_pa, block_tables_pa, context_lens_pa, block_size=block_sz)
    cuda_ms_5 = benchmark(
        lambda: paged_attention(q_pa, k_pa, v_pa, block_tables_pa, context_lens_pa, block_size=block_sz)
    )
with force_kernel_backend("pytorch"):
    out_ref_5 = paged_attention_pytorch_reference(
        q_pa, k_pa, v_pa, block_tables_pa, context_lens_pa, block_size=block_sz
    )
    ref_ms_5 = benchmark(
        lambda: paged_attention_pytorch_reference(
            q_pa, k_pa, v_pa, block_tables_pa, context_lens_pa, block_size=block_sz
        )
    )
diff_5 = (out_cuda_5 - out_ref_5).abs().max().item()
sp_5 = ref_ms_5 / cuda_ms_5 if cuda_ms_5 > 0 else 0
rows.append(("paged_attention", cuda_ms_5, ref_ms_5, sp_5, diff_5))

from neuralops.core.cuda_kernels.fused_dropout_add import fused_dropout_add, fused_dropout_add_pytorch_ref

x_da = torch.randn(128, 512, device=device, dtype=torch.float32)
res_da = torch.randn(128, 512, device=device, dtype=torch.float32)
with force_kernel_backend("cuda"):
    out_cuda_6 = fused_dropout_add(x_da, res_da, p=0.0, training=False, seed=42)
    cuda_ms_6 = benchmark(lambda: fused_dropout_add(x_da, res_da, p=0.0, training=False, seed=42))
with force_kernel_backend("pytorch"):
    out_ref_6 = fused_dropout_add_pytorch_ref(x_da, res_da, p=0.0, training=False)
    ref_ms_6 = benchmark(lambda: fused_dropout_add_pytorch_ref(x_da, res_da, p=0.0, training=False))
diff_6 = (out_cuda_6 - out_ref_6).abs().max().item()
sp_6 = ref_ms_6 / cuda_ms_6 if cuda_ms_6 > 0 else 0
rows.append(("fused_dropout_add", cuda_ms_6, ref_ms_6, sp_6, diff_6))

from neuralops.core.cuda_kernels.fused_ce import fused_cross_entropy, fused_cross_entropy_pytorch_ref

logits_ce = torch.randn(64, 1024, device=device, dtype=torch.float32)
targets_ce = torch.randint(0, 1024, (64,), device=device, dtype=torch.int64)
with force_kernel_backend("cuda"):
    out_cuda_7 = fused_cross_entropy(logits_ce, targets_ce)
    cuda_ms_7 = benchmark(lambda: fused_cross_entropy(logits_ce, targets_ce))
with force_kernel_backend("pytorch"):
    out_ref_7 = fused_cross_entropy_pytorch_ref(logits_ce, targets_ce)
    ref_ms_7 = benchmark(lambda: fused_cross_entropy_pytorch_ref(logits_ce, targets_ce))
diff_7 = (out_cuda_7 - out_ref_7).abs().item()
sp_7 = ref_ms_7 / cuda_ms_7 if cuda_ms_7 > 0 else 0
rows.append(("fused_ce", cuda_ms_7, ref_ms_7, sp_7, diff_7))

from neuralops.core.cuda_kernels.logcumsumexp import logcumsumexp, logcumsumexp_pytorch_ref

x_lcse = torch.randn(64, 128, device=device, dtype=torch.float32)
with force_kernel_backend("cuda"):
    out_cuda_8 = logcumsumexp(x_lcse)
    cuda_ms_8 = benchmark(lambda: logcumsumexp(x_lcse))
with force_kernel_backend("pytorch"):
    out_ref_8 = logcumsumexp_pytorch_ref(x_lcse)
    ref_ms_8 = benchmark(lambda: logcumsumexp_pytorch_ref(x_lcse))
diff_8 = (out_cuda_8 - out_ref_8).abs().max().item()
sp_8 = ref_ms_8 / cuda_ms_8 if cuda_ms_8 > 0 else 0
rows.append(("logcumsumexp", cuda_ms_8, ref_ms_8, sp_8, diff_8))

from neuralops.core.cuda_kernels.fused_norm_linear import fused_norm_linear, fused_norm_linear_pytorch_ref

M_fnl, K_fnl, N_fnl = 64, 256, 128
x_fnl = torch.randn(M_fnl, K_fnl, device=device, dtype=torch.float32)
w_fnl = torch.randn(N_fnl, K_fnl, device=device, dtype=torch.float32)
b_fnl = torch.randn(N_fnl, device=device, dtype=torch.float32)
with force_kernel_backend("cuda"):
    out_cuda_9 = fused_norm_linear(x_fnl, w_fnl, b_fnl)
    cuda_ms_9 = benchmark(lambda: fused_norm_linear(x_fnl, w_fnl, b_fnl))
with force_kernel_backend("pytorch"):
    out_ref_9 = fused_norm_linear_pytorch_ref(x_fnl, w_fnl, b_fnl)
    ref_ms_9 = benchmark(lambda: fused_norm_linear_pytorch_ref(x_fnl, w_fnl, b_fnl))
diff_9 = (out_cuda_9 - out_ref_9).abs().max().item()
sp_9 = ref_ms_9 / cuda_ms_9 if cuda_ms_9 > 0 else 0
rows.append(("fused_norm_linear", cuda_ms_9, ref_ms_9, sp_9, diff_9))

print("\n" + "=" * 90)
print(f"{'Module Name':<22} | {'CUDA (ms)':<10} | {'Ref (ms)':<10} | {'Speedup':<9} | {'Diff':<12} | {'Status'}")
print("-" * 90)
all_pass = True
for name, c_ms, r_ms, sp, diff in rows:
    status = "PASS" if diff < 1e-3 and c_ms > 0 and r_ms > 0 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"{name:<22} | {c_ms:<10.4f} | {r_ms:<10.4f} | {sp:<8.2f}x | {diff:<12.2e} | {status}")
print("=" * 90)
print(f"Overall Result: {'100% ALL PASS' if all_pass else 'SOME FAILED'}")
