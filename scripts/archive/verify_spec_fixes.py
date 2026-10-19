# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Verify spec fixes module.

Author: Taha Majlesi
"""

import sys
import traceback
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

results = {}

# 1. fwht_transform
print("\n--- 1. Testing fwht_transform ---")
try:
    from neuralops.core.cuda_kernels.fwht_transform import fwht_transform, fwht_transform_pytorch_ref

    x = torch.randn(8, 128, device=device, dtype=torch.float16)

    # Test invalid power of 2
    try:
        fwht_transform(x, B_dim=100)
        results["fwht_transform_invalid_pow2"] = "FAIL (Did not raise on non-power-of-2)"
    except ValueError as e:
        results["fwht_transform_invalid_pow2"] = f"PASS ({e})"

    # Test valid power of 2
    out_act = fwht_transform(x, B_dim=128)
    out_ref = fwht_transform_pytorch_ref(x, B_dim=128)
    diff = (out_act.float() - out_ref.float()).abs().max().item()
    print(f"fwht_transform valid diff: {diff:.6e}")
    assert diff < 1e-3, f"diff too large: {diff}"
    results["fwht_transform_valid"] = f"PASS (diff={diff:.2e})"
except Exception:
    results["fwht_transform"] = f"FAIL: {traceback.format_exc()}"

# 2. unified_rope_offset
print("\n--- 2. Testing unified_rope_offset ---")
try:
    from neuralops.core.cuda_kernels.unified_rope_offset import unified_rope_offset, fused_unified_rope_offset_pytorch

    B, H, N, D = 2, 4, 16, 64
    x = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    cos = torch.cos(torch.randn(N, D // 2, device=device, dtype=torch.float32))
    sin = torch.sin(torch.randn(N, D // 2, device=device, dtype=torch.float32))

    # Test bounds check failure
    try:
        # N + offset = 16 + 2 = 18 > cos.shape[0] (16)
        unified_rope_offset(x, cos, sin, rotary_dim=64, offset=2)
        results["unified_rope_offset_bounds"] = "FAIL (Did not raise on out of bounds offset)"
    except AssertionError as e:
        results["unified_rope_offset_bounds"] = f"PASS ({e})"

    # Test valid execution
    out_act = unified_rope_offset(x, cos, sin, rotary_dim=64, offset=0)
    out_ref = fused_unified_rope_offset_pytorch(x, cos, sin, rotary_dim=64)
    diff = (out_act - out_ref).abs().max().item()
    print(f"unified_rope_offset valid diff: {diff:.6e}")
    assert diff < 1e-3, f"diff too large: {diff}"
    results["unified_rope_offset_valid"] = f"PASS (diff={diff:.2e})"
except Exception:
    results["unified_rope_offset"] = f"FAIL: {traceback.format_exc()}"

# 3. moe_routing
print("\n--- 3. Testing moe_routing ---")
try:
    from neuralops.core.cuda_kernels.moe_routing import moe_dispatch_gather

    B, D = 16, 64
    tokens = torch.randn(B, D, device=device, dtype=torch.float16)
    gate_logits_8 = torch.randn(B, 8, device=device, dtype=torch.float16)
    gate_logits_16 = torch.randn(B, 16, device=device, dtype=torch.float16)
    expert = torch.nn.Linear(D, D, device=device, dtype=torch.float16)

    # Test top_k <= 8 (fused dispatch on CUDA)
    out_k2 = moe_dispatch_gather(tokens, gate_logits_8, expert, top_k=2)
    assert not out_k2.isnan().any(), "moe top_k=2 produced NaNs"
    results["moe_routing_topk2"] = "PASS"

    # Test top_k > 8 (fallback path without crashing)
    out_k9 = moe_dispatch_gather(tokens, gate_logits_16, expert, top_k=9)
    assert not out_k9.isnan().any(), "moe top_k=9 produced NaNs"
    results["moe_routing_topk9"] = "PASS"
except Exception:
    results["moe_routing"] = f"FAIL: {traceback.format_exc()}"

# 4. split_head_norm
print("\n--- 4. Testing split_head_norm ---")
try:
    from neuralops.core.cuda_kernels.split_head_norm import split_head_norm, split_head_norm_pytorch_ref

    x = torch.randn(16, 64, device=device, dtype=torch.float16)
    weight = torch.randn(64, device=device, dtype=torch.float16)

    # Test weight=None
    out_no_w = split_head_norm(x, weight=None)
    ref_no_w = split_head_norm_pytorch_ref(x, weight=None)
    diff1 = (out_no_w.float() - ref_no_w.float()).abs().max().item()
    assert diff1 < 1e-3, f"diff1 too large: {diff1}"

    # Test with weight
    out_w = split_head_norm(x, weight=weight)
    ref_w = split_head_norm_pytorch_ref(x, weight=weight)
    diff2 = (out_w.float() - ref_w.float()).abs().max().item()
    print(f"split_head_norm diffs: no_weight={diff1:.2e}, with_weight={diff2:.2e}")
    assert diff2 < 1e-3, f"diff2 too large: {diff2}"
    results["split_head_norm"] = f"PASS (diff_no_w={diff1:.2e}, diff_w={diff2:.2e})"
except Exception:
    results["split_head_norm"] = f"FAIL: {traceback.format_exc()}"

# 5. paged_attention
print("\n--- 5. Testing paged_attention ---")
try:
    from neuralops.core.cuda_kernels.paged_attention import paged_attention, paged_attention_pytorch_reference

    num_seqs, num_heads, num_kv_heads, head_dim, block_sz, max_blocks = 4, 8, 2, 64, 16, 8
    q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=torch.float32)
    num_total_blocks = num_seqs * max_blocks
    k_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_sz, device=device, dtype=torch.float32)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_sz, device=device, dtype=torch.float32)
    context_lens = torch.tensor([16, 32, 48, 64], device=device, dtype=torch.int32)
    block_tables = torch.zeros(num_seqs, max_blocks, device=device, dtype=torch.int32)
    for i in range(num_seqs):
        nb = int(context_lens[i].item()) // block_sz
        block_tables[i, :nb] = torch.arange(i * max_blocks, i * max_blocks + nb, device=device)

    out_act = paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size=block_sz)
    out_ref = paged_attention_pytorch_reference(q, k_cache, v_cache, block_tables, context_lens, block_size=block_sz)
    diff = (out_act - out_ref).abs().max().item()
    print(f"paged_attention diff: {diff:.6e}")
    assert diff < 1e-3, f"diff too large: {diff}"

    # Test autograd backward
    q_g = q.clone().requires_grad_(True)
    out_g = paged_attention(q_g, k_cache, v_cache, block_tables, context_lens, block_size=block_sz)
    out_g.sum().backward()
    assert q_g.grad is not None and not q_g.grad.isnan().any(), "paged_attention grad failed"
    results["paged_attention"] = f"PASS (diff={diff:.2e}, backward=OK)"
except Exception:
    results["paged_attention"] = f"FAIL: {traceback.format_exc()}"

# 6. fused_dropout_add
print("\n--- 6. Testing fused_dropout_add ---")
try:
    from neuralops.core.cuda_kernels.fused_dropout_add import fused_dropout_add

    x = torch.randn(32, 128, device=device, dtype=torch.float16)
    res = torch.randn(32, 128, device=device, dtype=torch.float16)

    # Test eval mode (training=False)
    out_eval = fused_dropout_add(x, res, p=0.5, training=False)
    ref_eval = x + res
    diff_eval = (out_eval.float() - ref_eval.float()).abs().max().item()
    assert diff_eval < 1e-4, f"diff_eval too large: {diff_eval}"

    # Test training mode with seed
    out_train1 = fused_dropout_add(x, res, p=0.5, training=True, seed=12345)
    out_train2 = fused_dropout_add(x, res, p=0.5, training=True, seed=12345)
    diff_seeds = (out_train1 - out_train2).abs().max().item()
    assert diff_seeds == 0.0, "Same seed produced different results"

    out_train3 = fused_dropout_add(x, res, p=0.5, training=True, seed=99999)
    diff_diff_seeds = (out_train1 - out_train3).abs().max().item()
    assert diff_diff_seeds > 0.0, "Different seeds produced identical results"
    results["fused_dropout_add"] = f"PASS (eval_diff={diff_eval:.2e}, seed_control=OK)"
except Exception:
    results["fused_dropout_add"] = f"FAIL: {traceback.format_exc()}"

# 7. fused_ce
print("\n--- 7. Testing fused_ce ---")
try:
    from neuralops.core.cuda_kernels.fused_ce import fused_cross_entropy, fused_cross_entropy_pytorch_ref

    # Test N == 0
    logits_0 = torch.empty(0, 100, device=device, dtype=torch.float32, requires_grad=True)
    targets_0 = torch.empty(0, device=device, dtype=torch.int64)
    loss_0 = fused_cross_entropy(logits_0, targets_0)
    print(f"fused_ce N=0 loss: {loss_0}")

    # Test N > 0 in float32 for high numerical fidelity
    logits = torch.randn(16, 100, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 100, (16,), device=device, dtype=torch.int64)
    loss_act = fused_cross_entropy(logits, targets)
    loss_ref = fused_cross_entropy_pytorch_ref(logits, targets)
    diff = (loss_act - loss_ref).abs().item()
    print(f"fused_ce diff: {diff:.6e}")
    assert diff < 1e-3, f"diff too large: {diff}"

    loss_act.backward()
    assert logits.grad is not None and not logits.grad.isnan().any(), "fused_ce backward failed"
    results["fused_ce"] = f"PASS (diff={diff:.2e}, N=0 handled)"
except Exception:
    results["fused_ce"] = f"FAIL: {traceback.format_exc()}"

# 8. logcumsumexp
print("\n--- 8. Testing logcumsumexp ---")
try:
    from neuralops.core.cuda_kernels.logcumsumexp import logcumsumexp, logcumsumexp_pytorch_ref

    # Test L=16 (within warp)
    x16 = torch.randn(8, 16, device=device, dtype=torch.float16)
    out16_act = logcumsumexp(x16)
    out16_ref = logcumsumexp_pytorch_ref(x16)
    diff16 = (out16_act.float() - out16_ref.float()).abs().max().item()
    print(f"logcumsumexp L=16 diff: {diff16:.6e}")
    assert diff16 < 1e-3, f"diff16 too large: {diff16}"

    # Test L=64 (multi-warp chunking)
    x64 = torch.randn(8, 64, device=device, dtype=torch.float16)
    out64_act = logcumsumexp(x64)
    out64_ref = logcumsumexp_pytorch_ref(x64)
    diff64 = (out64_act.float() - out64_ref.float()).abs().max().item()
    print(f"logcumsumexp L=64 diff: {diff64:.6e}")
    assert diff64 < 1e-3, f"diff64 too large: {diff64}"
    results["logcumsumexp"] = f"PASS (diff16={diff16:.2e}, diff64={diff64:.2e})"
except Exception:
    results["logcumsumexp"] = f"FAIL: {traceback.format_exc()}"

# 9. fused_norm_linear
print("\n--- 9. Testing fused_norm_linear ---")
try:
    from neuralops.core.cuda_kernels.fused_norm_linear import fused_norm_linear, fused_norm_linear_pytorch_ref

    M, K, N = 16, 64, 32
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)

    # With bias
    out_act = fused_norm_linear(x, w, b)
    out_ref = fused_norm_linear_pytorch_ref(x, w, b)
    diff_out = (out_act - out_ref).abs().max().item()
    print(f"fused_norm_linear with bias diff: out={diff_out:.6e}")
    assert diff_out < 1e-3, f"diff_out too large: {diff_out}"

    # Without bias
    out_act2 = fused_norm_linear(x, w, None)
    out_ref2 = fused_norm_linear_pytorch_ref(x, w, None)
    diff_out2 = (out_act2 - out_ref2).abs().max().item()
    print(f"fused_norm_linear without bias diff: out={diff_out2:.6e}")
    assert diff_out2 < 1e-3, f"diff_out2 too large: {diff_out2}"
    results["fused_norm_linear"] = f"PASS (diff_bias={diff_out:.2e}, diff_nobias={diff_out2:.2e})"
except Exception:
    results["fused_norm_linear"] = f"FAIL: {traceback.format_exc()}"

print("\n" + "=" * 50)
print("FINAL SUMMARY OF SPEC VERIFICATION:")
for k, v in results.items():
    print(f"  {k:<30}: {v}")
print("=" * 50)
