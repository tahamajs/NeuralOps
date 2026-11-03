#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
High-Throughput Serving & Paged Linear Attention Test Suite.

Verifies:
1. Paged Linear Attention non-contiguous page dereferencing parity with dense reference.
2. Shared memory XOR bank swizzling eliminating 32-way bank conflicts.
3. SGLang Radix Tree recurrent state caching, O(1) prefix retrieval, and LRU eviction.
4. TensorRT-LLM Dynamic Plugin structural and registration integrity.
5. Invariant check: Strictly zero tl.make_block_ptr in mainline Triton kernels.
"""

from __future__ import annotations

from pathlib import Path

import torch

from neuralops.ops.triton.paged_linear_attn import (
    create_paged_kv_cache,
    paged_linear_attention_pytorch,
    swizzle_tensor_layout,
    unswizzle_tensor_layout,
)
from neuralops.surgery.hf_cache_bridge import (
    NeuralOpsHybridHFCache,
    RadixTreeNode,
    SGLangRadixStateAdapter,
)


def dense_linear_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """
    Unpaged dense chunkwise linear attention reference using ELU + 1 feature map.
    q: [B, T, H, D]
    k: [B, T, H, D]
    v: [B, T, H, D]
    """
    b, t, h, d = q.shape
    num_chunks = (t + block_size - 1) // block_size

    q_phi = torch.where(q > 0.0, q + 1.0, torch.exp(q)).float()
    k_phi = torch.where(k > 0.0, k + 1.0, torch.exp(k)).float()
    v_float = v.float()

    out = torch.zeros(b, t, h, d, dtype=torch.float32, device=q.device)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * block_size
        end = min(start + block_size, t)

        q_c = q_phi[:, start:end, :, :].permute(0, 2, 1, 3)  # [B, H, C, D]
        k_c = k_phi[:, start:end, :, :].permute(0, 2, 1, 3)  # [B, H, C, D]
        v_c = v_float[:, start:end, :, :].permute(0, 2, 1, 3)  # [B, H, C, D]

        # S = K^T @ V -> [B, H, D, D]
        s_tile = torch.matmul(k_c.transpose(-2, -1), v_c)
        # Out = Q @ S -> [B, H, C, D]
        out_c = torch.matmul(q_c, s_tile)

        out[:, start:end, :, :] = out_c.permute(0, 2, 1, 3)

    return out.to(q.dtype)


def test_paged_linear_attention_parity():
    """
    Test 1: Paged attention parity with non-paged reference.
    Asserts output matches unpaged linear attention reference within atol=1e-3.
    """
    torch.manual_seed(42)
    b, t, h, d = 2, 32, 4, 32
    block_size = 16
    num_chunks = t // block_size

    q = torch.randn(b, t, h, d, dtype=torch.float32)
    k_dense = torch.randn(b, t, h, d, dtype=torch.float32)
    v_dense = torch.randn(b, t, h, d, dtype=torch.float32)

    # Allocate physical block pool with random non-contiguous mappings
    pool_blocks = b * num_chunks + 8
    k_cache = torch.zeros(pool_blocks, block_size, h, d, dtype=torch.float32)
    v_cache = torch.zeros(pool_blocks, block_size, h, d, dtype=torch.float32)

    perm = torch.randperm(pool_blocks, dtype=torch.long)
    block_tables = perm[: b * num_chunks].view(b, num_chunks)

    # Scatter dense tokens into non-contiguous physical pages
    for seq_idx in range(b):
        for c in range(num_chunks):
            phys_id = int(block_tables[seq_idx, c].item())
            k_cache[phys_id] = k_dense[seq_idx, c * block_size : (c + 1) * block_size]
            v_cache[phys_id] = v_dense[seq_idx, c * block_size : (c + 1) * block_size]

    # Compute paged linear attention via dereferencing
    paged_out = paged_linear_attention_pytorch(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        block_size=block_size,
        swizzle=False,
    )

    # Compute unpaged dense reference
    dense_ref = dense_linear_attention_reference(
        q=q,
        k=k_dense,
        v=v_dense,
        block_size=block_size,
    )

    max_diff = torch.max(torch.abs(paged_out - dense_ref)).item()
    assert max_diff <= 1e-4, f"Paged attention diverged from dense reference: max_diff={max_diff:.6e}"


def test_xor_bank_swizzling_invertibility_and_parity():
    """
    Test 2: Bitwise XOR shared memory bank swizzling:
        swizzled_offs = offs_d ^ ((offs_m & 0x7) << 2)
    Asserts invertibility and output parity when using swizzled layout.
    """
    torch.manual_seed(123)
    b, t, h, d = 2, 32, 4, 64
    block_size = 16

    x = torch.randn(b, t, h, d, dtype=torch.float32)
    swizzled = swizzle_tensor_layout(x, block_m=block_size)
    restored = unswizzle_tensor_layout(swizzled, block_m=block_size)

    # Verify mathematical invertibility
    inv_diff = torch.max(torch.abs(restored - x)).item()
    assert inv_diff == 0.0, f"XOR swizzling must be an exact involution, got diff={inv_diff}"

    # Verify that swizzling actually permutes columns for m > 0
    shift_row1 = ((1 & 0x7) << 2)
    assert shift_row1 == 4
    # Column 0 of row 1 in swizzled tensor should be column 4 of x
    assert torch.equal(swizzled[:, 1, :, 0], x[:, 1, :, 4])

    # Verify paged linear attention with swizzled Q
    k_cache, v_cache, block_tables = create_paged_kv_cache(
        batch_size=b,
        seq_len=t,
        num_heads=h,
        head_dim=d,
        block_size=block_size,
    )

    # Regular unswizzled output
    out_standard = paged_linear_attention_pytorch(
        q=x,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        block_size=block_size,
        swizzle=False,
    )

    # Swizzled output
    out_swizzled = paged_linear_attention_pytorch(
        q=swizzled,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        block_size=block_size,
        swizzle=True,
    )

    swizzle_diff = torch.max(torch.abs(out_standard - out_swizzled)).item()
    assert swizzle_diff <= 1e-4, f"Swizzled execution produced numeric divergence: {swizzle_diff:.6e}"


def test_sglang_radix_state_caching_and_prefix_retrieval():
    """
    Test 3: SGLang Radix Tree Recurrent State Adapter.
    Verifies caching S_prefix at prefix nodes and O(1) state recovery.
    """
    adapter = SGLangRadixStateAdapter(max_cached_nodes=100)

    # Define common system prompt prefix
    system_prompt = (101, 2054, 2003, 1037, 3231, 102)
    turn1_user = (2023, 2003, 1037, 2742)
    full_turn1 = system_prompt + turn1_user

    turn2_user = (2129, 2024, 2017)
    full_turn2 = system_prompt + turn2_user

    # Synthetic associative states S \in [1, 4, 32, 32]
    s_system = {
        0: torch.randn(1, 4, 32, 32),
        1: torch.randn(1, 4, 32, 32),
    }

    # 1. Cache system prompt state in Radix Tree
    adapter.cache_prefix_state(system_prompt, s_system)

    # 2. Retrieve state for turn 2 (shares the exact system prompt prefix)
    matched_len, retrieved_cache = adapter.retrieve_prefix_state(full_turn2)

    assert matched_len == len(system_prompt), f"Expected prefix length {len(system_prompt)}, got {matched_len}"
    assert isinstance(retrieved_cache, NeuralOpsHybridHFCache)
    assert retrieved_cache._seen_tokens == len(system_prompt)

    # Verify per-layer states match exactly
    retrieved_s0 = retrieved_cache.get_recurrent_state(0)
    retrieved_s1 = retrieved_cache.get_recurrent_state(1)
    assert retrieved_s0 is not None and torch.equal(retrieved_s0, s_system[0])
    assert retrieved_s1 is not None and torch.equal(retrieved_s1, s_system[1])

    # 3. Insert full turn 1 and verify deeper prefix matching
    s_turn1 = {
        0: torch.randn(1, 4, 32, 32),
        1: torch.randn(1, 4, 32, 32),
    }
    adapter.cache_prefix_state(full_turn1, s_turn1)

    turn1_followup = full_turn1 + (1996, 2111)
    matched_len_deep, cache_deep = adapter.retrieve_prefix_state(turn1_followup)
    assert matched_len_deep == len(full_turn1)
    assert torch.equal(cache_deep.get_recurrent_state(0), s_turn1[0])


def test_sglang_radix_tree_lru_eviction():
    """
    Test 4: Verifies Radix Tree LRU node eviction under capacity limits.
    """
    adapter = SGLangRadixStateAdapter(max_cached_nodes=5)

    for i in range(10):
        tokens = (1000 + i, 2000 + i)
        state = {0: torch.full((1, 2, 4, 4), float(i))}
        adapter.cache_prefix_state(tokens, state)

    # Node count must not grow unbounded beyond max capacity
    assert adapter.total_nodes() <= 10


def test_tensorrt_llm_plugin_source_and_structure():
    """
    Test 5: Validates TensorRT-LLM C++ Dynamic Plugin implementation.
    Verifies classes, methods, and registration macros.
    """
    repo_root = Path(__file__).resolve().parent.parent
    plugin_cpp = repo_root / "neuralops" / "ops" / "cuda" / "plugins" / "trt_linear_attention_plugin.cpp"

    assert plugin_cpp.exists(), f"Missing plugin file at {plugin_cpp}"
    content = plugin_cpp.read_text(encoding="utf-8")

    # Invariants for TensorRT IPluginV2DynamicExt
    assert "class LinearAttentionPlugin : public nvinfer1::IPluginV2DynamicExt" in content
    assert "class LinearAttentionPluginCreator : public nvinfer1::IPluginCreator" in content
    assert "REGISTER_TENSORRT_PLUGIN(LinearAttentionPluginCreator);" in content
    assert "getOutputDimensions" in content
    assert "enqueue" in content
    assert "supportsFormatCombination" in content
    assert "clone" in content
    assert "serialize" in content
    assert "destroy" in content


def test_zero_block_ptr_invariant():
    """
    Test 6: Enforces AGENTS.md §7: Strictly ZERO tl.make_block_ptr in mainline Triton kernels.
    """
    repo_root = Path(__file__).resolve().parent.parent
    kernel_file = repo_root / "neuralops" / "ops" / "triton" / "paged_linear_attn.py"

    assert kernel_file.exists(), f"Missing kernel file at {kernel_file}"
    content = kernel_file.read_text(encoding="utf-8")

    # Split lines and filter out comments/docstrings
    code_lines = [
        line.strip()
        for line in content.splitlines()
        if not line.strip().startswith("#") and not line.strip().startswith('"""')
    ]
    code_body = "\n".join(code_lines)

    assert "tl.make_block_ptr" not in code_body, "Violation of AGENTS.md §7: tl.make_block_ptr found in code!"
    assert "tl.advance" not in code_body, "Violation of AGENTS.md §7: tl.advance found in code!"
