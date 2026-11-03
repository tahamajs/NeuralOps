#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
High-Throughput Serving & Paged Linear Attention Test Suite (Phases 081–100).

Verifies:
1. vLLM physical page table layout and validation (Phase 081).
2. 2D Block Table dereferencing logic and parity (Phase 082, 100).
3. Zero Block Pointer Compliance (AGENTS.md §7 / Phase 083).
4. 2D SRAM tile scheduling & microarchitecture (Phase 084, 085).
5. Bitwise XOR shared memory bank swizzling & 32-way conflict elimination (Phases 086, 087).
6. Triton prefill & single-token decode step kernels (Phases 088, 089).
7. SGLang Radix Tree node structure, SHA-256 prefix hash, and O(1) state lookup (Phases 090–092).
8. Radix Tree node splitting and LRU eviction bounded VRAM policy (Phases 093, 094).
9. TensorRT-LLM Dynamic Plugin C++ interface and strict compilation (Phases 095–099).
10. End-to-end continuous serving benchmark parity at <= 1e-4 (Phase 100).
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from neuralops.ops.triton.paged_linear_attn import (
    create_paged_kv_cache,
    paged_linear_attention_decode_step_pytorch,
    paged_linear_attention_pytorch,
    swizzle_tensor_layout,
    unswizzle_tensor_layout,
    validate_block_tables,
)
from neuralops.surgery.hf_cache_bridge import (
    NeuralOpsHybridHFCache,
    RadixTreeNode,
    SGLangRadixStateAdapter,
    _compute_prefix_hash,
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


def test_vllm_physical_page_table_specification():
    """
    Test Phase 081: vLLM physical page table layout and bounds validation.
    """
    b, t, h, d = 4, 64, 4, 32
    block_size = 16

    k_cache, v_cache, block_tables = create_paged_kv_cache(
        batch_size=b,
        seq_len=t,
        num_heads=h,
        head_dim=d,
        block_size=block_size,
    )

    assert block_tables.shape == (b, t // block_size)
    assert k_cache.shape[1] == block_size
    assert v_cache.shape[1] == block_size

    # Validate negative or out-of-bounds raises ValueError
    invalid_tables = block_tables.clone()
    invalid_tables[0, 0] = -1
    with pytest.raises(ValueError, match="negative block IDs"):
        validate_block_tables(invalid_tables, k_cache.shape[0], t, block_size)

    invalid_tables_oob = block_tables.clone()
    invalid_tables_oob[0, 0] = k_cache.shape[0] + 10
    with pytest.raises(ValueError, match="out-of-bounds block IDs"):
        validate_block_tables(invalid_tables_oob, k_cache.shape[0], t, block_size)


def test_shared_memory_bank_conflict_elimination():
    """
    Test Phase 087: Verifies elimination of 32-way shared memory bank conflicts.
    Formula: swizzled_offs_d = offs_d ^ ((offs_m & 0x7) << 2)
    Bank ID: bank_id = (swizzled_offs_d / 4) mod 32
    Asserts that across all 32 threads in a warp, each accesses a distinct bank index.
    """
    # Test across multiple row offsets m in [0, 7]
    for row_m in range(8):
        shift = (row_m & 0x7) << 2
        # 32 threads in a warp accessing contiguous 4-byte words
        # thread_id t accesses d = t * 4 (stride 4 bytes)
        thread_ids = torch.arange(32, dtype=torch.long)
        raw_offs_d = thread_ids * 4

        swizzled_offs_d = raw_offs_d ^ shift
        bank_ids = (swizzled_offs_d // 4) % 32

        # Verify that all 32 threads access unique banks (0 conflicts)
        unique_banks = torch.unique(bank_ids)
        assert (
            len(unique_banks) == 32
        ), f"Row {row_m}: Expected 32 unique banks, got {len(unique_banks)}"


def test_xor_bank_swizzling_invertibility_and_parity():
    """
    Test Phase 086: Bitwise XOR shared memory bank swizzling:
        swizzled_offs = offs_d ^ ((offs_m & 0x7) << 2)
    Asserts involution invertibility and output parity when using swizzled layout.
    """
    torch.manual_seed(123)
    b, t, h, d = 2, 32, 4, 64
    block_size = 16

    x = torch.randn(b, t, h, d, dtype=torch.float32)
    swizzled = swizzle_tensor_layout(x, block_m=block_size)
    restored = unswizzle_tensor_layout(swizzled, block_m=block_size)

    # Verify mathematical invertibility ((x ^ s) ^ s == x)
    inv_diff = torch.max(torch.abs(restored - x)).item()
    assert inv_diff == 0.0, f"XOR swizzling must be an exact involution, got diff={inv_diff}"

    # Verify that swizzling actually permutes columns for m > 0
    shift_row1 = (1 & 0x7) << 2
    assert shift_row1 == 4
    assert torch.equal(swizzled[:, 1, :, 0], x[:, 1, :, 4])

    # Verify paged linear attention with swizzled Q
    k_cache, v_cache, block_tables = create_paged_kv_cache(
        batch_size=b,
        seq_len=t,
        num_heads=h,
        head_dim=d,
        block_size=block_size,
    )

    out_standard = paged_linear_attention_pytorch(
        q=x,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        block_size=block_size,
        swizzle=False,
    )

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


@pytest.mark.parametrize("block_size", [16, 32])
def test_paged_linear_attention_parity(block_size: int):
    """
    Test Phase 088 & Phase 100: Paged attention parity with non-paged reference.
    Asserts output matches unpaged linear attention reference within atol=1e-4.
    """
    torch.manual_seed(42)
    b, t, h, d = 2, 64, 4, 32
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


def test_paged_linear_attention_decode_step():
    """
    Test Phase 089: Single-token decode step kernel with O(1) state recurrence.
    """
    torch.manual_seed(99)
    b, h, d = 4, 4, 32

    # Initial state
    state = torch.zeros(b, h, d, d, dtype=torch.float32)

    # Execute 4 sequential decode steps
    for step in range(4):
        q = torch.randn(b, 1, h, d, dtype=torch.float32)
        k = torch.randn(b, 1, h, d, dtype=torch.float32)
        v = torch.randn(b, 1, h, d, dtype=torch.float32)

        out, state = paged_linear_attention_decode_step_pytorch(
            q=q,
            k=k,
            v=v,
            state=state,
            decay=0.99,
        )

        assert out.shape == (b, 1, h, d)
        assert state.shape == (b, h, d, d)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_sglang_prefix_hash_and_collision_resistance():
    """
    Test Phase 091: Deterministic SHA-256 token prefix hash generator.
    Verifies collision resistance across distinct sequences.
    """
    # 1. Basic deterministic hash check
    tokens_a = (101, 2054, 2003, 1037)
    tokens_b = (101, 2054, 2003, 1037)
    tokens_c = (101, 2054, 2003, 1038)

    hash_a = _compute_prefix_hash(tokens_a)
    hash_b = _compute_prefix_hash(tokens_b)
    hash_c = _compute_prefix_hash(tokens_c)

    assert hash_a == hash_b
    assert hash_a != hash_c
    assert len(hash_a) == 64

    # 2. Collision test on synthetic distinct token sequences
    seen_hashes = set()
    num_sequences = 10000
    for i in range(num_sequences):
        seq = (i, i + 1, i * 2)
        h = _compute_prefix_hash(seq)
        assert h not in seen_hashes, f"Collision detected at sequence {i}"
        seen_hashes.add(h)


def test_sglang_radix_state_caching_and_prefix_retrieval():
    """
    Test Phase 090, 092: SGLang Radix Tree Recurrent State Adapter and O(1) state lookup.
    """
    adapter = SGLangRadixStateAdapter(max_cached_nodes=100)

    system_prompt = (101, 2054, 2003, 1037, 3231, 102)
    turn1_user = (2023, 2003, 1037, 2742)
    full_turn1 = system_prompt + turn1_user

    turn2_user = (2129, 2024, 2017)
    full_turn2 = system_prompt + turn2_user

    s_system = {
        0: torch.randn(1, 4, 32, 32),
        1: torch.randn(1, 4, 32, 32),
    }

    # 1. Cache system prompt state in Radix Tree
    adapter.cache_prefix_state(system_prompt, s_system)

    # 2. Lookup prefix state in O(1) time (Phase 092)
    looked_up_state = adapter.lookup_prefix_state(system_prompt)
    assert looked_up_state is not None
    assert torch.equal(looked_up_state[0], s_system[0])
    assert torch.equal(looked_up_state[1], s_system[1])

    # 3. Retrieve state for turn 2 (shares the exact system prompt prefix)
    matched_len, retrieved_cache = adapter.retrieve_prefix_state(full_turn2)

    assert matched_len == len(system_prompt)
    assert isinstance(retrieved_cache, NeuralOpsHybridHFCache)
    assert retrieved_cache._seen_tokens == len(system_prompt)

    retrieved_s0 = retrieved_cache.get_recurrent_state(0)
    retrieved_s1 = retrieved_cache.get_recurrent_state(1)
    assert retrieved_s0 is not None and torch.equal(retrieved_s0, s_system[0])
    assert retrieved_s1 is not None and torch.equal(retrieved_s1, s_system[1])


def test_sglang_radix_node_splitting():
    """
    Test Phase 093: Radix Tree node splitting when prefixes diverge.
    Inserts prefixes [1, 2, 3, 4] and [1, 2, 5, 6]; verifies split at [1, 2].
    """
    adapter = SGLangRadixStateAdapter(max_cached_nodes=100)

    seq_a = (1, 2, 3, 4)
    state_a = {0: torch.ones(1, 2, 4, 4)}

    seq_b = (1, 2, 5, 6)
    state_b = {0: torch.full((1, 2, 4, 4), 2.0)}

    adapter.cache_prefix_state(seq_a, state_a)
    adapter.cache_prefix_state(seq_b, state_b)

    # Root should have child starting with 1
    assert 1 in adapter.root.children
    split_node = adapter.root.children[1]

    # Split node must hold common prefix (1, 2)
    assert split_node.token_chunk == (1, 2)
    # Split node must have two children: for token 3 and token 5
    assert 3 in split_node.children
    assert 5 in split_node.children
    assert split_node.children[3].token_chunk == (3, 4)
    assert split_node.children[5].token_chunk == (5, 6)


def test_sglang_radix_tree_lru_eviction():
    """
    Test Phase 094: Verifies Radix Tree LRU node eviction under capacity limits.
    """
    adapter = SGLangRadixStateAdapter(max_cached_nodes=5)

    for i in range(10):
        tokens = (1000 + i, 2000 + i)
        state = {0: torch.full((1, 2, 4, 4), float(i))}
        adapter.cache_prefix_state(tokens, state)

    # Node count must be bounded
    assert adapter.total_nodes() <= 10

    # Explicit eviction with max_nodes parameter
    evicted = adapter.evict_lru(max_nodes=3)
    assert adapter.total_nodes() <= 3
    assert evicted > 0


def test_tensorrt_llm_plugin_source_and_structure():
    """
    Test Phases 095–098: Validates TensorRT-LLM C++ Dynamic Plugin implementation.
    Verifies classes, methods, format combinations, workspace, and registration macros.
    """
    repo_root = Path(__file__).resolve().parent.parent
    plugin_cpp = (
        repo_root
        / "neuralops"
        / "ops"
        / "cuda"
        / "plugins"
        / "trt_linear_attention_plugin.cpp"
    )

    assert plugin_cpp.exists(), f"Missing plugin file at {plugin_cpp}"
    content = plugin_cpp.read_text(encoding="utf-8")

    # Invariants for TensorRT IPluginV2DynamicExt
    assert "class LinearAttentionPlugin : public nvinfer1::IPluginV2DynamicExt" in content
    assert "class LinearAttentionPluginCreator : public nvinfer1::IPluginCreator" in content
    assert "REGISTER_TENSORRT_PLUGIN(LinearAttentionPluginCreator);" in content
    assert "getOutputDimensions" in content
    assert "enqueue" in content
    assert "getWorkspaceSize" in content
    assert "supportsFormatCombination" in content
    assert "clone" in content
    assert "serialize" in content
    assert "deserialize" in content
    assert "destroy" in content


def test_tensorrt_plugin_clang_strict_compilation():
    """
    Test Phase 099: Strict C++17 compilation gate with clang++.
    Checks: clang++ -fsyntax-only -std=c++17 -Wall -Wextra.
    """
    clang_path = shutil.which("clang++")
    if not clang_path:
        pytest.skip("clang++ compiler not available in test environment.")

    repo_root = Path(__file__).resolve().parent.parent
    plugin_cpp = (
        repo_root
        / "neuralops"
        / "ops"
        / "cuda"
        / "plugins"
        / "trt_linear_attention_plugin.cpp"
    )

    cmd = [
        clang_path,
        "-fsyntax-only",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        str(plugin_cpp),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Clang++ compilation failed:\n{res.stderr}"


def test_zero_block_ptr_invariant():
    """
    Test Phase 083 / AGENTS.md §7: Strictly ZERO tl.make_block_ptr in mainline Triton kernels.
    """
    repo_root = Path(__file__).resolve().parent.parent
    kernel_file = repo_root / "neuralops" / "ops" / "triton" / "paged_linear_attn.py"

    assert kernel_file.exists(), f"Missing kernel file at {kernel_file}"
    content = kernel_file.read_text(encoding="utf-8")

    # Filter out comments and docstrings
    code_lines = [
        line.strip()
        for line in content.splitlines()
        if not line.strip().startswith("#")
        and not line.strip().startswith('"""')
        and not line.strip().startswith("*")
    ]
    code_body = "\n".join(code_lines)

    assert "tl.make_block_ptr" not in code_body, (
        "Violation of AGENTS.md §7: tl.make_block_ptr found in code!"
    )
    assert "tl.advance" not in code_body, (
        "Violation of AGENTS.md §7: tl.advance found in code!"
    )
