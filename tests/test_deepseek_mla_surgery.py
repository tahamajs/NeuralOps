#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and Regression Tests for DeepSeek Multi-Head Latent Attention (MLA) Surgery
and Hybrid Mixture of Attentions (MoA) Pareto Router (Phases 061 – 080).

Comprehensive Certification Matrix:
    - Phase 061: Low-Rank KV Compression (c_t^{KV} in R^512, 87.5% memory reduction).
    - Phase 062: Decoupled Rotary Key (k_t^R in R^64) separation pipeline & RoPE rotation.
    - Phase 063: Latent Content State (S_t^C in R^{512 x 128}) recurrence.
    - Phase 064: Decoupled Positional State (z_t^R in R^{64 x 64}) recurrence.
    - Phase 065: State Decay Exponential Smoothing (gamma = 0.99).
    - Phase 066: Parallel Prefill Chunk Associative Scan Operator.
    - Phase 067: Autoregressive Decode Step O(1) State Transition.
    - Phase 068: Compressed Output Reconstruction with W_{UKV} Fusion ([B, T, D_model]).
    - Phase 069: Prefill vs Decode Step-1 Numerical Boundary Parity Gate (|diff| < 1e-6) & multi-step causal parity.
    - Phase 070: 100-Step Autoregressive Memory Invariance Certification (0.0% growth).
    - Phase 071: Attention Distribution Entropy Metric (H_l) Formulator (ln(T) and 0.0 bounds).
    - Phase 072: Layer Sensitivity Profiler for Syntactic vs Semantic Routing.
    - Phase 073: Boundary Layer Safeguard (Layers 0-3 Full Attention).
    - Phase 074: Factual Logit Safeguard (Layers 30-31 Full Attention).
    - Phase 075: Intermediate Bulk Linear Attention Router (Layers 4-29, exactly 26 layers).
    - Phase 076: Pareto Frontier Latency-vs-Perplexity Solver (Delta PPL <= 0.15).
    - Phase 077: Dynamic Heterogeneous Layer Schedule Generator (12, 24, 32, 48 layers).
    - Phase 078: Endoskeletal Shell Bridge Dispatch Hook (full 32-layer heterogeneous model execution).
    - Phase 079: Weight Inheritance Engine for Low-Rank Projections (zero shape mismatch).
    - Phase 080: Multi-Seed Statistical Significance Test Suite (mean < 1e-6, var < 1e-12 across seeds).
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from neuralops.surgery.endoskeletal_shell import UniversalAttentionBridge
from neuralops.surgery.hf_cache_bridge import NeuralOpsHybridHFCache
from neuralops.surgery.mla_surgery import DeepSeekMLASurgeryBridge, DualStreamLatentState
from neuralops.surgery.moa_router import (
    HybridMoARouter,
    compute_attention_entropy,
    profile_layer_sensitivity,
)
from neuralops.surgery.scanner import AttentionTargetSpec


# =========================================================================
# Phase 061: DeepSeek MLA Low-Rank KV Compression (87.5% memory reduction)
# =========================================================================
def test_phase_061_mla_low_rank_kv_compression() -> None:
    """Certifies Phase 061: KV compression down to 512 dims achieving >= 87.5% memory reduction."""
    bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=512, decoupled_rope_dim=64)
    # Against standard uncompressed KV dimension 4096 (e.g. 32 heads * 128)
    reduction_4096 = bridge.get_kv_compression_ratio(uncompressed_kv_dim=4096)
    assert reduction_4096 == 0.875, f"Expected 87.5% reduction, got {reduction_4096 * 100}%"

    # Against full DeepSeek-V2 multi-head KV (128 heads * 128 = 16,384)
    reduction_16384 = bridge.get_kv_compression_ratio(uncompressed_kv_dim=16384)
    assert reduction_16384 == 1.0 - (512 / 16384)
    assert reduction_16384 > 0.95


# =========================================================================
# Phase 062: Decoupled Rotary Key Separation Pipeline (k_t^R in R^64)
# =========================================================================
def test_phase_062_decoupled_rope_key_separation() -> None:
    """Certifies Phase 062: RoPE key is cleanly separated with dimension strictly 64 and rotated."""
    bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=512, decoupled_rope_dim=64)
    batch_size = 2
    seq_len = 4
    # Fused KV projection output [B, T, 512 + 64]
    fused_kv = torch.randn(batch_size, seq_len, 512 + 64)
    c_kv, k_rope = bridge.separate_mla_projections(fused_kv)

    assert c_kv.shape == (batch_size, seq_len, 512)
    assert k_rope.shape == (batch_size, seq_len, 64)
    assert k_rope.shape[-1] == 64, f"RoPE key dimension must be strictly 64, got {k_rope.shape[-1]}"

    # Test rotary embedding application
    angles = torch.linspace(0, math.pi, 64)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)
    k_rotated = DeepSeekMLASurgeryBridge.apply_rotary_embedding(k_rope, cos=cos, sin=sin)
    assert k_rotated.shape == (batch_size, seq_len, 64)
    # Norm preservation under orthogonal rotation
    norm_before = torch.norm(k_rope, dim=-1)
    norm_after = torch.norm(k_rotated, dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-4)


# =========================================================================
# Phase 063: Latent Content State Recurrence (S_t^C in R^{512 x 128})
# =========================================================================
def test_phase_063_latent_content_state_shape() -> None:
    """Certifies Phase 063: Content state S_t^C maintains exact [B, 512, 128] shape."""
    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=512,
        decoupled_rope_dim=64,
        q_latent_dim=128,
    )
    batch_size = 3
    init_state = bridge.init_latent_state(batch_size=batch_size, device=torch.device("cpu"))
    assert init_state.state_c.shape == (batch_size, 512, 128)

    # Step update
    c_step = torch.randn(batch_size, 1, 512)
    k_step = torch.randn(batch_size, 1, 64)
    qc_step = torch.randn(batch_size, 1, 128)
    qr_step = torch.randn(batch_size, 1, 64)

    out, (new_sc, new_sr) = bridge.forward_decode_step(
        c_step, k_step, qc_step, qr_step, init_state.state_c, init_state.state_r
    )
    assert new_sc.shape == (batch_size, 512, 128)
    assert out.shape == (batch_size, 1, 512)


# =========================================================================
# Phase 064: Decoupled Positional State Recurrence (z_t^R in R^{64 x 64})
# =========================================================================
def test_phase_064_decoupled_rope_state_shape() -> None:
    """Certifies Phase 064: Decoupled positional state z_t^R maintains exact [B, 64, 64] shape."""
    bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=512, decoupled_rope_dim=64)
    batch_size = 2
    init_state = bridge.init_latent_state(batch_size=batch_size, device=torch.device("cpu"))
    assert init_state.state_r.shape == (batch_size, 64, 64)

    c_step = torch.randn(batch_size, 1, 512)
    k_step = torch.randn(batch_size, 1, 64)
    qc_step = torch.randn(batch_size, 1, 128)
    qr_step = torch.randn(batch_size, 1, 64)

    _, (_, new_sr) = bridge.forward_decode_step(
        c_step, k_step, qc_step, qr_step, init_state.state_c, init_state.state_r
    )
    assert new_sr.shape == (batch_size, 64, 64)


# =========================================================================
# Phase 065: State Decay Exponential Smoothing (gamma = 0.99)
# =========================================================================
def test_phase_065_state_decay_smoothing() -> None:
    """Certifies Phase 065: Historical tokens decay exponentially with factor gamma = 0.99."""
    gamma = 0.99
    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=64,
        decoupled_rope_dim=32,
        decay=gamma,
    )
    state = bridge.init_latent_state(batch_size=1, device=torch.device("cpu"))

    c_0 = torch.ones(1, 1, 64)
    k_0 = torch.ones(1, 1, 32)
    qc_0 = torch.ones(1, 1, 64)
    qr_0 = torch.ones(1, 1, 32)

    _, (sc_1, _) = bridge.forward_decode_step(c_0, k_0, qc_0, qr_0, state.state_c, state.state_r)

    # Inactive inputs at t=1 (F.elu(-20) + 1.0 ~ 0)
    c_zero = torch.full((1, 1, 64), -20.0)
    qc_zero = torch.full((1, 1, 64), -20.0)
    k_zero = torch.zeros(1, 1, 32)
    qr_zero = torch.zeros(1, 1, 32)

    _, (sc_2, _) = bridge.forward_decode_step(c_zero, k_zero, qc_zero, qr_zero, sc_1, state.state_r)

    expected_sc_2 = gamma * sc_1
    diff = torch.max(torch.abs(sc_2 - expected_sc_2)).item()
    assert diff < 1e-5, f"State decay mismatch: diff={diff}"


# =========================================================================
# Phase 066: Parallel Prefill Chunk Associative Scan Operator
# =========================================================================
def test_phase_066_parallel_prefill_chunk_scan() -> None:
    """Certifies Phase 066: Chunked associative scan processes variable sequences in parallel chunks."""
    torch.manual_seed(42)
    bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=64, decoupled_rope_dim=32, decay=0.98)
    batch_size = 2
    seq_len = 64
    chunk_size = 16

    c_kv = torch.randn(batch_size, seq_len, 64)
    k_rope = torch.randn(batch_size, seq_len, 32)
    qc = torch.randn(batch_size, seq_len, 64)
    qr = torch.randn(batch_size, seq_len, 32)

    out, (sc, sr) = bridge.forward_prefill(c_kv, k_rope, qc, qr, chunk_size=chunk_size, causal=True)
    assert out.shape == (batch_size, seq_len, 64)
    assert sc.shape == (batch_size, 64, 64)
    assert sr.shape == (batch_size, 32, 32)
    assert not torch.isnan(out).any()


# =========================================================================
# Phase 067: Autoregressive Decode Step O(1) State Transition
# =========================================================================
def test_phase_067_autoregressive_decode_step() -> None:
    """Certifies Phase 067: Single-token decode execution maintains O(1) constant state shape."""
    bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=64, decoupled_rope_dim=32)
    batch_size = 2
    state = bridge.init_latent_state(batch_size=batch_size, device=torch.device("cpu"))

    for _ in range(10):
        c_step = torch.randn(batch_size, 1, 64)
        k_step = torch.randn(batch_size, 1, 32)
        qc_step = torch.randn(batch_size, 1, 64)
        qr_step = torch.randn(batch_size, 1, 32)

        out, (state_c, state_r) = bridge.forward_decode_step(
            c_step, k_step, qc_step, qr_step, state.state_c, state.state_r
        )
        assert out.shape == (batch_size, 1, 64)
        assert state_c.shape == (batch_size, 64, 64)
        assert state_r.shape == (batch_size, 32, 32)
        state = DualStreamLatentState(state_c=state_c, state_r=state_r)


# =========================================================================
# Phase 068: Compressed Output Reconstruction with W_{UKV} Fusion
# =========================================================================
def test_phase_068_compressed_output_reconstruction_w_ukv() -> None:
    """Certifies Phase 068: Multi-head output reconstructed via W_{UKV} fusion to [B, T, D_model]."""
    hidden_dim = 256
    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=512,
        decoupled_rope_dim=64,
        q_latent_dim=128,
        hidden_dim=hidden_dim,
    )
    assert bridge.w_ukv is not None
    assert bridge.W_UKV is not None

    batch_size = 2
    seq_len = 5
    out_c = torch.randn(batch_size, seq_len, 512)
    out_r = torch.randn(batch_size, seq_len, 64)

    recon = bridge.reconstruct_output(out_c, out_r)
    assert recon.shape == (batch_size, seq_len, hidden_dim)

    # Test standalone forward from hidden_states
    x = torch.randn(batch_size, seq_len, hidden_dim)
    y, _ = bridge(x)
    assert y.shape == (batch_size, seq_len, hidden_dim)


# =========================================================================
# Phase 069: Prefill vs Decode Step-1 Numerical Boundary Parity Gate (< 1e-6)
# =========================================================================
def test_phase_069_prefill_decode_step1_parity_gate() -> None:
    """Certifies Phase 069: Step-1 numerical difference |out_prefill - out_decode| < 1e-6 and multi-token parity."""
    torch.manual_seed(42)
    latent_kv_dim = 128
    decoupled_rope_dim = 32
    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=latent_kv_dim,
        decoupled_rope_dim=decoupled_rope_dim,
        decay=0.99,
    )

    batch_size = 2
    # Single token T=1 in FP32
    c_kv = torch.randn(batch_size, 1, latent_kv_dim, dtype=torch.float32)
    k_rope = torch.randn(batch_size, 1, decoupled_rope_dim, dtype=torch.float32)
    q_content = torch.randn(batch_size, 1, latent_kv_dim, dtype=torch.float32)
    q_rope = torch.randn(batch_size, 1, decoupled_rope_dim, dtype=torch.float32)

    # 1. Parallel prefill at T=1
    out_prefill, (sc_pref, sr_pref) = bridge.forward_prefill(c_kv, k_rope, q_content, q_rope)

    # 2. Decode step from zero state at T=1
    init_state = bridge.init_latent_state(batch_size=batch_size, device=c_kv.device)
    out_decode, (sc_dec, sr_dec) = bridge.forward_decode_step(
        c_kv, k_rope, q_content, q_rope, init_state.state_c, init_state.state_r
    )

    diff_out = torch.max(torch.abs(out_prefill - out_decode)).item()
    diff_sc = torch.max(torch.abs(sc_pref - sc_dec)).item()
    diff_sr = torch.max(torch.abs(sr_pref - sr_dec)).item()

    assert diff_out < 1e-6, f"Step 1 output parity exceeded 1e-6: diff={diff_out}"
    assert diff_sc < 1e-6, f"Step 1 content state parity exceeded 1e-6: diff={diff_sc}"
    assert diff_sr < 1e-6, f"Step 1 RoPE state parity exceeded 1e-6: diff={diff_sr}"

    # Multi-token causal parity verification across T=8 tokens
    seq_len = 8
    c_seq = torch.randn(batch_size, seq_len, latent_kv_dim)
    k_seq = torch.randn(batch_size, seq_len, decoupled_rope_dim)
    qc_seq = torch.randn(batch_size, seq_len, latent_kv_dim)
    qr_seq = torch.randn(batch_size, seq_len, decoupled_rope_dim)

    prefill_out, (pref_sc, pref_sr) = bridge.forward_prefill(c_seq, k_seq, qc_seq, qr_seq, causal=True)

    curr_sc, curr_sr = init_state.state_c, init_state.state_r
    decode_outs = []
    for t in range(seq_len):
        step_out, (curr_sc, curr_sr) = bridge.forward_decode_step(
            c_seq[:, t : t + 1],
            k_seq[:, t : t + 1],
            qc_seq[:, t : t + 1],
            qr_seq[:, t : t + 1],
            curr_sc,
            curr_sr,
        )
        decode_outs.append(step_out)
    all_decode_out = torch.cat(decode_outs, dim=1)

    multi_step_diff = torch.max(torch.abs(prefill_out - all_decode_out)).item()
    assert multi_step_diff < 1e-5, f"Multi-step causal prefill vs decode mismatch: {multi_step_diff}"
    assert torch.max(torch.abs(pref_sc - curr_sc)).item() < 1e-5
    assert torch.max(torch.abs(pref_sr - curr_sr)).item() < 1e-5


# =========================================================================
# Phase 070: 100-Step Autoregressive Memory Invariance Certification (0.0% growth)
# =========================================================================
def test_phase_070_100_step_autoregressive_memory_invariance() -> None:
    """Certifies Phase 070: Memory growth between step 10 and step 100 is strictly 0.0%."""
    torch.manual_seed(42)
    bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=512, decoupled_rope_dim=64, q_latent_dim=128)
    batch_size = 4
    state = bridge.init_latent_state(batch_size=batch_size, device=torch.device("cpu"))

    state_memory_log = {}

    for step in range(1, 101):
        c_step = torch.randn(batch_size, 1, 512)
        k_step = torch.randn(batch_size, 1, 64)
        qc_step = torch.randn(batch_size, 1, 128)
        qr_step = torch.randn(batch_size, 1, 64)

        out, (new_sc, new_sr) = bridge.forward_decode_step(
            c_step, k_step, qc_step, qr_step, state.state_c, state.state_r
        )
        state = DualStreamLatentState(state_c=new_sc, state_r=new_sr)

        if step in (1, 10, 50, 100):
            # Calculate bytes occupied by active state tensors
            bytes_sc = new_sc.element_size() * new_sc.nelement()
            bytes_sr = new_sr.element_size() * new_sr.nelement()
            total_bytes = bytes_sc + bytes_sr
            state_memory_log[step] = total_bytes

    bytes_step10 = state_memory_log[10]
    bytes_step100 = state_memory_log[100]
    memory_growth = (bytes_step100 - bytes_step10) / float(bytes_step10)

    assert bytes_step10 == bytes_step100, f"State byte mismatch: step 10={bytes_step10}, step 100={bytes_step100}"
    assert memory_growth == 0.0, f"Memory growth was not 0.0%: {memory_growth * 100}%"


# =========================================================================
# Phase 071: Attention Distribution Entropy Metric Formulator
# =========================================================================
def test_phase_071_attention_entropy_theoretical_bounds() -> None:
    """Certifies Phase 071: Attention entropy equals ln(T) for uniform and 0.0 for one-hot."""
    seq_len = 16
    # 1. Uniform attention matrix: A_{i, j} = 1/T
    uniform_attn = torch.full((1, 1, seq_len, seq_len), 1.0 / seq_len)
    h_uniform = compute_attention_entropy(uniform_attn)
    expected_uniform = math.log(seq_len)
    assert abs(h_uniform - expected_uniform) < 1e-4, f"Uniform entropy mismatch: {h_uniform} vs {expected_uniform}"

    # 2. Sharp one-hot attention: A_{i, j} = delta_{i, j}
    one_hot_attn = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)
    h_one_hot = compute_attention_entropy(one_hot_attn)
    assert abs(h_one_hot - 0.0) < 1e-5, f"One-hot entropy mismatch: {h_one_hot}"


# =========================================================================
# Phase 072: Layer Sensitivity Profiler for Syntactic vs Semantic Routing
# =========================================================================
def test_phase_072_layer_sensitivity_profiler() -> None:
    """Certifies Phase 072: Profiler detects high entropy in boundary layers vs intermediate."""
    num_layers = 16
    seq_len = 8
    attn_list = []
    for l in range(num_layers):
        if l in (0, 1, 2, 3, num_layers - 2, num_layers - 1):
            # High entropy (uniform)
            attn = torch.full((1, 1, seq_len, seq_len), 1.0 / seq_len)
        else:
            # Low entropy (one-hot)
            attn = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)
        attn_list.append(attn)

    profile = profile_layer_sensitivity(attn_list)
    assert profile["num_layers"] == num_layers
    assert profile["mean_boundary_entropy"] > profile["mean_intermediate_entropy"]
    assert profile["mean_intermediate_entropy"] < 1e-4


# =========================================================================
# Phase 073: Boundary Layer Safeguard (Layers 0-3 Full Attention)
# =========================================================================
def test_phase_073_boundary_layer_safeguard_layers_0_to_3() -> None:
    """Certifies Phase 073: Layers 0, 1, 2, 3 retain Full Attention."""
    router = HybridMoARouter(num_layers=32)
    for l in (0, 1, 2, 3):
        assert router.is_full_attention(l) is True
        assert router.get_layer_type(l) == "full_attention"


# =========================================================================
# Phase 074: Factual Logit Safeguard (Layers 30-31 Full Attention)
# =========================================================================
def test_phase_074_factual_logit_safeguard_layers_30_31() -> None:
    """Certifies Phase 074: Layers 30 and 31 retain Full Attention."""
    router = HybridMoARouter(num_layers=32)
    assert router.is_full_attention(30) is True
    assert router.is_full_attention(31) is True
    assert router.get_layer_type(30) == "full_attention"
    assert router.get_layer_type(31) == "full_attention"


# =========================================================================
# Phase 075: Intermediate Bulk Linear Attention Router (Layers 4-29, 26 layers)
# =========================================================================
def test_phase_075_intermediate_bulk_linear_router() -> None:
    """Certifies Phase 075: Exactly 26 layers (4-29) allocated to Linear Attention in L=32."""
    router = HybridMoARouter(num_layers=32)
    linear_layers = router.get_linear_attention_layers()
    assert len(linear_layers) == 26
    assert linear_layers == list(range(4, 30))
    for l in range(4, 30):
        assert router.is_linear_attention(l) is True
        assert router.get_layer_type(l) == "linear_attention"


# =========================================================================
# Phase 076: Pareto Frontier Latency-vs-Perplexity Solver
# =========================================================================
def test_phase_076_pareto_frontier_solver() -> None:
    """Certifies Phase 076: Discovers Pareto optimal configuration with Delta PPL <= 0.15."""
    router, meta = HybridMoARouter.solve_pareto_frontier(
        num_layers=32,
        max_ppl_delta=0.15,
        min_full_layers=6,
    )
    assert meta["pareto_optimal"] is True
    assert meta["accumulated_ppl_delta"] <= 0.15
    assert meta["speedup_factor"] > 1.5
    # Safeguard layers must remain full
    for l in (0, 1, 2, 3, 30, 31):
        assert router.is_full_attention(l) is True


# =========================================================================
# Phase 077: Dynamic Heterogeneous Layer Schedule Generator
# =========================================================================
def test_phase_077_dynamic_schedule_generator_various_depths() -> None:
    """Certifies Phase 077: Generates valid schedules across 12, 24, 32, and 48 layers."""
    for depth in (12, 24, 32, 48):
        router = HybridMoARouter(num_layers=depth)
        schedule = router.partition_schedule()
        assert len(schedule) == depth
        assert all(t in ("full_attention", "linear_attention") for t in schedule.values())
        # First and last layers are always full attention
        assert schedule[0] == "full_attention"
        assert schedule[depth - 1] == "full_attention"


# =========================================================================
# Phase 078: Endoskeletal Shell Bridge Dispatch Hook
# =========================================================================
def test_phase_078_endoskeletal_shell_bridge_dispatch_hook() -> None:
    """Certifies Phase 078: UniversalAttentionBridge dispatches between full and linear layers."""
    d_model = 64
    num_heads = 4

    class MockFull(nn.Module):
        def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return x * 100.0

    class MockLinear(nn.Module):
        def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return x * 2.0

    orig = MockFull()
    lin = MockLinear()
    spec = AttentionTargetSpec(
        module_path="model.layers.0.self_attn",
        parent_module=nn.Module(),
        attr_name="self_attn",
        target_module=orig,
        pattern_type="split_4proj",
        embed_dim=d_model,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=d_model // num_heads,
        is_causal=True,
        has_bias=False,
    )

    router = HybridMoARouter(num_layers=32)

    # Layer 0: Full attention (retains orig)
    b0 = UniversalAttentionBridge(orig, spec, lin, inherit_weights=False, moa_router=router, layer_idx=0)
    # Layer 5: Linear attention (routes to lin)
    b5 = UniversalAttentionBridge(orig, spec, lin, inherit_weights=False, moa_router=router, layer_idx=5)

    x = torch.ones(2, 4, d_model)
    assert torch.allclose(b0(x), x * 100.0)
    assert torch.allclose(b5(x), x * 2.0)

    # Construct entire 32-layer heterogeneous model pipeline
    layers = []
    for l_idx in range(32):
        layer_spec = AttentionTargetSpec(
            module_path=f"model.layers.{l_idx}.self_attn",
            parent_module=nn.Module(),
            attr_name="self_attn",
            target_module=MockFull(),
            pattern_type="split_4proj",
            embed_dim=d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=d_model // num_heads,
            is_causal=True,
            has_bias=False,
        )
        bridge_layer = UniversalAttentionBridge(
            orig_module=MockFull(),
            spec=layer_spec,
            new_attention=MockLinear(),
            inherit_weights=False,
            moa_router=router,
            layer_idx=l_idx,
        )
        layers.append(bridge_layer)

    # Execute 32-layer pipeline pass
    cur = torch.ones(1, 2, d_model)
    full_count = 0
    linear_count = 0
    for l_idx, layer in enumerate(layers):
        out = layer(cur)
        if router.is_full_attention(l_idx):
            full_count += 1
            assert torch.allclose(out, cur * 100.0)
        else:
            linear_count += 1
            assert torch.allclose(out, cur * 2.0)

    assert full_count == 6
    assert linear_count == 26


def test_phase_078_hf_cache_bridge_interoperability() -> None:
    """Certifies Phase 078: Seamless interoperability between UniversalAttentionBridge MLA and NeuralOpsHybridHFCache."""
    d_model = 64
    d_c = 32
    d_r = 16

    class MockOrigMLA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kv_a_proj_with_mqa = nn.Linear(d_model, d_c + d_r, bias=False)
            self.q_a_proj = nn.Linear(d_model, d_c + d_r, bias=False)

        def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return x

    orig = MockOrigMLA()
    mla_bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=d_c, decoupled_rope_dim=d_r)
    spec = AttentionTargetSpec(
        module_path="model.layers.4.self_attn",
        parent_module=nn.Module(),
        attr_name="self_attn",
        target_module=orig,
        pattern_type="deepseek_mla",
        embed_dim=d_model,
        num_heads=4,
        num_kv_heads=1,
        head_dim=16,
        is_causal=True,
        has_bias=False,
    )

    bridge_layer = UniversalAttentionBridge(
        orig_module=orig,
        spec=spec,
        new_attention=mla_bridge,
        inherit_weights=False,
        layer_idx=4,
    )

    hf_cache = NeuralOpsHybridHFCache()

    # Step 1: Prefill phase with sequence length 4
    x_prefill = torch.randn(1, 4, d_model)
    out_prefill, _, cache_after_pref = bridge_layer(
        x_prefill,
        past_key_value=hf_cache,
        use_cache=True,
    )
    assert out_prefill.shape == (1, 4, d_c)
    assert cache_after_pref is hf_cache
    state_layer4 = hf_cache.get_recurrent_state(4)
    assert state_layer4 is not None
    assert isinstance(state_layer4, tuple)
    sc, sr = state_layer4
    assert sc.shape == (1, d_c, d_c)
    assert sr.shape == (1, d_r, d_r)

    # Step 2: Decode step with single token (sequence length 1)
    x_decode = torch.randn(1, 1, d_model)
    out_decode, _, cache_after_dec = bridge_layer(
        x_decode,
        past_key_value=hf_cache,
        use_cache=True,
    )
    assert out_decode.shape == (1, 1, d_c)
    assert cache_after_dec is hf_cache
    new_state_l4 = hf_cache.get_recurrent_state(4)
    assert new_state_l4 is not None
    assert new_state_l4[0].shape == (1, d_c, d_c)
    assert new_state_l4[1].shape == (1, d_r, d_r)


# =========================================================================
# Phase 079: Weight Inheritance Engine for Low-Rank Projections
# =========================================================================
def test_phase_079_mla_weight_inheritance_engine() -> None:
    """Certifies Phase 079: Pretrained MLA projection weights inherit with zero shape mismatch."""
    d_model = 128
    d_c = 512
    d_r = 64

    class MockPretrainedMLA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kv_a_proj_with_mqa = nn.Linear(d_model, d_c + d_r, bias=False)
            self.q_a_proj = nn.Linear(d_model, d_c + d_r, bias=False)
            self.o_proj = nn.Linear(d_c, d_model, bias=False)

    orig_mla = MockPretrainedMLA()
    nn.init.orthogonal_(orig_mla.kv_a_proj_with_mqa.weight)
    nn.init.orthogonal_(orig_mla.q_a_proj.weight)
    nn.init.orthogonal_(orig_mla.o_proj.weight)

    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=d_c,
        decoupled_rope_dim=d_r,
        q_latent_dim=d_c,
        hidden_dim=d_model,
    )

    bridge.inherit_from_mla_module(orig_mla)

    # Verify weights copied with zero shape mismatch
    assert torch.allclose(
        bridge.kv_latent_proj.weight.data,
        orig_mla.kv_a_proj_with_mqa.weight.data[:d_c, :],
    )
    assert torch.allclose(
        bridge.k_rope_proj.weight.data,
        orig_mla.kv_a_proj_with_mqa.weight.data[d_c : d_c + d_r, :],
    )
    assert torch.allclose(
        bridge.w_ukv.weight.data,
        orig_mla.o_proj.weight.data,
    )


# =========================================================================
# Phase 080: Multi-Seed Statistical Significance Test Suite (Seeds 42, 123, 999)
# =========================================================================
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_phase_080_multi_seed_statistical_significance(seed: int) -> None:
    """Certifies Phase 080: Multi-seed execution passes with zero flakiness across all seeds."""
    torch.manual_seed(seed)
    latent_kv_dim = 64
    decoupled_rope_dim = 32

    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=latent_kv_dim,
        decoupled_rope_dim=decoupled_rope_dim,
        decay=0.99,
    )

    batch_size = 2
    c_kv = torch.randn(batch_size, 1, latent_kv_dim)
    k_rope = torch.randn(batch_size, 1, decoupled_rope_dim)
    qc = torch.randn(batch_size, 1, latent_kv_dim)
    qr = torch.randn(batch_size, 1, decoupled_rope_dim)

    # 1. Prefill step 1
    out_pref, _ = bridge.forward_prefill(c_kv, k_rope, qc, qr)

    # 2. Decode step 1
    init_state = bridge.init_latent_state(batch_size=batch_size, device=c_kv.device)
    out_dec, _ = bridge.forward_decode_step(
        c_kv, k_rope, qc, qr, init_state.state_c, init_state.state_r
    )

    diff = torch.max(torch.abs(out_pref - out_dec)).item()
    assert diff < 1e-6, f"Seed {seed} step-1 parity failed: diff={diff}"

    # 3. MoA router stability
    router = HybridMoARouter(num_layers=32)
    assert len(router.get_full_attention_layers()) == 6
    assert len(router.get_linear_attention_layers()) == 26


def test_phase_080_aggregate_variance_across_seeds() -> None:
    """Certifies Phase 080 aggregate statistics: mean error < 1e-6, variance < 1e-12."""
    seeds = [42, 123, 999]
    diffs = []
    latent_kv_dim = 64
    decoupled_rope_dim = 32
    bridge = DeepSeekMLASurgeryBridge(latent_kv_dim=latent_kv_dim, decoupled_rope_dim=decoupled_rope_dim)

    for s in seeds:
        torch.manual_seed(s)
        c_kv = torch.randn(1, 1, latent_kv_dim)
        k_rope = torch.randn(1, 1, decoupled_rope_dim)
        qc = torch.randn(1, 1, latent_kv_dim)
        qr = torch.randn(1, 1, decoupled_rope_dim)

        out_pref, _ = bridge.forward_prefill(c_kv, k_rope, qc, qr)
        init_state = bridge.init_latent_state(batch_size=1, device=c_kv.device)
        out_dec, _ = bridge.forward_decode_step(c_kv, k_rope, qc, qr, init_state.state_c, init_state.state_r)
        diff = torch.max(torch.abs(out_pref - out_dec)).item()
        diffs.append(diff)

    mean_diff = sum(diffs) / len(diffs)
    variance_diff = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)

    assert mean_diff < 1e-6, f"Mean error across seeds exceeded threshold: {mean_diff}"
    assert variance_diff < 1e-12, f"Variance across seeds exceeded threshold: {variance_diff}"
