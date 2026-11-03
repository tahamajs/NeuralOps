#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and Regression Tests for DeepSeek Multi-Head Latent Attention (MLA) Surgery
and Hybrid Mixture of Attentions (MoA) Pareto Router.

Validates:
    1. Step-1 boundary numerical equivalence between parallel prefill and recurrent decoding.
    2. Zero memory growth (O(1) state footprint) across 100 autoregressive decoding steps.
    3. Dual-stream latent state updates (Content Stream S_t^C, Decoupled RoPE Stream z_t^R).
    4. Hybrid MoA Pareto optimal 6/26 layer partitioning.
    5. Attention entropy-based dynamic layer allocation.
    6. Seamless integration with UniversalAttentionBridge and Endoskeletal Shell.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.surgery.endoskeletal_shell import UniversalAttentionBridge
from neuralops.surgery.mla_surgery import DeepSeekMLASurgeryBridge, DualStreamLatentState
from neuralops.surgery.moa_router import HybridMoARouter
from neuralops.surgery.scanner import AttentionTargetSpec


def test_mla_prefill_decode_step1_parity() -> None:
    """Verifies that forward_prefill at T=1 exactly matches forward_decode_step from zero state."""
    torch.manual_seed(42)
    latent_kv_dim = 128
    decoupled_rope_dim = 32
    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=latent_kv_dim,
        decoupled_rope_dim=decoupled_rope_dim,
        decay=0.99,
    )

    batch_size = 2
    # Single token sequence T=1
    c_kv = torch.randn(batch_size, 1, latent_kv_dim)
    k_rope = torch.randn(batch_size, 1, decoupled_rope_dim)
    q_content = torch.randn(batch_size, 1, latent_kv_dim)
    q_rope = torch.randn(batch_size, 1, decoupled_rope_dim)

    # 1. Execute parallel prefill
    out_prefill, (s_c_prefill, z_r_prefill) = bridge.forward_prefill(
        c_kv=c_kv,
        k_rope=k_rope,
        q_content=q_content,
        q_rope=q_rope,
    )

    # 2. Execute recurrent decode step starting from zero state
    init_state = bridge.init_latent_state(
        batch_size=batch_size,
        device=c_kv.device,
        dtype=c_kv.dtype,
    )
    out_decode, (s_c_decode, z_r_decode) = bridge.forward_decode_step(
        c_kv_step=c_kv,
        k_rope_step=k_rope,
        q_content_step=q_content,
        q_rope_step=q_rope,
        state_c=init_state.state_c,
        state_r=init_state.state_r,
    )

    # 3. Assert mathematical equivalence
    diff_out = torch.max(torch.abs(out_prefill - out_decode)).item()
    diff_s_c = torch.max(torch.abs(s_c_prefill - s_c_decode)).item()
    diff_z_r = torch.max(torch.abs(z_r_prefill - z_r_decode)).item()

    assert diff_out < 1e-5, f"Step 1 output parity mismatch: max diff = {diff_out}"
    assert diff_s_c < 1e-5, f"Step 1 content state parity mismatch: max diff = {diff_s_c}"
    assert diff_z_r < 1e-5, f"Step 1 RoPE state parity mismatch: max diff = {diff_z_r}"


def test_mla_autoregressive_constant_memory() -> None:
    """Verifies that 100 decoding steps maintain strictly constant O(1) state memory footprint."""
    torch.manual_seed(42)
    latent_kv_dim = 64
    decoupled_rope_dim = 32
    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=latent_kv_dim,
        decoupled_rope_dim=decoupled_rope_dim,
        decay=0.95,
    )

    batch_size = 4
    state = bridge.init_latent_state(
        batch_size=batch_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_s_shape = (batch_size, latent_kv_dim, latent_kv_dim)
    expected_z_shape = (batch_size, decoupled_rope_dim, decoupled_rope_dim)

    # Run 100 autoregressive steps
    for step in range(100):
        c_step = torch.randn(batch_size, 1, latent_kv_dim)
        k_step = torch.randn(batch_size, 1, decoupled_rope_dim)
        qc_step = torch.randn(batch_size, 1, latent_kv_dim)
        qr_step = torch.randn(batch_size, 1, decoupled_rope_dim)

        out, (state_c, state_r) = bridge.forward_decode_step(
            c_kv_step=c_step,
            k_rope_step=k_step,
            q_content_step=qc_step,
            q_rope_step=qr_step,
            state_c=state.state_c,
            state_r=state.state_r,
        )

        assert out.shape == (batch_size, 1, latent_kv_dim)
        assert state_c.shape == expected_s_shape
        assert state_r.shape == expected_z_shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        # Update state container
        state = DualStreamLatentState(state_c=state_c, state_r=state_r)


def test_dual_stream_latent_recurrence_decay() -> None:
    """Verifies that the exponential decay factor gamma dampens historical content states."""
    decay = 0.9
    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=32,
        decoupled_rope_dim=16,
        decay=decay,
    )

    batch_size = 1
    state = bridge.init_latent_state(batch_size=1, device=torch.device("cpu"))

    # Non-zero input at t=0
    c_0 = torch.ones(batch_size, 1, 32)
    k_0 = torch.ones(batch_size, 1, 16)
    qc_0 = torch.ones(batch_size, 1, 32)
    qr_0 = torch.ones(batch_size, 1, 16)

    _, (s_c_1, _) = bridge.forward_decode_step(
        c_0, k_0, qc_0, qr_0, state.state_c, state.state_r
    )

    # Zero input at t=1: s_c_2 should equal decay * s_c_1 + k_phi^T @ q_phi
    c_1 = torch.full((batch_size, 1, 32), -10.0)  # ELU(-10)+1 ~= 0
    qc_1 = torch.full((batch_size, 1, 32), -10.0)
    k_1 = torch.zeros(batch_size, 1, 16)
    qr_1 = torch.zeros(batch_size, 1, 16)

    _, (s_c_2, _) = bridge.forward_decode_step(
        c_1, k_1, qc_1, qr_1, s_c_1, state.state_r
    )

    # s_c_2 should be very close to decay * s_c_1
    expected_s_c_2 = decay * s_c_1
    diff = torch.max(torch.abs(s_c_2 - expected_s_c_2)).item()
    assert diff < 1e-4, f"State decay mismatch: max diff = {diff}"


def test_hybrid_moa_router_pareto_partition() -> None:
    """Verifies Pareto optimal layer allocation for L=32: exactly 6 full and 26 linear attention layers."""
    router = HybridMoARouter(num_layers=32)
    schedule = router.partition_schedule()

    assert router.num_layers == 32
    assert len(schedule) == 32

    full_layers = router.get_full_attention_layers()
    linear_layers = router.get_linear_attention_layers()

    # Exact Pareto partition contract: layers 0-3 and 30-31 retain Full Attention
    expected_full = [0, 1, 2, 3, 30, 31]
    expected_linear = list(range(4, 30))

    assert full_layers == expected_full
    assert linear_layers == expected_linear
    assert len(full_layers) == 6
    assert len(linear_layers) == 26

    # Verify query predicates
    for l in expected_full:
        assert router.is_full_attention(l) is True
        assert router.is_linear_attention(l) is False
        assert router.get_layer_type(l) == "full_attention"

    for l in expected_linear:
        assert router.is_full_attention(l) is False
        assert router.is_linear_attention(l) is True
        assert router.get_layer_type(l) == "linear_attention"


def test_hybrid_moa_router_small_model_partition() -> None:
    """Verifies boundary partition behavior when total layers <= 6."""
    router = HybridMoARouter(num_layers=4)
    full_layers = router.get_full_attention_layers()
    linear_layers = router.get_linear_attention_layers()

    assert full_layers == [0, 3]
    assert linear_layers == [1, 2]


def test_hybrid_moa_router_custom_partition() -> None:
    """Verifies explicit custom full attention layer assignments."""
    router = HybridMoARouter(num_layers=12, full_attention_layers=[0, 6, 11])
    assert router.get_full_attention_layers() == [0, 6, 11]
    assert len(router.get_linear_attention_layers()) == 9

    with pytest.raises(ValueError):
        HybridMoARouter(num_layers=12, full_attention_layers=[-1, 5])


def test_hybrid_moa_entropy_allocation() -> None:
    """Verifies dynamic entropy-based Pareto allocation across simulated attention layers."""
    torch.manual_seed(42)
    num_layers = 8
    seq_len = 16

    # Create synthetic attention matrices with different entropies
    attn_matrices = []
    for l in range(num_layers):
        if l in (0, 7):
            # High entropy (uniform distribution)
            logits = torch.zeros(1, 4, seq_len, seq_len)
        else:
            # Low entropy (concentrated distribution)
            logits = torch.randn(1, 4, seq_len, seq_len) * 5.0
        attn = torch.softmax(logits, dim=-1)
        attn_matrices.append(attn)

    router = HybridMoARouter.compute_entropy_partition(
        attn_matrices,
        num_full_layers=2,
    )

    full_layers = router.get_full_attention_layers()
    assert len(full_layers) == 2
    # Layers 0 and 7 have highest entropy and must be selected
    assert 0 in full_layers
    assert 7 in full_layers


def test_endoskeletal_shell_hybrid_moa_integration() -> None:
    """Verifies that UniversalAttentionBridge honors HybridMoARouter decisions."""
    d_model = 64
    num_heads = 4

    class MockFullAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return x * 10.0  # Distinguishable marker

    class MockLinearAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return x * 2.0  # Distinguishable marker

    orig_layer = MockFullAttention()
    linear_layer = MockLinearAttention()
    spec = AttentionTargetSpec(
        module_path="layers.0.self_attn",
        parent_module=nn.Module(),
        attr_name="self_attn",
        target_module=orig_layer,
        pattern_type="split_4proj",
        embed_dim=d_model,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=d_model // num_heads,
        is_causal=True,
        has_bias=False,
    )

    router = HybridMoARouter(num_layers=10, full_attention_layers=[0, 9])

    # Bridge for Layer 0 (Full Attention)
    bridge_l0 = UniversalAttentionBridge(
        orig_module=orig_layer,
        spec=spec,
        new_attention=linear_layer,
        inherit_weights=False,
        moa_router=router,
        layer_idx=0,
    )

    # Bridge for Layer 1 (Linear Attention)
    bridge_l1 = UniversalAttentionBridge(
        orig_module=orig_layer,
        spec=spec,
        new_attention=linear_layer,
        inherit_weights=False,
        moa_router=router,
        layer_idx=1,
    )

    x = torch.ones(2, 8, d_model)
    out_l0 = bridge_l0(x)
    out_l1 = bridge_l1(x)

    # Layer 0 routed to orig_layer (x * 10.0)
    assert torch.allclose(out_l0, x * 10.0)
    # Layer 1 routed to linear_layer (x * 2.0)
    assert torch.allclose(out_l1, x * 2.0)


def test_endoskeletal_shell_deepseek_mla_integration() -> None:
    """Verifies that UniversalAttentionBridge routes MLA inputs through DeepSeekMLASurgeryBridge."""
    d_c = 32
    d_r = 16
    bridge_mla = DeepSeekMLASurgeryBridge(
        latent_kv_dim=d_c,
        decoupled_rope_dim=d_r,
    )

    class MockOrigDeepSeekAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kv_a_proj_with_mqa = nn.Linear(64, d_c + d_r, bias=False)
            self.q_a_proj = nn.Linear(64, d_c + d_r, bias=False)

    orig_mla = MockOrigDeepSeekAttn()
    spec = AttentionTargetSpec(
        module_path="model.layers.0.self_attn",
        parent_module=nn.Module(),
        attr_name="self_attn",
        target_module=orig_mla,
        pattern_type="deepseek_mla",
        embed_dim=64,
        num_heads=8,
        num_kv_heads=1,
        head_dim=16,
        is_causal=True,
        has_bias=False,
    )

    bridge = UniversalAttentionBridge(
        orig_module=orig_mla,
        spec=spec,
        new_attention=bridge_mla,
        inherit_weights=False,
    )

    x = torch.randn(2, 10, 64)
    out = bridge(x)
    assert out.shape == (2, 10, d_c)
    assert not torch.isnan(out).any()


def test_mla_prefill_chunk_scan() -> None:
    """Verifies that chunked associative scan in forward_prefill processes variable sequence chunks."""
    torch.manual_seed(42)
    latent_kv_dim = 32
    decoupled_rope_dim = 16
    bridge = DeepSeekMLASurgeryBridge(
        latent_kv_dim=latent_kv_dim,
        decoupled_rope_dim=decoupled_rope_dim,
        decay=0.98,
    )

    batch_size = 2
    seq_len = 32
    chunk_size = 8

    c_kv = torch.randn(batch_size, seq_len, latent_kv_dim)
    k_rope = torch.randn(batch_size, seq_len, decoupled_rope_dim)
    q_content = torch.randn(batch_size, seq_len, latent_kv_dim)
    q_rope = torch.randn(batch_size, seq_len, decoupled_rope_dim)

    # Chunked prefill
    out_chunked, (s_c_chunk, z_r_chunk) = bridge.forward_prefill(
        c_kv=c_kv,
        k_rope=k_rope,
        q_content=q_content,
        q_rope=q_rope,
        chunk_size=chunk_size,
    )

    assert out_chunked.shape == (batch_size, seq_len, latent_kv_dim)
    assert s_c_chunk.shape == (batch_size, latent_kv_dim, latent_kv_dim)
    assert z_r_chunk.shape == (batch_size, decoupled_rope_dim, decoupled_rope_dim)
    assert not torch.isnan(out_chunked).any()

